import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import Fastify from "fastify";
import Docker from "dockerode";
import { z } from "zod";
import { CommandRequestSchema } from "./assets/request_headers";
import { logError, logInfo } from "./shared/logging";

type RegistryEntry = { summary: string; cli: string };
type ActiveServer = {
  port: number;
  container: Docker.Container;
  status: "active" | "stopping";
  counter: number;
};

const OrchestratorResultSchema = z.object({
  status: z.number(),
  result: z.string().nullable(),
  info: z.string().optional(),
  id: z.string(),
});

const OrchestratorErrorSchema = z.object({
  status: z.number(),
  detail: z.string(),
  id: z.string(),
});

function sleep(ms: number) {
  return new Promise((r) => setTimeout(r, ms));
}

async function fetchJson(url: string, init?: RequestInit): Promise<any> {
  const res = await fetch(url, init);
  const text = await res.text();
  let json: any = null;
  try {
    json = text ? JSON.parse(text) : null;
  } catch {
    // ignore
  }
  if (!res.ok) {
    const detail = json?.detail ?? text ?? `HTTP ${res.status}`;
    const err: any = new Error(detail);
    err.status = res.status;
    err.detail = detail;
    throw err;
  }
  return json;
}

export class AIOrchestrator {
  docker = new Docker(); // /var/run/docker.sock
  registry: Record<string, RegistryEntry> = {};
  active_servers: Record<string, ActiveServer> = {};
  locks: Record<string, Promise<void>> = {};
  private next_port = 8001;

  constructor() {
    const docsPath = path.join(process.cwd(), "src", "cli_server", "docs");
    if (fs.existsSync(docsPath)) {
      for (const filename of fs.readdirSync(docsPath)) {
        const filePath = path.join(docsPath, filename);
        try {
          const content = fs.readFileSync(filePath, "utf8").split(/\r?\n/);
          const name = filename.split(".")[0];
          this.registry[name] = { summary: (content[0] ?? "").trim(), cli: (content[1] ?? "").trim() };
          logInfo("orchestrator", `Server summary and cli path read from ${filename}`);
        } catch (e: any) {
          logError("orchestrator", `Could not read ${filename}: ${e?.message ?? String(e)}`);
        }
      }
    } else {
      logWarnMaybe(`Docs path not found: ${docsPath}`);
    }
  }

  private async withLock<T>(name: string, fn: () => Promise<T>): Promise<T> {
    const prev = this.locks[name] ?? Promise.resolve();
    let release: () => void = () => {};
    const cur = new Promise<void>((r) => (release = r));
    this.locks[name] = prev.then(() => cur);
    await prev;
    try {
      return await fn();
    } finally {
      release();
    }
  }

  async list_servers() {
    const summary: Record<string, string> = {};
    for (const [name, info] of Object.entries(this.registry)) summary[name] = info.summary;
    return { status: 200, result: JSON.stringify(summary), info: "Fetched Tool Summaries" };
  }

  async activate_server(name: string, fetch_manual: boolean) {
    if (!this.registry[name]) return { status: 404, detail: `Server ${name} not found in registry.` };

    return await this.withLock(name, async () => {
      if (this.active_servers[name]) return { status: 422, detail: `Server ${name} is already active.` };

      const port = this.next_port;
      const image_tag = `${name}-server:test`;
      const container_name = `${name}-${port}`;

      let container: Docker.Container;
      try {
        logInfo("orchestrator", `Starting container: ${container_name} on host port ${port}...`);
        container = await this.docker.createContainer({
          Image: image_tag,
          name: container_name,
          HostConfig: {
            AutoRemove: true,
            PortBindings: { "8000/tcp": [{ HostPort: String(port) }] },
            Memory: 512 * 1024 * 1024,
            CpuQuota: 50000,
            NetworkMode: "bridge",
          },
          ExposedPorts: { "8000/tcp": {} },
        });
        await container.start();
      } catch (e: any) {
        logError("orchestrator", `Docker Error: ${e?.message ?? String(e)}`);
        return { status: 500, detail: `Failed to start Docker container: ${e?.message ?? String(e)}` };
      }

      const ok = await this.wait_for_server(port);
      if (!ok) {
        try {
          await container.stop();
        } catch {
          // ignore
        }
        return { status: 500, detail: "Server failed to start." };
      }

      this.active_servers[name] = { port, container, status: "active", counter: 0 };
      this.next_port += 1;

      if (!fetch_manual) return { status: 200, result: null, info: `Activated Server: ${name} - ${port}` };

      const manualPath = path.join(process.cwd(), "src", "cli_server", "docs", `${name}.md`);
      let manual = "";
      try {
        const lines = fs.readFileSync(manualPath, "utf8").split(/\r?\n/);
        manual = lines.slice(2).join("\n");
      } catch (e: any) {
        return { status: 500, detail: `Failed to read manual for ${name}: ${e?.message ?? String(e)}` };
      }

      return { status: 200, result: manual, info: `Activated Server: ${name} - ${port}` };
    });
  }

  private async wait_for_server(port: number, timeoutSec = 20): Promise<boolean> {
    const url = `http://127.0.0.1:${port}/openapi.json`;
    for (let i = 0; i < timeoutSec * 2; i++) {
      try {
        await fetch(url, { method: "GET" });
        return true;
      } catch {
        await sleep(500);
      }
    }
    return false;
  }

  async stop_server(name: string) {
    if (!this.registry[name]) return { status: 404, detail: `Server ${name} not found in registry.` };

    // mark stopping
    const mark = await this.withLock(name, async () => {
      if (!this.active_servers[name]) return { status: 404, detail: `Server ${name} already stopped or not active.` };
      this.active_servers[name].status = "stopping";
      return { status: 200 };
    });
    if ((mark as any).detail) return mark as any;

    // wait without lock for inflight executes
    let max_retries = 50;
    while ((this.active_servers[name]?.counter ?? 0) > 0 && max_retries > 0) {
      await sleep(100);
      max_retries -= 1;
    }

    return await this.withLock(name, async () => {
      const active = this.active_servers[name];
      if (!active) return { status: 404, detail: `Server ${name} already stopped or not active.` };
      const port = active.port;
      try {
        await active.container.stop();
      } catch {
        // ignore
      }
      delete this.active_servers[name];
      logInfo("orchestrator", `Stopped server: ${name} on port ${port}`);
      return { status: 200, result: null, info: `Stopped Server: ${name} - ${port}` };
    });
  }

  async execute(name: string, language: string, code: string) {
    if (!this.registry[name]) return { status: 404, detail: `Server ${name} not found in registry.` };

    // gate + increment counter
    const gate = await this.withLock(name, async () => {
      const active = this.active_servers[name];
      if (!active) return { status: 422, detail: `Server ${name} not activated` };
      if (active.status === "stopping")
        return { status: 503, detail: `Server ${name} is stopping and cannot accept new requests.` };
      active.counter += 1;
      return { status: 200, port: active.port };
    });
    if ((gate as any).detail) return gate as any;
    const port = (gate as any).port as number;

    const server_url = `http://127.0.0.1:${port}/execute`;
    const payload = { language, code };

    try {
      const resp = await fetchJson(server_url, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(payload),
      });
      return { status: 200, result: String(resp.stdout ?? "").trim(), info: `Executed Server: ${name} - ${port}` };
    } catch (e: any) {
      return { status: e.status ?? 500, detail: e.detail ?? e?.message ?? String(e) };
    } finally {
      await this.withLock(name, async () => {
        const active = this.active_servers[name];
        if (active) active.counter -= 1;
      });
    }
  }
}

function logWarnMaybe(msg: string) {
  process.stdout.write(`WARN:    [orchestrator] ${msg}\n`);
}

const app = Fastify({ logger: false });
const orchestrator = new AIOrchestrator();

app.post("/orchestrate", async (request, reply) => {
  const parsed = CommandRequestSchema.safeParse(request.body);
  if (!parsed.success) {
    reply.status(422);
    return { status: 422, detail: parsed.error.message, id: "unknown" };
  }

  const req = parsed.data;
  let result: any;
  if (req.command === "list_available_servers") result = await orchestrator.list_servers();
  else if (req.command === "activate_server") result = await orchestrator.activate_server(req.server_name ?? "", !!req.fetch_manual);
  else if (req.command === "stop_server") result = await orchestrator.stop_server(req.server_name ?? "");
  else if (req.command === "execute_server_code")
    result = await orchestrator.execute(req.server_name ?? "", req.language ?? "", req.code ?? "");
  else result = { status: 422, detail: `Unknown command: ${req.command}` };

  const responseBody = { ...result, id: req.id };
  const ok = OrchestratorResultSchema.safeParse(responseBody);
  const err = OrchestratorErrorSchema.safeParse(responseBody);
  if (!ok.success && !err.success) {
    reply.status(500);
    return { status: 500, detail: "Internal response schema error", id: req.id };
  }

  return responseBody;
});

if (require.main === module) {
  const host = "127.0.0.1";
  const port = 8000;
  app.listen({ host, port }).then(() => logInfo("orchestrator", `Listening on http://${host}:${port}`));
}

