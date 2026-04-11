import process from "node:process";
import { spawn } from "node:child_process";
import Fastify from "fastify";
import { z } from "zod";

function logInfo(message: string) {
  process.stdout.write(`INFO:    [weather_cli] ${message}\n`);
}

function logError(message: string) {
  process.stderr.write(`ERROR:   [weather_cli] ${message}\n`);
}

const CodeRequestSchema = z.object({
  language: z.string(),
  code: z.string(),
});

function validateCodeIntegrity(language: string, code: string) {
  const lang = language.toLowerCase();
  const markers: Record<string, RegExp[]> = {
    javascript: [/console\.log\(/, /await\s+/, /function\s+/, /=>/],
    node: [/console\.log\(/, /await\s+/, /require\(|from\s+["']/, /import\s+/],
  };

  if (markers[lang]) {
    if (!markers[lang].some((re) => re.test(code))) {
      throw new Error(`Code content does not appear to match the '${language}' label.`);
    }
  }
}

const app = Fastify({ logger: false });

app.get("/openapi.json", async () => {
  // Orchestrator readiness probe parity with FastAPI
  return { openapi: "3.0.0", info: { title: "weather-ts-server", version: "0.1.0" }, paths: {} };
});

app.post("/execute", async (request, reply) => {
  const parsed = CodeRequestSchema.safeParse(request.body);
  if (!parsed.success) {
    reply.status(422);
    return { detail: `Invalid request: ${parsed.error.message}` };
  }

  const { language, code } = parsed.data;
  const lang = language.toLowerCase();

  try {
    validateCodeIntegrity(lang, code);
  } catch (e: any) {
    reply.status(422);
    return { detail: `Invalid code grammar: ${e?.message ?? String(e)}` };
  }

  if (lang !== "javascript" && lang !== "node") {
    reply.status(422);
    return { detail: `Unsupported language: ${language}. Use 'javascript' (node).` };
  }

  const prelude = `
const { getForecast, getAlerts } = require("./weather");
(async () => {
${code}
})().catch((e) => { console.error(e && e.stack ? e.stack : String(e)); process.exitCode = 1; });
`.trim();

  const scriptDir = __dirname;
  const timeoutMs = 10_000;

  let killed = false;
  const child = spawn("node", ["-e", prelude], { cwd: scriptDir });
  const done = new Promise<{ stdout: string; stderr: string; code: number | null }>((resolve) => {
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (d) => (stdout += d.toString()));
    child.stderr.on("data", (d) => (stderr += d.toString()));
    child.on("close", (code) => resolve({ stdout, stderr, code }));
  });

  const timer = setTimeout(() => {
    killed = true;
    child.kill("SIGKILL");
  }, timeoutMs);

  const result = await done.finally(() => clearTimeout(timer));
  if (killed) {
    reply.status(408);
    return { detail: "Execution timed out." };
  }
  if (result.code !== 0) {
    logError(`Execution error:\n${result.stderr.trim()}`);
    reply.status(400);
    return { detail: `Execution error: ${result.stderr.trim()}` };
  }

  reply.status(200);
  return { stdout: result.stdout };
});

if (require.main === module) {
  const port = Number(process.argv[2] ?? "8000");
  app.listen({ host: "0.0.0.0", port }).then(() => logInfo(`Listening on 0.0.0.0:${port}`));
}

