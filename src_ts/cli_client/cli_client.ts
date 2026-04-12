import readline from "node:readline";
import process from "node:process";
import { ToolManualManager } from "./managers";
import { LanguageModel, ParsedResponse } from "./model";
import { logError, logInfo, logWarn } from "../shared/logging";
import { CoAdaptSession } from "../coadapt";

const SERVER_URL = "http://127.0.0.1:8000/orchestrate";

type ToolCall = { id: string; command: string; arguments: Record<string, any> };

async function postJson(url: string, body: any): Promise<any> {
  const res = await fetch(url, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  const text = await res.text();
  return text ? JSON.parse(text) : {};
}

export class CLIClient {
  private tool_manager = new ToolManualManager();
  private language_model: LanguageModel;
  private test_flag: boolean;
  private co_adapt: CoAdaptSession | null;

  constructor(opts?: { model_type?: string; model_name?: string; test_flag?: boolean; co_adapt?: boolean }) {
    const model_type = opts?.model_type ?? "google";
    const model_name = opts?.model_name;
    this.test_flag = !!opts?.test_flag;
    this.language_model = new LanguageModel(model_type, model_name);
    const enable =
      opts?.co_adapt !== false &&
      process.env.COADAPT !== "0" &&
      process.env.COADAPT_DISABLED !== "1";
    this.co_adapt = enable ? new CoAdaptSession() : null;
  }

  async agent_loop(): Promise<void> {
    const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
    const question = (q: string) => new Promise<string>((resolve) => rl.question(q, resolve));

    while (true) {
      const user_input = await question("--- User Input ---\nEnter your message for the agent (or 'quit' to exit):\n");
      if (user_input.trim().toLowerCase() === "quit") break;

      if (this.co_adapt) await this.co_adapt.onUserTurnStart(user_input);
      this.language_model.add_user_message(user_input);
      process.stdout.write("\n--- Agent Response ---\n");

      const setCoAdapt = async () => {
        if (!this.co_adapt) {
          this.language_model.set_co_adapt_context("");
          return;
        }
        this.language_model.set_co_adapt_context(await this.co_adapt.buildContextBlock(user_input));
      };
      await setCoAdapt();

      let turn_assistant = "";
      let tool_calls_total = 0;
      let llm_output: ParsedResponse = this.language_model.parse_response(
        await this.language_model.generate_response(this.tool_manager.getToolInfo())
      );

      tool_calls_total += ((llm_output.tool_calls as ToolCall[]) ?? []).length;

      if (this.test_flag) process.stdout.write(`LLM Output (with tool calls):\n${JSON.stringify(llm_output, null, 2)}\n\n`);

      if (llm_output.content) {
        process.stdout.write(`${llm_output.content}\n\n`);
        turn_assistant += `${llm_output.content}\n`;
      }

      let tool_calls: ToolCall[] = (llm_output.tool_calls as any) ?? [];
      while (tool_calls.length > 0) {
        await this.tool_callings(tool_calls);
        await setCoAdapt();
        llm_output = this.language_model.parse_response(
          await this.language_model.generate_response(this.tool_manager.getToolInfo())
        );
        tool_calls_total += ((llm_output.tool_calls as ToolCall[]) ?? []).length;
        if (this.test_flag) process.stdout.write(`LLM Output (with tool calls):\n${JSON.stringify(llm_output, null, 2)}\n\n`);
        if (llm_output.content) {
          process.stdout.write(`${llm_output.content}\n\n`);
          turn_assistant += `${llm_output.content}\n`;
        }
        tool_calls = (llm_output.tool_calls as any) ?? [];
      }

      if (this.co_adapt)
        await this.co_adapt.afterTurn(user_input, turn_assistant.trim(), { toolCallsCount: tool_calls_total });
    }

    rl.close();
  }

  async tool_callings(tool_calls: ToolCall[]) {
    logInfo("cli_client", `Processing ${tool_calls.length} tool calls...`);

    const command_records: Record<string, [string, string | null]> = {};
    const calls: any[] = [];

    for (const tool_call of tool_calls) {
      if (!tool_call.command || !tool_call.id) continue;
      const serverName = tool_call.arguments?.server_name ?? null;
      command_records[tool_call.id] = [tool_call.command, serverName];

      const call: any = { ...(tool_call.arguments ?? {}) };
      call.id = tool_call.id;
      call.command = tool_call.command;
      if (call.command === "activate_server") {
        call.fetch_manual = !this.tool_manager.checkTool(String(call.server_name ?? ""));
      }
      calls.push(call);
      logInfo("cli_client", `Calling command '${call.command}' with arguments ${JSON.stringify(call)}`);
    }

    const responses = await Promise.all(calls.map((c) => postJson(SERVER_URL, c)));

    for (const call_resp of responses) {
      const id = call_resp.id as string;
      const [command, server_name] = command_records[id] ?? ["", null];

      if (call_resp.status !== 200) {
        const error_status = call_resp.status;
        const detail = call_resp.detail;
        this.language_model.add_tool_response(`${error_status} - ${detail}`, command, id);
        logError("cli_client", `Error in tool call id ${id}: ${error_status} - ${detail}`);
        continue;
      }

      let resultForModel = "Unknown command response.";
      if (command === "list_available_servers") {
        try {
          this.tool_manager.registerSummary(JSON.parse(call_resp.result));
        } catch {
          // ignore
        }
        resultForModel = "Tool summaries have been registered.";
      } else if (command === "activate_server") {
        if (call_resp.result && server_name) this.tool_manager.registerTool(server_name, call_resp.result);
        if (server_name) this.tool_manager.injectTool(server_name);
        resultForModel = `Tool '${server_name}' has been activated.`;
      } else if (command === "stop_server") {
        if (server_name) this.tool_manager.pruneTool(server_name);
        resultForModel = `Tool '${server_name}' has been stopped.`;
      } else if (command === "execute_server_code") {
        resultForModel = String(call_resp.result ?? "");
      }

      this.language_model.add_tool_response(resultForModel, command, id);
      logInfo("cli_client", `Success in tool call id ${id}`);
      if (call_resp.info) logWarn("cli_client", `Info: ${call_resp.info}`);
    }
  }
}

if (require.main === module) {
  const client = new CLIClient({ model_type: process.env.MODEL_TYPE ?? "openai", test_flag: true });
  client.agent_loop().catch((e) => {
    logError("cli_client", e?.message ?? String(e));
    process.exitCode = 1;
  });
}

