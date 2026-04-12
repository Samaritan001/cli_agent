import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import dotenv from "dotenv";
import OpenAI from "openai";
import Anthropic from "@anthropic-ai/sdk";
import { GoogleGenAI } from "@google/genai";

type Provider = "openai" | "anthropic" | "google";

type ToolInfo = { tool_summaries: string; tool_manuals: string };

type ParsedToolCall = {
  id: string;
  command: string;
  arguments: Record<string, unknown>;
};

export type ParsedResponse = { content: string; tool_calls: ParsedToolCall[] };

const API_KEY_VARS: Record<Provider, string> = {
  openai: "OPENAI_API_KEY",
  anthropic: "ANTHROPIC_API_KEY",
  google: "GOOGLE_API_KEY",
};

function normalizeProvider(modelType: string): Provider {
  const t = modelType.toLowerCase();
  if (t === "gpt") return "openai";
  if (t === "claude") return "anthropic";
  if (t === "gemini") return "google";
  if (t === "openai" || t === "anthropic" || t === "google") return t;
  throw new Error(`Unsupported model type: ${modelType}`);
}

function hereDir(): string {
  // CommonJS `__dirname` matches both `tsx src_ts/...` and `node dist/...` layouts.
  return __dirname;
}

export class LanguageModel {
  history: any[] = [];
  model_type: Provider;
  model_name: string;
  system_instruction: string;
  tool_definitions: any;
  /** Appended to system prompt each turn (profiles + memory); Phase 1 co-adaptation. */
  private co_adapt_context = "";

  private openaiClient?: OpenAI;
  private anthropicClient?: Anthropic;
  private googleClient?: GoogleGenAI;

  constructor(model_type = "google", model_name?: string) {
    dotenv.config();

    this.model_type = normalizeProvider(model_type);
    const apiKey = process.env[API_KEY_VARS[this.model_type]];
    if (!apiKey) throw new Error(`Missing API key env var: ${API_KEY_VARS[this.model_type]}`);

    this.model_name = model_name ?? this.defaultModelName(this.model_type);

    if (this.model_type === "openai") this.openaiClient = new OpenAI({ apiKey });
    if (this.model_type === "anthropic") this.anthropicClient = new Anthropic({ apiKey });
    if (this.model_type === "google") this.googleClient = new GoogleGenAI({ apiKey });

    const baseDir = hereDir();
    const defsPath =
      this.model_type === "openai"
        ? path.join(baseDir, "tool_definitions_openai.json")
        : path.join(baseDir, "tool_definitions_google.json");
    this.tool_definitions = JSON.parse(fs.readFileSync(defsPath, "utf8"));

    const sysPath = path.join(baseDir, "system_instruction.md");
    this.system_instruction = fs.readFileSync(sysPath, "utf8");
  }

  private defaultModelName(p: Provider): string {
    // Keep parity with the Python defaults (approximate)
    if (p === "openai") return "gpt-5.4-nano";
    if (p === "anthropic") return "claude-4-6-opus";
    return "gemini-3.1-flash-lite-preview";
  }

  update_system_instruction(instruction: string) {
    this.system_instruction = instruction;
  }

  /** Set co-adaptation block (profiles + retrieved memory). Pass empty string to clear. */
  set_co_adapt_context(content: string) {
    this.co_adapt_context = content ?? "";
  }

  private system_with_co_adapt(): string {
    if (!this.co_adapt_context.trim()) return this.system_instruction;
    return `${this.system_instruction}\n\n---\nCo-adaptation context (memory + profiles):\n${this.co_adapt_context}`;
  }

  add_user_message(content: string) {
    if (this.model_type === "openai") this.history.push({ role: "user", content });
    else if (this.model_type === "anthropic")
      this.history.push({ role: "user", content: [{ type: "text", text: content }] });
    else this.history.push({ role: "user", parts: [{ text: content }] });
  }

  add_tool_response(content: string, command: string, tool_call_id: string) {
    if (this.model_type === "openai") {
      this.history.push({ role: "tool", content, tool_call_id });
    } else if (this.model_type === "anthropic") {
      this.history.push({
        role: "user",
        content: [{ type: "tool_result", tool_use_id: tool_call_id, content }],
      });
    } else {
      this.history.push({
        role: "user",
        parts: [
          {
            function_response: {
              name: command,
              response: { result: content },
            },
          },
        ],
      });
    }
  }

  private to_openai(tool_info: ToolInfo) {
    const msgs: any[] = [{ role: "system", content: this.system_with_co_adapt() }, ...this.history];
    if (!tool_info.tool_summaries) msgs.push({ role: "system", content: "No tools currently available." });
    else {
      msgs.push({ role: "system", content: `Available tools summaries:\n${tool_info.tool_summaries}` });
      if (tool_info.tool_manuals) {
        msgs.push({ role: "system", content: `Active tools full manuals:\n${tool_info.tool_manuals}` });
      }
    }
    return msgs;
  }

  private to_anthropic(tool_info: ToolInfo): { system: string; messages: any[] } {
    let system = this.system_with_co_adapt() + "\n";
    const messages = [...this.history];
    if (!tool_info.tool_summaries) system += "No tools currently available.\n";
    else {
      system += `Available tools summaries:\n${tool_info.tool_summaries}\n`;
      if (tool_info.tool_manuals) system += `Active tools full manuals:\n${tool_info.tool_manuals}\n`;
    }
    return { system, messages };
  }

  private to_google(tool_info: ToolInfo): { system: string; contents: any[] } {
    let system = this.system_with_co_adapt() + "\n";
    const contents = [...this.history];
    if (!tool_info.tool_summaries) system += "No tools currently available.\n";
    else {
      system += `Available tools summaries:\n${tool_info.tool_summaries}\n`;
      if (tool_info.tool_manuals) system += `Active tools full manuals:\n${tool_info.tool_manuals}\n`;
    }
    return { system, contents };
  }

  async generate_response(tool_info: ToolInfo, max_tokens = 1000): Promise<any> {
    if (this.model_type === "openai") {
      const response = await this.openaiClient!.chat.completions.create({
        model: this.model_name,
        messages: this.to_openai(tool_info),
        tools: this.tool_definitions,
        max_completion_tokens: max_tokens,
      });
      // record message in history (best-effort)
      this.history.push((response.choices?.[0]?.message as any) ?? {});
      return response;
    }

    if (this.model_type === "anthropic") {
      const { system, messages } = this.to_anthropic(tool_info);
      const response = await this.anthropicClient!.messages.create({
        model: this.model_name,
        max_tokens,
        system,
        messages,
      });
      this.history.push({ role: "assistant", content: response.content });
      return response;
    }

    const { system, contents } = this.to_google(tool_info);
    const response = await this.googleClient!.models.generateContent({
      model: this.model_name,
      contents,
      config: {
        systemInstruction: system,
        maxOutputTokens: max_tokens,
        tools: this.tool_definitions,
      } as any,
    });
    // record candidate content
    const cand = (response as any).candidates?.[0]?.content;
    if (cand) this.history.push(cand);
    return response;
  }

  parse_response(response: any): ParsedResponse {
    try {
      if (this.model_type === "openai") return this.from_openai(response);
      if (this.model_type === "anthropic") return this.from_anthropic(response);
      return this.from_google(response);
    } catch (e: any) {
      return { content: `Parsing Error: ${e?.message ?? String(e)}`, tool_calls: [] };
    }
  }

  private from_openai(response: any): ParsedResponse {
    const msg = response.choices?.[0]?.message;
    const tool_calls: ParsedToolCall[] = [];
    if (msg?.tool_calls) {
      for (const tc of msg.tool_calls) {
        tool_calls.push({
          id: tc.id,
          command: tc.function.name,
          arguments: JSON.parse(tc.function.arguments || "{}"),
        });
      }
    }
    return { content: msg?.content ?? "", tool_calls };
  }

  private from_anthropic(response: any): ParsedResponse {
    const tool_calls: ParsedToolCall[] = [];
    let content = "";
    for (const block of response.content ?? []) {
      if (block.type === "text") content += block.text;
      if (block.type === "tool_use") {
        tool_calls.push({ id: block.id, command: block.name, arguments: block.input ?? {} });
      }
    }
    return { content, tool_calls };
  }

  private from_google(response: any): ParsedResponse {
    const tool_calls: ParsedToolCall[] = [];
    let content = "";
    const candidate = response.candidates?.[0];
    for (const part of candidate?.content?.parts ?? []) {
      if (part.text) content += part.text;
      if (part.functionCall) {
        const fc = part.functionCall;
        tool_calls.push({
          id: fc.id ?? fc.name,
          command: fc.name,
          arguments: (fc.args as any) ?? {},
        });
      }
      // Some SDKs use snake_case
      if (part.function_call) {
        const fc = part.function_call;
        tool_calls.push({
          id: fc.id ?? fc.name,
          command: fc.name,
          arguments: fc.args ?? {},
        });
      }
    }
    return { content, tool_calls };
  }
}

