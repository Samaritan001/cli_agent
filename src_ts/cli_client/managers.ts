import { logInfo } from "../shared/logging";

export class ToolManualManager {
  private summaries: Record<string, string> = {};
  private registry: Record<string, string> = {};
  private activeTools = new Set<string>();

  registerSummary(summaries: Record<string, string>) {
    for (const [name, summary] of Object.entries(summaries)) {
      logInfo("managers", `Registering tool summary '${name}'`);
      this.summaries[name] = summary;
    }
  }

  registerTool(name: string, manual: string) {
    logInfo("managers", `Registering tool manual '${name}'`);
    this.registry[name] = manual;
  }

  checkTool(name: string): boolean {
    return Object.prototype.hasOwnProperty.call(this.registry, name);
  }

  injectTool(name: string): boolean {
    if (this.checkTool(name)) {
      this.activeTools.add(name);
      logInfo("managers", `Injecting tool manual '${name}'`);
      return true;
    }
    return false;
  }

  pruneTool(name: string) {
    if (this.activeTools.has(name)) {
      this.activeTools.delete(name);
      logInfo("managers", `Pruning tool manual '${name}'`);
    }
  }

  flushLoadout() {
    this.activeTools.clear();
    logInfo("managers", "Flushing all tool manuals from context");
  }

  private getAllSummaries(): string {
    return Object.entries(this.summaries)
      .map(([name, summary]) => `Tool Name: ${name}\nSummary: ${summary}`)
      .join("\n\n");
  }

  private getAllManuals(): string {
    const delimiter = `\n\n${"*".repeat(30)}\n\n`;
    return Array.from(this.activeTools)
      .map((name) => `Tool Name: ${name}\nManual: ${this.registry[name]}`)
      .join(delimiter);
  }

  getToolInfo(): { tool_summaries: string; tool_manuals: string } {
    return {
      tool_summaries: this.getAllSummaries(),
      tool_manuals: this.getAllManuals(),
    };
  }
}

