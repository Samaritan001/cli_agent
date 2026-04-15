import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.join(path.dirname(fileURLToPath(import.meta.url)), "..");
const srcDir = path.join(root, "src_ts", "cli_client");
const destDir = path.join(root, "dist", "cli_client");

fs.mkdirSync(destDir, { recursive: true });
for (const name of fs.readdirSync(srcDir)) {
  if (name.endsWith(".json") || name === "system_instruction.md") {
    fs.copyFileSync(path.join(srcDir, name), path.join(destDir, name));
  }
}
