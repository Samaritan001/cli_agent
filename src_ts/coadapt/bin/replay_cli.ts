/**
 * CLI entry: reads `eval.jsonl` (or argv path) and prints `analyzeEvalLog` JSON to stdout.
 */
import path from "node:path";
import process from "node:process";
import { analyzeEvalLog } from "../eval/replay";

const p = process.argv[2] ?? path.join(process.cwd(), ".cli_agent", "eval.jsonl");
const r = analyzeEvalLog(p);
console.log(JSON.stringify(r, null, 2));
