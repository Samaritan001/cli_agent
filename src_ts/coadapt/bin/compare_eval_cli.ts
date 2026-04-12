/**
 * CLI: compare two eval.jsonl files (stdout JSON). Usage:
 *   npx tsx src_ts/coadapt/bin/compare_eval_cli.ts <evalA.jsonl> <evalB.jsonl>
 */
import path from "node:path";
import process from "node:process";
import { compareEvalLogs } from "../eval/compare_eval";

const a = process.argv[2] ?? path.join(process.cwd(), ".cli_agent", "eval.jsonl");
const b = process.argv[3] ?? path.join(process.cwd(), ".cli_agent", "eval.off.jsonl");
const r = compareEvalLogs(a, b);
console.log(JSON.stringify(r, null, 2));
