import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    /** Co-adapt tests live under `coadapt/tests/` and mirror package structure (eval/, learning/, …). */
    include: ["src_ts/coadapt/tests/**/*.test.ts"],
    environment: "node",
  },
});
