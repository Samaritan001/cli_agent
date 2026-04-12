import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    include: ["src_ts/**/*.test.ts"],
    environment: "node",
  },
});
