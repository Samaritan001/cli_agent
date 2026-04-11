import { spawnSync } from "node:child_process";
import { logError, logInfo } from "../shared/logging";

export function buildWeatherImage() {
  logInfo("docker_image_builder", "Building weather-server:test image...");
  const r = spawnSync("docker", ["build", "-t", "weather-server:test", "-f", "src_ts/cli_server/weather_ts_server/Dockerfile", "."], {
    stdio: "inherit",
  });
  if (r.status !== 0) {
    throw new Error(`docker build failed with exit code ${r.status}`);
  }
  logInfo("docker_image_builder", "Image built successfully.");
}

if (require.main === module) {
  try {
    buildWeatherImage();
  } catch (e: any) {
    logError("docker_image_builder", e?.message ?? String(e));
    process.exitCode = 1;
  }
}

