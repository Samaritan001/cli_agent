export function logInfo(scope: string, message: string) {
  // Match the Python vibe: "INFO:    message"
  process.stdout.write(`INFO:    [${scope}] ${message}\n`);
}

export function logWarn(scope: string, message: string) {
  process.stdout.write(`WARN:    [${scope}] ${message}\n`);
}

export function logError(scope: string, message: string) {
  process.stderr.write(`ERROR:   [${scope}] ${message}\n`);
}

