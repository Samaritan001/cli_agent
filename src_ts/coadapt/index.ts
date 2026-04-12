/**
 * Co-adaptation package: session orchestration, profiles, learning, eval, memory, rules.
 * Subfolders expose `index.ts` barrels; this file re-exports the public API.
 */
export { CoAdaptSession, type CoAdaptSessionOptions } from "./session/co_adapt_session";
export * from "./memory";
export * from "./profile";
export * from "./eval";
export * from "./learning";
