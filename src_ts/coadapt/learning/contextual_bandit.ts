/**
 * Disjoint LinUCB contextual bandit over discrete arms (hand-crafted context only).
 * Persists ridge-regression sufficient stats per arm; migrates legacy v1 epsilon-greedy JSON.
 */
import fs from "node:fs";
import path from "node:path";
import { BANDIT_ARM_IDS, type BanditArmId } from "./strategies";
import { CONTEXT_FEATURE_DIM } from "./context_features";

export type LinucbArmState = {
  id: BanditArmId;
  updateCount: number;
  /** d×d — A accumulates λI + Σ x xᵀ */
  A: number[][];
  /** length d — b accumulates Σ r·x */
  b: number[];
};

export type ContextualBanditStateV2 = {
  version: 2;
  d: number;
  lambda: number;
  alpha: number;
  epsilon: number;
  arms: LinucbArmState[];
};

const DEFAULT_LAMBDA = 1;
const DEFAULT_ALPHA = 0.55;
const DEFAULT_EPSILON = 0.1;

function envNumber(name: string, fallback: number, min: number, max: number): number {
  const raw = process.env[name];
  if (raw === undefined || raw === "") return fallback;
  const n = Number(raw);
  return Number.isFinite(n) && n >= min && n <= max ? n : fallback;
}

function zeros2d(n: number): number[][] {
  return Array.from({ length: n }, () => new Array(n).fill(0));
}

function identityScaled(n: number, scale: number): number[][] {
  const M = zeros2d(n);
  for (let i = 0; i < n; i++) M[i]![i] = scale;
  return M;
}

function addOuter(A: number[][], x: number[]): void {
  const d = x.length;
  for (let i = 0; i < d; i++) {
    for (let j = 0; j < d; j++) A[i]![j]! += x[i]! * x[j]!;
  }
}

/** Solve A·z = rhs for square A (Gauss–Jordan, partial pivot). Returns null if singular. */
export function solveLinearSystem(Ain: number[][], rhs: number[]): number[] | null {
  const n = Ain.length;
  if (n === 0 || rhs.length !== n) return null;
  const M = Ain.map((row, i) => [...row, rhs[i]!]);
  for (let col = 0; col < n; col++) {
    let pivot = col;
    let best = Math.abs(M[pivot]![col]!);
    for (let r = col + 1; r < n; r++) {
      const v = Math.abs(M[r]![col]!);
      if (v > best) {
        best = v;
        pivot = r;
      }
    }
    if (best < 1e-12) return null;
    if (pivot !== col) [M[col], M[pivot]] = [M[pivot]!, M[col]!];
    const div = M[col]![col]!;
    for (let j = col; j <= n; j++) M[col]![j]! /= div;
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const f = M[r]![col]!;
      if (Math.abs(f) < 1e-15) continue;
      for (let j = col; j <= n; j++) M[r]![j]! -= f * M[col]![j]!;
    }
  }
  return M.map((row) => row[n]!);
}

function dot(a: number[], b: number[]): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i]! * b[i]!;
  return s;
}

function cloneMatrix(M: number[][]): number[][] {
  return M.map((row) => [...row]);
}

function initArm(lambda: number, d: number): Omit<LinucbArmState, "id"> {
  return {
    updateCount: 0,
    A: identityScaled(d, lambda),
    b: new Array(d).fill(0),
  };
}

function freshState(lambda: number, alpha: number, epsilon: number, d: number): ContextualBanditStateV2 {
  const arms: LinucbArmState[] = BANDIT_ARM_IDS.map((id) => ({
    id,
    ...initArm(lambda, d),
  }));
  return { version: 2, d, lambda, alpha, epsilon, arms };
}

export class ContextualLinUCBBandit {
  private state: ContextualBanditStateV2;
  private readonly filePath: string;

  constructor(filePath: string) {
    this.filePath = filePath;
    const lambda = envNumber("COADAPT_LINUCB_LAMBDA", DEFAULT_LAMBDA, 0.05, 10);
    const alpha = envNumber("COADAPT_LINUCB_ALPHA", DEFAULT_ALPHA, 0.05, 5);
    const epsilon = envNumber("COADAPT_BANDIT_EPSILON", DEFAULT_EPSILON, 0, 0.5);
    this.state = this.load(lambda, alpha, epsilon);
  }

  get epsilon(): number {
    return this.state.epsilon;
  }

  get featureDim(): number {
    return this.state.d;
  }

  private load(lambda: number, alpha: number, epsilon: number): ContextualBanditStateV2 {
    const d = CONTEXT_FEATURE_DIM;
    if (!fs.existsSync(this.filePath)) {
      return freshState(lambda, alpha, epsilon, d);
    }
    try {
      const raw = JSON.parse(fs.readFileSync(this.filePath, "utf8")) as unknown;
      if (raw && typeof raw === "object" && (raw as { version?: number }).version === 2) {
        const s = raw as ContextualBanditStateV2;
        if (s.d !== d || !Array.isArray(s.arms) || s.arms.length !== BANDIT_ARM_IDS.length) {
          return freshState(lambda, alpha, epsilon, d);
        }
        return {
          version: 2,
          d: s.d,
          lambda: s.lambda ?? lambda,
          alpha: s.alpha ?? alpha,
          epsilon: s.epsilon ?? epsilon,
          arms: s.arms.map((a, i) => ({
            id: BANDIT_ARM_IDS[i]!,
            updateCount: a.updateCount ?? 0,
            A: cloneMatrix(a.A),
            b: [...a.b],
          })),
        };
      }
      // Legacy v1 epsilon-greedy — cold-start LinUCB but keep file path
      return freshState(lambda, alpha, epsilon, d);
    } catch {
      return freshState(lambda, alpha, epsilon, d);
    }
  }

  private persist(): void {
    fs.mkdirSync(path.dirname(this.filePath), { recursive: true });
    fs.writeFileSync(this.filePath, JSON.stringify(this.state, null, 2), "utf8");
  }

  /**
   * UCB score for one arm: θᵀx + α·√(xᵀ A⁻¹ x) with θ = A⁻¹b.
   */
  private ucbScore(arm: LinucbArmState, x: number[]): number {
    const d = this.state.d;
    const A = cloneMatrix(arm.A);
    const theta = solveLinearSystem(A, arm.b);
    const mean = theta ? dot(theta, x) : 0;
    const Ainv_x = solveLinearSystem(cloneMatrix(arm.A), x);
    const conf = Ainv_x ? Math.sqrt(Math.max(0, dot(x, Ainv_x))) : 0;
    return mean + this.state.alpha * conf;
  }

  /** Pick arm using LinUCB; with probability ε explore uniformly. */
  selectArm(context: number[]): number {
    if (context.length !== this.state.d) {
      throw new Error(`contextual_bandit: expected context dim ${this.state.d}, got ${context.length}`);
    }
    const { arms, epsilon } = this.state;
    if (Math.random() < epsilon) {
      return Math.floor(Math.random() * arms.length);
    }
    let best = 0;
    let bestScore = this.ucbScore(arms[0]!, context);
    for (let i = 1; i < arms.length; i++) {
      const s = this.ucbScore(arms[i]!, context);
      if (s > bestScore) {
        bestScore = s;
        best = i;
      }
    }
    return best;
  }

  /** Ridge/LinUCB update for the arm that was played under this context. */
  update(armIndex: number, reward: number, context: number[]): void {
    if (context.length !== this.state.d) {
      throw new Error(`contextual_bandit: expected context dim ${this.state.d}, got ${context.length}`);
    }
    const r = Math.max(0, Math.min(1, reward));
    const arm = this.state.arms[armIndex];
    if (!arm) return;
    addOuter(arm.A, context);
    for (let i = 0; i < context.length; i++) arm.b[i]! += r * context[i]!;
    arm.updateCount += 1;
    this.persist();
  }
}
