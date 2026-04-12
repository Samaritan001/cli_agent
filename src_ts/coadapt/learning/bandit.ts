/**
 * Epsilon-greedy bandit over discrete arms with JSON persistence (running mean reward per arm).
 */
import fs from "node:fs";
import path from "node:path";
import { BANDIT_ARM_IDS, type BanditArmId } from "./strategies";

export type BanditArmStats = {
  id: BanditArmId;
  count: number;
  meanReward: number;
};

export type BanditState = {
  version: 1;
  epsilon: number;
  arms: BanditArmStats[];
};

const DEFAULT_EPSILON = 0.12;

export class EpsilonGreedyBandit {
  private state: BanditState;
  private readonly filePath: string;

  constructor(filePath: string, epsilon: number = DEFAULT_EPSILON) {
    this.filePath = filePath;
    this.state = this.load(epsilon);
  }

  private load(epsilon: number): BanditState {
    if (!fs.existsSync(this.filePath)) {
      return {
        version: 1,
        epsilon,
        arms: BANDIT_ARM_IDS.map((id) => ({ id, count: 0, meanReward: 0.5 })),
      };
    }
    try {
      const raw = JSON.parse(fs.readFileSync(this.filePath, "utf8")) as BanditState;
      if (raw.version !== 1 || !Array.isArray(raw.arms)) throw new Error("bad bandit file");
      const byId = new Map(raw.arms.map((a) => [a.id, a]));
      const arms = BANDIT_ARM_IDS.map((id) => {
        const x = byId.get(id);
        return x ? { ...x, id } : { id, count: 0, meanReward: 0.5 };
      });
      return { version: 1, epsilon: raw.epsilon ?? epsilon, arms };
    } catch {
      return {
        version: 1,
        epsilon,
        arms: BANDIT_ARM_IDS.map((id) => ({ id, count: 0, meanReward: 0.5 })),
      };
    }
  }

  private persist(): void {
    fs.mkdirSync(path.dirname(this.filePath), { recursive: true });
    fs.writeFileSync(this.filePath, JSON.stringify(this.state, null, 2), "utf8");
  }

  get epsilon(): number {
    return this.state.epsilon;
  }

  getArmsSnapshot(): BanditArmStats[] {
    return this.state.arms.map((a) => ({ ...a }));
  }

  /** 0 .. arms-1 */
  selectArm(): number {
    const { arms, epsilon } = this.state;
    if (Math.random() < epsilon) {
      return Math.floor(Math.random() * arms.length);
    }
    let best = 0;
    let bestMean = arms[0]!.meanReward;
    for (let i = 1; i < arms.length; i++) {
      if (arms[i]!.meanReward > bestMean) {
        bestMean = arms[i]!.meanReward;
        best = i;
      }
    }
    return best;
  }

  update(armIndex: number, reward: number): void {
    const r = Math.max(0, Math.min(1, reward));
    const arm = this.state.arms[armIndex];
    if (!arm) return;
    const n = arm.count + 1;
    arm.meanReward = arm.count === 0 ? r : arm.meanReward + (r - arm.meanReward) / n;
    arm.count = n;
    this.persist();
  }
}
