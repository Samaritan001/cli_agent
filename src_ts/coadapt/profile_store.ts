import fs from "node:fs";
import path from "node:path";
import {
  AIProfileSchema,
  type AIProfile,
  UserProfileSchema,
  type UserProfile,
} from "./schemas";
import { defaultAIProfile, defaultUserProfile } from "./defaults";

export type PersistedProfiles = {
  version: 1;
  user: UserProfile;
  ai: AIProfile;
  updatedAt: string;
};

function clamp01(n: number): number {
  return Math.max(0, Math.min(1, n));
}

function clampProfileUser(u: UserProfile): UserProfile {
  const x = UserProfileSchema.parse(u);
  return {
    preferences: {
      ...x.preferences,
      responseLength: clamp01(x.preferences.responseLength),
    },
    cognitiveStyle: {
      analyticalVsIntuitive: clamp01(x.cognitiveStyle.analyticalVsIntuitive),
      detailVsHighLevel: clamp01(x.cognitiveStyle.detailVsHighLevel),
    },
    behaviorPatterns: {
      ...x.behaviorPatterns,
      interactionFrequency: clamp01(x.behaviorPatterns.interactionFrequency),
      modalityTextVsVoice: clamp01(x.behaviorPatterns.modalityTextVsVoice),
    },
    emotionalTraits: {
      riskTolerance: clamp01(x.emotionalTraits.riskTolerance),
      patience: clamp01(x.emotionalTraits.patience),
      ambiguityTolerance: clamp01(x.emotionalTraits.ambiguityTolerance),
    },
    meta: {
      explorationVsExploitation: clamp01(x.meta.explorationVsExploitation),
      consistencyVsNovelty: clamp01(x.meta.consistencyVsNovelty),
    },
  };
}

function clampProfileAi(a: AIProfile): AIProfile {
  const x = AIProfileSchema.parse(a);
  return {
    style: { conciseVsExploratory: clamp01(x.style.conciseVsExploratory) },
    reasoning: {
      fastVsSlowThinking: clamp01(x.reasoning.fastVsSlowThinking),
      depthOfExplanation: clamp01(x.reasoning.depthOfExplanation),
    },
    initiative: { reactiveVsProactive: clamp01(x.initiative.reactiveVsProactive) },
    uncertaintyHandling: { hedgingVsDecisive: clamp01(x.uncertaintyHandling.hedgingVsDecisive) },
    personality: {
      skepticism: clamp01(x.personality.skepticism),
      empathy: clamp01(x.personality.empathy),
      literalism: clamp01(x.personality.literalism),
    },
  };
}

export function loadProfiles(filePath: string): PersistedProfiles {
  if (!fs.existsSync(filePath)) {
    const now = new Date().toISOString();
    return {
      version: 1,
      user: defaultUserProfile(),
      ai: defaultAIProfile(),
      updatedAt: now,
    };
  }
  const raw = JSON.parse(fs.readFileSync(filePath, "utf8")) as Partial<PersistedProfiles>;
  const du = defaultUserProfile();
  const da = defaultAIProfile();
  const ru = (raw.user ?? {}) as Partial<UserProfile>;
  const ra = (raw.ai ?? {}) as Partial<AIProfile>;
  const uMerged: UserProfile = {
    ...du,
    ...ru,
    preferences: { ...du.preferences, ...ru.preferences },
    cognitiveStyle: { ...du.cognitiveStyle, ...ru.cognitiveStyle },
    behaviorPatterns: { ...du.behaviorPatterns, ...ru.behaviorPatterns },
    emotionalTraits: { ...du.emotionalTraits, ...ru.emotionalTraits },
    meta: { ...du.meta, ...ru.meta },
  };
  const aMerged: AIProfile = {
    ...da,
    ...ra,
    style: { ...da.style, ...ra.style },
    reasoning: { ...da.reasoning, ...ra.reasoning },
    initiative: { ...da.initiative, ...ra.initiative },
    uncertaintyHandling: { ...da.uncertaintyHandling, ...ra.uncertaintyHandling },
    personality: { ...da.personality, ...ra.personality },
  };
  const u = UserProfileSchema.safeParse(uMerged);
  const a = AIProfileSchema.safeParse(aMerged);
  return {
    version: 1,
    user: clampProfileUser(u.success ? u.data : defaultUserProfile()),
    ai: clampProfileAi(a.success ? a.data : defaultAIProfile()),
    updatedAt: raw.updatedAt ?? new Date().toISOString(),
  };
}

export function saveProfiles(filePath: string, data: PersistedProfiles): void {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  const out: PersistedProfiles = {
    ...data,
    user: clampProfileUser(data.user),
    ai: clampProfileAi(data.ai),
    updatedAt: new Date().toISOString(),
  };
  fs.writeFileSync(filePath, JSON.stringify(out, null, 2), "utf8");
}
