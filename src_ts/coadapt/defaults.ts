import type { AIProfile, UserProfile } from "./schemas";

const mid = 0.5;

export function defaultUserProfile(): UserProfile {
  return {
    preferences: { topics: [], responseLength: mid, formatPreferences: [] },
    cognitiveStyle: { analyticalVsIntuitive: mid, detailVsHighLevel: mid },
    behaviorPatterns: { interactionFrequency: mid, modalityTextVsVoice: mid, taskTypes: [] },
    emotionalTraits: { riskTolerance: mid, patience: mid, ambiguityTolerance: mid },
    meta: { explorationVsExploitation: mid, consistencyVsNovelty: mid },
  };
}

export function defaultAIProfile(): AIProfile {
  return {
    style: { conciseVsExploratory: mid },
    reasoning: { fastVsSlowThinking: mid, depthOfExplanation: mid },
    initiative: { reactiveVsProactive: mid },
    uncertaintyHandling: { hedgingVsDecisive: mid },
    personality: { skepticism: mid, empathy: mid, literalism: mid },
  };
}
