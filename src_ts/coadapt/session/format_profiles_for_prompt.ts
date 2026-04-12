/**
 * Compact user + AI profile lines injected into the model context.
 */
import type { AIProfile, UserProfile } from "../profile/schemas";

export function formatProfilesForPrompt(user: UserProfile, ai: AIProfile): string {
  const lines: string[] = [];
  lines.push("User model (inferred, 0=low … 1=high):");
  lines.push(`- responseLength: ${user.preferences.responseLength.toFixed(2)}`);
  lines.push(`- detailVsHighLevel: ${user.cognitiveStyle.detailVsHighLevel.toFixed(2)}`);
  lines.push(`- analyticalVsIntuitive: ${user.cognitiveStyle.analyticalVsIntuitive.toFixed(2)}`);
  lines.push(`AI profile (control knobs):`);
  lines.push(`- conciseVsExploratory: ${ai.style.conciseVsExploratory.toFixed(2)}`);
  lines.push(`- depthOfExplanation: ${ai.reasoning.depthOfExplanation.toFixed(2)}`);
  lines.push(`- reactiveVsProactive: ${ai.initiative.reactiveVsProactive.toFixed(2)}`);
  return lines.join("\n");
}
