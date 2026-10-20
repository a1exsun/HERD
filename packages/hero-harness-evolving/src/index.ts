import type { DatasetItem } from "@hero/core";

// Harness evolution as in the stream runner (packages/hero-drift-constrained-opd/stream/hero.py, run.py): a
// model writes each task's skill from a bounded window of verified examples and the checkpoint's mistakes on it, and
// the candidate registry is committed only if it lowers the window's mean residual (verified-answer NLL with the
// routed skill and the bare prompt in context) and fits the registry budget. This package holds the pure parts; the
// DSH plugin runs the model calls through the local Qwen bridge.

export interface Skill { id: string; promptPrefix: string; instructions: string }

export interface HarnessSpecification {
  schemaVersion: "hero-harness-spec-v1";
  skills: Skill[];
}

/** The stream runner's settings (`run.py prepare`). */
export const HARNESS_SETTINGS = {
  windowTokens: 8192,
  candidates: 16,
  candidateInputTokens: 384,
  candidateOutputTokens: 128,
  mistakes: 4,
  mistakeNewTokens: 1536,
  proposalMaxNewTokens: 768,
  skillTokens: 768,
  registryTokenBudget: 24576
} as const;

export const HARNESS_SYSTEM = "Apply this task skill.\n<skill>\n{instructions}\n</skill>";
export const PROPOSER = `You write the skill for a task: guidance that a language model reads before solving one input of the task.

Candidate examples of the task, numbered, each with its verified correct output:

{candidates}
{mistakes}
The verified output is exactly the text inside <output>; a label at the end of an input (such as "Output:" or "Answer:") belongs to the input. The model will see only the skill and one new input, so do not restate the task instructions already in the input.

Write the skill under exactly these four headers, each on its own line and in this order: "Format:", "Rules:", "Avoid:", "Examples:".
Under Format, one sentence on the exact format, language and length of a correct output for this task.
Under Rules, one to three lines starting with "- ", on how to decide the correct output.
Under Avoid, three to five lines starting with "- ", each a typical wrong output, then ": ", then why it is wrong, based on the model's mistakes above when given.
Under Examples, five to ten lines starting with "- ", each a short input, then " => ", then its correct output. Take them from the candidates above, keep the task's language, and keep each example under 30 words: shorten, abstract or merge long inputs and outputs, mark cuts with "...", and keep only what decides the output.

Start your reply with "Format:". Keep the whole skill under 350 words.`;
export const MISTAKES = `
The model's current outputs for some of these inputs are wrong; base the Avoid list on mistakes like these:

{items}
`;

export class HarnessSpecificationError extends Error {}

const SKILL_ID = /^[A-Za-z0-9][A-Za-z0-9._:-]*$/;

export function parseHarnessSpecification(text: string): HarnessSpecification {
  let value: unknown;
  try { value = JSON.parse(text); }
  catch { throw new HarnessSpecificationError("Harness specification must be valid JSON."); }
  const object = value as Partial<HarnessSpecification> | null;
  if (!object || typeof object !== "object" || Array.isArray(object)
    || Object.keys(object).sort().join(",") !== "schemaVersion,skills"
    || object.schemaVersion !== "hero-harness-spec-v1" || !Array.isArray(object.skills)) {
    throw new HarnessSpecificationError("Harness specification requires schemaVersion and skills.");
  }
  const ids = new Set<string>(), prefixes = new Set<string>();
  for (const skill of object.skills) {
    if (!skill || typeof skill !== "object" || Object.keys(skill).sort().join(",") !== "id,instructions,promptPrefix"
      || typeof skill.id !== "string" || !SKILL_ID.test(skill.id)
      || typeof skill.promptPrefix !== "string" || typeof skill.instructions !== "string") {
      throw new HarnessSpecificationError("Every skill requires id, promptPrefix and instructions.");
    }
    const prefix = normalizeRoutingText(skill.promptPrefix);
    if (ids.has(skill.id) || prefixes.has(prefix)) throw new HarnessSpecificationError("Skill IDs and normalized prefixes must be unique.");
    ids.add(skill.id); prefixes.add(prefix);
  }
  return object as HarnessSpecification;
}

/** The skill with the longest matching prompt prefix (NFKC, collapsed whitespace, lowercase), if any. */
export function routeSkill(specification: HarnessSpecification, prompt: string): Skill | undefined {
  const key = normalizeRoutingText(prompt);
  let best: Skill | undefined;
  for (const skill of specification.skills) {
    const prefix = normalizeRoutingText(skill.promptPrefix);
    if (key.startsWith(prefix) && (best === undefined || prefix.length > normalizeRoutingText(best.promptPrefix).length)) best = skill;
  }
  return best;
}

/** The registry with `skill` added, replacing a skill with the same normalized prefix or the same id. */
export function withSkill(specification: HarnessSpecification, skill: Skill): HarnessSpecification {
  const prefix = normalizeRoutingText(skill.promptPrefix);
  return {
    schemaVersion: "hero-harness-spec-v1",
    skills: [...specification.skills.filter((s) => normalizeRoutingText(s.promptPrefix) !== prefix && s.id !== skill.id), skill]
  };
}

/** Python's `json.dumps(..., ensure_ascii=False, sort_keys=True)` of the registry, which the token budget counts. */
export function serializeHarness(specification: HarnessSpecification): string {
  const skill = (s: Skill) => `{"id": ${JSON.stringify(s.id)}, "instructions": ${JSON.stringify(s.instructions)}, "promptPrefix": ${JSON.stringify(s.promptPrefix)}}`;
  return `{"schemaVersion": "hero-harness-spec-v1", "skills": [${specification.skills.map(skill).join(", ")}]}`;
}

/** Longest shared prompt prefix, cut back to a whitespace boundary when one exists. */
export function commonPrefix(prompts: string[]): string {
  if (prompts.length === 0) return "";
  const sorted = [...prompts].sort();
  const first = sorted[0]!, last = sorted.at(-1)!;
  let size = 0;
  while (size < first.length && size < last.length && first[size] === last[size]) size += 1;
  const prefix = first.slice(0, size);
  const cut = Math.max(prefix.lastIndexOf(" "), prefix.lastIndexOf("\n"));
  return cut > 0 ? prefix.slice(0, cut + 1) : prefix;
}

/** Records whose answers open differently come first, so a task with several output conventions shows several. */
export function distinctFirst<T extends { answer?: string }>(records: T[]): T[] {
  const distinct: T[] = [], rest: T[] = [], openings = new Set<string>();
  for (const record of records) {
    const opening = (record.answer ?? "").toLowerCase().split(/\s+/u).filter(Boolean).slice(0, 2).join(" ");
    (openings.has(opening) ? rest : distinct).push(record);
    openings.add(opening);
  }
  return [...distinct, ...rest];
}

/** The proposer's request: numbered candidates with verified outputs and the model's current mistakes, all
 * already clipped. */
export function proposalPrompt(candidates: Array<{ input: string; output: string }>,
  mistakes: Array<{ input: string; prediction: string; output: string }> = []): string {
  const items = candidates.map(({ input, output }, k) =>
    `<candidate n="${k + 1}">\n<input>\n${input}\n</input>\n<output>\n${output}\n</output>\n</candidate>`).join("\n\n");
  const wrong = mistakes.map(({ input, prediction, output }) =>
    `<mistake>\n<input>\n${input}\n</input>\n<model_output>\n${prediction}\n</model_output>\n<output>\n${output}\n</output>\n</mistake>`)
    .join("\n\n");
  return PROPOSER.replace("{candidates}", () => items)
    .replace("{mistakes}", () => (mistakes.length ? MISTAKES.replace("{items}", () => wrong) : ""));
}

/** The skill in the proposer's reply: from its Format line on when there is one, else the whole reply. */
export function skillText(reply: string): string {
  const lines = reply.trim().split(/\r?\n/u);
  const first = lines.findIndex((line) => line.trim().replace(/^[*#]+/u, "").trim().toLowerCase().startsWith("format:"));
  return lines.slice(Math.max(first, 0)).join("\n").trim();
}

/** Residual context: only the routed skill as the system message and the bare prompt; no answer is replayed. */
export function residualMessages(prompt: string, skill: Skill | undefined): Array<{ role: string; content: string }> {
  return [
    ...(skill ? [{ role: "system", content: HARNESS_SYSTEM.replace("{instructions}", () => skill.instructions) }] : []),
    { role: "user", content: prompt }
  ];
}

export interface SkillGuardInput {
  registryTokens: number;
  registryTokenBudget: number;
  /** Mean residual (nats per token) of the window under the current and the candidate registry. */
  residual: { previous: number; candidate: number };
}

export interface SkillGuardResult {
  committed: boolean;
  /** previous minus candidate residual: positive when the skill helps the checkpoint predict the window. */
  gain: number;
  reasons: string[];
}

/** The candidate is committed only if it lowers the window's mean residual and fits the registry budget. */
export function guardSkill(input: SkillGuardInput): SkillGuardResult {
  for (const [name, value] of [["residual.previous", input.residual.previous], ["residual.candidate", input.residual.candidate]] as const) {
    if (!Number.isFinite(value) || value < 0) throw new Error(`${name} must be finite and non-negative.`);
  }
  assertNonNegativeInteger("registryTokens", input.registryTokens);
  assertPositiveInteger("registryTokenBudget", input.registryTokenBudget);
  const gain = input.residual.previous - input.residual.candidate;
  const reasons: string[] = [];
  if (!(gain > 0)) reasons.push("window-residual-not-lowered");
  if (input.registryTokens > input.registryTokenBudget) reasons.push("registry-budget-exceeded");
  return { committed: reasons.length === 0, gain, reasons };
}

export type EvidenceRecord = Required<Pick<DatasetItem, "id" | "prompt" | "answer">>;

function normalizeRoutingText(value: string): string {
  return value.normalize("NFKC").replace(/\s+/gu, " ").trim().toLowerCase();
}

function assertPositiveInteger(name: string, value: number): void {
  if (!Number.isSafeInteger(value) || value <= 0) throw new Error(`${name} must be a positive integer.`);
}

function assertNonNegativeInteger(name: string, value: number): void {
  if (!Number.isSafeInteger(value) || value < 0) throw new Error(`${name} must be a non-negative integer.`);
}
