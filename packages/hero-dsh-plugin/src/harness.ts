import {
  HARNESS_SETTINGS,
  commonPrefix,
  distinctFirst,
  guardSkill,
  parseHarnessSpecification,
  proposalPrompt,
  residualMessages,
  routeSkill,
  serializeHarness,
  skillText,
  withSkill,
  type EvidenceRecord,
  type HarnessSpecification,
  type Skill
} from "@hero/harness-evolving";

type Message = { role: string; content: string };

/** The local Qwen bridge (packages/hero-drift-constrained-opd/python/src/hero_qwen/bridge.py) serving the active version. */
export class QwenBridgeClient {
  constructor(private readonly url: string) {}

  async versionId(): Promise<string> {
    const response = await fetch(`${this.url}/health`, { signal: AbortSignal.timeout(5_000) });
    if (!response.ok) throw new Error(`The local Qwen bridge is unavailable (${response.status}).`);
    return ((await response.json()) as { versionId: string }).versionId;
  }

  async clip(texts: string[], tokens: number, ellipsis = true, tail = 0): Promise<string[]> {
    return (await this.post<{ texts: string[] }>("/hero/clip", { texts, tokens, ellipsis, tail })).texts;
  }

  async tokenCounts(texts: string[]): Promise<number[]> {
    return (await this.post<{ counts: number[] }>("/hero/token-counts", { texts })).counts;
  }

  /** Mean NLL of the verified answer after the chat prompt: the residual when the prompt holds only the routed skill
   * and the bare input. */
  async targetNll(messages: Message[], target: string): Promise<number> {
    return (await this.post<{ nll: number }>("/hero/target-nll", { messages, target })).nll;
  }

  /** One greedy completion with thinking off. */
  async complete(messages: Message[], maxTokens: number): Promise<string> {
    const response = await this.post<{ choices: Array<{ message: { content: string | null } }> }>("/v1/chat/completions",
      { model: "Qwen/Qwen3.5-4B", messages, max_tokens: maxTokens, temperature: 0, stream: false });
    return response.choices[0]?.message.content ?? "";
  }

  private async post<T>(path: string, body: unknown): Promise<T> {
    const response = await fetch(`${this.url}${path}`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(600_000)
    });
    if (!response.ok) throw new Error(`Local Qwen ${path} failed (${response.status}): ${await response.text()}`);
    return (await response.json()) as T;
  }
}

export interface ProposeInput {
  taskId: string;
  specification: string;
  evidence: EvidenceRecord[];
  /** Prompts that define the task's routing prefix; the evidence prompts when omitted. */
  taskPrompts?: string[];
}

/** The served model writes the task's skill from the window and its own mistakes on it. The stream runner writes
 * with the base model θ0 and finds the mistakes with the frozen checkpoint; activate the matching versions to
 * reproduce it. */
export async function proposeSkill(bridge: QwenBridgeClient, input: ProposeInput) {
  const specification = parseHarnessSpecification(input.specification);
  if (!input.taskId.trim()) throw new Error("taskId must be non-empty.");
  const evidence = checkEvidence(input.evidence);
  const window = await windowOf(bridge, evidence);
  const longest = Math.max(...(await bridge.tokenCounts(evidence.map((r) => r.answer))));
  const cap = Math.max(64, Math.min(HARNESS_SETTINGS.mistakeNewTokens, 2 * longest + 16));
  const mistakes: Array<{ record: EvidenceRecord; prediction: string }> = [];
  for (const record of window) {
    if (mistakes.length === HARNESS_SETTINGS.mistakes) break;
    const prediction = (await bridge.complete([{ role: "user", content: record.prompt }], cap)).trim();
    if (normalize(prediction) !== normalize(record.answer)) mistakes.push({ record, prediction });
  }
  const candidates = distinctFirst(window).slice(0, HARNESS_SETTINGS.candidates);
  const inputTokens = HARNESS_SETTINGS.candidateInputTokens, outputTokens = HARNESS_SETTINGS.candidateOutputTokens;
  const clipInputs = (texts: string[]) => bridge.clip(texts, inputTokens, true, Math.floor(inputTokens / 3));
  const [inputs, outputs, mistakeInputs, predictions, mistakeOutputs] = await Promise.all([
    clipInputs(candidates.map((r) => r.prompt)),
    bridge.clip(candidates.map((r) => r.answer), outputTokens),
    clipInputs(mistakes.map((m) => m.record.prompt)),
    bridge.clip(mistakes.map((m) => m.prediction || " "), outputTokens),
    bridge.clip(mistakes.map((m) => m.record.answer), outputTokens)
  ]);
  const proposal = await bridge.complete([{ role: "user", content: proposalPrompt(
    candidates.map((_, k) => ({ input: inputs[k]!, output: outputs[k]! })),
    mistakes.map((_, k) => ({ input: mistakeInputs[k]!, prediction: predictions[k]!, output: mistakeOutputs[k]! })))
  }], HARNESS_SETTINGS.proposalMaxNewTokens);
  const [instructions] = await bridge.clip([skillText(proposal) || " "], HARNESS_SETTINGS.skillTokens, false);
  const skill: Skill = {
    id: input.taskId,
    promptPrefix: commonPrefix(input.taskPrompts?.length ? input.taskPrompts : evidence.map((r) => r.prompt)),
    instructions: instructions!.trim()
  };
  return {
    proposal,
    skill,
    mistakes: mistakes.length,
    window: window.length,
    candidateSpecification: JSON.stringify(withSkill(specification, skill), null, 2),
    proposerVersionId: await bridge.versionId()
  };
}

export interface GuardInput {
  specification: string;
  candidateSpecification: string;
  evidence: EvidenceRecord[];
  registryTokenBudget?: number;
}

/** Commit only if the candidate registry lowers the window's mean residual and fits the registry budget. */
export async function guardProposal(bridge: QwenBridgeClient, input: GuardInput) {
  const previous = parseHarnessSpecification(input.specification);
  const candidate = parseHarnessSpecification(input.candidateSpecification);
  const window = await windowOf(bridge, checkEvidence(input.evidence));
  const unchanged = new Set(previous.skills.map((s) => JSON.stringify(s)));
  const added = candidate.skills.filter((s) => !unchanged.has(JSON.stringify(s)));
  if (added.length !== 1) throw new Error("The candidate must add or replace exactly one skill.");
  const residual = {
    previous: await meanResidual(bridge, previous, window),
    candidate: await meanResidual(bridge, candidate, window)
  };
  const [registryTokens] = await bridge.tokenCounts([serializeHarness(candidate)]);
  const result = guardSkill({
    registryTokens: registryTokens!,
    registryTokenBudget: input.registryTokenBudget ?? HARNESS_SETTINGS.registryTokenBudget,
    residual
  });
  return { ...result, residual, registryTokens, checkpointVersionId: await bridge.versionId() };
}

/** A_t: whole records in the given order, each counted as prompt + answer + 16 tokens, until the next record would
 * overflow the window budget. */
async function windowOf(bridge: QwenBridgeClient, evidence: EvidenceRecord[]): Promise<EvidenceRecord[]> {
  const counts = await bridge.tokenCounts(evidence.flatMap((r) => [r.prompt, r.answer]));
  const window: EvidenceRecord[] = [];
  let used = 0;
  for (const [k, record] of evidence.entries()) {
    const size = counts[2 * k]! + counts[2 * k + 1]! + 16;
    if (used + size > HARNESS_SETTINGS.windowTokens) break;
    window.push(record);
    used += size;
  }
  if (window.length === 0) throw new Error("The first evidence record overflows the window budget.");
  return window;
}

async function meanResidual(bridge: QwenBridgeClient, registry: HarnessSpecification, window: EvidenceRecord[]): Promise<number> {
  let total = 0;
  for (const record of window) {
    total += await bridge.targetNll(residualMessages(record.prompt, routeSkill(registry, record.prompt)), record.answer);
  }
  return total / window.length;
}

function normalize(text: string): string {
  return text.normalize("NFKC").replace(/\s+/gu, " ").trim().toLowerCase();
}

function checkEvidence(evidence: unknown): EvidenceRecord[] {
  if (!Array.isArray(evidence) || evidence.length === 0) throw new Error("evidence must be a non-empty array.");
  for (const record of evidence) {
    if (!record || typeof record.id !== "string" || typeof record.prompt !== "string" || typeof record.answer !== "string"
      || !record.prompt || !record.answer) {
      throw new Error("Every evidence record needs id, prompt and a verified answer.");
    }
  }
  return evidence as EvidenceRecord[];
}
