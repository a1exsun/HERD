import { describe, expect, it } from "vitest";
import { guardProposal, proposeSkill, type QwenBridgeClient } from "../src/harness.js";

const empty = JSON.stringify({ schemaVersion: "hero-harness-spec-v1", skills: [] });
const evidence = [
  { id: "1", prompt: "Classify the stance: a", answer: "A" },
  { id: "2", prompt: "Classify the stance: b", answer: "A" },
  { id: "3", prompt: "Classify the stance: c", answer: "B" }
];

function fakeBridge(nll: (messages: Array<{ role: string; content: string }>) => number, words = 1) {
  const calls: Record<string, unknown[]> = { clip: [], complete: [], targetNll: [] };
  const bridge = {
    versionId: async () => "base",
    clip: async (texts: string[], tokens: number, ellipsis = true, tail = 0) => { calls.clip!.push({ texts, tokens, ellipsis, tail }); return texts; },
    tokenCounts: async (texts: string[]) => texts.map(() => words),
    targetNll: async (messages: Array<{ role: string; content: string }>, target: string) => { calls.targetNll!.push({ messages, target }); return nll(messages); },
    complete: async (messages: unknown, maxTokens: number) => {
      calls.complete!.push({ messages, maxTokens });
      return maxTokens === 768 ? "Here it is.\nFormat: one letter.\nRules:\n- pick" : "A";
    }
  };
  return { bridge: bridge as unknown as QwenBridgeClient, calls };
}

describe("harness evolution through the local bridge", () => {
  it("writes the skill from the window and the mistakes under the stream runner's limits", async () => {
    const { bridge, calls } = fakeBridge(() => 0);
    const result = await proposeSkill(bridge, { taskId: "trace:C-STANCE", specification: empty, evidence });
    expect(calls.complete!.slice(0, 3)).toEqual(evidence.map((r) => ({ messages: [{ role: "user", content: r.prompt }], maxTokens: 64 })));
    expect(result.mistakes).toBe(1);
    expect(calls.clip![0]).toEqual({ texts: ["Classify the stance: a", "Classify the stance: c", "Classify the stance: b"], tokens: 384, ellipsis: true, tail: 128 });
    expect(calls.clip![1]).toMatchObject({ tokens: 128 });
    expect(calls.complete![3]).toMatchObject({ maxTokens: 768 });
    expect(calls.clip!.at(-1)).toEqual({ texts: ["Format: one letter.\nRules:\n- pick"], tokens: 768, ellipsis: false, tail: 0 });
    expect(result.skill).toEqual({ id: "trace:C-STANCE", promptPrefix: "Classify the stance: ", instructions: "Format: one letter.\nRules:\n- pick" });
    expect(JSON.parse(result.candidateSpecification).skills).toEqual([result.skill]);
    expect(result.proposerVersionId).toBe("base");
  });

  it("keeps the window within its budget", async () => {
    const { bridge } = fakeBridge(() => 0, 3000);
    expect((await proposeSkill(bridge, { taskId: "t", specification: empty, evidence })).window).toBe(1);
  });

  it("commits unless the routed skill raises the teacher's NLL beyond the tolerance", async () => {
    const candidate = JSON.stringify({ schemaVersion: "hero-harness-spec-v1",
      skills: [{ id: "trace:C-STANCE", promptPrefix: "Classify the stance: ", instructions: "Answer with one letter." }] });
    const mild = fakeBridge((messages) => (messages[0]!.role === "system" ? 0.2 : 0.05));
    const accepted = await guardProposal(mild.bridge, { specification: empty, candidateSpecification: candidate, evidence });
    expect(accepted).toMatchObject({ committed: true, teacherVersionId: "base" });
    expect(accepted.teacherNll.previous).toBeCloseTo(0.05);
    expect(accepted.teacherNll.candidate).toBeCloseTo(0.2);
    expect(mild.calls.targetNll).toHaveLength(6);
    const harmful = fakeBridge((messages) => (messages[0]!.role === "system" ? 0.6 : 0.05));
    expect(await guardProposal(harmful.bridge, { specification: empty, candidateSpecification: candidate, evidence }))
      .toMatchObject({ committed: false, reasons: ["teacher-nll-rise-exceeds-tolerance"] });
    await expect(guardProposal(mild.bridge, { specification: empty, candidateSpecification: empty, evidence }))
      .rejects.toThrow("exactly one skill");
  });
});
