import { describe, expect, it } from "vitest";
import {
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
  type HarnessSpecification
} from "../src/index.js";

const registry: HarnessSpecification = {
  schemaVersion: "hero-harness-spec-v1",
  skills: [
    { id: "trace:a", promptPrefix: "Solve the", instructions: "x" },
    { id: "trace:b", promptPrefix: "Solve the following math", instructions: "y" }
  ]
};

describe("harness evolving", () => {
  it("routes by the longest normalized prefix and replaces a skill by prefix or id", () => {
    expect(routeSkill(registry, "solve  the following MATH problem")?.id).toBe("trace:b");
    expect(routeSkill(registry, "Solve the riddle")?.id).toBe("trace:a");
    expect(routeSkill(registry, "Translate")).toBeUndefined();
    const replaced = withSkill(registry, { id: "trace:c", promptPrefix: "solve the", instructions: "z" });
    expect(replaced.skills.map((s) => s.id)).toEqual(["trace:b", "trace:c"]);
    expect(withSkill(registry, { id: "trace:a", promptPrefix: "Other", instructions: "w" }).skills.map((s) => s.id))
      .toEqual(["trace:b", "trace:a"]);
  });

  it("parses stream skill ids and rejects duplicate prefixes", () => {
    expect(parseHarnessSpecification(serializeHarness(registry))).toEqual(registry);
    const duplicate = { ...registry, skills: [...registry.skills, { id: "trace:d", promptPrefix: "SOLVE  the", instructions: "q" }] };
    expect(() => parseHarnessSpecification(JSON.stringify(duplicate))).toThrow("unique");
  });

  it("serializes like the stream runner's sorted-key JSON", () => {
    expect(serializeHarness({ schemaVersion: "hero-harness-spec-v1", skills: [{ id: "t", promptPrefix: "p", instructions: "i" }] }))
      .toBe('{"schemaVersion": "hero-harness-spec-v1", "skills": [{"id": "t", "instructions": "i", "promptPrefix": "p"}]}');
  });

  it("builds the proposal from distinct answer openings, the common prompt prefix and the mistakes", () => {
    const records = [{ answer: "A first" }, { answer: "a first again" }, { answer: "B other" }];
    expect(distinctFirst(records).map((r) => r.answer)).toEqual(["A first", "B other", "a first again"]);
    expect(commonPrefix(["Classify this text: one", "Classify this text: two"])).toBe("Classify this text: ");
    const prompt = proposalPrompt([{ input: "in", output: "out $&" }], [{ input: "x", prediction: "wrong", output: "B" }]);
    expect(prompt).toContain('<candidate n="1">\n<input>\nin\n</input>\n<output>\nout $&\n</output>\n</candidate>');
    expect(prompt).toContain("<model_output>\nwrong\n</model_output>");
    expect(prompt).toContain('Start your reply with "Format:". Keep the whole skill under 350 words.');
    expect(proposalPrompt([{ input: "in", output: "out" }])).not.toContain("<mistake>");
    expect(skillText("Sure.\n**Format:** one letter.\nRules:\n- x")).toBe("**Format:** one letter.\nRules:\n- x");
    expect(skillText("Answer with one letter.")).toBe("Answer with one letter.");
  });

  it("scores residuals with only the routed skill and the bare prompt", () => {
    expect(residualMessages("q", registry.skills[0])).toEqual([
      { role: "system", content: "Apply this task skill.\n<skill>\nx\n</skill>" },
      { role: "user", content: "q" }
    ]);
    expect(residualMessages("q", undefined)).toEqual([{ role: "user", content: "q" }]);
  });

  it("commits only a registry that lowers the window residual and fits the budget", () => {
    const base = { registryTokens: 100, registryTokenBudget: 24576 };
    expect(guardSkill({ ...base, residual: { previous: 1.2, candidate: 0.8 } })).toMatchObject({ committed: true });
    expect(guardSkill({ ...base, residual: { previous: 1.2, candidate: 0.8 } }).gain).toBeCloseTo(0.4);
    expect(guardSkill({ ...base, residual: { previous: 0.8, candidate: 0.8 } }))
      .toMatchObject({ committed: false, reasons: ["window-residual-not-lowered"] });
    expect(guardSkill({ ...base, registryTokens: 30000, residual: { previous: 1.2, candidate: 0.8 } }))
      .toMatchObject({ committed: false, reasons: ["registry-budget-exceeded"] });
  });
});
