import { describe, expect, it } from "vitest";
import {
  commonPrefix,
  distinctFirst,
  guardSkill,
  parseHarnessSpecification,
  proposalPrompt,
  routeSkill,
  serializeHarness,
  skillText,
  teacherMessages,
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

  it("builds the proposal from distinct answer openings and the common prompt prefix", () => {
    const records = [{ answer: "A first" }, { answer: "a first again" }, { answer: "B other" }];
    expect(distinctFirst(records).map((r) => r.answer)).toEqual(["A first", "B other", "a first again"]);
    expect(commonPrefix(["Classify this text: one", "Classify this text: two"])).toBe("Classify this text: ");
    const prompt = proposalPrompt([{ input: "in", output: "out" }]);
    expect(prompt).toContain('<candidate n="1">\n<input>\nin\n</input>\n<output>\nout\n</output>\n</candidate>');
    expect(prompt).toContain('"Format:", "Rules:", "Avoid:", "Examples:"');
    expect(skillText("Preamble\nFormat: one word")).toBe("Format: one word");
    expect(proposalPrompt([], [{ input: "x", prediction: "wrong", output: "right" }])).toContain("<model_output>\nwrong\n</model_output>");
  });

  it("gives the teacher the routed skill and the verified answer as its own previous turn", () => {
    expect(teacherMessages("q", registry.skills[0], "ans")).toEqual([
      { role: "system", content: "Apply this task skill.\n<skill>\nx\n</skill>" },
      { role: "user", content: "q" },
      { role: "assistant", content: "ans" },
      { role: "user", content: "Repeat your previous answer exactly, with no other text." }
    ]);
    expect(teacherMessages("q", undefined, "ans")[0]?.role).toBe("user");
  });

  it("commits unless the skill is empty, overflows the registry or raises teacher NLL beyond the tolerance", () => {
    const base = { instructions: "do it", registryTokens: 100, registryTokenBudget: 8192, harmTolerance: 0.3 };
    expect(guardSkill({ ...base, teacherNll: { previous: 0.02, candidate: 0.25 } })).toMatchObject({ committed: true });
    expect(guardSkill({ ...base, teacherNll: { previous: 0.02, candidate: 0.4 } }))
      .toMatchObject({ committed: false, reasons: ["teacher-nll-rise-exceeds-tolerance"] });
    expect(guardSkill({ ...base, instructions: " ", registryTokens: 9000, teacherNll: { previous: 0.1, candidate: 0.05 } }))
      .toMatchObject({ committed: false, reasons: ["empty-skill", "registry-budget-exceeded"] });
    expect(guardSkill({ ...base, teacherNll: { previous: 0.1, candidate: 0.05 } }).gain).toBeCloseTo(0.05);
  });

});
