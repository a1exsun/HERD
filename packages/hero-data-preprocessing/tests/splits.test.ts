import { describe, expect, it } from "vitest";
import { assertSplitIntegrity, coalesceItemsByPrompt, partitionItems } from "../src/index.js";

describe("dataset preparation", () => {
  it("coalesces normalized prompts and preserves references", () => {
    const result = coalesceItemsByPrompt("demo", [
      { id: "a", prompt: "Ａ  prompt", answer: "one" },
      { id: "b", prompt: "A prompt", answer: "two" }
    ]);
    expect(result.uniquePromptCount).toBe(1);
    expect(result.items[0]?.references).toEqual(["one", "two"]);
  });

  it("partitions deterministically without overlap", () => {
    const items = Array.from({ length: 8 }, (_, index) => ({
      id: String(index),
      prompt: `prompt ${index}`,
      answer: `answer ${index}`
    }));
    expect(partitionItems("demo", items, { train: 4, validation: 2, test: 2 }, 7))
      .toEqual(partitionItems("demo", items, { train: 4, validation: 2, test: 2 }, 7));
  });

  it("rejects test leakage", () => {
    expect(() => assertSplitIntegrity([{
      taskId: "demo",
      train: [{ id: "a", prompt: "same" }],
      validation: [],
      test: [{ id: "b", prompt: " same " }]
    }])).toThrow("Dataset contamination");
  });
});
