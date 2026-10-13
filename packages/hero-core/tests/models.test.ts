import { describe, expect, it } from "vitest";
import {
  CONTINUAL_LEARNING_MODEL,
  HARNESS_EVOLVING_MODEL,
  HERO_FULL_NAME,
  HERO_MODULES
} from "../src/index.js";

describe("HERO contracts", () => {
  it("uses the base model as its own skill proposer", () => {
    expect(CONTINUAL_LEARNING_MODEL.model).toBe("Qwen/Qwen3.5-4B");
    expect(HARNESS_EVOLVING_MODEL).toEqual({ provider: "hero-qwen-local", model: "Qwen/Qwen3.5-4B" });
  });

  it("exposes the workspace modules", () => {
    expect(HERO_FULL_NAME).toBe("Harness-Evolving Recursive Distillation for Continual Learning");
    expect(HERO_MODULES.map(({ name }) => name)).toEqual(["Harness Evolving", "Drift-Constrained OPD"]);
  });
});
