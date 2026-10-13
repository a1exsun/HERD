import { describe, expect, it } from "vitest";
import { QWEN_MODEL_REVISION } from "@hero/core";
import {
  streamResumeStage,
  type LocalModelRegistry,
  type LocalModelVersion
} from "../src/localModelManager.js";

const plan = "a".repeat(64);
const base: LocalModelVersion = {
  id: "base", stage: 0, parentVersionId: null, adapter: null, materializedSha256: "0".repeat(64), installedAt: ""
};

function registry(active: LocalModelVersion): LocalModelRegistry {
  return {
    schemaVersion: "hero-local-model-registry-v1",
    model: "Qwen/Qwen3.5-4B",
    modelRevision: QWEN_MODEL_REVISION,
    base: { path: "/base", sha256: "0".repeat(64), bytes: 1 },
    activeVersionId: active.id,
    activeCheckpointPath: "/active",
    versions: active.id === "base" ? [base] : [base, active]
  };
}

function streamStage(stage: number, planSha256: string): LocalModelVersion {
  return {
    ...base, id: `stage-${stage}-x`, stage, parentVersionId: "base",
    stream: {
      planSha256, taskId: `trace:T${stage}`, stageJsonSha256: "b".repeat(64), parentStageJsonSha256: null,
      weightsFingerprint: "c".repeat(64), origin: "/r"
    }
  };
}

describe("stream stage installation", () => {
  it("starts at stage 1 on base and after the active stage of the same plan", () => {
    expect(streamResumeStage(registry(base), plan, 3)).toBe(1);
    expect(streamResumeStage(registry(streamStage(2, plan)), plan, 3)).toBe(3);
  });

  it("refuses a different run's or a manual adapter's version, and stages already installed", () => {
    expect(() => streamResumeStage(registry(streamStage(2, "d".repeat(64))), plan, 3)).toThrow(/not a stage of this stream run/);
    expect(() => streamResumeStage(registry({ ...base, id: "stage-1-m", stage: 1 }), plan, 3)).toThrow(/not a stage of this stream run/);
    expect(() => streamResumeStage(registry(streamStage(3, plan)), plan, 3)).toThrow(/already at stage 3/);
  });
});
