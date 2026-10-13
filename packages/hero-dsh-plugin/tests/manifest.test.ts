import { describe, expect, it } from "vitest";
import { createManifest } from "../src/index.js";

describe("DSH plugin manifest", () => {
  it("publishes the installed runtime and fixed model roles", () => {
    const manifest = createManifest();
    expect(manifest.runtime.dsh).toBe("0.1.2-rc.1");
    expect(manifest.harnessEvolvingModel).toEqual({ provider: "hero-qwen-local", model: "Qwen/Qwen3.5-4B" });
    expect(manifest.continualLearningModel.model).toBe("Qwen/Qwen3.5-4B");
    expect(manifest.runtime.continualLearningModelRevision).toHaveLength(40);
    expect(manifest.runtime.localModel).toMatchObject({
      status: "missing",
      bridgeUrl: "http://127.0.0.1:8192",
      managedProcess: false
    });
    expect(manifest.modules).toHaveLength(2);
  });
});
