import { mkdtemp, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { createHash } from "node:crypto";
import { afterEach, describe, expect, it } from "vitest";
import {
  buildTracePlus17Dataset,
  verifyTracePlus17Dataset,
  SUPERNI_17_TASKS,
  TRACE_TASKS,
  type TracePlus17BuildConfig
} from "../src/tracePlus17.js";

const temporaryRoots: string[] = [];

afterEach(async () => {
  await Promise.all(temporaryRoots.splice(0).map((path) => rm(path, { recursive: true, force: true })));
});

describe("TRACE+17 builder", () => {
  it("declares the paper's exact ordered 8+17 stream", () => {
    expect(TRACE_TASKS.map(({ id }) => id)).toEqual([
      "C-STANCE", "FOMC", "MeetingBank", "Py150", "ScienceQA", "NumGLUE-cm", "NumGLUE-ds", "20Minuten"
    ]);
    expect(SUPERNI_17_TASKS.map(({ id }) => id)).toEqual([
      "024", "074", "195", "511", "1219", "574", "192", "177", "064", "346", "108", "111", "1415", "1447", "828", "1728", "217"
    ]);
  });

  it("writes isolated JSONL files and records shortages in the manifest", async () => {
    const root = await mkdtemp(join(tmpdir(), "hero-trace-plus-17-"));
    temporaryRoots.push(root);
    await createFixtureSources(root);
    const config: TracePlus17BuildConfig = {
      datasetVersion: "fixture-v1",
      sourceRoot: "raw",
      outputRoot: "datasets/fixture-v1",
      traceBenchmark: "trace-500",
      seed: 42,
      requestedSplits: { train: 2, validation: 2, test: 2 },
      sourceRevisions: { traceRepository: "trace-sha", superNiRepository: "superni-sha" }
    };
    const manifest = await buildTracePlus17Dataset(config, root);
    expect(manifest.taskCount).toBe(25);
    expect(manifest.totals.deviationCount).toBeGreaterThan(0);
    const first = manifest.tasks[0];
    expect(first?.realized).toEqual({ train: 2, validation: 1, test: 1 });
    expect(first?.deviations).toEqual([
      { split: "validation", requested: 2, realized: 1, reason: "source-exhausted-after-deduplication" },
      { split: "test", requested: 2, realized: 1, reason: "source-exhausted-after-deduplication" }
    ]);
    expect(first?.excludedRecords).toEqual({
      emptyPrompt: 0,
      emptyReference: 0,
      duplicatePrompt: 0,
      crossSplitPrompt: 0
    });
    const visibleValidation = await readFile(
      join(root, config.outputRoot, first?.files.validationInputs.path ?? "missing"),
      "utf8"
    );
    expect(visibleValidation).not.toContain("answer");
    expect(first?.files.testGold.visibility).toBe("hidden-scorer");
    expect(manifest.tasks[1]?.excludedRecords.crossSplitPrompt).toBe(1);

    // Repeated occurrences inside a split are valid; reuse in another task is not.
    const manifestPath = join(root, config.outputRoot, "manifest.json");
    const trainPath = join(root, config.outputRoot, first!.files.train.path);
    const original = JSON.parse((await readFile(trainPath, "utf8")).split("\n")[0]!);
    const repeated = { ...original, id: `${original.id}:repeat:1`,
      metadata: { ...original.metadata, resampledFrom: original.id } };
    const trainContent = await readFile(trainPath, "utf8") + JSON.stringify(repeated) + "\n";
    await writeFile(trainPath, trainContent);
    first!.files.train.sha256 = createHash("sha256").update(trainContent).digest("hex");
    first!.files.train.records += 1;
    first!.realized.train += 1;
    await writeFile(manifestPath, JSON.stringify(manifest));
    await expect(verifyTracePlus17Dataset(manifestPath)).resolves.toMatchObject({ taskCount: 25, fileCount: 100 });

    const other = manifest.tasks[1]!;
    const otherPath = join(root, config.outputRoot, other.files.train.path);
    const otherRows = (await readFile(otherPath, "utf8")).trim().split("\n").map((row) => JSON.parse(row));
    otherRows[0].prompt = original.prompt;
    const otherContent = otherRows.map((row) => JSON.stringify(row)).join("\n") + "\n";
    await writeFile(otherPath, otherContent);
    other.files.train.sha256 = createHash("sha256").update(otherContent).digest("hex");
    await writeFile(manifestPath, JSON.stringify(manifest));
    await expect(verifyTracePlus17Dataset(manifestPath)).rejects.toThrow("Prompt overlaps");
  });
});

async function createFixtureSources(root: string): Promise<void> {
  const traceRoot = join(root, "raw", "trace", "trace-500");
  for (const task of TRACE_TASKS) {
    const taskRoot = join(traceRoot, task.id);
    await mkdir(taskRoot, { recursive: true });
    await writeFile(join(taskRoot, "train.json"), JSON.stringify(traceRecords(task.id, "train", 2)), "utf8");
    await writeFile(join(taskRoot, "eval.json"), JSON.stringify(traceRecords(task.id, "validation", 1)), "utf8");
    const testRecords = traceRecords(task.id, "test", 1);
    if (task.id === "FOMC" && testRecords[0] !== undefined) {
      testRecords[0].prompt = "FOMC train prompt 0";
    }
    await writeFile(join(taskRoot, "test.json"), JSON.stringify(testRecords), "utf8");
  }
  const superNiRoot = join(root, "raw", "superni", "natural-instructions-v2.8", "tasks");
  await mkdir(superNiRoot, { recursive: true });
  for (const task of SUPERNI_17_TASKS) {
    await writeFile(join(superNiRoot, task.file), JSON.stringify({
      Definition: [`Definition for ${task.id}`],
      Instances: Array.from({ length: 4 }, (_, index) => ({
        id: `instance-${index}`,
        input: `input ${task.id} ${index}`,
        output: [`output ${task.id} ${index}`]
      }))
    }), "utf8");
  }
}

function traceRecords(taskId: string, split: string, count: number): Array<{ prompt: string; answer: string }> {
  return Array.from({ length: count }, (_, index) => ({
    prompt: `${taskId} ${split} prompt ${index}`,
    answer: `${taskId} ${split} answer ${index}`
  }));
}
