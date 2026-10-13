import { createHash } from "node:crypto";
import { access, mkdir, readFile, rename, rm, writeFile } from "node:fs/promises";
import { join, relative, resolve } from "node:path";
import type { DatasetItem, MetricName, TaskSplit } from "@hero/core";
import {
  assertSplitIntegrity,
  coalesceItemsByPrompt,
  normalizePromptContent,
  partitionItems,
  type SplitCounts
} from "./index.js";

export const TRACE_PLUS_17_SCHEMA_VERSION = "hero-trace-plus-17-manifest-v1" as const;

export interface TaskProtocol {
  id: string;
  metric: MetricName;
  outputType: "closed" | "open";
}

export interface SuperNiTaskProtocol extends TaskProtocol {
  file: string;
  category: string;
}

export const TRACE_TASKS = [
  { id: "C-STANCE", metric: "strict_exact_match", outputType: "closed" },
  { id: "FOMC", metric: "strict_exact_match", outputType: "closed" },
  { id: "MeetingBank", metric: "rouge_l", outputType: "open" },
  { id: "Py150", metric: "edit_similarity", outputType: "open" },
  { id: "ScienceQA", metric: "leading_choice_accuracy", outputType: "closed" },
  { id: "NumGLUE-cm", metric: "strict_exact_match", outputType: "closed" },
  { id: "NumGLUE-ds", metric: "strict_exact_match", outputType: "closed" },
  { id: "20Minuten", metric: "sari", outputType: "open" }
] as const satisfies readonly TaskProtocol[];

export const SUPERNI_17_TASKS = [
  superNi("024", "task024_cosmosqa_answer_generation.json", "Question Answering", "rouge_l", "open"),
  superNi("074", "task074_squad1.1_question_generation.json", "Question Generation", "rouge_l", "open"),
  superNi("195", "task195_sentiment140_classification.json", "Sentiment Analysis", "exact_match", "closed"),
  superNi("511", "task511_reddit_tifu_long_text_summarization.json", "Summarization", "rouge_l", "open"),
  superNi("1219", "task1219_ted_translation_en_es.json", "Translation", "rouge_l", "open"),
  superNi("574", "task574_air_dialogue_sentence_generation.json", "Dialogue Generation", "rouge_l", "open"),
  superNi("192", "task192_hotpotqa_sentence_generation.json", "Explanation", "rouge_l", "open"),
  superNi("177", "task177_para-nmt_paraphrasing.json", "Paraphrasing", "rouge_l", "open"),
  superNi("064", "task064_all_elements_except_first_i.json", "Program Execution", "exact_match", "closed"),
  superNi("346", "task346_hybridqa_classification.json", "POS Tagging", "exact_match", "closed"),
  superNi("108", "task108_contextualabusedetection_classification.json", "Toxic Language Detection", "exact_match", "closed"),
  superNi("111", "task111_asset_sentence_simplification.json", "Text Simplification", "rouge_l", "open"),
  superNi("1415", "task1415_youtube_caption_corrections_grammar_correction.json", "Grammar Error Correction", "rouge_l", "open"),
  superNi("1447", "task1447_drug_extraction_ade.json", "Named Entity Recognition", "exact_match", "closed"),
  superNi("828", "task828_copa_commonsense_cause_effect.json", "Cause Effect Classification", "exact_match", "closed"),
  superNi("1728", "task1728_web_nlg_data_to_text.json", "Data to Text", "rouge_l", "open"),
  superNi("217", "task217_rocstories_ordering_answer_generation.json", "Sentence Ordering", "exact_match", "closed")
] as const;

export interface TracePlus17BuildConfig {
  datasetVersion: string;
  sourceRoot?: string;
  outputRoot: string;
  traceBenchmark: string;
  seed: number;
  requestedSplits: SplitCounts;
  sourceRevisions: {
    traceRepository: string;
    superNiRepository: string;
  };
}

export interface SplitDeviation {
  split: keyof SplitCounts;
  requested: number;
  realized: number;
  reason: "source-exhausted-after-deduplication";
}

export interface DatasetFileDescriptor {
  path: string;
  records: number;
  sha256: string;
  visibility: "agent-visible" | "hidden-scorer";
}

export interface TracePlus17TaskManifest {
  position: number;
  id: string;
  source: "trace" | "superni";
  sourceTaskId: string;
  category: string;
  outputType: "closed" | "open";
  metric: MetricName;
  sourceRecords: number;
  uniquePrompts: number;
  excludedRecords: {
    emptyPrompt: number;
    emptyReference: number;
    duplicatePrompt: number;
    crossSplitPrompt: number;
  };
  requested: SplitCounts;
  realized: SplitCounts;
  deviations: SplitDeviation[];
  files: {
    train: DatasetFileDescriptor;
    validationInputs: DatasetFileDescriptor;
    validationGold: DatasetFileDescriptor;
    testGold: DatasetFileDescriptor;
  };
}

export interface TracePlus17Manifest {
  schemaVersion: typeof TRACE_PLUS_17_SCHEMA_VERSION;
  datasetVersion: string;
  name: "TRACE+17";
  seed: number;
  taskCount: 25;
  requestedSplits: SplitCounts;
  sourceRevisions: TracePlus17BuildConfig["sourceRevisions"];
  sourceLayout: {
    traceBenchmark: string;
    traceRoot: string;
    superNiTaskRoot: string;
  };
  isolation: {
    validationTargets: "hidden-scorer-only";
    testRecords: "hidden-scorer-only";
    testUse: "evaluation-only";
  };
  integrity: {
    normalizedPromptRule: "NFKC-whitespace";
    crossSplitOverlap: 0;
    status: "passed";
  };
  totals: {
    requested: SplitCounts;
    realized: SplitCounts;
    deviationCount: number;
  };
  tasks: TracePlus17TaskManifest[];
}

interface SuperNiSource {
  Definition?: string[];
  Instances: Array<{ id?: string; input: string; output: Array<string | null> }>;
}

interface PreparedTask {
  id: string;
  source: "trace" | "superni";
  sourceTaskId: string;
  category: string;
  outputType: "closed" | "open";
  metric: MetricName;
  sourceRecords: number;
  uniquePrompts: number;
  excludedRecords: TracePlus17TaskManifest["excludedRecords"];
  split: TaskSplit;
}

export async function buildTracePlus17Dataset(
  config: TracePlus17BuildConfig,
  repositoryRoot: string
): Promise<TracePlus17Manifest> {
  validateBuildConfig(config);
  const rawSourceRoot = config.sourceRoot ?? process.env.HERO_TRACE_SOURCE_ROOT;
  if (!rawSourceRoot) {
    throw new Error(
      "HERO_TRACE_SOURCE_ROOT is required to build the TRACE+17 dataset.\n" +
      "Set HERO_TRACE_SOURCE_ROOT=/path/to/raw or pass --source-root <path>.\n" +
      "The directory must contain upstream source repositories:\n" +
      `  - TRACE: https://github.com/BeyonderXX/TRACE (revision: ${config.sourceRevisions.traceRepository})\n` +
      `  - SuperNI: https://github.com/allenai/natural-instructions (revision: ${config.sourceRevisions.superNiRepository})\n` +
      `Expected layout: <sourceRoot>/trace/${config.traceBenchmark}/ and <sourceRoot>/superni/natural-instructions-v2.8/tasks/`
    );
  }
  const sourceRoot = resolve(repositoryRoot, rawSourceRoot);
  const outputRoot = resolve(repositoryRoot, config.outputRoot);
  if (await exists(outputRoot)) throw new Error(`Dataset version already exists: ${outputRoot}.`);
  const stagingRoot = `${outputRoot}.building-${process.pid}`;
  await rm(stagingRoot, { recursive: true, force: true });
  await mkdir(stagingRoot, { recursive: true });

  try {
    const traceRoot = join(sourceRoot, "trace", config.traceBenchmark);
    const superNiTaskRoot = join(sourceRoot, "superni", "natural-instructions-v2.8", "tasks");
    const prepared = [
      ...await Promise.all(TRACE_TASKS.map((task) => prepareTraceTask(traceRoot, task, config))),
      ...await Promise.all(SUPERNI_17_TASKS.map((task) => prepareSuperNiTask(superNiTaskRoot, task, config)))
    ];
    assertSplitIntegrity(prepared.map((task) => ({ taskId: task.id, ...task.split })));

    const tasks: TracePlus17TaskManifest[] = [];
    for (let index = 0; index < prepared.length; index += 1) {
      const task = prepared[index];
      if (task === undefined) throw new Error(`Missing prepared task at position ${index + 1}.`);
      tasks.push(await writePreparedTask(stagingRoot, task, index + 1, config.requestedSplits));
    }
    const totals = totalSplitCounts(tasks);
    const manifest: TracePlus17Manifest = {
      schemaVersion: TRACE_PLUS_17_SCHEMA_VERSION,
      datasetVersion: config.datasetVersion,
      name: "TRACE+17",
      seed: config.seed,
      taskCount: 25,
      requestedSplits: config.requestedSplits,
      sourceRevisions: config.sourceRevisions,
      sourceLayout: {
        traceBenchmark: config.traceBenchmark,
        traceRoot: relative(repositoryRoot, traceRoot),
        superNiTaskRoot: relative(repositoryRoot, superNiTaskRoot)
      },
      isolation: {
        validationTargets: "hidden-scorer-only",
        testRecords: "hidden-scorer-only",
        testUse: "evaluation-only"
      },
      integrity: {
        normalizedPromptRule: "NFKC-whitespace",
        crossSplitOverlap: 0,
        status: "passed"
      },
      totals: {
        requested: multiplyCounts(config.requestedSplits, prepared.length),
        realized: totals.realized,
        deviationCount: totals.deviationCount
      },
      tasks
    };
    await writeFile(join(stagingRoot, "manifest.json"), `${JSON.stringify(manifest, null, 2)}\n`, "utf8");
    await rename(stagingRoot, outputRoot);
    return manifest;
  } catch (error) {
    await rm(stagingRoot, { recursive: true, force: true });
    throw error;
  }
}

export async function loadTracePlus17Manifest(path: string): Promise<TracePlus17Manifest> {
  return JSON.parse(await readFile(path, "utf8")) as TracePlus17Manifest;
}

export async function verifyTracePlus17Dataset(manifestPath: string): Promise<{
  taskCount: number;
  fileCount: number;
  recordCount: number;
}> {
  const manifest = await loadTracePlus17Manifest(manifestPath);
  if (manifest.schemaVersion !== TRACE_PLUS_17_SCHEMA_VERSION || manifest.taskCount !== 25 || manifest.tasks.length !== 25) {
    throw new Error("TRACE+17 manifest identity or task count is invalid.");
  }
  const datasetRoot = resolve(manifestPath, "..");
  const expectedOrder = [
    ...TRACE_TASKS.map(({ id }) => `trace:${id}`),
    ...SUPERNI_17_TASKS.map(({ id }) => `superni:${id}`)
  ];
  let fileCount = 0;
  let recordCount = 0;
  const globalIds = new Set<string>();
  const promptOwners = new Map<string, string>();
  for (let index = 0; index < manifest.tasks.length; index += 1) {
    const task = manifest.tasks[index];
    if (task === undefined || task.position !== index + 1 || task.id !== expectedOrder[index]) {
      throw new Error(`TRACE+17 task order mismatch at position ${index + 1}.`);
    }
    const loaded = new Map<keyof TracePlus17TaskManifest["files"], unknown[]>();
    for (const [name, descriptor] of Object.entries(task.files) as Array<[
      keyof TracePlus17TaskManifest["files"],
      DatasetFileDescriptor
    ]>) {
      const filePath = resolve(datasetRoot, descriptor.path);
      if (!filePath.startsWith(`${datasetRoot}/`)) throw new Error(`Dataset file escapes its version root: ${descriptor.path}.`);
      const content = await readFile(filePath, "utf8");
      const digest = createHash("sha256").update(content).digest("hex");
      if (digest !== descriptor.sha256) throw new Error(`Dataset hash mismatch: ${descriptor.path}.`);
      const records = parseJsonLines(content, descriptor.path);
      if (records.length !== descriptor.records) throw new Error(`Dataset record count mismatch: ${descriptor.path}.`);
      loaded.set(name, records);
      fileCount += 1;
      recordCount += records.length;
    }
    const visibleValidation = loaded.get("validationInputs") ?? [];
    if (visibleValidation.some((record) => hasTarget(record))) {
      throw new Error(`Validation targets leaked into agent-visible task ${task.id}.`);
    }
    const validationGold = (loaded.get("validationGold") ?? []) as DatasetItem[];
    if (JSON.stringify(visibleValidation) !== JSON.stringify(validationGold.map(({ id, prompt, metadata }) => ({ id, prompt, metadata })))) {
      throw new Error(`Validation inputs and gold do not align for ${task.id}.`);
    }
    for (const [split, name] of [["train", "train"], ["validation", "validationGold"], ["test", "testGold"]] as const) {
      const records = loaded.get(name) as DatasetItem[];
      if (records.length !== task.realized[split]) throw new Error(`Realized count mismatch for ${task.id}/${split}.`);
      const owner = `${task.id}/${split}`;
      for (const record of records) {
        if (globalIds.has(record.id)) throw new Error(`Duplicate global id: ${record.id}.`);
        globalIds.add(record.id);
        const prompt = normalizePromptContent(record.prompt);
        const previous = promptOwners.get(prompt);
        if (previous !== undefined && previous !== owner) {
          throw new Error(`Prompt overlaps ${previous} and ${owner}: ${record.id}.`);
        }
        promptOwners.set(prompt, owner);
      }
    }
    assertSplitIntegrity([{
      taskId: task.id,
      train: (loaded.get("train") ?? []) as DatasetItem[],
      validation: (loaded.get("validationGold") ?? []) as DatasetItem[],
      test: (loaded.get("testGold") ?? []) as DatasetItem[]
    }]);
  }
  return { taskCount: manifest.tasks.length, fileCount, recordCount };
}

async function prepareTraceTask(
  traceRoot: string,
  protocol: (typeof TRACE_TASKS)[number],
  config: TracePlus17BuildConfig
): Promise<PreparedTask> {
  const splitNames = { train: "train.json", validation: "eval.json", test: "test.json" } as const;
  let sourceRecords = 0;
  const excludedRecords = { emptyPrompt: 0, emptyReference: 0, duplicatePrompt: 0, crossSplitPrompt: 0 };
  const entries = await Promise.all((Object.keys(splitNames) as Array<keyof SplitCounts>).map(async (splitName) => {
    const records = JSON.parse(await readFile(join(traceRoot, protocol.id, splitNames[splitName]), "utf8")) as Array<{
      prompt: string;
      answer: string;
    }>;
    sourceRecords += records.length;
    const items = records.flatMap((record, index) => {
      if (!normalizePromptContent(record.prompt)) {
        excludedRecords.emptyPrompt += 1;
        return [];
      }
      if (!record.answer.trim()) {
        excludedRecords.emptyReference += 1;
        return [];
      }
      return [{
        id: `trace:${protocol.id}:${splitName}:${index}`,
        prompt: record.prompt,
        answer: record.answer,
        references: [record.answer],
        metadata: { source: "trace", sourceTaskId: protocol.id, sourceSplit: splitName, sourceIndex: index }
      }];
    });
    const coalesced = coalesceItemsByPrompt(`trace:${protocol.id}:${splitName}`, items);
    excludedRecords.duplicatePrompt += items.length - coalesced.uniquePromptCount;
    return [splitName, coalesced.items] as const;
  }));
  const sourceSplits = Object.fromEntries(entries) as unknown as TaskSplit;
  const testPrompts = promptSet(sourceSplits.test);
  const validation = sourceSplits.validation.filter((item) => !testPrompts.has(normalizePromptContent(item.prompt)));
  const validationPrompts = promptSet(validation);
  const train = sourceSplits.train.filter((item) => {
    const prompt = normalizePromptContent(item.prompt);
    return !testPrompts.has(prompt) && !validationPrompts.has(prompt);
  });
  excludedRecords.crossSplitPrompt =
    sourceSplits.validation.length - validation.length + sourceSplits.train.length - train.length;
  const decontaminated: TaskSplit = { train, validation, test: sourceSplits.test };
  const split: TaskSplit = {
    train: sampleUpTo(decontaminated.train, config.requestedSplits.train, stableSeed(config.seed, protocol.id, "train")),
    validation: sampleUpTo(
      decontaminated.validation,
      config.requestedSplits.validation,
      stableSeed(config.seed, protocol.id, "validation")
    ),
    test: sampleUpTo(decontaminated.test, config.requestedSplits.test, stableSeed(config.seed, protocol.id, "test"))
  };
  return {
    id: `trace:${protocol.id}`,
    source: "trace",
    sourceTaskId: protocol.id,
    category: traceCategory(protocol.id),
    outputType: protocol.outputType,
    metric: protocol.metric,
    sourceRecords,
    uniquePrompts: promptSet([...sourceSplits.train, ...sourceSplits.validation, ...sourceSplits.test]).size,
    excludedRecords,
    split
  };
}

async function prepareSuperNiTask(
  taskRoot: string,
  protocol: (typeof SUPERNI_17_TASKS)[number],
  config: TracePlus17BuildConfig
): Promise<PreparedTask> {
  const source = JSON.parse(
    (await readFile(join(taskRoot, protocol.file), "utf8")).replace(/\bNaN\b/g, "null")
  ) as SuperNiSource;
  const definition = source.Definition?.[0]?.trim();
  if (!definition) throw new Error(`SuperNI ${protocol.id} has no task definition.`);
  const excludedRecords = { emptyPrompt: 0, emptyReference: 0, duplicatePrompt: 0, crossSplitPrompt: 0 };
  const items = source.Instances.flatMap((instance, index) => {
    const references = instance.output.filter((value): value is string => typeof value === "string" && Boolean(value.trim()));
    if (references.length === 0) {
      excludedRecords.emptyReference += 1;
      return [];
    }
    const prompt = `${definition}\n\nInput:\n${instance.input}\n\nOutput:`;
    if (!normalizePromptContent(prompt)) {
      excludedRecords.emptyPrompt += 1;
      return [];
    }
    return [{
      id: `superni:${protocol.id}:${instance.id ?? index}`,
      prompt,
      answer: references[0],
      references,
      metadata: { source: "superni", sourceTaskId: protocol.id, sourceFile: protocol.file, sourceIndex: index }
    } satisfies DatasetItem];
  });
  const coalesced = coalesceItemsByPrompt(`superni:${protocol.id}`, items);
  excludedRecords.duplicatePrompt = items.length - coalesced.uniquePromptCount;
  const realized = fitCounts(coalesced.items.length, config.requestedSplits);
  return {
    id: `superni:${protocol.id}`,
    source: "superni",
    sourceTaskId: protocol.id,
    category: protocol.category,
    outputType: protocol.outputType,
    metric: protocol.metric,
    sourceRecords: source.Instances.length,
    uniquePrompts: coalesced.items.length,
    excludedRecords,
    split: partitionItems(
      `superni:${protocol.id}`,
      coalesced.items,
      realized,
      stableSeed(config.seed, protocol.id)
    )
  };
}

async function writePreparedTask(
  stagingRoot: string,
  task: PreparedTask,
  position: number,
  requested: SplitCounts
): Promise<TracePlus17TaskManifest> {
  const directoryName = `${String(position).padStart(2, "0")}-${task.source}-${sanitize(task.sourceTaskId)}`;
  const visibleRoot = join(stagingRoot, "agent-visible", "tasks", directoryName);
  const hiddenRoot = join(stagingRoot, "hidden-scorer", "tasks", directoryName);
  await mkdir(visibleRoot, { recursive: true });
  await mkdir(hiddenRoot, { recursive: true });
  const validationInputs = task.split.validation.map(({ id, prompt, metadata }) => ({ id, prompt, metadata }));
  const files = {
    train: await writeJsonLines(stagingRoot, join(visibleRoot, "train.jsonl"), task.split.train, "agent-visible"),
    validationInputs: await writeJsonLines(
      stagingRoot,
      join(visibleRoot, "validation.jsonl"),
      validationInputs,
      "agent-visible"
    ),
    validationGold: await writeJsonLines(
      stagingRoot,
      join(hiddenRoot, "validation.jsonl"),
      task.split.validation,
      "hidden-scorer"
    ),
    testGold: await writeJsonLines(stagingRoot, join(hiddenRoot, "test.jsonl"), task.split.test, "hidden-scorer")
  };
  const realized: SplitCounts = {
    train: task.split.train.length,
    validation: task.split.validation.length,
    test: task.split.test.length
  };
  return {
    position,
    id: task.id,
    source: task.source,
    sourceTaskId: task.sourceTaskId,
    category: task.category,
    outputType: task.outputType,
    metric: task.metric,
    sourceRecords: task.sourceRecords,
    uniquePrompts: task.uniquePrompts,
    excludedRecords: task.excludedRecords,
    requested,
    realized,
    deviations: deviations(requested, realized),
    files
  };
}

async function writeJsonLines(
  root: string,
  path: string,
  records: unknown[],
  visibility: DatasetFileDescriptor["visibility"]
): Promise<DatasetFileDescriptor> {
  const content = records.map((record) => JSON.stringify(record)).join("\n") + (records.length === 0 ? "" : "\n");
  await writeFile(path, content, "utf8");
  return {
    path: relative(root, path),
    records: records.length,
    sha256: createHash("sha256").update(content).digest("hex"),
    visibility
  };
}

function fitCounts(available: number, requested: SplitCounts): SplitCounts {
  const train = Math.min(requested.train, available);
  const validation = Math.min(requested.validation, available - train);
  const test = Math.min(requested.test, available - train - validation);
  return { train, validation, test };
}

function sampleUpTo<T>(items: T[], count: number, seed: number): T[] {
  const output = [...items];
  let state = seed >>> 0;
  for (let index = output.length - 1; index > 0; index -= 1) {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    const swap = state % (index + 1);
    const left = output[index];
    const right = output[swap];
    if (left !== undefined && right !== undefined) {
      output[index] = right;
      output[swap] = left;
    }
  }
  return output.slice(0, Math.min(count, output.length));
}

function promptSet(items: DatasetItem[]): Set<string> {
  return new Set(items.map(({ prompt }) => normalizePromptContent(prompt)));
}

function deviations(requested: SplitCounts, realized: SplitCounts): SplitDeviation[] {
  return (Object.keys(requested) as Array<keyof SplitCounts>)
    .filter((split) => realized[split] < requested[split])
    .map((split) => ({
      split,
      requested: requested[split],
      realized: realized[split],
      reason: "source-exhausted-after-deduplication"
    }));
}

function totalSplitCounts(tasks: TracePlus17TaskManifest[]): { realized: SplitCounts; deviationCount: number } {
  return tasks.reduce((total, task) => ({
    realized: {
      train: total.realized.train + task.realized.train,
      validation: total.realized.validation + task.realized.validation,
      test: total.realized.test + task.realized.test
    },
    deviationCount: total.deviationCount + task.deviations.length
  }), { realized: { train: 0, validation: 0, test: 0 }, deviationCount: 0 });
}

function multiplyCounts(counts: SplitCounts, factor: number): SplitCounts {
  return { train: counts.train * factor, validation: counts.validation * factor, test: counts.test * factor };
}

function stableSeed(seed: number, ...parts: string[]): number {
  let value = seed >>> 0;
  for (const character of parts.join(":")) value = (Math.imul(value, 31) + character.charCodeAt(0)) >>> 0;
  return value;
}

function traceCategory(taskId: string): string {
  const categories: Record<string, string> = {
    "C-STANCE": "Stance Classification",
    FOMC: "Monetary Policy Classification",
    MeetingBank: "Meeting Summarization",
    Py150: "Code Completion",
    ScienceQA: "Science Question Answering",
    "NumGLUE-cm": "Numerical Reasoning",
    "NumGLUE-ds": "Numerical Reasoning",
    "20Minuten": "Text Simplification"
  };
  const category = categories[taskId];
  if (category === undefined) throw new Error(`Unknown TRACE task ${taskId}.`);
  return category;
}

function superNi(
  id: string,
  file: string,
  category: string,
  metric: MetricName,
  outputType: "closed" | "open"
): SuperNiTaskProtocol {
  return { id, file, category, metric, outputType };
}

function sanitize(value: string): string {
  return value.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
}

function validateBuildConfig(config: TracePlus17BuildConfig): void {
  if (!/^[a-z0-9]+(?:[._-][a-z0-9]+)*$/.test(config.datasetVersion)) {
    throw new Error("datasetVersion must be a filesystem-safe version identifier.");
  }
  if (!config.outputRoot || !config.traceBenchmark) {
    throw new Error("outputRoot and traceBenchmark are required.");
  }
  if (!Number.isSafeInteger(config.seed)) throw new Error("seed must be an integer.");
  for (const [split, count] of Object.entries(config.requestedSplits)) {
    if (!Number.isSafeInteger(count) || count <= 0) throw new Error(`${split} count must be positive.`);
  }
  if (!config.sourceRevisions.traceRepository || !config.sourceRevisions.superNiRepository) {
    throw new Error("Both source revisions are required.");
  }
}

async function exists(path: string): Promise<boolean> {
  try {
    await access(path);
    return true;
  } catch {
    return false;
  }
}

function parseJsonLines(content: string, path: string): unknown[] {
  return content.split("\n").filter(Boolean).map((line, index) => {
    try {
      return JSON.parse(line) as unknown;
    } catch (error) {
      throw new Error(`Invalid JSONL at ${path}:${index + 1}.`, { cause: error });
    }
  });
}

function hasTarget(record: unknown): boolean {
  if (record === null || typeof record !== "object" || Array.isArray(record)) return true;
  return "answer" in record || "references" in record;
}
