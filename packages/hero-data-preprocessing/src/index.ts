import type { DatasetItem, TaskSplit } from "@hero/core";

export * from "./tracePlus17.js";

export interface SplitCounts {
  train: number;
  validation: number;
  test: number;
}

export interface IntegrityUnit extends TaskSplit {
  taskId: string;
}

export interface CoalescedItems {
  items: DatasetItem[];
  sourceItemCount: number;
  uniquePromptCount: number;
}

type SplitName = keyof TaskSplit;

export function normalizePromptContent(prompt: string): string {
  return prompt.normalize("NFKC").replace(/\s+/gu, " ").trim();
}

export function coalesceItemsByPrompt(taskId: string, items: DatasetItem[]): CoalescedItems {
  const groups = new Map<string, { item: DatasetItem; references: Map<string, string> }>();
  for (const item of items) {
    const promptKey = normalizePromptContent(item.prompt);
    if (!promptKey) {
      throw new Error(`Task ${taskId} contains an empty normalized prompt for ${item.id}.`);
    }
    const references = referenceMap(item);
    const existing = groups.get(promptKey);
    if (existing === undefined) {
      groups.set(promptKey, { item: withReferences(item, references), references });
      continue;
    }
    for (const [key, value] of references) existing.references.set(key, value);
    existing.item = withReferences(existing.item, existing.references);
  }
  return {
    items: [...groups.values()].map(({ item }) => item),
    sourceItemCount: items.length,
    uniquePromptCount: groups.size
  };
}

export function partitionItems(
  taskId: string,
  items: DatasetItem[],
  counts: SplitCounts,
  seed: number
): TaskSplit {
  for (const [name, count] of Object.entries(counts)) {
    if (!Number.isSafeInteger(count) || count < 0) {
      throw new Error(`Task ${taskId} has an invalid ${name} count: ${count}.`);
    }
  }
  const sourceIds = new Set<string>();
  for (const item of items) {
    if (!item.id) throw new Error(`Task ${taskId} contains an item without a stable id.`);
    if (sourceIds.has(item.id)) throw new Error(`Task ${taskId} contains duplicate id ${item.id}.`);
    sourceIds.add(item.id);
    if (!normalizePromptContent(item.prompt)) {
      throw new Error(`Task ${taskId} contains an empty normalized prompt for ${item.id}.`);
    }
  }
  const uniquePrompts = new Set(items.map(({ prompt }) => normalizePromptContent(prompt)));
  if (uniquePrompts.size !== items.length) {
    throw new Error(`Task ${taskId} must coalesce normalized prompt duplicates before partitioning.`);
  }
  const requested = counts.train + counts.validation + counts.test;
  if (requested > items.length) {
    throw new Error(`Task ${taskId} requests ${requested} records but only ${items.length} are available.`);
  }
  const shuffled = seededShuffle(items, seed);
  const trainEnd = counts.train;
  const validationEnd = trainEnd + counts.validation;
  const split: TaskSplit = {
    train: shuffled.slice(0, trainEnd),
    validation: shuffled.slice(trainEnd, validationEnd),
    test: shuffled.slice(validationEnd, requested)
  };
  assertSplitIntegrity([{ taskId, ...split }]);
  return split;
}

export function assertSplitIntegrity(units: IntegrityUnit[]): void {
  const issues: string[] = [];
  const pairs: Array<[SplitName, SplitName]> = [
    ["train", "validation"],
    ["train", "test"],
    ["validation", "test"]
  ];
  for (const unit of units) {
    for (const splitName of ["train", "validation", "test"] as const) {
      const ids = new Set<string>();
      for (const item of unit[splitName]) {
        if (ids.has(item.id)) issues.push(`${unit.taskId}:${splitName} duplicate id ${item.id}`);
        ids.add(item.id);
        if (!normalizePromptContent(item.prompt)) {
          issues.push(`${unit.taskId}:${splitName} empty prompt ${item.id}`);
        }
      }
    }
    for (const [leftName, rightName] of pairs) {
      const leftIds = new Set(unit[leftName].map(({ id }) => id));
      const leftPrompts = new Set(unit[leftName].map(({ prompt }) => normalizePromptContent(prompt)));
      for (const item of unit[rightName]) {
        if (leftIds.has(item.id)) issues.push(`${unit.taskId}:${leftName}/${rightName} id ${item.id}`);
        if (leftPrompts.has(normalizePromptContent(item.prompt))) {
          issues.push(`${unit.taskId}:${leftName}/${rightName} prompt ${item.id}`);
        }
      }
    }
  }
  if (issues.length > 0) {
    throw new Error(`Dataset contamination detected: ${issues.join("; ")}.`);
  }
}

function referenceMap(item: DatasetItem): Map<string, string> {
  const values = [...(item.references ?? []), ...(item.answer === undefined ? [] : [item.answer])];
  return new Map(
    values
      .filter((value) => value.trim())
      .map((value) => [normalizePromptContent(value), value])
  );
}

function withReferences(item: DatasetItem, references: Map<string, string>): DatasetItem {
  const values = [...references.values()];
  return values.length === 0 ? { ...item } : { ...item, answer: values[0], references: values };
}

function seededShuffle<T>(items: T[], seed: number): T[] {
  const output = [...items];
  let state = seed >>> 0;
  for (let index = output.length - 1; index > 0; index -= 1) {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    const swapIndex = state % (index + 1);
    const current = output[index];
    const replacement = output[swapIndex];
    if (current !== undefined && replacement !== undefined) {
      output[index] = replacement;
      output[swapIndex] = current;
    }
  }
  return output;
}
