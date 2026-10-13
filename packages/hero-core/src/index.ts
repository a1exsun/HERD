export const HERO_NAME = "HERD" as const;
export const HERO_FULL_NAME = "Harness-Evolving Recursive Distillation for Continual Learning" as const;
export const HERO_VERSION = "0.1.0" as const;

export const QWEN_MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a" as const;
export const DSH_RUNTIME_VERSION = "0.1.2-rc.1" as const;

export const CONTINUAL_LEARNING_MODEL = {
  provider: "huggingface",
  model: "Qwen/Qwen3.5-4B"
} as const;

/** The frozen base model writes each task's skill, served locally through DSH. */
export const HARNESS_EVOLVING_MODEL = {
  provider: "hero-qwen-local",
  model: CONTINUAL_LEARNING_MODEL.model
} as const;

export const HERO_MODULES = [
  {
    id: "harness-evolving",
    name: "Harness Evolving",
    description: "Propose and guard one bounded harness update per stage."
  },
  {
    id: "drift-constrained-opd",
    name: "Drift-Constrained OPD",
    description: "Install and serve the checkpoints of a stream run."
  }
] as const;

export type HeroModuleId = (typeof HERO_MODULES)[number]["id"];

export type MetricName =
  | "exact_match"
  | "strict_exact_match"
  | "leading_choice_accuracy"
  | "rouge_l"
  | "edit_similarity"
  | "sari";

export interface DatasetItem {
  id: string;
  prompt: string;
  answer?: string;
  references?: string[];
  metadata?: Record<string, unknown>;
}

export interface TaskSplit {
  train: DatasetItem[];
  validation: DatasetItem[];
  test: DatasetItem[];
}
