import { spawn, type ChildProcess } from "node:child_process";
import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { readFile } from "node:fs/promises";
import { isAbsolute, join, resolve, sep } from "node:path";
import { setTimeout as delay } from "node:timers/promises";
import { CONTINUAL_LEARNING_MODEL, QWEN_MODEL_REVISION } from "@hero/core";

export type LocalModelStatus = "missing" | "stopped" | "starting" | "ready" | "installing" | "activating" | "error";

export interface LocalModelVersion {
  id: string;
  stage: number;
  parentVersionId: string | null;
  adapter: null | {
    path: string;
    sha256: string;
    bytes: number;
  };
  materializedSha256: string;
  installedAt: string;
  sourceJobId?: string;
  /** Provenance of a stage installed from a stream run (hero_qwen.model_registry install-adapter --stream-run). */
  stream?: {
    planSha256: string;
    taskId: string;
    stageJsonSha256: string;
    parentStageJsonSha256: string | null;
    weightsFingerprint: string;
    adapterSha256?: string;
    origin: string;
  };
}

export interface LocalModelRegistry {
  schemaVersion: "hero-local-model-registry-v1";
  model: typeof CONTINUAL_LEARNING_MODEL.model;
  modelRevision: typeof QWEN_MODEL_REVISION;
  base: { path: string; sha256: string; bytes: number };
  activeVersionId: string;
  activeCheckpointPath: string;
  versions: LocalModelVersion[];
}

export interface LocalModelSnapshot {
  status: LocalModelStatus;
  bridgeUrl: string;
  managedProcess: boolean;
  activeVersionId?: string;
  error?: string;
  registry?: LocalModelRegistry;
}

export class LocalModelManager {
  private readonly modelRoot: string;
  private readonly pythonProject: string;
  readonly bridgeUrl = "http://127.0.0.1:8192";
  private child?: ChildProcess;
  private status: LocalModelStatus = "stopped";
  private error?: string;
  private operation?: Promise<void>;

  constructor(private readonly repositoryRoot: string) {
    this.modelRoot = join(repositoryRoot, "experiments", "checkpoints", "qwen3.5-4b");
    this.pythonProject = join(repositoryRoot, "packages", "hero-drift-constrained-opd", "python");
  }

  initialize(autostart: boolean): void {
    void this.serial(async () => {
      const registry = await this.readRegistry(false);
      if (registry === undefined) {
        this.status = "missing";
        return;
      }
      if (autostart) await this.startUnlocked(registry);
      else this.status = "stopped";
    }).catch(() => undefined);
  }

  async snapshot(): Promise<LocalModelSnapshot> {
    const registry = await this.readRegistry(false);
    return {
      status: registry === undefined ? "missing" : this.status,
      bridgeUrl: this.bridgeUrl,
      managedProcess: this.child !== undefined,
      ...(registry === undefined ? {} : { activeVersionId: registry.activeVersionId, registry }),
      ...(this.error === undefined ? {} : { error: this.error })
    };
  }

  async restart(): Promise<void> {
    await this.serial(async () => {
      const registry = await this.readRegistry(true);
      await this.stopUnlocked();
      await this.startUnlocked(registry);
    });
  }

  /** Install stages of a stream run already under experiments/, up to and including throughStage. */
  async installStreamRun(input: { runPath: string; throughStage: number }): Promise<void> {
    assertStage(input.throughStage);
    this.assertHeroArtifactPath(input.runPath);
    const runPath = resolve(input.runPath);
    await this.serial(async () => {
      await this.stopUnlocked();
      const planSha256 = await sha256File(join(runPath, "bundle", "plan.json"));
      const fromStage = streamResumeStage(await this.readRegistry(true), planSha256, input.throughStage);
      await this.installStreamStages(runPath, fromStage, input.throughStage, runPath);
      await this.startUnlocked(await this.readRegistry(true));
    });
  }

  async activate(versionId: string): Promise<void> {
    if (!versionId.trim()) throw new Error("versionId must be non-empty.");
    await this.serial(async () => {
      await this.stopUnlocked();
      this.status = "activating";
      await this.runRegistryCommand(["activate", "--version", versionId]);
      await this.startUnlocked(await this.readRegistry(true));
    });
  }

  async dispose(): Promise<void> {
    if (this.child !== undefined) await this.stopUnlocked();
  }

  private async serial(action: () => Promise<void>): Promise<void> {
    if (this.operation !== undefined) throw new Error("A local model operation is already running.");
    const operation = action().catch((error: unknown) => {
      this.status = "error";
      this.error = error instanceof Error ? error.message : String(error);
      throw error;
    }).finally(() => {
      this.operation = undefined;
    });
    this.operation = operation;
    await operation;
  }

  private async startUnlocked(registry: LocalModelRegistry): Promise<void> {
    const current = await this.probe();
    if (current !== undefined) {
      if (current.model === CONTINUAL_LEARNING_MODEL.model && current.versionId === registry.activeVersionId) {
        this.status = "ready";
        this.error = undefined;
        return;
      }
      throw new Error(`Port 8192 is occupied by a different model service (${current.model}/${current.versionId}).`);
    }
    this.status = "starting";
    this.error = undefined;
    let stderr = "";
    const child = spawn("uv", [
      "run", "--project", this.pythonProject, "--python", "3.12", "--extra", "bridge",
      "hero-qwen-bridge",
      "--checkpoint", registry.activeCheckpointPath,
      "--version-id", registry.activeVersionId,
      "--port", "8192"
    ], {
      cwd: this.repositoryRoot,
      env: { ...process.env },
      stdio: ["ignore", "ignore", "pipe"]
    });
    this.child = child;
    child.stderr?.setEncoding("utf8");
    child.stderr?.on("data", (chunk: string) => {
      stderr = `${stderr}${chunk}`.slice(-16_384);
    });
    child.once("exit", (code) => {
      if (this.child === child) {
        this.child = undefined;
        if (this.status !== "stopped") {
          this.status = "error";
          this.error = `Local Qwen bridge exited with ${code ?? "signal"}. ${stderr}`.trim();
        }
      }
    });
    const deadline = Date.now() + 300_000;
    while (Date.now() < deadline) {
      if (child.exitCode !== null) throw new Error(`Local Qwen bridge exited during startup. ${stderr}`);
      const health = await this.probe();
      if (health?.model === CONTINUAL_LEARNING_MODEL.model && health.versionId === registry.activeVersionId) {
        this.status = "ready";
        return;
      }
      await delay(1_000);
    }
    throw new Error(`Local Qwen bridge did not become ready within five minutes. ${stderr}`);
  }

  private async stopUnlocked(): Promise<void> {
    const child = this.child;
    if (child === undefined) {
      if (await this.probe() !== undefined) {
        throw new Error("The active Qwen bridge is not owned by this HERO process and cannot be restarted safely.");
      }
      this.status = "stopped";
      return;
    }
    this.status = "stopped";
    child.kill("SIGTERM");
    await Promise.race([
      new Promise<void>((resolveExit) => child.once("exit", () => resolveExit())),
      delay(30_000).then(() => { child.kill("SIGKILL"); })
    ]);
    this.child = undefined;
  }

  /** The registry verifies each stage against its stage.json, the plan and the run's parent chain. */
  private async installStreamStages(runPath: string, fromStage: number, throughStage: number, origin: string): Promise<void> {
    this.status = "installing";
    for (let stage = fromStage; stage <= throughStage; stage += 1) {
      try {
        await this.runRegistryCommand([
          "install-adapter", "--stream-run", runPath, "--stage", String(stage), "--stream-origin", origin
        ]);
      } catch (error) {
        const installed = stage > fromStage ? `Stages ${fromStage}-${stage - 1} were installed. ` : "";
        throw new Error(`${installed}Stream stage ${stage}: ${error instanceof Error ? error.message : String(error)}`);
      }
    }
  }

  private async runRegistryCommand(arguments_: string[]): Promise<void> {
    await this.runCommand("uv", [
      "run", "--project", this.pythonProject, "--python", "3.12", "python",
      "-m", "hero_qwen.model_registry", "--root", this.modelRoot, ...arguments_
    ], "Local model registry command");
  }

  private async runCommand(command: string, arguments_: string[], label: string, input?: string): Promise<{ stdout: string; stderr: string }> {
    return await new Promise<{ stdout: string; stderr: string }>((resolveRun, rejectRun) => {
      let stdout = "";
      let stderr = "";
      const child = spawn(command, arguments_, {
        cwd: this.repositoryRoot,
        env: { ...process.env },
        stdio: [input === undefined ? "ignore" : "pipe", "pipe", "pipe"]
      });
      child.stdout?.setEncoding("utf8");
      child.stdout?.on("data", (chunk: string) => { stdout = `${stdout}${chunk}`.slice(-1_048_576); });
      child.stderr?.setEncoding("utf8");
      child.stderr?.on("data", (chunk: string) => { stderr = `${stderr}${chunk}`.slice(-16_384); });
      child.once("error", rejectRun);
      child.once("exit", (code) => {
        if (code === 0) resolveRun({ stdout, stderr });
        else rejectRun(new Error(`${label} failed with ${code}. ${stderr}`.trim()));
      });
      child.stdin?.end(input);
    });
  }

  private async probe(): Promise<{ model?: string; versionId?: string } | undefined> {
    try {
      const response = await fetch(`${this.bridgeUrl}/health`, { signal: AbortSignal.timeout(1_000) });
      if (!response.ok) return undefined;
      return await response.json() as { model?: string; versionId?: string };
    } catch {
      return undefined;
    }
  }

  private async readRegistry(required: true): Promise<LocalModelRegistry>;
  private async readRegistry(required: false): Promise<LocalModelRegistry | undefined>;
  private async readRegistry(required: boolean): Promise<LocalModelRegistry | undefined> {
    try {
      const parsed: unknown = JSON.parse(await readFile(join(this.modelRoot, "registry.json"), "utf8"));
      if (parsed === null || typeof parsed !== "object" || Array.isArray(parsed)) throw new Error("Local model registry is invalid.");
      const registry = parsed as LocalModelRegistry;
      if (registry.schemaVersion !== "hero-local-model-registry-v1"
        || registry.model !== CONTINUAL_LEARNING_MODEL.model
        || registry.modelRevision !== QWEN_MODEL_REVISION) {
        throw new Error("Local model registry identity is invalid.");
      }
      return registry;
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === "ENOENT" && !required) return undefined;
      throw error;
    }
  }

  private assertHeroArtifactPath(candidate: string): void {
    if (!isAbsolute(candidate)) throw new Error("runPath must be absolute.");
    const artifactRoot = resolve(this.repositoryRoot, "experiments");
    const resolved = resolve(candidate);
    if (resolved !== artifactRoot && !resolved.startsWith(`${artifactRoot}${sep}`)) {
      throw new Error("runPath must stay inside the project experiments artifact root.");
    }
  }
}

/**
 * First stream stage to install: 1 on base, otherwise the stage after the active version when that version came
 * from the same plan (the registry then checks it is this run's preceding stage, by its stage.json digest).
 */
export function streamResumeStage(registry: LocalModelRegistry, planSha256: string, throughStage: number): number {
  const active = registry.versions.find(({ id }) => id === registry.activeVersionId);
  if (active === undefined) throw new Error("The active local model version is absent from the registry.");
  if (active.id !== "base" && active.stream?.planSha256 !== planSha256) {
    throw new Error(`Active version ${active.id} is not a stage of this stream run; activate base or the run's latest installed stage first.`);
  }
  if (throughStage <= active.stage) {
    throw new Error(`Active version ${active.id} is already at stage ${active.stage}; nothing to install through stage ${throughStage}.`);
  }
  return active.stage + 1;
}

function assertStage(stage: number): void {
  if (!Number.isSafeInteger(stage) || stage <= 0) throw new Error("throughStage must be a positive integer.");
}

async function sha256File(path: string): Promise<string> {
  const digest = createHash("sha256");
  for await (const chunk of createReadStream(path)) digest.update(chunk);
  return digest.digest("hex");
}
