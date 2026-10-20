import React from "react";
import { createRoot } from "react-dom/client";
import type { HeroModuleId } from "@hero/core";
import { Icon, Workspace, useModuleNavigation } from "./workspace.js";
import "./styles.css";

interface Manifest {
  name: string;
  fullName: string;
  version: string;
  harnessEvolvingModel: { provider: string; model: string };
  continualLearningModel: { provider: string; model: string };
  modules: Array<{ id: HeroModuleId; name: string; description: string }>;
  runtime: {
    dsh: string;
    continualLearningModelRevision: string;
    harnessModelReady: boolean;
    harnessModelName?: string;
    localModel: LocalModelSnapshot;
  };
}

type LocalModelStatus = "missing" | "stopped" | "starting" | "ready" | "installing" | "activating" | "error";

interface LocalModelVersion {
  id: string;
  stage: number;
  parentVersionId: string | null;
  adapter: null | { path: string; sha256: string; bytes: number };
  materializedSha256: string;
  installedAt: string;
  sourceJobId?: string;
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

interface LocalModelSnapshot {
  status: LocalModelStatus;
  bridgeUrl: string;
  managedProcess: boolean;
  activeVersionId?: string;
  error?: string;
  registry?: {
    schemaVersion: string;
    model: string;
    modelRevision: string;
    base: { path: string; sha256: string; bytes: number };
    activeVersionId: string;
    activeCheckpointPath: string;
    versions: LocalModelVersion[];
  };
}

interface HarnessProposal {
  proposal: string;
  skill: { id: string; promptPrefix: string; instructions: string };
  mistakes: number;
  window: number;
  candidateSpecification: string;
  proposerVersionId: string;
}

interface HarnessGuard {
  committed: boolean;
  gain: number;
  reasons: string[];
  residual: { previous: number; candidate: number };
  registryTokens: number;
  checkpointVersionId: string;
}

function parseEvidence(text: string): unknown[] {
  const value: unknown = JSON.parse(text);
  if (!Array.isArray(value)) throw new Error("Evidence must be a JSON array.");
  return value;
}

const MODULE_DETAILS: Record<HeroModuleId, { eyebrow: string; body: string; bullets: string[] }> = {
  "harness-evolving": {
    eyebrow: "Bounded non-parametric update",
    body: "The base model writes one skill per task from a bounded window of verified examples and the checkpoint's mistakes; the registry is committed only if it lowers the window's residual.",
    bullets: ["Base-model skill writer", "Window, skill and registry budgets", "High-residual retention"]
  },
  "drift-constrained-opd": {
    eyebrow: "Recursive parameter update",
    body: "Distill the committed harness into Qwen while the previous checkpoint and immutable base model constrain drift.",
    bullets: ["Distillation on verified answers, no rollouts", "Checkpoint + general anchors", "Lagrange multipliers on worst-group drift"]
  }
};

function App() {
  const [manifest, setManifest] = React.useState<Manifest>();
  const selected = useModuleNavigation();
  const [previousSpecification, setPreviousSpecification] = React.useState(
    JSON.stringify({ schemaVersion: "hero-harness-spec-v1", skills: [] }, null, 2)
  );
  const [taskId, setTaskId] = React.useState("trace:FOMC");
  const [evidence, setEvidence] = React.useState('[{"id":"example-1","prompt":"Task input","answer":"Verified output"}]');
  const [proposal, setProposal] = React.useState<HarnessProposal>();
  const [guard, setGuard] = React.useState<HarnessGuard>();
  const [status, setStatus] = React.useState("Loading DSH runtime…");
  const [startupError, setStartupError] = React.useState("");
  const [busy, setBusy] = React.useState(false);
  const [localModel, setLocalModel] = React.useState<LocalModelSnapshot>();

  React.useEffect(() => {
    apiGet<Manifest>("/hero/api/manifest")
      .then((runtime) => {
        setManifest(runtime);
        setLocalModel(runtime.runtime.localModel);
        setStatus(runtime.runtime.harnessModelReady ? "Harness model route ready" : "Harness model route unavailable");
      })
      .catch((error: unknown) => {
        const message = error instanceof Error ? error.message : String(error);
        setStatus(message);
        setStartupError(message);
      });
  }, []);

  React.useEffect(() => {
    let active = true;
    const refresh = async () => {
      try {
        const snapshot = await apiGet<LocalModelSnapshot>("/hero/api/models/local");
        if (active) setLocalModel(snapshot);
      } catch {
        // Keep the last trustworthy state while DSH or the model service restarts.
      }
    };
    const timer = window.setInterval(() => void refresh(), 3_000);
    return () => { active = false; window.clearInterval(timer); };
  }, []);

  async function proposeHarness() {
    setBusy(true);
    setProposal(undefined);
    setGuard(undefined);
    setStatus("The active local Qwen version is writing the skill…");
    try {
      const result = await apiPost<HarnessProposal>("/hero/api/harness-evolving/propose", {
        input: { taskId, specification: previousSpecification, evidence: parseEvidence(evidence) }
      });
      setProposal(result);
      setStatus(`Skill written by version ${result.proposerVersionId} from ${result.window} records and ${result.mistakes} mistakes; check it with the guard before committing.`);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : String(error));
    } finally {
      setBusy(false);
    }
  }

  async function guardHarness() {
    if (!proposal) return;
    setBusy(true);
    setStatus("Scoring the window's residual with and without the skill…");
    try {
      const result = await apiPost<HarnessGuard>("/hero/api/harness-evolving/guard", {
        input: { specification: previousSpecification, candidateSpecification: proposal.candidateSpecification, evidence: parseEvidence(evidence) }
      });
      setGuard(result);
      setStatus(result.committed ? "The guard commits this skill." : `The guard keeps the current registry: ${result.reasons.join(", ")}.`);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : String(error));
    } finally {
      setBusy(false);
    }
  }

  function commitHarness() {
    if (!proposal || !guard?.committed) return;
    setPreviousSpecification(proposal.candidateSpecification);
    setProposal(undefined);
    setGuard(undefined);
    setStatus("Skill committed to the current registry.");
  }

  const details = MODULE_DETAILS[selected];

  return (
    <Workspace selected={selected} version={manifest?.version} dshVersion={manifest?.runtime.dsh} harnessReady={manifest?.runtime.harnessModelReady} connectionFailed={Boolean(startupError)}>
      {startupError ? <div className="connectionError" role="alert"><div><strong>Could not load the workspace</strong><p>{startupError}</p></div><button className="secondary" type="button" onClick={() => window.location.reload()}>Reload workspace</button></div> : !manifest ? <p className="connectionLoading" role="status">Connecting to the DSH runtime…</p> : null}
      <div className="modelRail" aria-label="Learning environment">
        <div><span>Harness model</span><strong>{manifest ? `${manifest.harnessEvolvingModel.provider} / ${manifest.harnessEvolvingModel.model}` : startupError ? "Unavailable" : "Connecting…"}</strong></div>
        <div><span>Learning model</span><strong>{manifest?.continualLearningModel.model ?? (startupError ? "Unavailable" : "Connecting…")}</strong><small>Active version: {localModel?.activeVersionId ?? localModel?.status ?? (startupError ? "unknown" : "loading")}</small></div>
      </div>

      <details className="moduleProtocol" key={selected}>
        <summary>About this workflow <span>{details.eyebrow}</span></summary>
        <div><p>{details.body}</p><ul>{details.bullets.map((item) => <li key={item}>{item}</li>)}</ul></div>
      </details>

      <section hidden={selected !== "harness-evolving"} aria-label="Harness Evolving workspace">
        <div className="panel workbench">
          <div className="panelHeader"><div><h2>Write and guard a task skill</h2><p>The local Qwen writes the task's skill from a window of verified examples and its own mistakes on them. The guard commits it only if it lowers the window's mean residual and fits the registry budget.</p></div><span className="subtleTag">Instruction learning</span></div>
          <div className="harnessGrid">
            <div className="harnessInputs">
              <label><span className="fieldTitle">Task id <span>skill id</span></span>
                <input spellCheck={false} value={taskId} onChange={(event) => setTaskId(event.target.value)} />
              </label>
              <label><span className="fieldTitle">Current skill registry <span>JSON · K(t−1)</span></span>
                <textarea spellCheck={false} rows={8} value={previousSpecification} onChange={(event) => setPreviousSpecification(event.target.value)} />
              </label>
              <label><span className="fieldTitle">Task evidence <span>JSON array · A(t)</span></span>
                <textarea spellCheck={false} rows={6} value={evidence} onChange={(event) => setEvidence(event.target.value)} />
                <span className="fieldHint">Each record needs id, prompt and its verified answer. Records are taken in order up to 8,192 tokens; up to 16 of them, distinct answer openings first, and up to 4 mistakes are shown to the writer. The routing prefix is the prompts' common prefix.</span>
              </label>
            </div>
            <div className="candidatePanel">
              <div className="candidateHeading"><h3>Proposed skill</h3><span className="subtleTag">K′(t)</span></div>
              {proposal ? <>
                <label className="candidateOutput"><span className="srOnly">Proposed skill</span><textarea spellCheck={false} readOnly rows={6} value={proposal.skill.instructions} /></label>
                <p className="fieldHint">Routing prefix <code>{JSON.stringify(proposal.skill.promptPrefix)}</code> · written by version <code>{proposal.proposerVersionId}</code>{proposal.proposerVersionId !== "base" ? " (the stream runner writes with the base model; activate base for an identical proposal)" : ""}</p>
                {guard ? <p className="fieldHint">Window residual {guard.residual.previous.toFixed(4)} → {guard.residual.candidate.toFixed(4)} nats/token · registry {guard.registryTokens} tokens · checkpoint version <code>{guard.checkpointVersionId}</code> · <strong>{guard.committed ? "commit" : `keep current (${guard.reasons.join(", ")})`}</strong></p> : <p className="fieldHint">Run the guard before committing.</p>}
              </> : <div className="candidateEmpty"><Icon name="harness-evolving" /><strong>{busy ? "Writing the skill…" : "Your next skill starts here"}</strong><p>{busy ? "The local Qwen is reading the verified examples." : "Submit the task evidence to see a proposed skill here."}</p></div>}
            </div>
          </div>
          <div className="actionBar">
            <button className="primary" disabled={busy || !manifest?.runtime.harnessModelReady} onClick={() => void proposeHarness()} type="button">{busy && !proposal ? "Writing…" : "Write the skill"}<Icon name="arrow" /></button>
            <button className="secondary" disabled={busy || !proposal} onClick={() => void guardHarness()} type="button">Run the guard</button>
            <button className="secondary" disabled={busy || !guard?.committed} onClick={commitHarness} type="button">Commit</button>
            <p className="operationMessage" role="status">{status}</p>
          </div>
        </div>
      </section>

      <section hidden={selected !== "drift-constrained-opd"} aria-label="Drift-Constrained OPD workspace">
        <div className="opdWorkspace">
          <details className="moduleProtocol">
            <summary>Training protocol <span>Verified answers → stream job</span></summary>
            <div><strong>Drift-constrained distillation in one stream job</strong><p>The frozen previous checkpoint, given the routed harness skill and the verified answer replayed as its own previous turn, teaches a rank-32 LoRA student that sees only the bare prompt: full-vocabulary KL at every answer position, with no rollouts. Checkpoint and base drift (worst probe group's mean KL) each stay under 0.05 through Lagrange multipliers.</p><small>32 updates × 32 records · lr 1e-4 → 0 · BF16 · 1×40GB A100 · every stage in one job</small></div>
          </details>
          <LocalModelPanel snapshot={localModel} onSnapshot={setLocalModel} />
        </div>
      </section>

    </Workspace>
  );
}

function LocalModelPanel(props: {
  snapshot?: LocalModelSnapshot;
  onSnapshot: (snapshot: LocalModelSnapshot) => void;
}) {
  const activeRegistry = props.snapshot?.registry;
  const nextStage = (activeRegistry?.versions.find(({ id }) => id === activeRegistry.activeVersionId)?.stage ?? 0) + 1;
  const [stage, setStage] = React.useState(nextStage);
  const [localPath, setLocalPath] = React.useState("");
  const [message, setMessage] = React.useState("");
  const [busy, setBusy] = React.useState(false);

  React.useEffect(() => setStage(nextStage), [nextStage]);

  async function operate(path: string, body: unknown, progress: string) {
    setBusy(true);
    setMessage(progress);
    try {
      const snapshot = await apiPost<LocalModelSnapshot>(path, body);
      props.onSnapshot(snapshot);
      setMessage(`Active model is now ${snapshot.activeVersionId ?? snapshot.status}.`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : String(error));
    } finally {
      setBusy(false);
    }
  }

  async function smoke() {
    setBusy(true);
    setMessage("Sending a fixed completion through the DSH provider registry…");
    try {
      const result = await apiPost<{ response: string; localModel: LocalModelSnapshot }>(
        "/hero/api/models/local/dsh-smoke",
        {}
      );
      props.onSnapshot(result.localModel);
      setMessage(`DSH provider smoke passed: ${result.response}`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : String(error));
    } finally {
      setBusy(false);
    }
  }

  const snapshot = props.snapshot;
  const registry = snapshot?.registry;
  return <section className="modelManager" aria-labelledby="model-versions-title">
    <div className="modelManagerHeader">
      <div><p className="eyebrow">Local model</p><h2 id="model-versions-title">Qwen LoRA versions</h2></div>
      <span className={`statusPill ${snapshot?.status ?? "missing"}`}>
        <span className={`dot ${snapshot?.status === "ready" ? "ready" : ""}`} />
        {snapshot?.status ?? "loading"}
      </span>
    </div>
    {snapshot?.error ? <p className="modelError" role="alert">{snapshot.error}</p> : null}
    {registry === undefined ? (
      <div className="modelEmpty">
        <strong>Pinned Qwen base is not registered yet.</strong>
        <p>HERO will expose the model after the exact revision has downloaded and passed digest registration.</p>
      </div>
    ) : <>
      <div className="modelFacts">
        <span>Active <strong>{registry.activeVersionId}</strong></span>
        <span>Base <strong>{formatBytes(registry.base.bytes)}</strong></span>
        <span>Revision <strong><code>{registry.modelRevision.slice(0, 12)}</code></strong></span>
        <span>Endpoint <strong><code>{snapshot?.bridgeUrl}</code></strong></span>
      </div>
      <div className="modelActions">
        <button className="secondary" disabled={busy} onClick={() => void operate("/hero/api/models/local/restart", {}, "Restarting the active Qwen checkpoint…")} type="button">Restart local model</button>
        <button className="secondary" disabled={busy || snapshot?.status !== "ready"} onClick={() => void smoke()} type="button">Test through DSH</button>
      </div>
      <details className="adapterInstaller">
        <summary>Install stream stages from a run under this project’s experiments directory</summary>
        <div className="installerGrid">
          <label className="wide">Absolute stream run directory<input value={localPath} onChange={(event) => setLocalPath(event.target.value)} placeholder="/absolute/project/experiments/runs/hero-stream-pilots/p9F" /></label>
          <label>Install through stage<input min="1" type="number" value={stage} onChange={(event) => setStage(Number(event.target.value))} /></label>
          <button className="primary" disabled={busy || !localPath} onClick={() => void operate(
            "/hero/api/models/local/install-stream",
            { runPath: localPath, throughStage: stage },
            "Verifying and merging the stream stages into the active checkpoint, then restarting Qwen…"
          )} type="button">Install & activate</button>
        </div>
      </details>
      {message ? <p className="operationMessage" role="status">{message}</p> : null}
      <div className="tableScroll versionTable" tabIndex={0} role="region" aria-label="Model version history"><table>
        <thead><tr><th>Version</th><th>Stage</th><th>Parent</th><th>LoRA</th><th>Stream task</th><th>Source job</th><th>Installed</th><th><span className="srOnly">Actions</span></th></tr></thead>
        <tbody>{[...registry.versions].reverse().map((version) => <tr key={version.id}>
          <td><code>{version.id}</code>{version.id === registry.activeVersionId ? <span className="activeTag">active</span> : null}</td>
          <td>{version.stage}</td><td><code>{version.parentVersionId ?? "—"}</code></td>
          <td>{version.adapter ? formatBytes(version.adapter.bytes) : "immutable base"}</td>
          <td>{version.stream ? <span title={`${version.stream.origin}\nplan ${version.stream.planSha256}\nstage.json ${version.stream.stageJsonSha256}`}>{version.stream.taskId}</span> : "—"}</td>
          <td>{version.sourceJobId ?? "—"}</td><td>{new Date(version.installedAt).toLocaleString()}</td>
          <td><button className="tableAction" disabled={busy || version.id === registry.activeVersionId} onClick={() => void operate(
            "/hero/api/models/local/activate", { versionId: version.id }, `Reconstructing ${version.id}…`
          )} aria-label={`Activate model version ${version.id}`} type="button">Activate</button></td>
        </tr>)}</tbody>
      </table></div>
      <p className="modelFootnote">The base checkpoint is immutable. HERO retains each LoRA and keeps one materialized active checkpoint; activating an older version deterministically rebuilds it from the base and its adapter chain.</p>
    </>}
  </section>;
}

function formatBytes(value: number): string {
  if (value < 1024) return `${value} B`;
  const units = ["KiB", "MiB", "GiB", "TiB"];
  let amount = value;
  let unit = -1;
  do {
    amount /= 1024;
    unit += 1;
  } while (amount >= 1024 && unit < units.length - 1);
  return `${amount.toFixed(amount >= 10 ? 1 : 2)} ${units[unit]}`;
}

async function apiGet<T>(path: string): Promise<T> {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`${path} failed (${response.status}).`);
  return await response.json() as T;
}

async function apiPost<T>(path: string, body: unknown): Promise<T> {
  const response = await fetch(path, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body)
  });
  const result = await response.json() as T & { error?: string };
  if (!response.ok) throw new Error(result.error ?? `HERO request failed (${response.status}).`);
  return result;
}

const root = document.getElementById("root");
if (root === null) throw new Error("Missing HERO root element.");
createRoot(root).render(<React.StrictMode><App /></React.StrictMode>);
