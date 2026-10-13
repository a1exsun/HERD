import { spawnSync } from "node:child_process";
import { copyFileSync, mkdirSync } from "node:fs";
import { createRequire } from "node:module";
import { connect } from "node:net";
import { homedir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const repositoryRoot = dirname(dirname(fileURLToPath(import.meta.url)));
const pluginPath = join(repositoryRoot, "packages", "hero-dsh-plugin");
const workspacePath = join(repositoryRoot, "config", "dsh", "profile-pnpm-workspace.yaml");
const hostProfilePath = join(homedir(), ".dsh", "profiles", "web");
const hostOrigin = "http://127.0.0.1:3080";
const environment = { ...process.env };
delete environment.DSH_HOME;
const dshVersion = createRequire(import.meta.url)("@deepseek-ai/dsh/package.json").version;

mkdirSync(hostProfilePath, { recursive: true });
copyFileSync(workspacePath, join(hostProfilePath, "pnpm-workspace.yaml"));
runDsh([
  "plugin", "--profile", "web", "add",
  `@deepseek-ai/dsh-base@${dshVersion}`,
  `@deepseek-ai/dsh-web-app@${dshVersion}`,
  pluginPath
]);
const hostState = await inspectHost();
if (hostState === "ready") {
  throw new Error("HERO is already running on port 3080. Stop it before restarting the current build.");
}
if (hostState === "occupied") {
  throw new Error("Port 3080 is occupied by a process that has not loaded the HERO plugin.");
}
runDsh(["web", "--port", "3080", "--no-open"], true);

async function inspectHost() {
  if (!await portIsOpen()) return "free";
  try {
    const response = await fetch(`${hostOrigin}/hero/api/manifest`, {
      signal: AbortSignal.timeout(2_000)
    });
    if (!response.ok) return "occupied";
    const manifest = await response.json();
    return manifest?.name === "HERD" ? "ready" : "occupied";
  } catch {
    return "occupied";
  }
}

function portIsOpen() {
  return new Promise((resolveOpen) => {
    const socket = connect({ host: "127.0.0.1", port: 3080 });
    const finish = (open) => {
      socket.destroy();
      resolveOpen(open);
    };
    socket.setTimeout(1_000, () => finish(false));
    socket.once("connect", () => finish(true));
    socket.once("error", () => finish(false));
  });
}

function runDsh(arguments_, allowInterrupt = false) {
  const result = spawnSync("npx", ["--no-install", "dsh", ...arguments_], {
    cwd: repositoryRoot,
    env: environment,
    stdio: "inherit"
  });
  if (result.error !== undefined) throw result.error;
  if (allowInterrupt && (result.status === 130 || result.signal === "SIGINT")) return;
  if (result.status !== 0) throw new Error(`dsh ${arguments_.join(" ")} exited with ${result.status}.`);
}
