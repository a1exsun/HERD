import { copyFileSync, mkdirSync, rmSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";

const repositoryRoot = dirname(dirname(fileURLToPath(import.meta.url)));
const runtimeRoot = join(repositoryRoot, "experiments", "runtime");
const testHome = resolve(process.env.HERO_DSH_PROFILE_HOME ?? join(runtimeRoot, "dsh-test"));
if (!testHome.startsWith(`${runtimeRoot}${sep}`)) throw new Error("DSH test profiles must stay inside experiments/runtime.");
const pluginPath = join(repositoryRoot, "packages", "hero-dsh-plugin");
const settingsPath = join(repositoryRoot, "config", "dsh", "test-settings.yaml");
const workspacePath = join(repositoryRoot, "config", "dsh", "profile-pnpm-workspace.yaml");
const mode = process.argv[2] ?? "setup";
const environment = { ...process.env, DSH_HOME: testHome };
const dshVersion = createRequire(import.meta.url)("@deepseek-ai/dsh/package.json").version;

rmSync(testHome, { recursive: true, force: true });
mkdirSync(testHome, { recursive: true });
runDsh(["--profile", "web", "--dump-default-config"], "ignore");
copyFileSync(workspacePath, join(testHome, "profiles", "web", "pnpm-workspace.yaml"));
runDsh([
  "plugin", "--profile", "web", "add",
  `@deepseek-ai/dsh-base@${dshVersion}`,
  `@deepseek-ai/dsh-web-app@${dshVersion}`,
  pluginPath
], "ignore");
copyFileSync(settingsPath, join(testHome, "settings.yaml"));

if (mode === "setup") process.exit(0);
if (mode === "config") runDsh(["--profile", "web", "--dump-config"], "inherit");
else throw new Error(`Unknown DSH test-profile mode: ${mode}.`);

function runDsh(arguments_, stdio) {
  const result = spawnSync("npx", ["--no-install", "dsh", ...arguments_], {
    cwd: repositoryRoot,
    env: environment,
    stdio
  });
  if (result.error !== undefined) throw result.error;
  if (result.status !== 0) throw new Error(`dsh ${arguments_.join(" ")} exited with ${result.status}.`);
}
