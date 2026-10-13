import { spawn, spawnSync } from "node:child_process";
import { setTimeout as delay } from "node:timers/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const repositoryRoot = dirname(dirname(fileURLToPath(import.meta.url)));
const testHome = join(repositoryRoot, "experiments", "runtime", "dsh-test");
const dshBinary = join(repositoryRoot, "node_modules", ".bin", "dsh");
const environment = {
  ...process.env,
  DSH_HOME: testHome,
  HERO_QWEN_LOCAL_API_KEY: "local-loopback-only"
};

const setup = spawnSync(process.execPath, [join(repositoryRoot, "scripts", "dsh-test-profile.mjs")], {
  cwd: repositoryRoot,
  env: environment,
  stdio: "inherit"
});
if (setup.error !== undefined) throw setup.error;
if (setup.status !== 0) throw new Error(`DSH test-profile setup exited with ${setup.status}.`);

const child = spawn(dshBinary, ["web", "--port", "0", "--no-open"], {
  cwd: testHome,
  env: environment,
  stdio: ["ignore", "pipe", "pipe"]
});
let stderr = "";
child.stderr.setEncoding("utf8");
child.stderr.on("data", (chunk) => { stderr = `${stderr}${chunk}`.slice(-32_768); });

try {
  const authenticatedUrl = await waitForUrl(child);
  const baseUrl = new URL(authenticatedUrl).origin;
  const snapshot = await waitForModel(baseUrl);
  if (snapshot.activeVersionId !== "base" && !snapshot.activeVersionId?.startsWith("stage-")) {
    throw new Error(`Unexpected active model version: ${snapshot.activeVersionId}`);
  }
  const smoke = await postJson(`${baseUrl}/hero/api/models/local/dsh-smoke`, {});
  if (smoke.response !== "HERO_DSH_READY") throw new Error("DSH provider returned the wrong smoke response.");
  process.stdout.write(`HERO local Qwen DSH smoke passed at ${baseUrl} (${snapshot.activeVersionId}).\n`);
} finally {
  child.kill("SIGINT");
  await new Promise((resolve) => {
    if (child.exitCode !== null) resolve();
    else child.once("exit", resolve);
  });
}

async function waitForModel(baseUrl) {
  const deadline = Date.now() + 300_000;
  let last;
  while (Date.now() < deadline) {
    try {
      last = await getJson(`${baseUrl}/hero/api/models/local`);
      if (last.status === "ready") return last;
      if (last.status === "error") throw new Error(last.error ?? "Local model startup failed.");
    } catch (error) {
      if (last?.status === "error") throw error;
    }
    await delay(1_000);
  }
  throw new Error(`Local Qwen did not become ready within five minutes. ${JSON.stringify(last)}`);
}

async function waitForUrl(process_) {
  process_.stdout.setEncoding("utf8");
  return await new Promise((resolve, reject) => {
    const timeout = setTimeout(() => reject(new Error(`DSH did not start within 15 seconds. ${stderr}`)), 15_000);
    let output = "";
    process_.stdout.on("data", (chunk) => {
      output += chunk;
      const match = output.match(/dsh web: (http:\/\/127\.0\.0\.1:\d+\/\?token=[^\s]+)/);
      if (match?.[1] !== undefined) {
        clearTimeout(timeout);
        resolve(match[1]);
      }
    });
    process_.once("exit", (code) => {
      clearTimeout(timeout);
      reject(new Error(`DSH exited before startup with ${code}. ${stderr}`));
    });
  });
}

async function getJson(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url} returned HTTP ${response.status}.`);
  return await response.json();
}

async function postJson(url, body) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body)
  });
  if (!response.ok) throw new Error(`${url} returned HTTP ${response.status}: ${await response.text()}`);
  return await response.json();
}
