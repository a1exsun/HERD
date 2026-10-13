import { spawn, spawnSync } from "node:child_process";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { createRequire } from "node:module";
import { writeFile } from "node:fs/promises";

const repositoryRoot = dirname(dirname(fileURLToPath(import.meta.url)));
const dshVersion = createRequire(import.meta.url)("@deepseek-ai/dsh/package.json").version;
const testHome = join(repositoryRoot, "experiments", "runtime", "dsh-test");
const dshBinary = join(repositoryRoot, "node_modules", ".bin", "dsh");
const environment = {
  ...process.env,
  DSH_HOME: testHome,
  HERO_LOCAL_QWEN_AUTOSTART: "0",
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
child.stderr.on("data", (chunk) => { stderr += chunk; });

try {
  const authenticatedUrl = await waitForUrl(child);
  const baseUrl = new URL(authenticatedUrl).origin;
  const manifest = await getJson(`${baseUrl}/hero/api/manifest`);
  assert(manifest.name === "HERD", "manifest has the wrong plugin name");
  assert(manifest.runtime?.dsh === dshVersion, "manifest has the wrong DSH version");
  assert(manifest.runtime?.harnessModelReady === true, "Harness model route is not ready");
  assert(manifest.harnessEvolvingModel?.model === "Qwen/Qwen3.5-4B", "Harness driver model is wrong");
  assert(manifest.continualLearningModel?.model === "Qwen/Qwen3.5-4B", "CL subject model is wrong");
  assert(["missing", "stopped"].includes(manifest.runtime?.localModel?.status), "local model has the wrong smoke-test state");
  assert(manifest.modules?.length === 2, "manifest does not expose two modules");

  const consoleResponse = await fetch(`${baseUrl}/hero/`);
  assert(consoleResponse.ok, `console returned HTTP ${consoleResponse.status}`);
  assert((await consoleResponse.text()).includes('<div id="root"></div>'), "console HTML is incomplete");

  const dshResponse = await authenticate(authenticatedUrl);
  assert(dshResponse.ok, `DSH root returned HTTP ${dshResponse.status}`);
  assert((await dshResponse.text()).includes("hero-dsh-launch"), "DSH root has no HERO launcher");

  // The guard scores verified answers with the local model, which this smoke does not start; it must still reject
  // an invalid request before any model call.
  const emptyRegistry = JSON.stringify({ schemaVersion: "hero-harness-spec-v1", skills: [] });
  const guard = await fetch(`${baseUrl}/hero/api/harness-evolving/guard`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ input: { specification: emptyRegistry, candidateSpecification: emptyRegistry, evidence: [] } })
  });
  assert(guard.status === 400 && (await guard.text()).includes("evidence must be a non-empty array"),
    "harness guard route did not reject empty evidence");
  await writeFile(join(testHome, "dsh-smoke.json"), `${JSON.stringify({
    status: "passed", recordedAt: new Date().toISOString(), dshVersion,
    checks: ["manifest", "console", "authenticated-dsh-root", "harness-guard-validation"]
  }, null, 2)}\n`);
  process.stdout.write(`HERO DSH smoke passed at ${baseUrl}\n`);
} finally {
  child.kill("SIGINT");
  await new Promise((resolve) => {
    if (child.exitCode !== null) resolve();
    else child.once("exit", resolve);
  });
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

async function authenticate(url) {
  const exchange = await fetch(url, { redirect: "manual" });
  assert(exchange.status >= 300 && exchange.status < 400, "DSH token exchange did not redirect");
  const cookie = exchange.headers.get("set-cookie")?.split(";", 1)[0];
  assert(cookie, "DSH token exchange returned no session cookie");
  const destination = new URL(exchange.headers.get("location") ?? "/", url);
  return await fetch(destination, { headers: { cookie } });
}

async function getJson(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url} returned HTTP ${response.status}.`);
  return await response.json();
}

function assert(condition, message) {
  if (!condition) throw new Error(`HERO DSH smoke failed: ${message}.`);
}
