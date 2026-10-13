import { readFile } from "node:fs/promises";
import type { IncomingMessage, ServerResponse } from "node:http";
import { createRequire } from "node:module";
import { extname, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";
import type { Context } from "@deepseek-ai/cordis";
import type {} from "@deepseek-ai/dsh-host-webserver";
import {
  BlockAssembler,
  createUserMessage,
  type LlmResolvedModelInfo
} from "@deepseek-ai/dsh-llm";
import {
  CONTINUAL_LEARNING_MODEL,
  DSH_RUNTIME_VERSION,
  HARNESS_EVOLVING_MODEL,
  HERO_FULL_NAME,
  HERO_MODULES,
  HERO_NAME,
  HERO_VERSION,
  QWEN_MODEL_REVISION
} from "@hero/core";
import { QwenBridgeClient, guardProposal, proposeSkill, type GuardInput, type ProposeInput } from "./harness.js";
import { LocalModelManager, type LocalModelSnapshot } from "./localModelManager.js";

export const name = "hero";
export const inject = ["webServer", "llm"];

const MAX_BODY_BYTES = 4 * 1024 * 1024;
const LOCAL_QWEN_PROVIDER = HARNESS_EVOLVING_MODEL.provider;
const REPOSITORY_ROOT = fileURLToPath(new URL("../../../", import.meta.url));
const WEB_ROOT = fileURLToPath(new URL("./web", import.meta.url));
const dshPackage = createRequire(import.meta.url)("@deepseek-ai/dsh/package.json") as { version?: unknown };
if (dshPackage.version !== DSH_RUNTIME_VERSION) {
  throw new Error(`HERO requires @deepseek-ai/dsh ${DSH_RUNTIME_VERSION}, found ${String(dshPackage.version)}.`);
}

export interface HeroManifest {
  name: typeof HERO_NAME;
  fullName: typeof HERO_FULL_NAME;
  version: typeof HERO_VERSION;
  harnessEvolvingModel: typeof HARNESS_EVOLVING_MODEL;
  continualLearningModel: typeof CONTINUAL_LEARNING_MODEL;
  modules: typeof HERO_MODULES;
  runtime: {
    dsh: typeof DSH_RUNTIME_VERSION;
    continualLearningModelRevision: typeof QWEN_MODEL_REVISION;
    harnessModelReady: boolean;
    harnessModelName?: string;
    localModel: LocalModelSnapshot;
  };
}

export function createManifest(modelInfo?: LlmResolvedModelInfo, localModel?: LocalModelSnapshot): HeroManifest {
  return {
    name: HERO_NAME,
    fullName: HERO_FULL_NAME,
    version: HERO_VERSION,
    harnessEvolvingModel: HARNESS_EVOLVING_MODEL,
    continualLearningModel: CONTINUAL_LEARNING_MODEL,
    modules: HERO_MODULES,
    runtime: {
      dsh: DSH_RUNTIME_VERSION,
      continualLearningModelRevision: QWEN_MODEL_REVISION,
      harnessModelReady: modelInfo !== undefined,
      ...(modelInfo === undefined ? {} : { harnessModelName: modelInfo.name }),
      localModel: localModel ?? {
        status: "missing",
        bridgeUrl: "http://127.0.0.1:8192",
        managedProcess: false
      }
    }
  };
}

export function apply(ctx: Context): void {
  process.env.HERO_QWEN_LOCAL_API_KEY ??= "local-loopback-only";
  const localModels = new LocalModelManager(REPOSITORY_ROOT);
  localModels.initialize(process.env.HERO_LOCAL_QWEN_AUTOSTART !== "0");
  ctx.effect(() => async () => localModels.dispose());
  ctx.effect(() => ctx.webServer.register({
    kind: "prefix",
    path: "/hero/api",
    handler: (request, response) => handleApi(ctx, localModels, request, response)
  }));
  ctx.effect(() => ctx.webServer.register({
    kind: "prefix",
    path: "/hero",
    handler: serveConsole
  }));
  ctx.on("webserver/index-inject", (table) => {
    table.push({
      kind: "style",
      text: ".hero-dsh-launch{position:fixed;right:18px;bottom:18px;z-index:100;display:inline-flex;align-items:center;min-height:36px;padding:6px 16px;border:1px solid var(--dsw-alias-border-l2);border-radius:18px;background:var(--dsw-alias-button-floating-fill);color:var(--dsw-alias-label-primary);text-decoration:none;font:500 13px/22px var(--dsw-font-family);box-shadow:var(--dsw-shadow-lv1)}.hero-dsh-launch:hover{background:var(--dsw-alias-button-floating-hover)}.hero-dsh-launch:focus-visible{outline:2px solid var(--dsw-static-deepseek-450);outline-offset:3px}"
    });
    table.push({
      kind: "html",
      placement: "body",
      html: `<a class="hero-dsh-launch" href="/hero/" aria-label="Open HERD project learning workspace" title="HERD: ${HERO_FULL_NAME}">HERD</a>`
    });
  });
}

async function handleApi(
  ctx: Context,
  localModels: LocalModelManager,
  request: IncomingMessage,
  response: ServerResponse
): Promise<void> {
  try {
    const path = new URL(request.url ?? "/", "http://hero.local").pathname;
    if (request.method === "GET" && path === "/hero/api/manifest") {
      let modelInfo: LlmResolvedModelInfo | undefined;
      try {
        modelInfo = await ctx.llm.resolveModelInfo(
          HARNESS_EVOLVING_MODEL.provider,
          HARNESS_EVOLVING_MODEL.model
        );
      } catch {
        modelInfo = undefined;
      }
      sendJson(response, 200, createManifest(modelInfo, await localModels.snapshot()));
      return;
    }
    if (request.method === "GET" && path === "/hero/api/models/local") {
      sendJson(response, 200, await localModels.snapshot());
      return;
    }
    if (request.method !== "POST") {
      sendJson(response, 405, { error: "Method not allowed." });
      return;
    }
    const body = await readJsonBody(request);
    if (path === "/hero/api/models/local/restart") {
      await localModels.restart();
      sendJson(response, 200, await localModels.snapshot());
      return;
    }
    if (path === "/hero/api/models/local/dsh-smoke") {
      sendJson(response, 200, {
        response: await smokeLocalModel(ctx),
        localModel: await localModels.snapshot()
      });
      return;
    }
    if (path === "/hero/api/models/local/install-stream") {
      await localModels.installStreamRun({
        runPath: assertString(body, "runPath"),
        throughStage: assertNumber(body, "throughStage")
      });
      sendJson(response, 200, await localModels.snapshot());
      return;
    }
    if (path === "/hero/api/models/local/activate") {
      await localModels.activate(assertString(body, "versionId"));
      sendJson(response, 200, await localModels.snapshot());
      return;
    }
    if (path === "/hero/api/harness-evolving/propose") {
      sendJson(response, 200, await proposeSkill(new QwenBridgeClient(localModels.bridgeUrl), assertObject<ProposeInput>(body, "input")));
      return;
    }
    if (path === "/hero/api/harness-evolving/guard") {
      sendJson(response, 200, await guardProposal(new QwenBridgeClient(localModels.bridgeUrl), assertObject<GuardInput>(body, "input")));
      return;
    }
    sendJson(response, 404, { error: "Unknown HERO API route." });
  } catch (error) {
    sendJson(response, 400, { error: error instanceof Error ? error.message : String(error) });
  }
}

async function smokeLocalModel(ctx: Context): Promise<string> {
  const assembler = new BlockAssembler();
  for await (const chunk of ctx.llm.stream({
    provider: LOCAL_QWEN_PROVIDER,
    model: CONTINUAL_LEARNING_MODEL.model,
    messages: [createUserMessage({
      content: [{ type: "text", text: "Reply with exactly: HERO_DSH_READY" }],
      source: { kind: "plugin", plugin: "hero/local-model-smoke", form: "instructions" }
    })]
  })) assembler.push(chunk);
  if (assembler.finish.kind === "error" || assembler.finish.kind === "aborted") {
    throw new Error(`Local Qwen DSH smoke failed: ${assembler.finish.failure.message}`);
  }
  const output = assembler.blocks()
    .filter((block): block is { type: "text"; text: string } => block.type === "text")
    .map(({ text }) => text)
    .join("")
    .trim();
  if (output !== "HERO_DSH_READY") {
    throw new Error(`Local Qwen returned an unexpected DSH smoke response: ${JSON.stringify(output)}`);
  }
  return output;
}

async function serveConsole(request: IncomingMessage, response: ServerResponse): Promise<void> {
  if (request.method !== "GET" && request.method !== "HEAD") {
    response.writeHead(405);
    response.end();
    return;
  }
  const pathname = new URL(request.url ?? "/hero/", "http://hero.local").pathname;
  const suffix = pathname === "/hero" || pathname === "/hero/"
    ? "index.html"
    : decodeURIComponent(pathname.slice("/hero/".length));
  const filePath = resolve(WEB_ROOT, suffix);
  if (filePath !== WEB_ROOT && !filePath.startsWith(`${WEB_ROOT}${sep}`)) {
    response.writeHead(404);
    response.end();
    return;
  }
  try {
    const body = await readFile(filePath);
    response.writeHead(200, {
      "content-type": contentType(filePath),
      "cache-control": suffix === "index.html" ? "no-cache" : "public, max-age=31536000, immutable"
    });
    response.end(request.method === "HEAD" ? undefined : body);
  } catch {
    response.writeHead(404);
    response.end();
  }
}

async function readJsonBody(request: IncomingMessage): Promise<Record<string, unknown>> {
  const chunks: Buffer[] = [];
  let bytes = 0;
  for await (const chunk of request) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
    bytes += buffer.length;
    if (bytes > MAX_BODY_BYTES) throw new Error("Request body exceeds 4 MiB.");
    chunks.push(buffer);
  }
  const parsed: unknown = JSON.parse(Buffer.concat(chunks).toString("utf8"));
  if (parsed === null || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new Error("Request body must be a JSON object.");
  }
  return parsed as Record<string, unknown>;
}

function sendJson(response: ServerResponse, status: number, value: unknown): void {
  response.writeHead(status, { "content-type": "application/json; charset=utf-8" });
  response.end(`${JSON.stringify(value)}\n`);
}

function assertString(source: Record<string, unknown>, key: string): string {
  const value = source[key];
  if (typeof value !== "string") throw new Error(`${key} must be a string.`);
  return value;
}

function assertNumber(source: Record<string, unknown>, key: string): number {
  const value = source[key];
  if (typeof value !== "number") throw new Error(`${key} must be a number.`);
  return value;
}

function assertObject<T extends object>(source: Record<string, unknown>, key: string): T {
  const value = source[key];
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`${key} must be an object.`);
  }
  return value as T;
}

function contentType(path: string): string {
  const types: Record<string, string> = {
    ".css": "text/css; charset=utf-8",
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml"
  };
  return types[extname(path)] ?? "application/octet-stream";
}
