import assert from "node:assert/strict";
import { createWriteStream } from "node:fs";
import { access, mkdir, readFile, writeFile } from "node:fs/promises";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";
import { resolve } from "node:path";

const directory = resolve("experiments/data/raw/trace");
await mkdir(directory, { recursive: true });
const destination = `${directory}/TRACE-Benchmark.zip`;
if (await access(destination).then(() => true, () => false)) {
  console.log(destination);
} else {
  const source = "https://drive.usercontent.google.com/download?id=1S0SmU0WEw5okW_XvP2Ns0URflNzZq6sV&export=download";
  const response = await fetch(source);
  assert(response.ok);
  const html = await response.text();
  const uuid = html.match(/name="uuid" value="([^"]+)"/)?.[1];
  assert(uuid, "Official archive download form was not found");
  const download = await fetch(`${source}&confirm=t&uuid=${encodeURIComponent(uuid)}`);
  assert(download.ok && !download.headers.get("content-type")?.includes("text/html"), "Official archive unavailable");
  await pipeline(Readable.fromWeb(download.body), createWriteStream(destination, { flags: "wx" }));
  const bytes = await readFile(destination);
  assert.equal(bytes.subarray(0, 2).toString(), "PK", "Expected ZIP archive");
  await writeFile(`${directory}/download.json`, JSON.stringify({ source, downloadedAt: new Date().toISOString(), bytes: bytes.length }, null, 2));
  console.log(destination);
}
