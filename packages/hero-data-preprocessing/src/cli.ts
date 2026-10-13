#!/usr/bin/env node
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import {
  buildTracePlus17Dataset,
  verifyTracePlus17Dataset,
  type TracePlus17BuildConfig
} from "./tracePlus17.js";

const configFlag = process.argv.indexOf("--config");
const verifyFlag = process.argv.indexOf("--verify");
const sourceRootFlag = process.argv.indexOf("--source-root");
const verifyArgument = verifyFlag < 0 ? undefined : process.argv[verifyFlag + 1];
const configArgument = configFlag < 0 ? undefined : process.argv[configFlag + 1];
const sourceRootArgument = sourceRootFlag < 0 ? undefined : process.argv[sourceRootFlag + 1];
const repositoryRoot = process.cwd();
if (verifyArgument !== undefined) {
  const result = await verifyTracePlus17Dataset(resolve(repositoryRoot, verifyArgument));
  process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
  process.exit(0);
}
if (configArgument === undefined) throw new Error("Usage: hero-build-trace-plus-17 --config <path> [--source-root <path>] | --verify <manifest>.");
const configPath = resolve(repositoryRoot, configArgument);
const config = JSON.parse(await readFile(configPath, "utf8")) as TracePlus17BuildConfig;
if (sourceRootArgument !== undefined) {
  config.sourceRoot = sourceRootArgument;
}
const manifest = await buildTracePlus17Dataset(config, repositoryRoot);
process.stdout.write(`${JSON.stringify({
  datasetVersion: manifest.datasetVersion,
  taskCount: manifest.taskCount,
  realized: manifest.totals.realized,
  deviations: manifest.totals.deviationCount,
  manifest: resolve(repositoryRoot, config.outputRoot, "manifest.json")
}, null, 2)}\n`);
