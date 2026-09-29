import { readdirSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const assetsDirectory = fileURLToPath(new URL("../dist/assets/", import.meta.url));
const javascriptAssets = readdirSync(assetsDirectory)
  .filter((name) => name.endsWith(".js"))
  .sort();

if (javascriptAssets.length === 0) {
  throw new Error(`No JavaScript bundles were found in ${assetsDirectory}.`);
}

for (const asset of javascriptAssets) {
  const assetPath = fileURLToPath(new URL(`../dist/assets/${asset}`, import.meta.url));
  const result = spawnSync(process.execPath, ["--check", assetPath], { encoding: "utf8" });
  if (result.status !== 0) {
    process.stderr.write(result.stderr);
    throw new Error(`Generated JavaScript bundle failed syntax validation: ${asset}`);
  }
}

process.stdout.write(`Validated ${javascriptAssets.length} generated JavaScript bundle(s).\n`);
