/**
 * Vite cannot resolve modules when "#" appears in the project path (URL fragment).
 * On Windows, map the repo to a drive letter via SUBST and run Vite from there.
 */
import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const frontendRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const repoRoot = path.resolve(frontendRoot, "..", "..");
const viteBin = path.join(frontendRoot, "node_modules", "vite", "bin", "vite.js");

function pathHasHash(value) {
  return value.includes("#");
}

function substFrontendRoot(driveLetter) {
  const root = `${driveLetter}:\\`;
  const candidate = path.join(root, "09_WEB", "front_end");
  return existsSync(path.join(candidate, "package.json")) ? candidate : null;
}

function runVite(cwd) {
  const child = spawn(process.execPath, [viteBin], {
    cwd,
    stdio: "inherit",
    env: process.env,
  });
  child.on("close", (code) => process.exit(code ?? 1));
}

function runSubst(driveLetter, target) {
  return new Promise((resolve, reject) => {
    const child = spawn("subst", [`${driveLetter}:`, target], {
      stdio: "inherit",
      shell: false,
    });
    child.on("close", (code) => {
      if (code === 0) {
        resolve();
        return;
      }
      // Drive may already be mapped from a previous dev session.
      if (substFrontendRoot(driveLetter)) {
        resolve();
        return;
      }
      reject(new Error(`subst ${driveLetter}: failed with exit code ${code}`));
    });
  });
}

async function main() {
  const needsSubst =
    process.platform === "win32" && (pathHasHash(repoRoot) || pathHasHash(frontendRoot));

  if (!needsSubst) {
    runVite(frontendRoot);
    return;
  }

  for (const letter of ["V", "W", "U", "T", "S"]) {
    const mapped = substFrontendRoot(letter);
    if (mapped) {
      console.log(`Using existing ${letter}:\\ → repo (Vite cannot use paths containing "#").`);
      runVite(mapped);
      return;
    }
  }

  const driveLetter = "V";
  console.log(
    `Project path contains "#", which Vite treats as a URL fragment.\n` +
      `Mapping ${driveLetter}:\\ → ${repoRoot}`,
  );
  await runSubst(driveLetter, repoRoot);
  const mappedRoot = substFrontendRoot(driveLetter);
  if (!mappedRoot) {
    console.error("SUBST succeeded but front_end was not found on the mapped drive.");
    process.exit(1);
  }
  runVite(mappedRoot);
}

main().catch((error) => {
  console.error(error.message);
  process.exit(1);
});
