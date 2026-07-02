// Calibrates the matcher against REAL ground-truth: the player's recorded
// performances (karisma-*.poses.json) vs the reference dances they were
// performing (poses_original/*). Replays each recording through the exact
// live pipeline (createPoseSample -> sliding window -> audio-synced match)
// and reports score distributions for matched pairs (should be ~100%) and
// mismatched pairs (should stay low).
//
// Usage:
//   node scripts/calibrate-real.js            # summary
//   node scripts/calibrate-real.js --offsets  # also scan global time offsets
import { readFile, readdir } from "node:fs/promises";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";
import {
  createPoseSample,
  matchPoseSequenceToCatalogue,
  prepareDance,
} from "../src/utils/catalogue-matcher.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PROJECT_ROOT = path.resolve(__dirname, "..");
const PLAYER_DIR = path.join(PROJECT_ROOT, "public/catalogue/poses");
const TARGET_DIR = path.join(PROJECT_ROOT, "public/catalogue/poses/poses_original");

const WINDOW_SECONDS = 4;
const STEP_SECONDS = 1;

// player recording keyword -> target file keyword
const PAIR_KEYWORDS = {
  whiplash: "whiplash",
  redred: "redred",
  magnetic: "magnetic",
  jenny: "jennie",
  wicked: "wicked",
};

async function loadJson(file) {
  return JSON.parse(await readFile(file, "utf8"));
}

async function loadTargets() {
  const index = await loadJson(path.join(TARGET_DIR, "index.json"));
  const targets = [];
  for (const entry of index.dances) {
    const data = await loadJson(path.join(TARGET_DIR, entry.file));
    targets.push({ id: entry.id, prepared: prepareDance({ ...entry, data }) });
  }
  return targets;
}

async function loadPlayers() {
  const files = (await readdir(PLAYER_DIR)).filter((f) => f.startsWith("karisma-") && f.endsWith(".poses.json"));
  const players = [];
  for (const file of files) {
    const data = await loadJson(path.join(PLAYER_DIR, file));
    players.push({
      id: file.replace(".poses.json", ""),
      frames: data.frames.filter((frame) => frame.person?.keypoints?.length),
    });
  }
  return players;
}

function targetFor(playerId, targets) {
  const keyword = Object.entries(PAIR_KEYWORDS).find(([k]) => playerId.includes(k))?.[1];
  return targets.find((t) => keyword && t.id.includes(keyword)) ?? null;
}

// Replays the recording as sliding live windows, exactly like the app's
// rolling buffer, with an optional global offset added to every timestamp.
function scoreRecording(playerFrames, target, offset) {
  const samples = playerFrames
    .map((frame) => createPoseSample(frame.person, frame.timestamp + offset))
    .filter(Boolean);
  if (samples.length === 0) return [];

  const results = [];
  const lastT = samples.at(-1).timestamp;
  for (let start = samples[0].timestamp; start + WINDOW_SECONDS <= lastT + 1e-9; start += STEP_SECONDS) {
    const window = samples.filter((s) => s.timestamp >= start && s.timestamp <= start + WINDOW_SECONDS);
    if (window.length < 8) continue;
    const match = matchPoseSequenceToCatalogue(window, { dances: [target.prepared] }, { syncToTimeline: true });
    if (match.best) results.push(match.best);
  }
  return results;
}

function percentile(values, p) {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.min(sorted.length - 1, Math.floor((p / 100) * sorted.length))];
}

function fmt(v) {
  return v === null ? "   -" : v.toFixed(1).padStart(5);
}

function summarize(label, results) {
  const display = results.map((r) => r.score);
  const raw = results.map((r) => r.rawScore);
  const ratio = results.map((r) => r.motionRatio).filter((v) => v !== null);
  const corr = results.map((r) => r.correlation).filter((v) => v !== null && v !== undefined);
  const mirroredRate = results.length ? results.filter((r) => r.mirrored).length / results.length : 0;
  console.log(
    `${label.padEnd(46)} n=${String(results.length).padStart(3)}  ` +
    `display p10/p50/p90: ${fmt(percentile(display, 10))} ${fmt(percentile(display, 50))} ${fmt(percentile(display, 90))}  ` +
    `raw p10/p50/p90: ${fmt(percentile(raw, 10))} ${fmt(percentile(raw, 50))} ${fmt(percentile(raw, 90))}  ` +
    `corr p10/p50/p90: ${fmt(percentile(corr, 10))} ${fmt(percentile(corr, 50))} ${fmt(percentile(corr, 90))}  ` +
    `motion p50: ${fmt(percentile(ratio, 50))}  mirror: ${(mirroredRate * 100).toFixed(0)}%`,
  );
}

async function main() {
  const scanOffsets = process.argv.includes("--offsets");
  const targets = await loadTargets();
  const players = await loadPlayers();

  if (scanOffsets) {
    console.log("=== Global offset scan (mean raw per offset) ===");
    for (const player of players) {
      const target = targetFor(player.id, targets);
      if (!target) continue;
      const line = [];
      for (let offset = -2; offset <= 2.001; offset += 0.5) {
        const results = scoreRecording(player.frames, target, offset);
        const mean = results.length ? results.reduce((a, r) => a + r.rawScore, 0) / results.length : 0;
        line.push(`${offset >= 0 ? "+" : ""}${offset.toFixed(1)}:${mean.toFixed(1)}`);
      }
      console.log(player.id.padEnd(20), line.join("  "));
    }
    console.log("");
  }

  console.log("=== Matched pairs (player vs their own dance) - should be ~100% ===");
  for (const player of players) {
    const target = targetFor(player.id, targets);
    if (!target) {
      console.log(`${player.id}: no target found`);
      continue;
    }
    summarize(`${player.id} vs ${target.id.slice(0, 24)}`, scoreRecording(player.frames, target, 0));
  }

  console.log("\n=== Mismatched pairs (player vs a different dance) - should stay low ===");
  for (const player of players) {
    const own = targetFor(player.id, targets);
    for (const target of targets) {
      if (!own || target.id === own.id) continue;
      summarize(`${player.id} vs ${target.id.slice(0, 24)}`, scoreRecording(player.frames, target, 0));
    }
  }
}

main().catch((error) => {
  console.error(error.stack);
  process.exitCode = 1;
});
