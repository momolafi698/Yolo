import { readFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";
import { matchPoseSequenceToCatalogue, prepareDance } from "../src/utils/catalogue-matcher.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PROJECT_ROOT = path.resolve(__dirname, "..");

const DEFAULTS = {
  catalogueDir: "public/catalogue/poses",
  windowSeconds: 3,
  trueStepSeconds: 1,
  falseStepSeconds: 2,
};

// Perturbation severities used to build synthetic "live" sequences from a
// dance's own frames, modeling the real single-dance/audio-synced gameplay
// path: reactionLag simulates the dancer being behind the beat, tempoAmplitude
// simulates local (non-uniform) tempo drift, posSigma simulates per-joint
// position noise, dropProb simulates brief keypoint occlusion.
const SEVERITIES = {
  none: { reactionLag: 0, tempoAmplitude: 0, posSigma: 0, dropProb: 0 },
  low: { reactionLag: 0.08, tempoAmplitude: 0.05, posSigma: 0.015, dropProb: 0.02 },
  medium: { reactionLag: 0.18, tempoAmplitude: 0.12, posSigma: 0.035, dropProb: 0.06 },
  high: { reactionLag: 0.32, tempoAmplitude: 0.22, posSigma: 0.06, dropProb: 0.12 },
};

function printHelp() {
  console.log(`
Evaluate the pose-matching algorithm against the real catalogue data.

Computes score/detection distributions for:
  - true-positive matches: a dance's own frames replayed against itself at
    several injected noise severities (reaction lag, tempo drift, positional
    jitter, occlusion) - models the primary single-dance/audio-synced path.
  - false-positive matches: a dance's frames replayed against every other
    dance in the catalogue - models cross-dance confusion.

Re-run this after any change to matching thresholds/formulas in
src/utils/catalogue-matcher.js to confirm true/false separation improves,
rather than tuning against live gameplay by feel.

Usage:
  node scripts/evaluate-matcher.js
  node scripts/evaluate-matcher.js --window 4 --true-step 0.5

Options:
  --catalogue <dir>     Catalogue directory. Default: ${DEFAULTS.catalogueDir}
  --window <seconds>    Live-sequence window length. Default: ${DEFAULTS.windowSeconds}
  --true-step <seconds> Slide step for true-positive windows. Default: ${DEFAULTS.trueStepSeconds}
  --false-step <seconds> Slide step for false-positive windows. Default: ${DEFAULTS.falseStepSeconds}
  --help                Show this help.
`);
}

function parseArgs(argv) {
  const options = { ...DEFAULTS, help: false };

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    const readValue = () => {
      const value = argv[++i];
      if (!value) throw new Error(`Missing value after ${arg}`);
      return value;
    };

    switch (arg) {
      case "--catalogue":
        options.catalogueDir = readValue();
        break;
      case "--window":
        options.windowSeconds = Number(readValue());
        break;
      case "--true-step":
        options.trueStepSeconds = Number(readValue());
        break;
      case "--false-step":
        options.falseStepSeconds = Number(readValue());
        break;
      case "--help":
      case "-h":
        options.help = true;
        break;
      default:
        throw new Error(`Unknown argument: ${arg}`);
    }
  }

  return options;
}

async function loadCatalogue(catalogueDir) {
  const root = path.resolve(PROJECT_ROOT, catalogueDir);
  const index = JSON.parse(await readFile(path.join(root, "index.json"), "utf8"));
  const dances = await Promise.all(
    index.dances.map(async (entry) => {
      const data = JSON.parse(await readFile(path.join(root, entry.file), "utf8"));
      return prepareDance({ ...entry, data });
    }),
  );

  return { ...index, dances };
}

// Deterministic PRNG (mulberry32) so runs are reproducible.
function createRng(seed) {
  let state = seed;
  return function rng() {
    state |= 0;
    state = (state + 0x6d2b79f5) | 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gaussianRandom(rng) {
  let u = 0;
  let v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function jitterKeypoints(keypoints, sigma, dropProbability, rng) {
  if (sigma === 0 && dropProbability === 0) return keypoints;
  return keypoints.map((point) => {
    if (dropProbability > 0 && rng() < dropProbability) {
      return { ...point, score: 0 };
    }
    if (sigma === 0) return point;
    return {
      ...point,
      x: point.x + gaussianRandom(rng) * sigma,
      y: point.y + gaussianRandom(rng) * sigma,
    };
  });
}

function buildPerturbedWindow(dance, startIndex, frameCount, severity, rng) {
  const frames = dance.frames.slice(startIndex, startIndex + frameCount);
  if (frames.length < frameCount) return null;

  const baseTimestamp = frames[0].timestamp;
  return frames.map((frame) => {
    const tempoOffset = severity.tempoAmplitude *
      Math.sin((2 * Math.PI * (frame.timestamp - baseTimestamp)) / 2.5);
    return {
      timestamp: frame.timestamp + severity.reactionLag + tempoOffset,
      keypoints: jitterKeypoints(frame.keypoints, severity.posSigma, severity.dropProb, rng),
      angles: frame.angles,
    };
  });
}

function computeTrueScores(catalogue, config, rng) {
  const results = {};
  for (const severityName of Object.keys(SEVERITIES)) results[severityName] = [];

  for (const dance of catalogue.dances) {
    const sampledFps = dance.sampledFps || 30;
    const frameCount = Math.round(config.windowSeconds * sampledFps);
    const stepFrames = Math.max(1, Math.round(config.trueStepSeconds * sampledFps));

    for (let start = 0; start + frameCount <= dance.frames.length; start += stepFrames) {
      for (const [severityName, severity] of Object.entries(SEVERITIES)) {
        const window = buildPerturbedWindow(dance, start, frameCount, severity, rng);
        if (!window) continue;

        const match = matchPoseSequenceToCatalogue(window, { dances: [dance] }, { syncToTimeline: true });
        if (match.best) {
          results[severityName].push({
            score: match.best.score,
            detected: match.detected,
            coverage: match.best.informativeCoverageFraction,
          });
        }
      }
    }
  }

  return results;
}

function computeFalseScores(catalogue, config, rng) {
  const scores = [];

  for (const liveDance of catalogue.dances) {
    const sampledFps = liveDance.sampledFps || 30;
    const frameCount = Math.round(config.windowSeconds * sampledFps);
    const stepFrames = Math.max(1, Math.round(config.falseStepSeconds * sampledFps));

    for (let start = 0; start + frameCount <= liveDance.frames.length; start += stepFrames) {
      const window = buildPerturbedWindow(liveDance, start, frameCount, SEVERITIES.low, rng);
      if (!window) continue;

      for (const referenceDance of catalogue.dances) {
        if (referenceDance.id === liveDance.id) continue;

        const match = matchPoseSequenceToCatalogue(
          window,
          { dances: [referenceDance] },
          { syncToTimeline: true },
        );
        if (match.best) {
          scores.push({
            score: match.best.score,
            detected: match.detected,
            coverage: match.best.informativeCoverageFraction,
          });
        }
      }
    }
  }

  return scores;
}

function percentile(values, p) {
  if (values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const index = Math.min(sorted.length - 1, Math.floor((p / 100) * sorted.length));
  return sorted[index];
}

function formatScore(value) {
  return value === null ? "-" : value.toFixed(1);
}

function summarize(label, entries) {
  const scores = entries.map((entry) => entry.score);
  const detectedRate = entries.length > 0
    ? entries.filter((entry) => entry.detected).length / entries.length
    : 0;

  console.log(`\n${label} (n=${entries.length})`);
  console.log(
    `  p50=${formatScore(percentile(scores, 50))}  ` +
    `p90=${formatScore(percentile(scores, 90))}  ` +
    `p95=${formatScore(percentile(scores, 95))}  ` +
    `p99=${formatScore(percentile(scores, 99))}  ` +
    `detectedRate=${(detectedRate * 100).toFixed(1)}%`,
  );
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) {
    printHelp();
    return;
  }

  console.log(`Loading catalogue from ${args.catalogueDir}...`);
  const catalogue = await loadCatalogue(args.catalogueDir);
  console.log(`Loaded ${catalogue.dances.length} dances.`);

  const rng = createRng(42);

  console.log("\n=== True-positive distributions (self-replay, syncToTimeline) ===");
  const trueScores = computeTrueScores(catalogue, args, rng);
  for (const [severity, entries] of Object.entries(trueScores)) {
    summarize(`true/${severity}`, entries);
  }

  console.log("\n=== False-positive distribution (cross-dance, low-severity noise) ===");
  const falseScores = computeFalseScores(catalogue, args, rng);
  summarize("false/cross-dance", falseScores);

  console.log(
    "\nThresholds (minConfidence, minCoverageFraction, minMargin, syncBandSeconds, " +
    "fallbackBandSeconds, keypointSigma, angleSigma) live in DEFAULT_OPTIONS, " +
    "src/utils/catalogue-matcher.js. Pick them so the false-positive p90/p95 sits " +
    "below the true-positive (medium/high severity) p10/p50, then adjust and re-run.",
  );
}

main().catch((error) => {
  console.error(`\nEvaluation failed: ${error.message}`);
  process.exitCode = 1;
});
