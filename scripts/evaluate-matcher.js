import { readFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";
import {
  matchPoseSequenceToCatalogue,
  prepareDance,
  createPoseSample,
  normalizePoseKeypoints,
  retargetSkeleton,
  BONE_LENGTH,
} from "../src/utils/catalogue-matcher.js";

// Must match DEFAULT_OPTIONS.keypointThreshold in catalogue-matcher.js.
const KEYPOINT_THRESHOLD = 0.2;

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PROJECT_ROOT = path.resolve(__dirname, "..");

const DEFAULTS = {
  catalogueDir: "public/catalogue/poses",
  windowSeconds: 3,
  trueStepSeconds: 2,
  falseStepSeconds: 4,
};

// COCO left/right index pairs, for synthesizing a mirrored player.
const SWAP_PAIRS = [
  [1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16],
];

// Perturbation scenarios applied to a dance's own RAW pixel keypoints to
// synthesize a live webcam player, then run through the exact same
// createPoseSample pipeline the app uses. Unlike the old harness (which
// jittered already-normalized frames), these model the real domain gap:
//   reactionLag  - seconds behind the beat
//   tempoAmp     - local (sinusoidal) tempo drift amplitude, seconds
//   posSigma     - per-keypoint Gaussian noise, as a FRACTION OF SHOULDER
//                  WIDTH (0.05 = each joint wanders by ~5% shoulder width)
//   dropProb     - probability a keypoint is lost (occlusion / low conf)
//   rotDeg       - whole-body rotation, models camera tilt / lens distortion
//   mirror       - the player follows the routine in the opposite chirality
//   bodySigma    - per-bone length multiplier stddev (different build)
const SCENARIOS = {
  "true/none": {},
  "true/low": { lag: 0.1, tempoAmp: 0.05, posSigma: 0.03, dropProb: 0.02, rotDeg: 2 },
  "true/medium": { lag: 0.2, tempoAmp: 0.12, posSigma: 0.06, dropProb: 0.06, rotDeg: 5 },
  "true/high": { lag: 0.35, tempoAmp: 0.22, posSigma: 0.1, dropProb: 0.12, rotDeg: 9 },
  "true/medium+mirror": { lag: 0.2, tempoAmp: 0.12, posSigma: 0.06, dropProb: 0.06, rotDeg: 5, mirror: true },
  "true/medium+body": { lag: 0.2, tempoAmp: 0.12, posSigma: 0.06, dropProb: 0.06, rotDeg: 5, bodySigma: 0.25 },
  // Anti-cheese scenarios: a player who isn't dancing at all. "freeze" holds
  // one pose taken from the middle of the window (the hardest case - it's a
  // genuine pose from this very choreography); "stand" is a neutral person
  // standing with arms down. Both should score LOW against a dance.
  "idle/freeze": { mode: "freeze", posSigma: 0.03 },
  "idle/stand": { mode: "stand", posSigma: 0.03 },
};

const FALSE_SCENARIO = SCENARIOS["true/low"];

function printHelp() {
  console.log(`
Evaluate the pose-matching algorithm against the real catalogue data.

Synthesizes "live webcam players" from each dance's raw extracted keypoints
(timing lag + tempo drift + positional noise + occlusion + camera tilt +
mirroring + different body proportions) and scores them through the same
pipeline the app uses, both against the correct dance (true positives) and
against every other dance (false positives / cross-dance).

Interpretation: pick staticSigmaDeg / scoreFloor / scoreCeil (DEFAULT_OPTIONS
in src/utils/catalogue-matcher.js) so the false/cross-dance DISPLAY scores sit
near 0-25% while true/medium-high sit comfortably above ~55%.

Usage:
  npm run evaluate:matcher
  node scripts/evaluate-matcher.js --window 4 --true-step 1

Options:
  --catalogue <dir>      Catalogue directory. Default: ${DEFAULTS.catalogueDir}
  --window <seconds>     Live-sequence window length. Default: ${DEFAULTS.windowSeconds}
  --true-step <seconds>  Slide step for true-positive windows. Default: ${DEFAULTS.trueStepSeconds}
  --false-step <seconds> Slide step for false-positive windows. Default: ${DEFAULTS.falseStepSeconds}
  --help                 Show this help.
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
      return {
        prepared: prepareDance({ ...entry, data }),
        rawFrames: data.frames.filter((frame) => frame.person?.keypoints?.length),
        sampledFps: entry.sampledFps || 30,
        id: entry.id,
      };
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

function randomBodyBoneLengths(rng, sigma) {
  const lengths = {};
  for (const [bone, base] of Object.entries(BONE_LENGTH)) {
    const factor = 1 + gaussianRandom(rng) * sigma;
    lengths[bone] = base * Math.max(0.4, factor);
  }
  return lengths;
}

function shoulderWidth(keypoints) {
  const left = keypoints[5];
  const right = keypoints[6];
  if (!left || !right) return null;
  const d = Math.hypot(left.x - right.x, left.y - right.y);
  return d > 1e-6 ? d : null;
}

function centroid(keypoints) {
  let x = 0;
  let y = 0;
  let n = 0;
  for (const point of keypoints) {
    if (!point) continue;
    x += point.x;
    y += point.y;
    n += 1;
  }
  return n > 0 ? { x: x / n, y: y / n } : { x: 0, y: 0 };
}

// A neutral standing pose (hip-centered, shoulder-width units): arms
// hanging, legs straight, built from the canonical segment lengths.
function standingKeypoints() {
  const shoulderY = -BONE_LENGTH.torso;
  const named = {
    nose: { x: 0, y: shoulderY - 0.5 },
    leftEye: { x: 0.08, y: shoulderY - 0.55 },
    rightEye: { x: -0.08, y: shoulderY - 0.55 },
    leftEar: { x: 0.16, y: shoulderY - 0.5 },
    rightEar: { x: -0.16, y: shoulderY - 0.5 },
    leftShoulder: { x: BONE_LENGTH.shoulderHalf, y: shoulderY },
    rightShoulder: { x: -BONE_LENGTH.shoulderHalf, y: shoulderY },
    leftElbow: { x: BONE_LENGTH.shoulderHalf, y: shoulderY + BONE_LENGTH.upperArm },
    rightElbow: { x: -BONE_LENGTH.shoulderHalf, y: shoulderY + BONE_LENGTH.upperArm },
    leftWrist: { x: BONE_LENGTH.shoulderHalf, y: shoulderY + BONE_LENGTH.upperArm + BONE_LENGTH.forearm },
    rightWrist: { x: -BONE_LENGTH.shoulderHalf, y: shoulderY + BONE_LENGTH.upperArm + BONE_LENGTH.forearm },
    leftHip: { x: BONE_LENGTH.hipHalf, y: 0 },
    rightHip: { x: -BONE_LENGTH.hipHalf, y: 0 },
    leftKnee: { x: BONE_LENGTH.hipHalf, y: BONE_LENGTH.thigh },
    rightKnee: { x: -BONE_LENGTH.hipHalf, y: BONE_LENGTH.thigh },
    leftAnkle: { x: BONE_LENGTH.hipHalf, y: BONE_LENGTH.thigh + BONE_LENGTH.shin },
    rightAnkle: { x: -BONE_LENGTH.hipHalf, y: BONE_LENGTH.thigh + BONE_LENGTH.shin },
  };
  return [
    "nose", "leftEye", "rightEye", "leftEar", "rightEar",
    "leftShoulder", "rightShoulder", "leftElbow", "rightElbow",
    "leftWrist", "rightWrist", "leftHip", "rightHip",
    "leftKnee", "rightKnee", "leftAnkle", "rightAnkle",
  ].map((name) => ({ name, ...named[name], score: 0.9 }));
}

// Synthesizes one live window from a slice of a dance's raw frames.
// Per-window draws (fixed for the whole window, like a real player/camera):
// body proportions, camera tilt sign. Per-frame draws: positional noise,
// keypoint drops, tempo drift.
function buildLiveWindow(rawFrames, startIndex, frameCount, scenario, rng) {
  const slice = rawFrames.slice(startIndex, startIndex + frameCount);
  if (slice.length < frameCount) return null;

  // Idle modes: keep the slice's real timestamps but replace every pose
  // with one fixed pose (the player is not dancing).
  const frozenKeypoints = scenario.mode === "freeze"
    ? slice[Math.floor(slice.length / 2)].person.keypoints
    : scenario.mode === "stand"
      ? standingKeypoints()
      : null;

  const bodyLengths = scenario.bodySigma
    ? randomBodyBoneLengths(rng, scenario.bodySigma)
    : null;
  const rotation = scenario.rotDeg
    ? (rng() < 0.5 ? -1 : 1) * scenario.rotDeg * (Math.PI / 180)
    : 0;
  const baseTimestamp = slice[0].timestamp;

  const samples = [];
  for (const frame of slice) {
    let keypoints = (frozenKeypoints ?? frame.person.keypoints).map((point) => ({ ...point }));

    if (bodyLengths) {
      const normalized = normalizePoseKeypoints(keypoints, KEYPOINT_THRESHOLD);
      if (!normalized) continue;
      keypoints = retargetSkeleton(normalized, bodyLengths, KEYPOINT_THRESHOLD);
    }

    // Positional noise scale: fraction of THIS pose's shoulder width, so the
    // same scenario means the same relative sloppiness in pixel or
    // normalized space.
    const width = shoulderWidth(keypoints) ?? 1;
    const sigma = (scenario.posSigma ?? 0) * width;

    if (rotation !== 0) {
      const center = centroid(keypoints);
      const cos = Math.cos(rotation);
      const sin = Math.sin(rotation);
      keypoints = keypoints.map((point) => ({
        ...point,
        x: center.x + (point.x - center.x) * cos - (point.y - center.y) * sin,
        y: center.y + (point.x - center.x) * sin + (point.y - center.y) * cos,
      }));
    }

    keypoints = keypoints.map((point) => {
      if ((scenario.dropProb ?? 0) > 0 && rng() < scenario.dropProb) {
        return { ...point, score: 0 };
      }
      if (sigma === 0) return point;
      return {
        ...point,
        x: point.x + gaussianRandom(rng) * sigma,
        y: point.y + gaussianRandom(rng) * sigma,
      };
    });

    if (scenario.mirror) {
      keypoints = keypoints.map((point) => ({ ...point, x: -point.x }));
      for (const [left, right] of SWAP_PAIRS) {
        const tmp = keypoints[left];
        keypoints[left] = keypoints[right];
        keypoints[right] = tmp;
      }
    }

    const tempoOffset = (scenario.tempoAmp ?? 0) *
      Math.sin((2 * Math.PI * (frame.timestamp - baseTimestamp)) / 2.5);
    const timestamp = frame.timestamp + (scenario.lag ?? 0) + tempoOffset;

    const sample = createPoseSample({ keypoints }, timestamp);
    if (sample) samples.push(sample);
  }

  return samples.length >= 8 ? samples : null;
}

function computeTrueScores(catalogue, config, rng) {
  const results = {};
  for (const name of Object.keys(SCENARIOS)) results[name] = [];

  for (const dance of catalogue.dances) {
    const frameCount = Math.round(config.windowSeconds * dance.sampledFps);
    const stepFrames = Math.max(1, Math.round(config.trueStepSeconds * dance.sampledFps));

    for (let start = 0; start + frameCount <= dance.rawFrames.length; start += stepFrames) {
      for (const [name, scenario] of Object.entries(SCENARIOS)) {
        const window = buildLiveWindow(dance.rawFrames, start, frameCount, scenario, rng);
        if (!window) continue;

        const match = matchPoseSequenceToCatalogue(
          window,
          { dances: [dance.prepared] },
          { syncToTimeline: true },
        );
        if (match.best) {
          results[name].push({
            display: match.best.score,
            raw: match.best.rawScore,
            detected: match.detected,
            mirrored: match.best.mirrored,
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
    const frameCount = Math.round(config.windowSeconds * liveDance.sampledFps);
    const stepFrames = Math.max(1, Math.round(config.falseStepSeconds * liveDance.sampledFps));

    for (let start = 0; start + frameCount <= liveDance.rawFrames.length; start += stepFrames) {
      const window = buildLiveWindow(liveDance.rawFrames, start, frameCount, FALSE_SCENARIO, rng);
      if (!window) continue;

      for (const referenceDance of catalogue.dances) {
        if (referenceDance.id === liveDance.id) continue;

        const match = matchPoseSequenceToCatalogue(
          window,
          { dances: [referenceDance.prepared] },
          { syncToTimeline: true },
        );
        if (match.best) {
          scores.push({
            display: match.best.score,
            raw: match.best.rawScore,
            detected: match.detected,
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
  const displays = entries.map((entry) => entry.display);
  const raws = entries.map((entry) => entry.raw);
  const detectedRate = entries.length > 0
    ? entries.filter((entry) => entry.detected).length / entries.length
    : 0;
  const mirroredRate = entries.length > 0
    ? entries.filter((entry) => entry.mirrored).length / entries.length
    : 0;

  console.log(`\n${label} (n=${entries.length})`);
  console.log(
    `  display: p10=${formatScore(percentile(displays, 10))}  ` +
    `p50=${formatScore(percentile(displays, 50))}  ` +
    `p90=${formatScore(percentile(displays, 90))}  ` +
    `p99=${formatScore(percentile(displays, 99))}  ` +
    `detected=${(detectedRate * 100).toFixed(1)}%` +
    (mirroredRate > 0 ? `  mirroredPick=${(mirroredRate * 100).toFixed(1)}%` : ""),
  );
  console.log(
    `  raw:     p10=${formatScore(percentile(raws, 10))}  ` +
    `p50=${formatScore(percentile(raws, 50))}  ` +
    `p90=${formatScore(percentile(raws, 90))}  ` +
    `p99=${formatScore(percentile(raws, 99))}`,
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

  console.log("\n=== True-positive scenarios (correct dance, synthesized live player) ===");
  const trueScores = computeTrueScores(catalogue, args, rng);
  for (const [name, entries] of Object.entries(trueScores)) {
    summarize(name, entries);
  }

  console.log("\n=== False-positive distribution (wrong dance, low-severity player) ===");
  const falseScores = computeFalseScores(catalogue, args, rng);
  summarize("false/cross-dance", falseScores);

  console.log(
    "\nKnobs live in DEFAULT_OPTIONS, src/utils/catalogue-matcher.js: " +
    "staticSigmaDeg (pose tolerance), dynamicsWeight/velocitySigmaDegPerSec " +
    "(motion term), scoreFloor/scoreCeil (raw->display stretch). Aim for " +
    "false/cross-dance display p90 well below the true/medium display p10.",
  );
}

main().catch((error) => {
  console.error(`\nEvaluation failed: ${error.message}`);
  console.error(error.stack);
  process.exitCode = 1;
});
