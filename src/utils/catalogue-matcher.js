import { bandedDtw } from "./dtw.js";

const KP = {
  leftShoulder: 5,
  rightShoulder: 6,
  leftElbow: 7,
  rightElbow: 8,
  leftWrist: 9,
  rightWrist: 10,
  leftHip: 11,
  rightHip: 12,
  leftKnee: 13,
  rightKnee: 14,
  leftAnkle: 15,
  rightAnkle: 16,
};

const ANGLE_NAMES = [
  "leftElbow",
  "rightElbow",
  "leftShoulder",
  "rightShoulder",
  "leftHip",
  "rightHip",
  "leftKnee",
  "rightKnee",
];

// Per-joint tolerance tiers. Core joints (hips/shoulders) anchor the pose and
// get a tight tolerance; extremities (wrists/ankles, plus the face points we
// don't otherwise score) are naturally noisier - both in real dancers'
// precision and in pose-estimation confidence - so they get more room.
const JOINT_TIER = {
  leftShoulder: "core",
  rightShoulder: "core",
  leftHip: "core",
  rightHip: "core",
  leftElbow: "mid",
  rightElbow: "mid",
  leftKnee: "mid",
  rightKnee: "mid",
  leftWrist: "extremity",
  rightWrist: "extremity",
  leftAnkle: "extremity",
  rightAnkle: "extremity",
  nose: "extremity",
  leftEye: "extremity",
  rightEye: "extremity",
  leftEar: "extremity",
  rightEar: "extremity",
};

const ANGLE_JOINT_TIER = {
  leftShoulder: "core",
  rightShoulder: "core",
  leftHip: "core",
  rightHip: "core",
  leftElbow: "mid",
  rightElbow: "mid",
  leftKnee: "mid",
  rightKnee: "mid",
};

const DEFAULT_OPTIONS = {
  // Minimum keypoint confidence (0-1, from the pose model) for a joint to be
  // treated as "visible" at all - anywhere below this, the joint is skipped
  // rather than compared. Lower it if webcam pose detection tends to report
  // low-confidence-but-correct joints (e.g. dim lighting); raise it if you'd
  // rather ignore shaky/noisy joints than compare against a bad estimate.
  keypointThreshold: 0.20,
  // How many of the 12 tracked joints must be visible (per keypointThreshold)
  // in BOTH the live pose and the reference frame for that frame's comparison
  // to count at all ("informative"). Below this, the frame is worth 0 and
  // doesn't count toward score or coverage. Lower it to tolerate more
  // occlusion/cropping (e.g. a webcam that only frames the upper body);
  // raising it demands a fuller-body view before trusting any single frame.
  minComparableKeypoints: 7,
  // The final score (0-100) a dance candidate must reach to be considered
  // "detected" at all. This is the main false-positive knob: lower it and
  // more borderline/sloppy matches get accepted (more forgiving, but risks
  // matching the wrong dance or a half-right move); raise it to demand a
  // cleaner match before showing anything. Use `npm run evaluate:matcher`
  // after changing this - it prints the cross-dance false-positive score
  // distribution (p90/p95/p99) so you can see how close you're cutting it.
  minConfidence: 50,
  // Minimum number of live pose samples buffered before matching even
  // starts. Too low and single-frame noise can trigger a match; too high
  // delays the first score after the player starts moving. Tied to
  // sequenceWindowSeconds and the live sampling rate (roughly, the camera's
  // effective FPS) - don't set this above what a full window can hold.
  minSequenceSamples: 8,
  // Length (seconds) of the rolling live-pose buffer matched against the
  // catalogue each tick. Bigger = smoother/more stable scores but slower to
  // react to a new move; smaller = more responsive but noisier frame-to-frame.
  sequenceWindowSeconds: 3,
  // Time-based coverage gate: what fraction of the live window must be
  // backed by informative (enough visible keypoints, see
  // minComparableKeypoints) comparisons for the match to count as detected.
  // Lower it to tolerate more dropped/occluded frames within an otherwise
  // decent take; raise it to require the player stay fully visible
  // throughout the whole window.
  minCoverageFraction: 0.5,
  // Score-point gap required over the second-best dance candidate. Only
  // meaningful when more than one dance is being compared at once (the
  // video-mode fallback path) - a single pre-selected dance has no "other
  // candidate" to be confused with. Lower it if the fallback path rejects
  // matches that are obviously right just because two dances score close;
  // raise it if it's flip-flopping between two similar-looking dances.
  minMargin: 6,
  // Local time-warping room around the audio-clock anchor, in seconds, for
  // the single-dance/audio-synced path. This is the direct replacement for
  // the old rigid +/-0.22s nearest-frame snap with no warping at all - it's
  // the main "temporal room" knob: how far off-beat (reaction lag, audio
  // latency, natural tempo drift) a player can be while still being aligned
  // to the right reference frame. Raise it to forgive more timing slop;
  // lower it if the matcher seems to be "catching up" to moves from the
  // wrong part of the song. Also consider calibrating
  // AUDIO_SYNC_OFFSET_SECONDS in App.jsx (currently 0, uncalibrated) if
  // there's a consistent lag/lead rather than random jitter.
  syncBandSeconds: 0.9,
  // Same time-warping room as syncBandSeconds, but for the fallback
  // multi-dance search (no dance pre-selected / no audio clock) - kept
  // tighter since this path also has to search for the right start offset,
  // and a wide band there makes the search both slower and more likely to
  // drift onto the wrong choreography.
  fallbackBandSeconds: 0.35,
  // How finely the fallback path scans candidate start offsets (seconds
  // between scan points) before running the banded alignment at each one.
  // Smaller = more thorough/slower search; larger = faster but can skip
  // past the true starting offset entirely.
  fallbackAnchorStepSeconds: 0.3,
  // How far (seconds, before/after the live sequence's own start time) the
  // fallback path is willing to search for a starting offset. Raise it if
  // players can start dancing well before/after the reference video's
  // timeline; lower it to keep the search fast and avoid matching a
  // coincidentally-similar moment far away in the song.
  fallbackTimeTolerance: 2.0,
  // Small extra cost added for every non-diagonal (insertion/deletion) DTW
  // step, to discourage the alignment from taking long degenerate runs
  // (e.g. freezing on one frame) just to dodge a costly comparison. Raise it
  // to force more diagonal (one-to-one, real-time) steps; lower it to allow
  // more aggressive local speed-up/slow-down warping.
  stepPenalty: 0.02,
  // Per-joint positional tolerance (Gaussian sigma, in normalized
  // hip-to-shoulder units) - core joints (hips/shoulders) are tightest since
  // they anchor the pose, extremities (wrists/ankles) are loosest since
  // both real dancers' precision and pose-estimation noise are highest
  // there. Raising these makes positional matching more forgiving, but
  // they're the most sensitive knob for false positives - cross-dance
  // choreography mostly differs by "how far off is each joint," so widening
  // sigma erases that signal fast. Always re-run
  // `npm run evaluate:matcher` after touching these; widening them from
  // {0.12, 0.16, 0.22} to {0.16, 0.20, 0.28} alone pushed the cross-dance
  // false-positive detection rate from 0% to ~28-59% in testing.
  keypointSigma: { core: 0.12, mid: 0.16, extremity: 0.22 },
  // Same idea as keypointSigma but for joint angles (degrees) instead of
  // joint positions - core joints (shoulders/hips) tighter, elbows/knees
  // ("mid") looser. Same false-positive sensitivity warning applies.
  angleSigma: { core: 18, mid: 24 },
};

function mergeConfig(options) {
  return {
    ...DEFAULT_OPTIONS,
    ...options,
    keypointSigma: { ...DEFAULT_OPTIONS.keypointSigma, ...(options.keypointSigma ?? {}) },
    angleSigma: { ...DEFAULT_OPTIONS.angleSigma, ...(options.angleSigma ?? {}) },
  };
}

export async function loadPoseCatalogue(baseUrl = "/") {
  const root = joinUrl(baseUrl, "catalogue/poses");
  const index = await fetchJson(joinUrl(root, "index.json"));
  const dances = await Promise.all(
    index.dances.map(async (entry) => {
      const data = await fetchJson(joinUrl(root, entry.file));
      return prepareDance({ ...entry, data });
    }),
  );

  return {
    ...index,
    dances,
  };
}

export function matchPoseToCatalogue(pose, catalogue, options = {}) {
  const config = mergeConfig(options);
  if (!pose?.keypoints || !catalogue?.dances?.length) {
    return emptyMatch();
  }

  const normalized = normalizeKeypoints(pose.keypoints, pose.bbox, config.keypointThreshold);
  if (!normalized) return emptyMatch();

  const live = {
    keypoints: normalized,
    angles: calculateAngles(pose.keypoints, config.keypointThreshold),
  };

  const candidates = catalogue.dances
    .map((dance) => findBestFrame(live, dance, config))
    .sort((a, b) => b.score - a.score);

  const best = candidates[0] ?? null;
  const second = candidates[1] ?? null;
  const margin = best && second ? best.score - second.score : best?.score ?? 0;

  return {
    best,
    candidates,
    detected: Boolean(best && best.score >= config.minConfidence),
    margin,
  };
}

const SWAP_PAIRS = [
  [1, 2],   // eyes
  [3, 4],   // ears
  [5, 6],   // shoulders
  [7, 8],   // elbows
  [9, 10],  // wrists
  [11, 12], // hips
  [13, 14], // knees
  [15, 16]  // ankles
];

function swapLeftRightKeypoints(keypoints) {
  if (!keypoints) return keypoints;
  const swapped = keypoints.map(kp => ({ ...kp }));
  for (const [left, right] of SWAP_PAIRS) {
    if (left < keypoints.length && right < keypoints.length) {
      const tempX = swapped[left].x;
      const tempY = swapped[left].y;
      const tempScore = swapped[left].score;

      swapped[left].x = swapped[right].x;
      swapped[left].y = swapped[right].y;
      swapped[left].score = swapped[right].score;

      swapped[right].x = tempX;
      swapped[right].y = tempY;
      swapped[right].score = tempScore;
    }
  }
  return swapped;
}

export function createPoseSample(pose, timestamp, options = {}) {
  const config = mergeConfig(options);
  if (!pose?.keypoints) return null;

  let keypoints = pose.keypoints;
  if (config.mirror) {
    keypoints = swapLeftRightKeypoints(keypoints);
  }

  const normalized = normalizeKeypoints(keypoints, pose.bbox, config.keypointThreshold);
  if (!normalized) return null;

  return {
    timestamp,
    keypoints: normalized,
    angles: calculateAngles(keypoints, config.keypointThreshold),
  };
}

export function matchPoseSequenceToCatalogue(samples, catalogue, options = {}) {
  const config = mergeConfig(options);
  const usableSamples = samples.filter((sample) => sample?.keypoints?.length);

  if (
    usableSamples.length < config.minSequenceSamples ||
    !catalogue?.dances?.length
  ) {
    return emptyMatch();
  }

  const candidates = catalogue.dances
    .map((dance) => findBestSequence(usableSamples, dance, config))
    .sort((a, b) => b.score - a.score);

  const best = candidates[0] ?? null;
  const second = candidates[1] ?? null;
  const margin = best && second ? best.score - second.score : best?.score ?? 0;
  const singleCandidate = candidates.length <= 1;

  const detected = Boolean(
    best &&
    best.score >= config.minConfidence &&
    best.informativeCoverageFraction >= config.minCoverageFraction &&
    (singleCandidate || margin >= config.minMargin),
  );

  return {
    best,
    candidates,
    detected,
    margin,
  };
}

export function stabilizeMatches(history, currentMatch, now, windowMs = 2200) {
  const nextHistory = history
    .filter((entry) => now - entry.time <= windowMs)
    .concat(
      currentMatch?.detected && currentMatch?.best
        ? [{ time: now, match: currentMatch.best }]
        : [],
    );

  const grouped = new Map();
  for (const entry of nextHistory) {
    const current = grouped.get(entry.match.id) ?? {
      ...entry.match,
      hits: 0,
      scoreSum: 0,
      bestScore: 0,
    };
    current.hits += 1;
    current.scoreSum += entry.match.score;
    current.bestScore = Math.max(current.bestScore, entry.match.score);
    grouped.set(entry.match.id, current);
  }

  const stableCandidates = [...grouped.values()]
    .map((candidate) => ({
      ...candidate,
      score: candidate.scoreSum / candidate.hits,
    }))
    .sort((a, b) => b.score - a.score || b.hits - a.hits);

  return {
    history: nextHistory,
    stable: stableCandidates[0] ?? null,
  };
}

// Turns raw extracted-pose JSON for one dance into the in-memory shape the
// matcher operates on. Pure/fetch-independent so it can be shared between
// the browser catalogue loader above and offline node tooling (e.g. an
// evaluation/calibration script) that reads the same JSON files from disk.
export function prepareDance(entry) {
  const frames = entry.data.frames
    .filter((frame) => {
      if (!frame.person) return false;
      // Reject frames whose normalization scale is below the reliable threshold.
      // This catches existing data where shoulders were misdetected (scale ~2–14px)
      // which would blow up all normalized coordinates.
      const scale = frame.person.normalized?.scale ?? 0;
      return scale >= MIN_NORM_SCALE;
    })
    .map((frame) => ({
      frameIndex: frame.frameIndex,
      timestamp: frame.timestamp,
      keypoints: frame.person.normalized?.keypoints ?? [],
      angles: frame.person.angles ?? {},
    }));

  return {
    id: entry.id,
    title: entry.title,
    source: entry.source,
    audioUrl: entry.audioUrl,
    sampledFps: entry.sampledFps,
    detectedFrames: entry.detectedFrames,
    sampledFrames: entry.sampledFrames,
    detectionRate: entry.detectionRate,
    frames,
  };
}

function findBestFrame(live, dance, config) {
  let best = {
    id: dance.id,
    title: dance.title,
    score: 0,
    frameIndex: null,
    timestamp: null,
    keypointScore: 0,
    angleScore: 0,
    comparableKeypoints: 0,
  };

  for (const frame of dance.frames) {
    const comparison = compareFrame(live, frame, config);
    if (comparison.score > best.score) {
      best = {
        id: dance.id,
        title: dance.title,
        frameIndex: frame.frameIndex,
        timestamp: frame.timestamp,
        ...comparison,
      };
    }
  }

  return best;
}

function emptySequenceResult(dance) {
  return {
    id: dance.id,
    title: dance.title,
    score: 0,
    matchedSamples: 0,
    keypointScore: 0,
    angleScore: null,
    coverageSeconds: 0,
    informativeCoverageFraction: 0,
    alignedFrameIndex: null,
    alignedTimestamp: null,
  };
}

function findBestSequence(samples, dance, config) {
  if (!dance.frames.length || samples.length === 0) {
    return emptySequenceResult(dance);
  }

  if (config.syncToTimeline) {
    return alignSyncedSequence(samples, dance, config);
  }

  return alignFallbackSequence(samples, dance, config);
}

// Primary path: a single dance has already been selected and the audio (or
// video) element's clock tells us, for every live sample's timestamp,
// approximately which reference frame it should correspond to. We still run
// a small banded DTW rather than a rigid nearest-frame lookup, because real
// dancers have reaction lag and micro-tempo drift relative to the track -
// the band is exactly the "temporal room" that was previously nonexistent.
function alignSyncedSequence(samples, dance, config) {
  const sampledFps = dance.sampledFps || 30;
  const bandFrames = Math.max(1, Math.round(config.syncBandSeconds * sampledFps));
  const anchorForRow = (i) => nearestFrameIndex(dance.frames, samples[i].timestamp);

  return runAlignment(samples, dance, config, anchorForRow, bandFrames);
}

// Fallback path: no dance has been pre-selected (video mode without a
// manual selection), so we don't know where in the reference timeline - or
// even which dance - the live sequence corresponds to. Coarsely scan
// candidate start offsets (replacing the old offset x speedFactor sweep),
// and run a banded DTW around each candidate to absorb local tempo
// variation instead of only 3 discrete global speeds.
function alignFallbackSequence(samples, dance, config) {
  const sampledFps = dance.sampledFps || 30;
  const bandFrames = Math.max(1, Math.round(config.fallbackBandSeconds * sampledFps));
  const firstLiveTimestamp = samples[0].timestamp;
  const liveDuration = samples.at(-1).timestamp - firstLiveTimestamp;

  const latestStart = Math.max(0, dance.frames.at(-1).timestamp - liveDuration * 0.85);
  const tolerance = config.fallbackTimeTolerance;
  const minStart = Math.max(0, firstLiveTimestamp - tolerance);
  const maxStart = Math.min(latestStart, firstLiveTimestamp + tolerance);
  const step = Math.max(config.fallbackAnchorStepSeconds, 1 / sampledFps);

  let best = null;
  for (let start = minStart; start <= maxStart; start += step) {
    const anchorForRow = (i) => nearestFrameIndex(
      dance.frames,
      start + (samples[i].timestamp - firstLiveTimestamp),
    );
    const result = runAlignment(samples, dance, config, anchorForRow, bandFrames);
    if (!best || result.score > best.score) best = result;
  }

  return best ?? emptySequenceResult(dance);
}

// Shared alignment core: runs the banded DTW for one dance against one
// anchor line, then aggregates score/coverage from the resulting path.
function runAlignment(samples, dance, config, anchorForRow, bandFrames) {
  const frames = dance.frames;
  const cellCache = new Map();

  const cost = (i, j) => {
    const comparison = compareFrame(samples[i], frames[j], config);
    cellCache.set(i * frames.length + j, comparison);
    return 1 - comparison.score / 100;
  };

  const alignment = bandedDtw({
    n: samples.length,
    m: frames.length,
    cost,
    anchorForRow,
    bandRadius: bandFrames,
    stepPenalty: config.stepPenalty,
  });

  if (!alignment.path.length) return emptySequenceResult(dance);

  // Collapse the path to one comparison per live sample (a live sample can
  // appear multiple times when the path takes a horizontal - reference
  // advances faster than the dancer - step; keep the last, which is what
  // the alignment settled on).
  const rowResult = new Map();
  for (const [i, j] of alignment.path) {
    rowResult.set(i, cellCache.get(i * frames.length + j));
  }

  let scoreSum = 0;
  let keypointScoreSum = 0;
  let angleScoreSum = 0;
  let angleScoreCount = 0;
  let informativeRows = 0;

  for (const comparison of rowResult.values()) {
    if (!comparison?.informative) continue;
    informativeRows += 1;
    scoreSum += comparison.score;
    keypointScoreSum += comparison.keypointScore;
    if (comparison.angleScore !== null) {
      angleScoreSum += comparison.angleScore;
      angleScoreCount += 1;
    }
  }

  if (informativeRows === 0) return emptySequenceResult(dance);

  const totalDuration = samples.at(-1).timestamp - samples[0].timestamp;
  const informativeCoverageFraction = informativeRows / samples.length;
  const coverageSeconds = totalDuration > 0
    ? informativeCoverageFraction * totalDuration
    : 0;

  const [, lastJ] = alignment.path.at(-1);
  const alignedFrame = frames[lastJ] ?? null;

  return {
    id: dance.id,
    title: dance.title,
    score: round(scoreSum / informativeRows, 1),
    matchedSamples: informativeRows,
    keypointScore: round(keypointScoreSum / informativeRows, 1),
    angleScore: angleScoreCount > 0 ? round(angleScoreSum / angleScoreCount, 1) : null,
    coverageSeconds: round(coverageSeconds, 2),
    informativeCoverageFraction: round(informativeCoverageFraction, 3),
    alignedFrameIndex: alignedFrame?.frameIndex ?? null,
    alignedTimestamp: alignedFrame?.timestamp ?? null,
  };
}

// Binary search for the reference frame whose timestamp is closest to the
// given timestamp; used to center the DTW band on a per-row anchor.
function nearestFrameIndex(frames, timestamp) {
  let low = 0;
  let high = frames.length - 1;

  while (low <= high) {
    const mid = (low + high) >> 1;
    if (frames[mid].timestamp < timestamp) low = mid + 1;
    else high = mid - 1;
  }

  if (high < 0) return 0;
  if (low >= frames.length) return frames.length - 1;
  return Math.abs(frames[high].timestamp - timestamp) <= Math.abs(frames[low].timestamp - timestamp)
    ? high
    : low;
}

function compareFrame(live, frame, config) {
  const pointComparison = compareKeypoints(live.keypoints, frame.keypoints, config);
  const angleScore = compareAngles(live.angles, frame.angles, config);
  const informative = pointComparison.count >= config.minComparableKeypoints;

  if (!informative) {
    return {
      score: 0,
      keypointScore: round(pointComparison.score, 1),
      angleScore: angleScore === null ? null : round(angleScore, 1),
      comparableKeypoints: pointComparison.count,
      informative: false,
    };
  }

  const score = angleScore === null
    ? pointComparison.score
    : pointComparison.score * 0.68 + angleScore * 0.32;

  return {
    score: round(score, 1),
    keypointScore: round(pointComparison.score, 1),
    angleScore: angleScore === null ? null : round(angleScore, 1),
    comparableKeypoints: pointComparison.count,
    informative: true,
  };
}

// Per-joint Gaussian falloff, applied before aggregating. This replaces the
// old approach of pooling every joint into one average distance and
// applying a single linear threshold to that scalar - which made it
// impossible to give core joints tighter tolerance than extremities, and
// meant fixing false negatives required uniformly loosening every joint at
// once (exactly the tuning churn this rework replaces).
function compareKeypoints(live, reference, config) {
  let weightedScore = 0;
  let totalWeight = 0;
  let count = 0;

  for (let i = 0; i < Math.min(live.length, reference.length); i++) {
    const a = live[i];
    const b = reference[i];
    if (!visible(a, config.keypointThreshold) || !visible(b, config.keypointThreshold)) {
      continue;
    }

    const tier = JOINT_TIER[a.name] ?? "mid";
    const sigma = config.keypointSigma[tier];
    const d = Math.hypot(a.x - b.x, a.y - b.y);
    const pointScore = 100 * Math.exp(-(d * d) / (2 * sigma * sigma));

    const weight = (a.score + b.score) / 2;
    weightedScore += pointScore * weight;
    totalWeight += weight;
    count += 1;
  }

  if (count === 0 || totalWeight === 0) {
    return { score: 0, count: 0 };
  }

  return { score: weightedScore / totalWeight, count };
}

function compareAngles(liveAngles, referenceAngles, config) {
  let scoreSum = 0;
  let count = 0;

  for (const name of ANGLE_NAMES) {
    const live = liveAngles[name];
    const reference = referenceAngles[name];
    if (live === null || live === undefined || reference === null || reference === undefined) {
      continue;
    }

    const tier = ANGLE_JOINT_TIER[name] ?? "mid";
    const sigma = config.angleSigma[tier];
    const diff = Math.abs(live - reference);
    scoreSum += 100 * Math.exp(-(diff * diff) / (2 * sigma * sigma));
    count += 1;
  }

  if (count === 0) {
    return null;
  }

  return scoreSum / count;
}

const MIN_NORM_SCALE = 15;

function normalizeKeypoints(keypoints, bbox, threshold) {
  const leftHip = keypoints[KP.leftHip];
  const rightHip = keypoints[KP.rightHip];
  const leftShoulder = keypoints[KP.leftShoulder];
  const rightShoulder = keypoints[KP.rightShoulder];

  // Both hips required for a position-invariant origin.
  if (!visible(leftHip, threshold) || !visible(rightHip, threshold)) return null;
  const origin = midpoint(leftHip, rightHip);

  // Primary scale: shoulder width.
  let scale = null;
  if (visible(leftShoulder, threshold) && visible(rightShoulder, threshold)) {
    const d = distance(leftShoulder, rightShoulder);
    if (d >= MIN_NORM_SCALE) scale = d;
  }

  // Secondary scale: torso height.
  if (!scale) {
    const visShoulders = [
      visible(leftShoulder, threshold) ? leftShoulder : null,
      visible(rightShoulder, threshold) ? rightShoulder : null,
    ].filter(Boolean);
    if (visShoulders.length > 0) {
      const shoulderMid = visShoulders.length === 2
        ? midpoint(leftShoulder, rightShoulder)
        : visShoulders[0];
      const torso = distance(origin, shoulderMid);
      if (torso >= MIN_NORM_SCALE) scale = torso;
    }
  }

  if (!scale) return null;

  return keypoints.map((point) => ({
    name: point.name,
    x: (point.x - origin.x) / scale,
    y: (point.y - origin.y) / scale,
    score: point.score,
  }));
}

function calculateAngles(keypoints, threshold) {
  return {
    leftElbow: calculateAngle(keypoints[KP.leftShoulder], keypoints[KP.leftElbow], keypoints[KP.leftWrist], threshold),
    rightElbow: calculateAngle(keypoints[KP.rightShoulder], keypoints[KP.rightElbow], keypoints[KP.rightWrist], threshold),
    leftShoulder: calculateAngle(keypoints[KP.leftHip], keypoints[KP.leftShoulder], keypoints[KP.leftElbow], threshold),
    rightShoulder: calculateAngle(keypoints[KP.rightHip], keypoints[KP.rightShoulder], keypoints[KP.rightElbow], threshold),
    leftHip: calculateAngle(keypoints[KP.leftShoulder], keypoints[KP.leftHip], keypoints[KP.leftKnee], threshold),
    rightHip: calculateAngle(keypoints[KP.rightShoulder], keypoints[KP.rightHip], keypoints[KP.rightKnee], threshold),
    leftKnee: calculateAngle(keypoints[KP.leftHip], keypoints[KP.leftKnee], keypoints[KP.leftAnkle], threshold),
    rightKnee: calculateAngle(keypoints[KP.rightHip], keypoints[KP.rightKnee], keypoints[KP.rightAnkle], threshold),
  };
}

function calculateAngle(a, b, c, threshold) {
  if (!visible(a, threshold) || !visible(b, threshold) || !visible(c, threshold)) {
    return null;
  }

  const ba = { x: a.x - b.x, y: a.y - b.y };
  const bc = { x: c.x - b.x, y: c.y - b.y };
  const angleA = Math.atan2(ba.y, ba.x);
  const angleC = Math.atan2(bc.y, bc.x);
  let diff = Math.abs(angleA - angleC) * (180 / Math.PI);
  if (diff > 180) diff = 360 - diff;
  return diff;
}

function visible(point, threshold) {
  return point && point.score >= threshold;
}

function midpoint(a, b) {
  return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
}

function distance(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function emptyMatch() {
  return {
    best: null,
    candidates: [],
    detected: false,
    margin: 0,
  };
}

function joinUrl(base, part) {
  return `${base.replace(/\/+$/, "")}/${part.replace(/^\/+/, "")}`;
}

async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Could not load ${url} (${response.status})`);
  }
  return response.json();
}

function round(value, digits = 1) {
  const factor = 10 ** digits;
  return Math.round(value * factor) / factor;
}
