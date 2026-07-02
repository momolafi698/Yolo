import { bandedDtw } from "./dtw.js";

const KEYPOINT_NAMES = [
  "nose",
  "leftEye",
  "rightEye",
  "leftEar",
  "rightEar",
  "leftShoulder",
  "rightShoulder",
  "leftElbow",
  "rightElbow",
  "leftWrist",
  "rightWrist",
  "leftHip",
  "rightHip",
  "leftKnee",
  "rightKnee",
  "leftAnkle",
  "rightAnkle",
];

const KP = Object.fromEntries(KEYPOINT_NAMES.map((name, index) => [name, index]));

// A pose is represented as the DIRECTION (unit vector) of each major limb
// segment, nothing else. Direction is invariant to where the person stands,
// how far from the camera they are, and their body proportions (a long and a
// short forearm pointing the same way produce the same feature) - so none of
// those differences between the reference performer and the live player can
// cost points. Each bone only needs its own two endpoints to be visible,
// so a webcam that crops the legs still scores the arms instead of dropping
// the whole frame.
//
// Weights express how much each segment carries choreography: arms most,
// legs slightly less (often supporting weight rather than gesturing), torso
// lean least.
const BONES = [
  // Torso is deliberately low-weight: it mostly points up on everyone, so it
  // says little about the choreography - arms and legs carry the score.
  { name: "torso", from: "hipMid", to: "shoulderMid", weight: 0.3 },
  { name: "leftUpperArm", from: "leftShoulder", to: "leftElbow", weight: 1.0 },
  { name: "rightUpperArm", from: "rightShoulder", to: "rightElbow", weight: 1.0 },
  { name: "leftForearm", from: "leftElbow", to: "leftWrist", weight: 1.0 },
  { name: "rightForearm", from: "rightElbow", to: "rightWrist", weight: 1.0 },
  { name: "leftThigh", from: "leftHip", to: "leftKnee", weight: 0.9 },
  { name: "rightThigh", from: "rightHip", to: "rightKnee", weight: 0.9 },
  { name: "leftShin", from: "leftKnee", to: "leftAnkle", weight: 0.7 },
  { name: "rightShin", from: "rightKnee", to: "rightAnkle", weight: 0.7 },
];

const BONE_NAMES = BONES.map((bone) => bone.name);

const MIRROR_BONE = {
  torso: "torso",
  leftUpperArm: "rightUpperArm",
  rightUpperArm: "leftUpperArm",
  leftForearm: "rightForearm",
  rightForearm: "leftForearm",
  leftThigh: "rightThigh",
  rightThigh: "leftThigh",
  leftShin: "rightShin",
  rightShin: "leftShin",
};

// "Rest" direction of each bone in image coordinates (y grows downward):
// limbs hang/point down on an idle human, the torso points up. A reference
// bone near its rest direction and not moving says almost nothing about the
// choreography - any person standing in frame matches it - so it gets less
// say in the score (see boneInformativeness).
const REST_DIRECTION = {
  torso: { x: 0, y: -1 },
  leftUpperArm: { x: 0, y: 1 },
  rightUpperArm: { x: 0, y: 1 },
  leftForearm: { x: 0, y: 1 },
  rightForearm: { x: 0, y: 1 },
  leftThigh: { x: 0, y: 1 },
  rightThigh: { x: 0, y: 1 },
  leftShin: { x: 0, y: 1 },
  rightShin: { x: 0, y: 1 },
};

// How much a reference bone should count, in [restingBoneWeight, 1]: full
// weight when it deviates from its rest direction (a held gesture) or is
// moving (a beat), low weight when it just hangs there. Attached to
// reference frames once at load; live samples never need it.
function attachInformativeness(frames, config) {
  for (const frame of frames) {
    for (const name of BONE_NAMES) {
      const bone = frame.bones[name];
      if (!bone) continue;

      const rest = REST_DIRECTION[name];
      const dot = clamp(bone.x * rest.x + bone.y * rest.y, -1, 1);
      const deviationDeg = Math.acos(dot) * (180 / Math.PI);
      const deviationTerm = Math.min(1, deviationDeg / config.informativeDeviationDeg);

      const vel = frame.vel?.[name];
      const motionTerm = vel === null || vel === undefined
        ? 0
        : Math.min(1, Math.abs(vel) / config.activityFloorDegPerSec);

      bone.info = config.restingBoneWeight +
        (1 - config.restingBoneWeight) * Math.max(deviationTerm, motionTerm);
    }
  }
}

const DEFAULT_OPTIONS = {
  // Minimum keypoint confidence (0-1, from the pose model) for a joint to
  // participate in a bone. Below this the bone that needs it is skipped -
  // never compared against a bad estimate.
  keypointThreshold: 0.2,
  // How many of the 9 bones must be measurable in BOTH the live sample and
  // the reference frame for that comparison to count ("informative").
  // 4 = both arms, or one arm plus torso and a leg - enough signal to score.
  minComparableBones: 4,
  // Minimum live samples buffered before matching starts.
  minSequenceSamples: 8,
  // Length (seconds) of the rolling live-pose window matched each tick.
  sequenceWindowSeconds: 3,
  // Fraction of the live window that must be informative for the match to
  // count as a confident detection. Only gates the `detected` flag (used for
  // dance identification / rank labels) - the displayed score is never gated.
  minCoverageFraction: 0.35,
  // Display-score (0-100) threshold for the `detected` flag. Same caveat:
  // this no longer hides the score, it only marks confidence.
  minConfidence: 40,
  // Display-score gap over the second-best dance required in multi-dance
  // (fallback) mode for `detected`.
  minMargin: 6,
  // Local time-warping room (seconds) around the audio-clock anchor for the
  // single-dance/audio-synced path: how far off-beat a player can be while
  // still matched to the right reference frame.
  syncBandSeconds: 1.25,
  // After alignment, each live sample is scored against the BEST reference
  // frame within this many seconds of its aligned position, not just the
  // exact aligned frame. This is the per-pose "human timing slop": hitting
  // a move slightly early or late costs nothing.
  scorePoolSeconds: 0.25,
  // Same, for the multi-dance fallback search (kept tighter - this path also
  // scans start offsets, and a wide band there drifts onto wrong dances).
  fallbackBandSeconds: 0.35,
  // Scan granularity (seconds) for fallback start offsets.
  fallbackAnchorStepSeconds: 0.3,
  // How far (seconds) fallback searches for a start offset.
  fallbackTimeTolerance: 5.0,
  // Extra DTW cost per non-diagonal step. Deliberately not tiny: a player
  // holding one pose could otherwise "freeze" the alignment on whichever
  // single reference frame matches that pose best and coast on it for the
  // whole window.
  stepPenalty: 0.06,
  // Reference videos can have unusable stretches (close-up shots, failed
  // detections) that leave holes in the frame timeline. A live sample whose
  // expected reference time falls more than this far from any existing
  // frame is excluded from scoring entirely - better no score than being
  // graded against a different moment of the song.
  referenceGapToleranceSeconds: 0.75,
  // Tolerance (degrees) on each bone's direction error. This is the main
  // "how forgiving is a pose" knob. 30 deg means a bone held ~30 deg off
  // still earns ~61% of its points; ~60 deg off earns ~14%.
  staticSigmaDeg: 30,
  // Motion term: bones' angular velocities (deg/s) are compared so that
  // moving with the choreography scores and freezing during a move doesn't.
  // Velocity is measured over velocityDeltaSeconds; the comparison tolerance
  // is velocitySigmaDegPerSec. The term is weighted by how active the bone
  // is (relative to activityFloorDegPerSec), so held poses aren't penalized
  // for having no motion to compare.
  velocityDeltaSeconds: 0.3,
  // Loose on purpose: live webcam keypoints jitter, and differentiating
  // jittery positions inflates velocity error even for a perfect dancer -
  // and human timing offsets already show up in the static term, so a tight
  // velocity match would punish the same mistake twice. The anti-idle work
  // is done by the sequence-level motion-ratio penalty and the
  // informativeness weighting, not by this per-frame term.
  velocitySigmaDegPerSec: 140,
  activityFloorDegPerSec: 45,
  // Share of the per-frame score taken by the motion term when available.
  dynamicsWeight: 0.25,
  // Reference-bone informativeness (see attachInformativeness): a bone at
  // rest and not moving only counts restingBoneWeight; full weight is
  // reached at informativeDeviationDeg away from rest (or when moving).
  restingBoneWeight: 0.25,
  informativeDeviationDeg: 60,
  // Sequence-level anti-idle penalty: over the aligned window, the player's
  // total limb motion is compared to the reference's. Full credit from
  // motionRatioFloor of the reference's motion upward; below that the
  // displayed score scales down linearly (a frozen pose ends up near 0 no
  // matter how well it matches one frame). No penalty when the reference
  // itself is nearly still (below motionPenaltyMinRefDegPerSec) - holding a
  // pose the choreography holds is correct.
  // 0.45: casual players dance "smaller" than the pro in the reference
  // video; moving with the routine at ~half amplitude should not be
  // punished - only genuinely standing still should.
  motionRatioFloor: 0.45,
  motionPenaltyMinRefDegPerSec: 12,
  // Live pose estimates jitter, and differentiating jitter looks like ~10
  // deg/s of motion even on a player standing perfectly still. Deducted
  // from the live side before computing the ratio so "frozen" reads as
  // frozen; the reference (extracted from stable video) needs no deduction.
  liveMotionNoiseFloorDegPerSec: 8,
  // Raw similarity -> displayed percentage mapping. Raw similarity has a
  // high floor (two random dance poses share gravity, a standing torso,
  // legs pointing down...), so it's stretched: scoreFloor maps to 0% and
  // scoreCeil to 100%. Calibrated with `npm run evaluate:matcher` so that
  // cross-dance (wrong dance) raw scores land near 0-25% displayed and
  // heavily-perturbed correct dancing lands above ~60%.
  scoreFloor: 78,
  scoreCeil: 96,
  // Concave display curve exponent (see toDisplayScore); < 1 lifts the
  // mid-range so decent-but-human dancing reads as a motivating score.
  displayGamma: 0.7,
};

function mergeConfig(options) {
  return { ...DEFAULT_OPTIONS, ...options };
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

// Load-time cleanup of extracted pose data. Older catalogues were extracted
// with a "largest bbox wins" person picker, which sometimes captured a
// spurious whole-frame detection with wild keypoints; those frames poison
// both the static comparison and (worse) the reference motion statistics.
// Drops: whole-frame bboxes, detections with garbage keypoint confidence,
// and single-frame "teleports" where the person jumps far from BOTH
// temporal neighbors while the neighbors agree with each other.
function sanitizeRawFrames(rawFrames, source) {
  const frameArea = source?.width && source?.height ? source.width * source.height : null;

  const frames = rawFrames.map((frame) => {
    const person = frame.person;
    if (!person?.keypoints?.length) return { ...frame, person: null };

    const bbox = person.bbox;
    if (frameArea && bbox?.length === 4 && bbox[2] * bbox[3] > frameArea * 0.9) {
      return { ...frame, person: null };
    }

    const topScores = person.keypoints
      .map((point) => point.score ?? 0)
      .sort((a, b) => b - a)
      .slice(0, 9);
    const meanTop = topScores.reduce((sum, s) => sum + s, 0) / topScores.length;
    if (meanTop < 0.25) return { ...frame, person: null };

    return frame;
  });

  const center = (bbox) => ({ x: bbox[0] + bbox[2] / 2, y: bbox[1] + bbox[3] / 2 });
  for (let i = 1; i < frames.length - 1; i++) {
    const prev = frames[i - 1]?.person;
    const cur = frames[i]?.person;
    const next = frames[i + 1]?.person;
    if (!prev?.bbox || !cur?.bbox || !next?.bbox) continue;

    const scale = Math.max(
      Math.hypot(cur.bbox[2], cur.bbox[3]),
      Math.hypot(prev.bbox[2], prev.bbox[3]),
    );
    if (scale <= 0) continue;

    const cPrev = center(prev.bbox);
    const cCur = center(cur.bbox);
    const cNext = center(next.bbox);
    const dPrev = Math.hypot(cCur.x - cPrev.x, cCur.y - cPrev.y) / scale;
    const dNext = Math.hypot(cCur.x - cNext.x, cCur.y - cNext.y) / scale;
    const dNeighbors = Math.hypot(cNext.x - cPrev.x, cNext.y - cPrev.y) / scale;

    if (dPrev > 0.45 && dNext > 0.45 && dNeighbors < 0.45) {
      frames[i] = { ...frames[i], person: null };
    }
  }

  return frames;
}

// Turns one raw extracted-pose JSON (pixel-space keypoints) into the
// in-memory shape the matcher uses: per-frame bone direction vectors plus
// per-bone angular velocities. Pure/fetch-independent so the browser loader
// above and the offline evaluation script share it.
export function prepareDance(entry) {
  const config = DEFAULT_OPTIONS;
  const frames = sanitizeRawFrames(entry.data.frames, entry.data.source)
    .map((frame) => {
      if (!frame.person?.keypoints?.length) return null;
      const bones = extractBones(frame.person.keypoints, config.keypointThreshold);
      if (!bones || bones.count < config.minComparableBones) return null;
      return {
        frameIndex: frame.frameIndex,
        timestamp: frame.timestamp,
        bones: bones.bones,
        boneCount: bones.count,
        // Hip-centered/shoulder-scaled coordinates kept only for the debug
        // overlay tools that draw the reference stick figure.
        keypoints: normalizePoseKeypoints(frame.person.keypoints, config.keypointThreshold),
      };
    })
    .filter(Boolean);

  attachVelocities(frames, config);
  attachInformativeness(frames, config);

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

// Builds a live sample from one raw pose detection. Coordinates can be in
// any consistent space (canvas pixels, letterboxed model space...) - only
// bone directions are kept. Returns null when nothing usable is visible.
export function createPoseSample(pose, timestamp, options = {}) {
  const config = mergeConfig(options);
  if (!pose?.keypoints?.length) return null;

  const bones = extractBones(pose.keypoints, config.keypointThreshold);
  if (!bones || bones.count === 0) return null;

  return {
    timestamp,
    bones: bones.bones,
    boneCount: bones.count,
    vel: null,
  };
}

export function matchPoseSequenceToCatalogue(samples, catalogue, options = {}) {
  const config = mergeConfig(options);
  const usableSamples = samples.filter((sample) => sample?.bones);

  if (
    usableSamples.length < config.minSequenceSamples ||
    !catalogue?.dances?.length
  ) {
    return emptyMatch();
  }

  attachVelocities(usableSamples, config);
  // The mirror convention (mirrored camera preview, "mirrored" dance-cover
  // sources, which way the player learned the routine...) is unknowable in
  // general, so don't guess: score both chiralities of the live window and
  // keep whichever matches better. A correct dancer always matches one.
  const mirroredSamples = usableSamples.map(mirrorSample);

  const candidates = catalogue.dances
    .map((dance) => findBestSequence(usableSamples, mirroredSamples, dance, config))
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

// ---------------------------------------------------------------------------
// Feature extraction
// ---------------------------------------------------------------------------

function visible(point, threshold) {
  return Boolean(point) &&
    Number.isFinite(point.x) &&
    Number.isFinite(point.y) &&
    (point.score ?? 0) >= threshold;
}

// Returns { bones: {name -> {x, y, w} | null}, count } where (x, y) is the
// bone's unit direction and w its confidence weight, or null if fewer than
// one bone is measurable.
function extractBones(keypoints, threshold) {
  const joint = (name) => {
    const point = keypoints[KP[name]];
    return visible(point, threshold) ? point : null;
  };

  // Virtual midpoints for the torso segment. If only one hip/shoulder is
  // visible, use it alone: the direction error this introduces is small
  // (half a hip width over a full torso length) and beats losing the torso.
  const virtualMid = (leftName, rightName) => {
    const left = joint(leftName);
    const right = joint(rightName);
    if (left && right) {
      return {
        x: (left.x + right.x) / 2,
        y: (left.y + right.y) / 2,
        score: Math.min(left.score ?? 1, right.score ?? 1),
      };
    }
    return left ?? right;
  };

  const points = {};
  for (const name of KEYPOINT_NAMES) points[name] = joint(name);
  points.hipMid = virtualMid("leftHip", "rightHip");
  points.shoulderMid = virtualMid("leftShoulder", "rightShoulder");

  const bones = {};
  let count = 0;
  for (const bone of BONES) {
    const a = points[bone.from];
    const b = points[bone.to];
    bones[bone.name] = null;
    if (!a || !b) continue;

    const dx = b.x - a.x;
    const dy = b.y - a.y;
    const length = Math.hypot(dx, dy);
    if (length < 1e-6) continue;

    bones[bone.name] = {
      x: dx / length,
      y: dy / length,
      w: Math.min(a.score ?? 1, b.score ?? 1),
    };
    count += 1;
  }

  if (count === 0) return null;
  return { bones, count };
}

// Per-bone angular velocity (deg/s, signed), measured against the sample
// roughly velocityDeltaSeconds earlier in the same sequence. Idempotent -
// safe to re-run on a rolling buffer every tick.
function attachVelocities(sequence, config) {
  const delta = config.velocityDeltaSeconds;

  for (let i = 0; i < sequence.length; i++) {
    const current = sequence[i];
    let previous = null;

    for (let k = i - 1; k >= 0; k--) {
      const gap = current.timestamp - sequence[k].timestamp;
      if (gap > delta * 1.8) break;
      if (gap >= delta * 0.5) previous = sequence[k];
      if (gap >= delta) break;
    }

    if (!previous) {
      current.vel = null;
      continue;
    }

    const gap = current.timestamp - previous.timestamp;
    const vel = {};
    let any = false;
    for (const name of BONE_NAMES) {
      const a = previous.bones[name];
      const b = current.bones[name];
      if (!a || !b) {
        vel[name] = null;
        continue;
      }
      const cross = a.x * b.y - a.y * b.x;
      const dot = a.x * b.x + a.y * b.y;
      vel[name] = (Math.atan2(cross, dot) * (180 / Math.PI)) / gap;
      any = true;
    }

    current.vel = any ? vel : null;
  }

  return sequence;
}

// Mirror a sample's features (swap left/right bones, negate x, flip angular
// velocity sign) - equivalent to having mirrored the input image. Works on
// anything carrying {bones, vel}, including prepared reference frames -
// exported (as mirrorPoseFeatures) so the debug overlay can compare in the
// same chirality the matcher picked.
export function mirrorPoseFeatures(sample) {
  return mirrorSample(sample);
}

function mirrorSample(sample) {
  const bones = {};
  for (const name of BONE_NAMES) {
    const source = sample.bones[MIRROR_BONE[name]];
    bones[name] = source ? { x: -source.x, y: source.y, w: source.w } : null;
  }

  let vel = null;
  if (sample.vel) {
    vel = {};
    for (const name of BONE_NAMES) {
      const value = sample.vel[MIRROR_BONE[name]];
      vel[name] = value === null || value === undefined ? null : -value;
    }
  }

  return { ...sample, bones, vel };
}

// ---------------------------------------------------------------------------
// Frame comparison
// ---------------------------------------------------------------------------

function compareFrame(live, reference, config) {
  const sigma = config.staticSigmaDeg;
  const velSigma = config.velocitySigmaDegPerSec;

  let staticSum = 0;
  let staticWeight = 0;
  let dynSum = 0;
  let dynWeight = 0;
  let count = 0;
  let liveMotion = 0;
  let refMotion = 0;
  let motionWeight = 0;

  for (const bone of BONES) {
    const a = live.bones[bone.name];
    const b = reference.bones[bone.name];
    if (!a || !b) continue;

    count += 1;
    const dot = clamp(a.x * b.x + a.y * b.y, -1, 1);
    const theta = Math.acos(dot) * (180 / Math.PI);
    const weight = bone.weight * Math.min(a.w, b.w) * (b.info ?? 1);
    staticSum += weight * Math.exp(-(theta * theta) / (2 * sigma * sigma));
    staticWeight += weight;

    const liveVel = live.vel?.[bone.name];
    const refVel = reference.vel?.[bone.name];
    if (liveVel !== null && liveVel !== undefined && refVel !== null && refVel !== undefined) {
      // A bone nobody is moving carries no rhythm information; scale the
      // motion term by how active the bone is so held poses aren't judged
      // on noise. A reference bone that IS moving while the player's isn't
      // gets full activity weight - and a large velocity difference.
      const activity = Math.min(
        1,
        Math.max(Math.abs(liveVel), Math.abs(refVel)) / config.activityFloorDegPerSec,
      );
      const diff = liveVel - refVel;
      const velWeight = weight * activity;
      dynSum += velWeight * Math.exp(-(diff * diff) / (2 * velSigma * velSigma));
      dynWeight += velWeight;

      liveMotion += bone.weight * Math.abs(liveVel);
      refMotion += bone.weight * Math.abs(refVel);
      motionWeight += bone.weight;
    }
  }

  if (count === 0 || staticWeight === 0) {
    return {
      score: 0,
      staticScore: 0,
      dynamicScore: null,
      comparableBones: 0,
      informative: false,
      liveMotion: null,
      refMotion: null,
    };
  }

  const staticScore = (100 * staticSum) / staticWeight;
  const dynamicScore = dynWeight > 0 ? (100 * dynSum) / dynWeight : null;
  const informative = count >= config.minComparableBones;

  const blended = dynamicScore === null
    ? staticScore
    : staticScore * (1 - config.dynamicsWeight) + dynamicScore * config.dynamicsWeight;

  return {
    score: informative ? blended : 0,
    staticScore: round(staticScore, 1),
    dynamicScore: dynamicScore === null ? null : round(dynamicScore, 1),
    comparableBones: count,
    informative,
    liveMotion: motionWeight > 0 ? liveMotion / motionWeight : null,
    refMotion: motionWeight > 0 ? refMotion / motionWeight : null,
  };
}

// Per-bone breakdown of one live-vs-reference comparison for the debug
// panel. Runs once per rendered frame, not in the DTW hot path.
export function compareFrameDetailed(liveSample, referenceFrame, options = {}) {
  const config = mergeConfig(options);
  const sigma = config.staticSigmaDeg;

  const bones = BONES.map((bone) => {
    const a = liveSample?.bones?.[bone.name];
    const b = referenceFrame?.bones?.[bone.name];
    if (!a || !b) {
      return { name: bone.name, present: false, thetaDeg: null, score: null, liveVel: null, refVel: null };
    }

    const dot = clamp(a.x * b.x + a.y * b.y, -1, 1);
    const theta = Math.acos(dot) * (180 / Math.PI);
    const liveVel = liveSample.vel?.[bone.name] ?? null;
    const refVel = referenceFrame.vel?.[bone.name] ?? null;
    return {
      name: bone.name,
      present: true,
      thetaDeg: round(theta, 1),
      score: round(100 * Math.exp(-(theta * theta) / (2 * sigma * sigma)), 1),
      liveVel: liveVel === null ? null : round(liveVel, 0),
      refVel: refVel === null ? null : round(refVel, 0),
    };
  });

  return {
    bones,
    summary: liveSample && referenceFrame ? compareFrame(liveSample, referenceFrame, config) : null,
  };
}

// ---------------------------------------------------------------------------
// Sequence alignment
// ---------------------------------------------------------------------------

function emptySequenceResult(dance) {
  return {
    id: dance.id,
    title: dance.title,
    score: 0,
    rawScore: 0,
    motionRatio: null,
    staticScore: 0,
    dynamicScore: null,
    mirrored: false,
    matchedSamples: 0,
    coverageSeconds: 0,
    informativeCoverageFraction: 0,
    alignedFrameIndex: null,
    alignedTimestamp: null,
  };
}

function findBestSequence(samples, mirroredSamples, dance, config) {
  if (!dance.frames.length || samples.length === 0) {
    return emptySequenceResult(dance);
  }

  const align = config.syncToTimeline ? alignSyncedSequence : alignFallbackSequence;
  const direct = align(samples, dance, config);
  const mirrored = align(mirroredSamples, dance, config);

  if (mirrored.rawScore > direct.rawScore) {
    return { ...mirrored, mirrored: true };
  }
  return direct;
}

// Primary path: a single dance is selected and the audio clock anchors every
// live sample to a reference region; a banded DTW inside that region absorbs
// reaction lag and local tempo drift.
function alignSyncedSequence(samples, dance, config) {
  const sampledFps = dance.sampledFps || 30;
  const bandFrames = Math.max(1, Math.round(config.syncBandSeconds * sampledFps));

  return runAlignment(samples, dance, config, (i) => samples[i].timestamp, bandFrames);
}

// Fallback path: no dance/audio clock, so coarsely scan candidate start
// offsets and run a banded DTW around each.
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
    const anchorTimeForRow = (i) => start + (samples[i].timestamp - firstLiveTimestamp);
    const result = runAlignment(samples, dance, config, anchorTimeForRow, bandFrames);
    if (!best || result.rawScore > best.rawScore) best = result;
  }

  return best ?? emptySequenceResult(dance);
}

function runAlignment(samples, dance, config, anchorTimeForRow, bandFrames) {
  const frames = dance.frames;
  const cellCache = new Map();

  // Resolve each live sample's expected reference position, and mark rows
  // whose expected time falls into a hole in the reference timeline (no
  // frame within referenceGapToleranceSeconds) as unusable: they still pass
  // through the DTW (band continuity) but are excluded from scoring.
  const anchorIndex = new Int32Array(samples.length);
  const rowUsable = new Array(samples.length);
  let usableRows = 0;
  for (let i = 0; i < samples.length; i++) {
    const anchorTime = anchorTimeForRow(i);
    const j = nearestFrameIndex(frames, anchorTime);
    anchorIndex[i] = j;
    rowUsable[i] = Math.abs(frames[j].timestamp - anchorTime) <= config.referenceGapToleranceSeconds;
    if (rowUsable[i]) usableRows += 1;
  }

  if (usableRows === 0) return emptySequenceResult(dance);

  const cost = (i, j) => {
    const comparison = compareFrame(samples[i], frames[j], config);
    cellCache.set(i * frames.length + j, comparison);
    return 1 - comparison.score / 100;
  };

  const alignment = bandedDtw({
    n: samples.length,
    m: frames.length,
    cost,
    anchorForRow: (i) => anchorIndex[i],
    bandRadius: bandFrames,
    stepPenalty: config.stepPenalty,
  });

  if (!alignment.path.length) return emptySequenceResult(dance);

  // Collapse the path to one comparison per live sample, max-pooling over a
  // small temporal neighborhood of the aligned frame (all those cells were
  // already computed by the banded DTW): a pose hit slightly early or late
  // gets full credit. Human timing slop, not robot frame-matching.
  const sampledFps = dance.sampledFps || 30;
  const poolRadius = Math.max(0, Math.round(config.scorePoolSeconds * sampledFps));
  const rowResult = new Map();
  for (const [i, j] of alignment.path) {
    let best = rowResult.get(i) ?? null;
    for (let dj = -poolRadius; dj <= poolRadius; dj++) {
      const cell = cellCache.get(i * frames.length + (j + dj));
      if (!cell) continue;
      if (
        !best ||
        (cell.informative && !best.informative) ||
        (cell.informative === best.informative && cell.score > best.score)
      ) {
        best = cell;
      }
    }
    rowResult.set(i, best);
  }

  let scoreSum = 0;
  let staticSum = 0;
  let dynSum = 0;
  let dynCount = 0;
  let informativeRows = 0;
  const liveMotions = [];
  const refMotions = [];

  for (const [i, comparison] of rowResult.entries()) {
    if (!rowUsable[i] || !comparison?.informative) continue;
    informativeRows += 1;
    scoreSum += comparison.score;
    staticSum += comparison.staticScore;
    if (comparison.dynamicScore !== null) {
      dynSum += comparison.dynamicScore;
      dynCount += 1;
    }
    if (comparison.liveMotion !== null) {
      liveMotions.push(comparison.liveMotion);
      refMotions.push(comparison.refMotion);
    }
  }

  if (informativeRows === 0) return emptySequenceResult(dance);

  const totalDuration = samples.at(-1).timestamp - samples[0].timestamp;
  const informativeCoverageFraction = informativeRows / usableRows;
  const coverageSeconds = totalDuration > 0
    ? informativeCoverageFraction * totalDuration
    : 0;

  const [, lastJ] = alignment.path.at(-1);
  const alignedFrame = frames[lastJ] ?? null;
  const rawScore = scoreSum / informativeRows;

  // Anti-idle: how much did the player actually move, relative to what the
  // choreography demanded over this window? Medians, not means - a single
  // glitchy frame (bad extraction, keypoint teleport) produces a velocity
  // spike that would otherwise dominate the ratio.
  let motionRatio = null;
  let motionPenalty = 1;
  if (refMotions.length > 0) {
    const refMotionMedian = median(refMotions);
    if (refMotionMedian >= config.motionPenaltyMinRefDegPerSec) {
      const liveMotionMedian = Math.max(
        0,
        median(liveMotions) - config.liveMotionNoiseFloorDegPerSec,
      );
      motionRatio = liveMotionMedian / refMotionMedian;
      motionPenalty = clamp(motionRatio / config.motionRatioFloor, 0, 1);
    }
  }

  return {
    id: dance.id,
    title: dance.title,
    score: round(toDisplayScore(rawScore, config) * motionPenalty, 1),
    rawScore: round(rawScore, 1),
    motionRatio: motionRatio === null ? null : round(motionRatio, 2),
    staticScore: round(staticSum / informativeRows, 1),
    dynamicScore: dynCount > 0 ? round(dynSum / dynCount, 1) : null,
    mirrored: false,
    matchedSamples: informativeRows,
    coverageSeconds: round(coverageSeconds, 2),
    informativeCoverageFraction: round(informativeCoverageFraction, 3),
    alignedFrameIndex: alignedFrame?.frameIndex ?? null,
    alignedTimestamp: alignedFrame?.timestamp ?? null,
  };
}

// Raw bone similarity has a high floor (any two humans standing under
// gravity share most of a pose), so stretch [scoreFloor, scoreCeil] onto
// the 0-100% the player sees. The concave exponent makes mid-range
// performances read rewarding (t=0.5 -> ~62%) instead of linearly stingy -
// this is a game, not a metrology lab.
function toDisplayScore(raw, config) {
  const t = clamp((raw - config.scoreFloor) / (config.scoreCeil - config.scoreFloor), 0, 1);
  return round(100 * Math.pow(t, config.displayGamma), 1);
}

// Binary search for the reference frame whose timestamp is closest.
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

// ---------------------------------------------------------------------------
// Display / synthesis helpers
// ---------------------------------------------------------------------------

// Hip-centered, shoulder-width-scaled keypoints. Not used for matching
// (bone directions are) - kept for drawing reference stick figures and for
// the evaluation script's synthetic-body generation.
export function normalizePoseKeypoints(keypoints, threshold) {
  const leftHip = keypoints[KP.leftHip];
  const rightHip = keypoints[KP.rightHip];
  const leftShoulder = keypoints[KP.leftShoulder];
  const rightShoulder = keypoints[KP.rightShoulder];

  if (!visible(leftHip, threshold) || !visible(rightHip, threshold)) return null;
  const origin = midpoint(leftHip, rightHip);

  // Scale: shoulder width alone collapses whenever the dancer turns
  // sideways (the projected width goes toward zero, so dividing by it
  // explodes every coordinate - the "teleporting joints / giant bbox"
  // artifact in overlays). Torso length barely changes with body rotation,
  // so take the max of the two, converting torso to shoulder-width units
  // (torso =~ 1.11 shoulder widths on a standard body).
  let scale = 0;
  if (visible(leftShoulder, threshold) && visible(rightShoulder, threshold)) {
    scale = Math.hypot(leftShoulder.x - rightShoulder.x, leftShoulder.y - rightShoulder.y);
  }
  const shoulders = [leftShoulder, rightShoulder].filter((point) => visible(point, threshold));
  if (shoulders.length > 0) {
    const shoulderMid = shoulders.length === 2 ? midpoint(leftShoulder, rightShoulder) : shoulders[0];
    const torso = Math.hypot(origin.x - shoulderMid.x, origin.y - shoulderMid.y);
    scale = Math.max(scale, torso / BONE_LENGTH.torso);
  }
  if (scale < 1e-6) return null;

  return keypoints.map((point, index) => ({
    name: point.name ?? KEYPOINT_NAMES[index],
    x: (point.x - origin.x) / scale,
    y: (point.y - origin.y) / scale,
    score: point.score,
  }));
}

// Kinematic tree + canonical bone lengths, kept for the evaluation script:
// retargeting a pose onto randomized lengths synthesizes "a differently
// proportioned dancer performing the same choreography".
const KINEMATIC_CHAIN = [
  ["hipMid", "shoulderMid", "torso"],
  ["shoulderMid", "leftShoulder", "shoulderHalf"],
  ["shoulderMid", "rightShoulder", "shoulderHalf"],
  ["hipMid", "leftHip", "hipHalf"],
  ["hipMid", "rightHip", "hipHalf"],
  ["leftShoulder", "leftElbow", "upperArm"],
  ["rightShoulder", "rightElbow", "upperArm"],
  ["leftElbow", "leftWrist", "forearm"],
  ["rightElbow", "rightWrist", "forearm"],
  ["leftHip", "leftKnee", "thigh"],
  ["rightHip", "rightKnee", "thigh"],
  ["leftKnee", "leftAnkle", "shin"],
  ["rightKnee", "rightAnkle", "shin"],
];

export const BONE_LENGTH = {
  torso: 1.11,
  shoulderHalf: 0.5,
  hipHalf: 0.37,
  upperArm: 0.72,
  forearm: 0.56,
  thigh: 0.95,
  shin: 0.95,
};

function unitVector(dx, dy) {
  const length = Math.hypot(dx, dy);
  if (length < 1e-6) return null;
  return { x: dx / length, y: dy / length };
}

function actualMidpoint(a, b, threshold) {
  const aVisible = visible(a, threshold);
  const bVisible = visible(b, threshold);
  if (aVisible && bVisible) return midpoint(a, b);
  if (aVisible) return { x: a.x, y: a.y };
  if (bVisible) return { x: b.x, y: b.y };
  return null;
}

// Rebuilds a hip-centered pose so every bone has the given fixed length
// while keeping each bone's measured direction. Expects named, hip-centered
// keypoints (see normalizePoseKeypoints).
export function retargetSkeleton(keypoints, boneLengths, threshold) {
  const actual = new Map(keypoints.map((point) => [point.name, point]));
  actual.set("hipMid", { x: 0, y: 0, score: 1 });
  const shoulderMid = actualMidpoint(actual.get("leftShoulder"), actual.get("rightShoulder"), threshold);
  if (shoulderMid) actual.set("shoulderMid", shoulderMid);

  const canonical = new Map();
  canonical.set("hipMid", { x: 0, y: 0 });

  for (const [parentName, childName, boneKey] of KINEMATIC_CHAIN) {
    const parentActual = actual.get(parentName);
    const childActual = actual.get(childName);
    const parentCanonical = canonical.get(parentName);
    if (!parentActual || !childActual || !parentCanonical) continue;

    const direction = unitVector(childActual.x - parentActual.x, childActual.y - parentActual.y);
    canonical.set(
      childName,
      direction
        ? {
            x: parentCanonical.x + direction.x * boneLengths[boneKey],
            y: parentCanonical.y + direction.y * boneLengths[boneKey],
          }
        : { x: childActual.x, y: childActual.y },
    );
  }

  return keypoints.map((point) => {
    const target = canonical.get(point.name);
    return target ? { name: point.name, x: target.x, y: target.y, score: point.score } : point;
  });
}

// ---------------------------------------------------------------------------
// Misc
// ---------------------------------------------------------------------------

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function median(values) {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = sorted.length >> 1;
  return sorted.length % 2 === 1 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

function midpoint(a, b) {
  return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
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
