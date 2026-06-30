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

const DEFAULT_OPTIONS = {
  keypointThreshold: 0.25,
  maxPointDistance: 1.15,
  minComparableKeypoints: 8,
  minConfidence: 52,
};

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
  const config = { ...DEFAULT_OPTIONS, ...options };
  if (!pose?.keypoints || !catalogue?.dances?.length) {
    return emptyMatch();
  }

  const live = {
    keypoints: normalizeKeypoints(pose.keypoints, pose.bbox, config.keypointThreshold),
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

export function stabilizeMatches(history, currentMatch, now, windowMs = 2200) {
  const nextHistory = history
    .filter((entry) => now - entry.time <= windowMs)
    .concat(currentMatch?.best ? [{ time: now, match: currentMatch.best }] : []);

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

function prepareDance(entry) {
  const frames = entry.data.frames
    .filter((frame) => frame.person)
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

function compareFrame(live, frame, config) {
  const pointComparison = compareKeypoints(live.keypoints, frame.keypoints, config);
  const angleScore = compareAngles(live.angles, frame.angles);

  if (pointComparison.count < config.minComparableKeypoints) {
    return {
      score: 0,
      keypointScore: pointComparison.score,
      angleScore,
      comparableKeypoints: pointComparison.count,
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
  };
}

function compareKeypoints(live, reference, config) {
  let weightedDistance = 0;
  let totalWeight = 0;
  let count = 0;

  for (let i = 0; i < Math.min(live.length, reference.length); i++) {
    const a = live[i];
    const b = reference[i];
    if (!visible(a, config.keypointThreshold) || !visible(b, config.keypointThreshold)) {
      continue;
    }

    const weight = (a.score + b.score) / 2;
    weightedDistance += Math.hypot(a.x - b.x, a.y - b.y) * weight;
    totalWeight += weight;
    count += 1;
  }

  if (count === 0 || totalWeight === 0) {
    return { score: 0, count: 0 };
  }

  const averageDistance = weightedDistance / totalWeight;
  const score = Math.max(0, 100 * (1 - averageDistance / config.maxPointDistance));
  return { score, count };
}

function compareAngles(liveAngles, referenceAngles) {
  let diffSum = 0;
  let count = 0;

  for (const name of ANGLE_NAMES) {
    const live = liveAngles[name];
    const reference = referenceAngles[name];
    if (live === null || live === undefined || reference === null || reference === undefined) {
      continue;
    }
    diffSum += Math.abs(live - reference);
    count += 1;
  }

  if (count === 0) {
    return null;
  }

  const averageDiff = diffSum / count;
  return Math.max(0, 100 * (1 - averageDiff / 90));
}

function normalizeKeypoints(keypoints, bbox, threshold) {
  const leftHip = keypoints[KP.leftHip];
  const rightHip = keypoints[KP.rightHip];
  const leftShoulder = keypoints[KP.leftShoulder];
  const rightShoulder = keypoints[KP.rightShoulder];
  const [boxX = 0, boxY = 0, boxW = 1, boxH = 1] = bbox ?? [];

  const origin = visible(leftHip, threshold) && visible(rightHip, threshold)
    ? midpoint(leftHip, rightHip)
    : { x: boxX + boxW / 2, y: boxY + boxH / 2 };

  let scale = null;
  if (visible(leftShoulder, threshold) && visible(rightShoulder, threshold)) {
    scale = distance(leftShoulder, rightShoulder);
  }
  if (!scale || scale < 1) {
    scale = Math.max(boxW, boxH, 1);
  }

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
