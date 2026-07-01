import { spawn } from "node:child_process";
import { Buffer } from "node:buffer";
import { existsSync } from "node:fs";
import { mkdir, readdir, stat, writeFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";
import ffmpegPath from "ffmpeg-static";
import ffprobeStatic from "ffprobe-static";
import ort from "onnxruntime-node";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PROJECT_ROOT = path.resolve(__dirname, "..");
const VIDEO_EXTENSIONS = new Set([".mp4", ".mov", ".m4v", ".webm", ".avi"]);
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

const DEFAULTS = {
  input: "videos",
  output: "public/catalogue/poses",
  model: "public/models/yolo26n-pose.onnx",
  fps: 10,
  imageSize: 640,
  scoreThreshold: 0.35,
  iouThreshold: 0.45,
  keypointThreshold: 0.2,
  maxFrames: Number.POSITIVE_INFINITY,
  keepEmpty: true,
};

function printHelp() {
  console.log(`
Extract YOLO pose tracks from dance videos.

Usage:
  bun run extract:poses
  bun scripts/extract-poses.js --input videos --fps 12

Options:
  --input <path>       Video file or directory. Default: ${DEFAULTS.input}
  --output <dir>       Output catalogue directory. Default: ${DEFAULTS.output}
  --model              Removed. Pose extraction always uses YOLO26-n.
  --fps <number>       Sampling FPS. Default: ${DEFAULTS.fps}
  --size <number>      Model input square size. Default: ${DEFAULTS.imageSize}
  --score <number>     Person confidence threshold. Default: ${DEFAULTS.scoreThreshold}
  --iou <number>       NMS IoU threshold. Default: ${DEFAULTS.iouThreshold}
  --kp-score <number>  Keypoint confidence used by angles/normalization. Default: ${DEFAULTS.keypointThreshold}
  --max-frames <n>     Stop after n sampled frames per video.
  --drop-empty         Omit frames where no person was detected.
  --help               Show this help.
`);
}

function parseArgs(argv) {
  const options = { ...DEFAULTS };

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    const readValue = () => {
      const value = argv[++i];
      if (!value) throw new Error(`Missing value after ${arg}`);
      return value;
    };

    switch (arg) {
      case "--input":
      case "-i":
        options.input = readValue();
        break;
      case "--output":
      case "-o":
        options.output = readValue();
        break;
      case "--model":
      case "-m":
        throw new Error("Model selection was removed; pose extraction always uses YOLO26-n.");
      case "--fps":
        options.fps = Number(readValue());
        break;
      case "--size":
        options.imageSize = Number(readValue());
        break;
      case "--score":
        options.scoreThreshold = Number(readValue());
        break;
      case "--iou":
        options.iouThreshold = Number(readValue());
        break;
      case "--kp-score":
        options.keypointThreshold = Number(readValue());
        break;
      case "--max-frames":
        options.maxFrames = Number(readValue());
        break;
      case "--drop-empty":
        options.keepEmpty = false;
        break;
      case "--help":
      case "-h":
        options.help = true;
        break;
      default:
        throw new Error(`Unknown option: ${arg}`);
    }
  }

  if (!Number.isFinite(options.fps) || options.fps <= 0) {
    throw new Error("--fps must be a positive number");
  }
  if (!Number.isInteger(options.imageSize) || options.imageSize <= 0) {
    throw new Error("--size must be a positive integer");
  }
  if (!Number.isFinite(options.maxFrames) || options.maxFrames <= 0) {
    options.maxFrames = Number.POSITIVE_INFINITY;
  }

  options.input = path.resolve(PROJECT_ROOT, options.input);
  options.output = path.resolve(PROJECT_ROOT, options.output);
  options.model = path.resolve(PROJECT_ROOT, options.model);

  return options;
}

async function findVideos(inputPath) {
  if (!existsSync(inputPath)) {
    throw new Error(`Input path does not exist: ${inputPath}`);
  }

  const inputStat = await stat(inputPath);
  if (inputStat.isFile()) {
    if (!VIDEO_EXTENSIONS.has(path.extname(inputPath).toLowerCase())) {
      throw new Error(`Input file is not a supported video: ${inputPath}`);
    }
    return [inputPath];
  }

  const entries = await readdir(inputPath, { withFileTypes: true });
  return entries
    .filter((entry) => entry.isFile())
    .map((entry) => path.join(inputPath, entry.name))
    .filter((file) => VIDEO_EXTENSIONS.has(path.extname(file).toLowerCase()))
    .sort((a, b) => a.localeCompare(b));
}

function runJsonCommand(command, args) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, { stdio: ["ignore", "pipe", "pipe"] });
    let stdout = "";
    let stderr = "";

    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");
    child.stdout.on("data", (chunk) => {
      stdout += chunk;
    });
    child.stderr.on("data", (chunk) => {
      stderr += chunk;
    });
    child.on("error", reject);
    child.on("close", (code) => {
      if (code !== 0) {
        reject(new Error(`${path.basename(command)} exited with ${code}: ${stderr}`));
        return;
      }
      try {
        resolve(JSON.parse(stdout));
      } catch (error) {
        reject(new Error(`Could not parse JSON from ${path.basename(command)}: ${error.message}`));
      }
    });
  });
}

async function probeVideo(videoPath) {
  const data = await runJsonCommand(ffprobeStatic.path, [
    "-v",
    "error",
    "-select_streams",
    "v:0",
    "-show_entries",
    "stream=width,height,avg_frame_rate,r_frame_rate,duration,nb_frames",
    "-show_entries",
    "format=duration",
    "-of",
    "json",
    videoPath,
  ]);

  const stream = data.streams?.[0];
  if (!stream?.width || !stream?.height) {
    throw new Error(`Could not read video dimensions for ${videoPath}`);
  }

  return {
    width: Number(stream.width),
    height: Number(stream.height),
    duration: Number(stream.duration || data.format?.duration || 0),
    frameRate: parseRate(stream.avg_frame_rate || stream.r_frame_rate),
    frameCount: stream.nb_frames ? Number(stream.nb_frames) : null,
  };
}

function parseRate(rate) {
  if (!rate || rate === "0/0") return null;
  const [num, den] = rate.split("/").map(Number);
  if (!den) return num || null;
  return num / den;
}

function slugify(name) {
  return name
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLowerCase()
    .replace(/['"]/g, "")
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

function getLetterbox(width, height, imageSize) {
  const ratio = Math.min(imageSize / width, imageSize / height);
  const resizedWidth = Math.round(width * ratio);
  const resizedHeight = Math.round(height * ratio);
  return {
    ratio,
    resizedWidth,
    resizedHeight,
    padX: Math.floor((imageSize - resizedWidth) / 2),
    padY: Math.floor((imageSize - resizedHeight) / 2),
  };
}

function frameToTensor(frame, width, height, letterbox, imageSize) {
  const channelSize = imageSize * imageSize;
  const input = new Float32Array(channelSize * 3);
  input.fill(114 / 255);

  const { ratio, resizedWidth, resizedHeight, padX, padY } = letterbox;

  for (let y = 0; y < resizedHeight; y++) {
    const srcY = Math.min(height - 1, Math.max(0, (y + 0.5) / ratio - 0.5));
    const y0 = Math.floor(srcY);
    const y1 = Math.min(height - 1, y0 + 1);
    const wy = srcY - y0;

    for (let x = 0; x < resizedWidth; x++) {
      const srcX = Math.min(width - 1, Math.max(0, (x + 0.5) / ratio - 0.5));
      const x0 = Math.floor(srcX);
      const x1 = Math.min(width - 1, x0 + 1);
      const wx = srcX - x0;

      const topLeft = (y0 * width + x0) * 3;
      const topRight = (y0 * width + x1) * 3;
      const bottomLeft = (y1 * width + x0) * 3;
      const bottomRight = (y1 * width + x1) * 3;
      const outIndex = (padY + y) * imageSize + padX + x;

      for (let channel = 0; channel < 3; channel++) {
        const top = frame[topLeft + channel] * (1 - wx) + frame[topRight + channel] * wx;
        const bottom = frame[bottomLeft + channel] * (1 - wx) + frame[bottomRight + channel] * wx;
        input[channel * channelSize + outIndex] = (top * (1 - wy) + bottom * wy) / 255;
      }
    }
  }

  return new ort.Tensor("float32", input, [1, 3, imageSize, imageSize]);
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function unletterboxPoint(x, y, letterbox, width, height) {
  return {
    x: clamp((x - letterbox.padX) / letterbox.ratio, 0, width),
    y: clamp((y - letterbox.padY) / letterbox.ratio, 0, height),
  };
}

function unletterboxBox(cx, cy, boxWidth, boxHeight, letterbox, width, height) {
  const topLeft = unletterboxPoint(cx - boxWidth / 2, cy - boxHeight / 2, letterbox, width, height);
  const bottomRight = unletterboxPoint(cx + boxWidth / 2, cy + boxHeight / 2, letterbox, width, height);
  return [
    topLeft.x,
    topLeft.y,
    Math.max(0, bottomRight.x - topLeft.x),
    Math.max(0, bottomRight.y - topLeft.y),
  ];
}

function unletterboxCorners(x1, y1, x2, y2, letterbox, width, height) {
  const topLeft = unletterboxPoint(x1, y1, letterbox, width, height);
  const bottomRight = unletterboxPoint(x2, y2, letterbox, width, height);
  return [
    topLeft.x,
    topLeft.y,
    Math.max(0, bottomRight.x - topLeft.x),
    Math.max(0, bottomRight.y - topLeft.y),
  ];
}

function postProcessPose(rawTensor, options, videoMeta, letterbox) {
  const dims = rawTensor.dims;
  if (dims.length !== 3) {
    throw new Error(`Unsupported YOLO output shape: [${dims.join(", ")}]`);
  }

  const second = dims[1];
  const third = dims[2];
  const attrs = second <= third ? second : third;
  const predictions = second <= third ? third : second;
  const channelsFirst = second <= third;

  if (attrs < 56) {
    throw new Error(`Expected at least 56 pose attributes, got shape [${dims.join(", ")}]`);
  }

  const data = rawTensor.data;
  const get = channelsFirst
    ? (predictionIndex, attrIndex) => data[attrIndex * predictions + predictionIndex]
    : (predictionIndex, attrIndex) => data[predictionIndex * attrs + attrIndex];

  const isEndToEndPose = attrs >= 57;
  const keypointOffset = isEndToEndPose ? 6 : 5;
  const detections = [];
  for (let i = 0; i < predictions; i++) {
    const score = get(i, 4);
    if (score < options.scoreThreshold) continue;

    const bbox = isEndToEndPose
      ? unletterboxCorners(
        get(i, 0),
        get(i, 1),
        get(i, 2),
        get(i, 3),
        letterbox,
        videoMeta.width,
        videoMeta.height,
      )
      : unletterboxBox(
        get(i, 0),
        get(i, 1),
        get(i, 2),
        get(i, 3),
        letterbox,
        videoMeta.width,
        videoMeta.height,
      );

    const keypoints = KEYPOINT_NAMES.map((name, kpIndex) => {
      const attr = keypointOffset + kpIndex * 3;
      const point = unletterboxPoint(
        get(i, attr),
        get(i, attr + 1),
        letterbox,
        videoMeta.width,
        videoMeta.height,
      );
      return {
        name,
        x: round(point.x),
        y: round(point.y),
        score: round(get(i, attr + 2), 5),
      };
    });

    detections.push({
      bbox: bbox.map((value) => round(value)),
      score: round(score, 5),
      keypoints,
    });
  }

  return applyNms(detections, options.iouThreshold);
}

function applyNms(detections, iouThreshold) {
  const sorted = [...detections].sort((a, b) => b.score - a.score);
  const selected = [];

  for (const detection of sorted) {
    if (selected.every((existing) => calculateIou(existing.bbox, detection.bbox) <= iouThreshold)) {
      selected.push(detection);
    }
  }

  return selected;
}

function calculateIou(a, b) {
  const [ax, ay, aw, ah] = a;
  const [bx, by, bw, bh] = b;
  const x1 = Math.max(ax, bx);
  const y1 = Math.max(ay, by);
  const x2 = Math.min(ax + aw, bx + bw);
  const y2 = Math.min(ay + ah, by + bh);
  const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const union = aw * ah + bw * bh - intersection;
  return union > 0 ? intersection / union : 0;
}

function pickBestPerson(detections) {
  if (!detections.length) return null;
  return detections.reduce((best, current) => {
    const bestArea = best.bbox[2] * best.bbox[3];
    const currentArea = current.bbox[2] * current.bbox[3];
    return currentArea > bestArea ? current : best;
  });
}

function distance(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function visible(point, threshold) {
  return point && point.score >= threshold;
}

function midpoint(a, b) {
  return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
}

function normalizeKeypoints(detection, threshold) {
  const kps = detection.keypoints;
  const leftHip = kps[KP.leftHip];
  const rightHip = kps[KP.rightHip];
  const leftShoulder = kps[KP.leftShoulder];
  const rightShoulder = kps[KP.rightShoulder];
  const [boxX, boxY, boxW, boxH] = detection.bbox;

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

  return {
    origin: { x: round(origin.x), y: round(origin.y) },
    scale: round(scale),
    keypoints: kps.map((point) => ({
      name: point.name,
      x: round((point.x - origin.x) / scale, 6),
      y: round((point.y - origin.y) / scale, 6),
      score: point.score,
    })),
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
  return round(diff, 2);
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

function enrichDetection(detection, options) {
  if (!detection) return null;
  return {
    ...detection,
    normalized: normalizeKeypoints(detection, options.keypointThreshold),
    angles: calculateAngles(detection.keypoints, options.keypointThreshold),
  };
}

function round(value, digits = 3) {
  const factor = 10 ** digits;
  return Math.round(value * factor) / factor;
}

async function* fixedChunks(stream, size) {
  let buffer = Buffer.alloc(0);
  for await (const chunk of stream) {
    buffer = buffer.length ? Buffer.concat([buffer, chunk]) : chunk;
    while (buffer.length >= size) {
      yield buffer.subarray(0, size);
      buffer = buffer.subarray(size);
    }
  }

  if (buffer.length > 0) {
    throw new Error(`Decoder ended with a partial frame (${buffer.length}/${size} bytes)`);
  }
}

async function extractVideo(videoPath, session, options) {
  const videoMeta = await probeVideo(videoPath);
  const letterbox = getLetterbox(videoMeta.width, videoMeta.height, options.imageSize);
  const frameSize = videoMeta.width * videoMeta.height * 3;
  const frames = [];
  const startedAt = Date.now();

  const ffmpeg = spawn(ffmpegPath, [
    "-hide_banner",
    "-loglevel",
    "error",
    "-i",
    videoPath,
    "-vf",
    `fps=${options.fps}`,
    "-f",
    "rawvideo",
    "-pix_fmt",
    "rgb24",
    "pipe:1",
  ], {
    stdio: ["ignore", "pipe", "pipe"],
  });

  let stderr = "";
  const exitPromise = new Promise((resolve, reject) => {
    ffmpeg.on("error", reject);
    ffmpeg.on("close", resolve);
  });

  ffmpeg.stderr.setEncoding("utf8");
  ffmpeg.stderr.on("data", (chunk) => {
    stderr += chunk;
  });

  let frameIndex = 0;
  try {
    for await (const frame of fixedChunks(ffmpeg.stdout, frameSize)) {
      if (frameIndex >= options.maxFrames) {
        ffmpeg.kill("SIGTERM");
        break;
      }

      const inputTensor = frameToTensor(frame, videoMeta.width, videoMeta.height, letterbox, options.imageSize);
      const outputs = await session.run({ images: inputTensor });
      const outputTensor = outputs[session.outputNames[0]];
      const detections = postProcessPose(outputTensor, options, videoMeta, letterbox);
      const bestPerson = enrichDetection(pickBestPerson(detections), options);

      inputTensor.dispose?.();
      for (const tensor of Object.values(outputs)) {
        tensor.dispose?.();
      }

      if (bestPerson || options.keepEmpty) {
        frames.push({
          frameIndex,
          timestamp: round(frameIndex / options.fps, 3),
          person: bestPerson,
        });
      }

      frameIndex++;
      if (frameIndex % Math.max(1, options.fps) === 0) {
        process.stdout.write(`\r  processed ${frameIndex} sampled frames`);
      }
    }
  } finally {
    if (!ffmpeg.killed) ffmpeg.kill("SIGTERM");
  }

  const exitCode = await exitPromise;

  if (exitCode !== 0 && frameIndex < options.maxFrames) {
    throw new Error(`ffmpeg failed for ${path.basename(videoPath)}: ${stderr}`);
  }

  process.stdout.write(`\r  processed ${frameIndex} sampled frames\n`);

  const detectedFrames = frames.filter((frame) => frame.person).length;
  return {
    schemaVersion: 1,
    danceId: slugify(path.basename(videoPath, path.extname(videoPath))),
    title: path.basename(videoPath, path.extname(videoPath)),
    createdAt: new Date().toISOString(),
    source: {
      file: path.relative(PROJECT_ROOT, videoPath).replaceAll("\\", "/"),
      width: videoMeta.width,
      height: videoMeta.height,
      duration: videoMeta.duration ? round(videoMeta.duration, 3) : null,
      frameRate: videoMeta.frameRate ? round(videoMeta.frameRate, 3) : null,
      sampledFps: options.fps,
    },
    model: {
      file: path.relative(PROJECT_ROOT, options.model).replaceAll("\\", "/"),
      inputSize: options.imageSize,
      scoreThreshold: options.scoreThreshold,
      iouThreshold: options.iouThreshold,
      keypointThreshold: options.keypointThreshold,
    },
    keypointFormat: "COCO-17",
    keypointNames: KEYPOINT_NAMES,
    stats: {
      sampledFrames: frameIndex,
      writtenFrames: frames.length,
      detectedFrames,
      detectionRate: frameIndex > 0 ? round(detectedFrames / frameIndex, 4) : 0,
      elapsedSeconds: round((Date.now() - startedAt) / 1000, 2),
    },
    frames,
  };
}

async function createSession(modelPath) {
  if (!existsSync(modelPath)) {
    throw new Error(`Model does not exist: ${modelPath}`);
  }

  try {
    return await ort.InferenceSession.create(modelPath, {
      executionProviders: ["dml", "cpu"],
      graphOptimizationLevel: "all",
    });
  } catch (error) {
    console.warn(`DirectML failed, falling back to CPU: ${error.message}`);
    return ort.InferenceSession.create(modelPath, {
      executionProviders: ["cpu"],
      graphOptimizationLevel: "all",
    });
  }
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  if (options.help) {
    printHelp();
    return;
  }

  const videos = await findVideos(options.input);
  if (videos.length === 0) {
    throw new Error(`No supported videos found in ${options.input}`);
  }

  await mkdir(options.output, { recursive: true });
  console.log(`Loading model: ${path.relative(PROJECT_ROOT, options.model)}`);
  const session = await createSession(options.model);

  const index = {
    schemaVersion: 1,
    createdAt: new Date().toISOString(),
    outputDirectory: path.relative(PROJECT_ROOT, options.output).replaceAll("\\", "/"),
    dances: [],
  };

  for (const videoPath of videos) {
    console.log(`\nExtracting ${path.basename(videoPath)} at ${options.fps} FPS`);
    const catalogue = await extractVideo(videoPath, session, options);
    const outputFile = `${catalogue.danceId}.poses.json`;
    const outputPath = path.join(options.output, outputFile);
    await writeFile(outputPath, `${JSON.stringify(catalogue, null, 2)}\n`);

    index.dances.push({
      id: catalogue.danceId,
      title: catalogue.title,
      file: outputFile,
      source: catalogue.source.file,
      sampledFps: catalogue.source.sampledFps,
      detectedFrames: catalogue.stats.detectedFrames,
      sampledFrames: catalogue.stats.sampledFrames,
      detectionRate: catalogue.stats.detectionRate,
    });

    console.log(`  wrote ${path.relative(PROJECT_ROOT, outputPath)}`);
    console.log(`  detection rate ${(catalogue.stats.detectionRate * 100).toFixed(1)}%`);
  }

  const indexPath = path.join(options.output, "index.json");
  await writeFile(indexPath, `${JSON.stringify(index, null, 2)}\n`);
  console.log(`\nWrote ${path.relative(PROJECT_ROOT, indexPath)}`);
}

main().catch((error) => {
  console.error(`\nPose extraction failed: ${error.message}`);
  process.exitCode = 1;
});
