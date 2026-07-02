import "./assets/App.css";
import { useCallback, useEffect, useRef, useState } from "react";
import classes from "./utils/yolo_classes.json";
import { renderOverlay } from "./utils/render-overlay";
import {
  createPoseSample,
  loadPoseCatalogue,
  matchPoseSequenceToCatalogue,
  stabilizeMatches,
} from "./utils/catalogue-matcher";

import { useInferenceWorker } from "./hooks/useInferenceWorker";
import { useWebcam } from "./hooks/useWebcam";

import ImageDisplay from "./components/ImageDisplay";
import ModelStatus from "./components/ModelStatus";
import PoseOverlayTool from "./components/PoseOverlayTool";
import SettingsPanel from "./components/SettingsPanel";

// Cap source frames at this dimension before sending to the inference worker.
// The letterbox targets 640px anyway, so sending full 1080p is pure waste.
// CSS object-fit:contain on the overlay canvas handles the upscale visually.
const MAX_INFER_DIM = 640;

const DEFAULT_MODEL_CONFIG = {
  inputShape: [1, 3, 640, 640],
  overlaySize: [640, 640],
  iouThreshold: 0.45,
  scoreThreshold: 0.35,
  backend: "webgpu",
  numThreads: 1,
  enableNMS: false,
  model: "yolo26n",
  modelPath: "",
  task: "pose",
  imgszType: "letterbox",
  classes,
};

const COUNTDOWN_SECONDS = 5;
const SEQUENCE_WINDOW_SECONDS = 4;
const AUDIO_SYNC_OFFSET_SECONDS = 0;
const CATALOGUE_SYNC_FPS = 30;

const formatTime = (seconds) => {
  if (seconds === null || seconds === undefined || isNaN(seconds)) return "--:--";
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
};

const DEBUG_TARGET_COLORS = {
  bboxColor: "rgba(34, 211, 238, 0.95)",
  labelBackground: "rgba(8, 145, 178, 0.85)",
  skeletonColor: "rgba(34, 211, 238, 0.95)",
  keypointColor: "rgba(244, 114, 182, 0.95)",
  skeletonLineWidth: 3,
  label: "target",
};
const DEBUG_TARGET_SCALE = 0.5;
const DEBUG_TARGET_X_OFFSET = -100;
const POSE_KEYPOINTS = {
  leftShoulder: 5,
  rightShoulder: 6,
  leftHip: 11,
  rightHip: 12,
};

function getHighestScorePose(results) {
  if (!Array.isArray(results) || results.length === 0) {
    return null;
  }

  return results.reduce((best, current) => {
    const bestScore = best?.score ?? best?.confidence ?? 0;
    const currentScore = current?.score ?? current?.confidence ?? 0;
    return currentScore > bestScore ? current : best;
  }, results[0]);
}

function findNearestCatalogueFrame(frames, timestamp, frameIndex = null) {
  if (!frames?.length) return null;

  if (frameIndex !== null) {
    let nearest = frames[0];
    let nearestDistance = Math.abs((nearest.frameIndex ?? 0) - frameIndex);

    for (const frame of frames) {
      const distance = Math.abs((frame.frameIndex ?? 0) - frameIndex);
      if (distance < nearestDistance) {
        nearest = frame;
        nearestDistance = distance;
      }
    }

    return nearest;
  }

  let low = 0;
  let high = frames.length - 1;

  while (low <= high) {
    const mid = Math.floor((low + high) / 2);
    if (frames[mid].timestamp < timestamp) low = mid + 1;
    else high = mid - 1;
  }

  const previous = frames[high];
  const next = frames[low];
  if (!previous) return next ?? null;
  if (!next) return previous;
  return Math.abs(previous.timestamp - timestamp) <= Math.abs(next.timestamp - timestamp)
    ? previous
    : next;
}

function isVisibleKeypoint(point, threshold = 0.2) {
  return point && Number.isFinite(point.x) && Number.isFinite(point.y) && (point.score ?? 0) >= threshold;
}

function midpoint(a, b) {
  return {
    x: (a.x + b.x) / 2,
    y: (a.y + b.y) / 2,
  };
}

function getPoseProjectionAnchor(pose) {
  const keypoints = pose?.keypoints;
  if (!keypoints?.length) return null;

  const leftHip = keypoints[POSE_KEYPOINTS.leftHip];
  const rightHip = keypoints[POSE_KEYPOINTS.rightHip];
  const leftShoulder = keypoints[POSE_KEYPOINTS.leftShoulder];
  const rightShoulder = keypoints[POSE_KEYPOINTS.rightShoulder];

  if (!isVisibleKeypoint(leftHip) || !isVisibleKeypoint(rightHip)) return null;

  const origin = midpoint(leftHip, rightHip);
  let scale = null;

  if (isVisibleKeypoint(leftShoulder) && isVisibleKeypoint(rightShoulder)) {
    const shoulderWidth = Math.hypot(leftShoulder.x - rightShoulder.x, leftShoulder.y - rightShoulder.y);
    if (shoulderWidth >= 15) scale = shoulderWidth;
  }

  if (!scale) {
    const shoulders = [leftShoulder, rightShoulder].filter((point) => isVisibleKeypoint(point));
    if (shoulders.length > 0) {
      const shoulderMid = shoulders.length === 2 ? midpoint(leftShoulder, rightShoulder) : shoulders[0];
      const torsoHeight = Math.hypot(origin.x - shoulderMid.x, origin.y - shoulderMid.y);
      if (torsoHeight >= 15) scale = torsoHeight;
    }
  }

  return scale ? { origin, scale } : null;
}

function drawPoseSkeletonOnCanvas(ctx, keypoints, scale, centerX, centerY, color, lineWidth) {
  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";

  const getCanvasCoords = (kp) => {
    if (!kp || typeof kp.x !== "number" || typeof kp.y !== "number") return null;
    return {
      x: centerX + kp.x * scale,
      y: centerY + kp.y * scale
    };
  };

  const SKELETON = [
    [15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
    [5, 11], [6, 12], [5, 6], [5, 7], [6, 8],
    [7, 9], [8, 10]
  ];

  for (const [fromIdx, toIdx] of SKELETON) {
    const fromKp = keypoints[fromIdx];
    const toKp = keypoints[toIdx];
    
    if (fromKp && toKp && (fromKp.score === undefined || fromKp.score > 0.15) && (toKp.score === undefined || toKp.score > 0.15)) {
      const fromProj = getCanvasCoords(fromKp);
      const toProj = getCanvasCoords(toKp);
      if (fromProj && toProj) {
        ctx.beginPath();
        ctx.moveTo(fromProj.x, fromProj.y);
        ctx.lineTo(toProj.x, toProj.y);
        ctx.stroke();
      }
    }
  }

  ctx.fillStyle = color === "rgba(6, 182, 212, 0.9)" ? "rgba(244, 114, 182, 0.95)" : color;
  keypoints.forEach((kp, idx) => {
    if (idx < 5) return; // Skip head/face keypoints (nose, eyes, ears)
    if (kp && (kp.score === undefined || kp.score > 0.15)) {
      const proj = getCanvasCoords(kp);
      if (proj) {
        ctx.beginPath();
        ctx.arc(proj.x, proj.y, lineWidth * 1.1, 0, 2 * Math.PI);
        ctx.fill();
      }
    }
  });
}

const PoseReplayCanvas = ({ frames }) => {
  const canvasRef = useRef(null);
  const animFrameRef = useRef(null);
  const frameIdxRef = useRef(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !frames || frames.length === 0) return;

    const ctx = canvas.getContext("2d");
    const w = canvas.width;
    const h = canvas.height;

    // Compute a bounding box that covers ALL frames to keep a stable viewport
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (const f of frames) {
      for (const kp of [...(f.userPose || []), ...(f.targetPose || [])]) {
        if (kp && typeof kp.x === "number" && (kp.score === undefined || kp.score > 0.1)) {
          if (kp.x < minX) minX = kp.x;
          if (kp.x > maxX) maxX = kp.x;
          if (kp.y < minY) minY = kp.y;
          if (kp.y > maxY) maxY = kp.y;
        }
      }
    }

    // Fallback range if no valid keypoints found
    if (!isFinite(minX)) { minX = -1; maxX = 1; minY = -1; maxY = 1; }

    const pad = 0.25;
    const rangeX = (maxX - minX) + pad * 2;
    const rangeY = (maxY - minY) + pad * 2;
    const scaleX = w / rangeX;
    const scaleY = h / rangeY;
    const scale = Math.min(scaleX, scaleY) * 0.85;
    const centerX = w / 2 - ((minX + maxX) / 2) * scale;
    const centerY = h / 2 - ((minY + maxY) / 2) * scale;

    const drawFrame = () => {
      const frameData = frames[frameIdxRef.current];
      ctx.fillStyle = "#FFFFFF";
      ctx.fillRect(0, 0, w, h);

      if (frameData) {
        // Draw target pose (CYAN) first so user pose renders on top
        if (frameData.targetPose) {
          drawPoseSkeletonOnCanvas(ctx, frameData.targetPose, scale, centerX, centerY, "rgba(6, 182, 212, 0.85)", 5);
        }
        // Draw user pose (FUCHSIA)
        if (frameData.userPose) {
          drawPoseSkeletonOnCanvas(ctx, frameData.userPose, scale, centerX, centerY, "rgba(217, 70, 239, 0.9)", 5);
        }

        // Frame score badge (top-right corner)
        if (typeof frameData.score === "number") {
          const scoreColor = frameData.score >= 60 ? "#10b981" : frameData.score >= 40 ? "#f59e0b" : "#ef4444";
          ctx.fillStyle = scoreColor;
          ctx.font = "bold 13px Inter, sans-serif";
          ctx.textAlign = "right";
          ctx.fillText(`${frameData.score}%`, w - 8, 20);
        }

        // Progress bar at bottom
        const progress = (frameIdxRef.current + 1) / frames.length;
        ctx.fillStyle = "#e2e8f0";
        ctx.fillRect(0, h - 4, w, 4);
        ctx.fillStyle = "#8b5cf6";
        ctx.fillRect(0, h - 4, w * progress, 4);
      }

      frameIdxRef.current = (frameIdxRef.current + 1) % frames.length;
      // ~20 FPS playback
      animFrameRef.current = setTimeout(drawFrame, 50);
    };

    frameIdxRef.current = 0;
    drawFrame();

    return () => {
      if (animFrameRef.current) clearTimeout(animFrameRef.current);
    };
  }, [frames]);

  if (!frames || frames.length === 0) return null;

  return (
    <div className="flex flex-col items-center gap-2 bg-white rounded-2xl p-3 shadow-lg border border-violet-200 flex-1 min-w-0 max-w-xs w-full">
      <div className="flex justify-between w-full text-[10px] font-bold text-slate-700 px-1">
        <span className="flex items-center gap-1.5">
          <span className="w-2.5 h-2.5 rounded-full bg-[#d946ef] flex-shrink-0 shadow-[0_0_4px_#d946ef]"></span>
          Ta silhouette
        </span>
        <span className="flex items-center gap-1.5">
          <span className="w-2.5 h-2.5 rounded-full bg-[#06b6d4] flex-shrink-0 shadow-[0_0_4px_#06b6d4]"></span>
          Cible attendue
        </span>
      </div>
      <canvas
        ref={canvasRef}
        width={280}
        height={300}
        className="rounded-xl border border-slate-100 w-full"
        style={{ imageRendering: "pixelated" }}
      />
      <span className="text-[9px] text-slate-400 font-semibold italic">
        Replay de {frames.length} frames analysées • En boucle
      </span>
    </div>
  );
};


function createDebugTargetPrediction(
  dance,
  match,
  liveSequence,
  detectedPose,
  elapsedSeconds,
  canvas,
  options = {},
) {
  if (!dance?.frames?.length || !detectedPose?.keypoints || !canvas) return null;

  const firstLiveTimestamp = liveSequence?.[0]?.timestamp ?? elapsedSeconds;
  const targetTimestamp = options.syncToTimeline
    ? elapsedSeconds
    : (
      match?.startTimestamp !== null && match?.startTimestamp !== undefined
        ? match.startTimestamp + (elapsedSeconds - firstLiveTimestamp) * (match.speedFactor ?? 1)
        : elapsedSeconds
    );
  const syncFps = dance.sampledFps || CATALOGUE_SYNC_FPS;
  const targetFrameIndex = options.syncToTimeline
    ? Math.round(Math.max(0, targetTimestamp) * syncFps)
    : null;
  const frame = findNearestCatalogueFrame(
    dance.frames,
    Math.max(0, targetTimestamp),
    targetFrameIndex,
  );
  const anchor = getPoseProjectionAnchor(detectedPose);

  if (!frame?.keypoints?.length || !anchor) return null;

  const keypoints = frame.keypoints.map((point) => ({
    name: point.name,
    x: anchor.origin.x + DEBUG_TARGET_X_OFFSET + point.x * anchor.scale * DEBUG_TARGET_SCALE,
    y: anchor.origin.y + point.y * anchor.scale * DEBUG_TARGET_SCALE,
    score: point.score ?? 1,
  }));
  const visiblePoints = keypoints.filter((point) => isVisibleKeypoint(point, 0.5));

  if (visiblePoints.length === 0) return null;

  const minX = Math.min(...visiblePoints.map((point) => point.x));
  const minY = Math.min(...visiblePoints.map((point) => point.y));
  const maxX = Math.max(...visiblePoints.map((point) => point.x));
  const maxY = Math.max(...visiblePoints.map((point) => point.y));
  const padding = Math.max(8, Math.min(canvas.width, canvas.height) * 0.02);

  return {
    bbox: [
      Math.max(0, minX - padding),
      Math.max(0, minY - padding),
      Math.min(canvas.width, maxX + padding) - Math.max(0, minX - padding),
      Math.min(canvas.height, maxY + padding) - Math.max(0, minY - padding),
    ],
    score: 1,
    keypoints,
  };
}

let mirrorCanvas = null;
let mirrorCtx = null;

const RainbowText = ({ text }) => {
  const colors = [
    "text-neon-pink",
    "text-neon-blue",
    "text-neon-yellow",
    "text-neon-purple",
    "text-neon-green",
    "text-neon-orange",
  ];
  return (
    <span className="font-display inline-block">
      {text.split("").map((char, index) => {
        if (char === " ") return <span key={index}>&nbsp;</span>;
        const colorClass = colors[index % colors.length];
        return (
          <span
            key={index}
            className={`${colorClass} inline-block animate-float font-black`}
            style={{
              animationDelay: `${index * 0.08}s`,
              textShadow: "0 0 8px currentColor, 0 2px 4px rgba(0,0,0,0.8)",
            }}
          >
            {char}
          </span>
        );
      })}
    </span>
  );
};

function App() {
  const [processingStatus, setProcessingStatus] = useState({
    warnUpTime: 0,
    inferenceTime: 0,
    statusMsg: "Model not loaded",
    statusColor: "inherit",
  });
  const [catalogue, setCatalogue] = useState(null);
  const [catalogueStatus, setCatalogueStatus] = useState({
    state: "loading",
    message: "Chargement du catalogue local...",
  });
  const [customClasses] = useState([]);
  const [imgSrc] = useState(null);
  const [videoSrc, setVideoSrc] = useState("");
  const [videoName, setVideoName] = useState("");
  const [activeFeature, setActiveFeature] = useState(null);
  const [selectedDanceId, setSelectedDanceId] = useState(null);
  const [debugTargetOverlay, setDebugTargetOverlay] = useState(false);
  const [gameState, setGameState] = useState("idle");
  const [countdown, setCountdown] = useState(COUNTDOWN_SECONDS);
  const [sequenceSampleCount, setSequenceSampleCount] = useState(0);
  const [currentMatch, setCurrentMatch] = useState({
    best: null,
    candidates: [],
    detected: false,
    margin: 0,
  });
  const [stableMatch, setStableMatch] = useState(null);
  const [danceScore, setDanceScore] = useState(0);
  const [dancePrecision, setDancePrecision] = useState(0);
  const [lastDanceInfo, setLastDanceInfo] = useState(null);
  const [pauseCountdown, setPauseCountdown] = useState(0);
  const [audioTimeLeft, setAudioTimeLeft] = useState(null);
  const [scoreHistory, setScoreHistory] = useState([]);
  const [isFullscreen, setIsFullscreen] = useState(false);

  const [performanceRating, setPerformanceRating] = useState({
    text: "PRET",
    color: "text-slate-400",
  });
  const [, setCoachComments] = useState([
    "Active la camera et reproduis un mouvement extrait d'une des deux videos.",
  ]);
  const [showAdvancedSettings, setShowAdvancedSettings] = useState(false);

  const modelConfigRef = useRef(DEFAULT_MODEL_CONFIG);
  const cameraSelectorRef = useRef(null);
  const imgszTypeSelectorRef = useRef(null);
  const fileVideoRef = useRef(null);
  const imgRef = useRef(null);
  const overlayRef = useRef(null);
  const cameraRef = useRef(null);
  const videoRef = useRef(null);
  const cameraContainerRef = useRef(null);

  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(document.fullscreenElement === cameraContainerRef.current);
    };
    document.addEventListener("fullscreenchange", handleFullscreenChange);
    return () => {
      document.removeEventListener("fullscreenchange", handleFullscreenChange);
    };
  }, [cameraContainerRef]);

  const activeFeatureRef = useRef(activeFeature);
  const gameStateRef = useRef(gameState);
  const isProcessingRef = useRef(false);
  const mediaLoopTokenRef = useRef(0);
  const matchHistoryRef = useRef([]);
  const liveSequenceRef = useRef([]);
  const sequenceStartRef = useRef(0);
  const lastCommentRef = useRef("");
  const lastCommentTimeRef = useRef(0);
  const currentAudioRef = useRef(null);
  const audioTimeoutRef = useRef(null);
  const danceScoreRef = useRef(danceScore);
  const playedDanceIdsRef = useRef([]);
  const sessionPrecisionsRef = useRef([]);

  const [preciseAnalysisProgress, setPreciseAnalysisProgress] = useState(0);
  const [preciseAnalysisStatus, setPreciseAnalysisStatus] = useState("");
  const [poseReplayFrames, setPoseReplayFrames] = useState([]);

  const recordedFramesRef = useRef([]);
  const lastRecordedTimeRef = useRef(0);
  const inferenceResolverRef = useRef(null);
  const modelLoadedResolverRef = useRef(null);
  const shouldAnalyzeRef = useRef(false);
  const currentDanceTitleRef = useRef("");
  const postInferenceMessageRef = useRef(null);

  useEffect(() => {
    danceScoreRef.current = danceScore;
  }, [danceScore]);

  useEffect(() => {
    activeFeatureRef.current = activeFeature;
  }, [activeFeature]);

  useEffect(() => {
    gameStateRef.current = gameState;
  }, [gameState]);

  useEffect(() => () => {
    if (videoSrc) URL.revokeObjectURL(videoSrc);
  }, [videoSrc]);



  useEffect(() => {
    let cancelled = false;

    loadPoseCatalogue(import.meta.env.BASE_URL)
      .then((loadedCatalogue) => {
        if (cancelled) return;
        setCatalogue(loadedCatalogue);
        setCatalogueStatus({
          state: "ready",
          message: `${loadedCatalogue.dances.length} danses chargees depuis le catalogue local.`,
        });
      })
      .catch((error) => {
        if (cancelled) return;
        setCatalogueStatus({
          state: "error",
          message: error.message,
        });
      });

    return () => {
      cancelled = true;
    };
  }, []);

  const pushCoachComment = useCallback((message) => {
    const now = performance.now();
    if (lastCommentRef.current === message || now - lastCommentTimeRef.current < 1400) {
      return;
    }

    lastCommentRef.current = message;
    lastCommentTimeRef.current = now;
    setCoachComments((previous) => [message, ...previous].slice(0, 5));
  }, []);

  const getSyncTimeSeconds = useCallback(() => {
    if (activeFeatureRef.current === "camera") {
      const audio = currentAudioRef.current;
      if (audio && Number.isFinite(audio.currentTime)) {
        return Math.max(0, audio.currentTime + AUDIO_SYNC_OFFSET_SECONDS);
      }
    }

    if (activeFeatureRef.current === "video") {
      const video = videoRef.current;
      if (video && Number.isFinite(video.currentTime)) {
        return Math.max(0, video.currentTime + AUDIO_SYNC_OFFSET_SECONDS);
      }
    }

    if (!sequenceStartRef.current) return 0;
    return Math.max(0, (performance.now() - sequenceStartRef.current) / 1000);
  }, []);

  const resetLiveComparison = useCallback(() => {
    matchHistoryRef.current = [];
    liveSequenceRef.current = [];
    sessionPrecisionsRef.current = [];
    sequenceStartRef.current = performance.now();
    setCurrentMatch({ best: null, candidates: [], detected: false, margin: 0 });
    setStableMatch(null);
    setDancePrecision(0);
    setSequenceSampleCount(0);
  }, []);

  const stopMediaLoop = useCallback(() => {
    mediaLoopTokenRef.current += 1;
  }, []);

  const abortRecording = useCallback(() => {
    shouldAnalyzeRef.current = false;
    recordedFramesRef.current = [];
    setPoseReplayFrames([]);
  }, []);

  const stopRecording = useCallback(() => {
    shouldAnalyzeRef.current = false;
    lastRecordedTimeRef.current = 0;
  }, []);

  const prepareCountdown = useCallback(() => {
    resetLiveComparison();
    abortRecording();
    sequenceStartRef.current = 0;
    setDanceScore(0);
    setCountdown(COUNTDOWN_SECONDS);
    setGameState("countdown");
    setPerformanceRating({
      text: `${COUNTDOWN_SECONDS}`,
      color: "text-cyan-300 font-black",
    });
    setCoachComments([
      "Prepare-toi. Detection temporelle dans 5 secondes.",
    ]);
    if (overlayRef.current) {
      const ctx = overlayRef.current.getContext("2d");
      ctx?.clearRect(0, 0, overlayRef.current.width, overlayRef.current.height);
    }
  }, [resetLiveComparison, abortRecording]);

  const stopMusic = useCallback(() => {
    if (audioTimeoutRef.current) {
      window.clearInterval(audioTimeoutRef.current);
      window.clearTimeout(audioTimeoutRef.current);
      audioTimeoutRef.current = null;
    }
    if (currentAudioRef.current) {
      currentAudioRef.current.pause();
      currentAudioRef.current.currentTime = 0;
      currentAudioRef.current = null;
    }
    setLastDanceInfo(null);
    setPauseCountdown(0);
    setAudioTimeLeft(null);
    playedDanceIdsRef.current = [];
  }, []);

  const loadModelPromise = useCallback((modelName) => {
    return new Promise((resolve) => {
      modelLoadedResolverRef.current = resolve;
      const config = {
        ...DEFAULT_MODEL_CONFIG,
        model: modelName,
        modelPath: `${import.meta.env.BASE_URL}models/${modelName}-pose.onnx`
      };
      postInferenceMessageRef.current?.({
        type: "LOAD_MODEL",
        config
      });
    });
  }, []);

  const processFramePromise = useCallback((bitmap, config) => {
    return new Promise((resolve) => {
      // Use a ref-like object so the timeout closure can access wrappedResolve after declaration
      const slot = { fn: null };

      const timeoutId = setTimeout(() => {
        if (inferenceResolverRef.current === slot.fn) {
          inferenceResolverRef.current = null;
          console.warn("[PreciseAnalysis] Worker timeout on frame – skipping");
          resolve({ results: [] });
        }
      }, 30000);

      slot.fn = (data) => {
        clearTimeout(timeoutId);
        resolve(data);
      };

      inferenceResolverRef.current = slot.fn;
      postInferenceMessageRef.current?.(
        {
          type: "INFERENCE",
          config,
          bitmap,
        },
        [bitmap]
      );
    });
  }, []);

  const startPauseCountdown = useCallback((finalScore, finalSampleCount, danceTitle) => {
    let rank = "C";
    if (finalScore > 60) rank = "A";
    else if (finalScore >= 45) rank = "B";

    const historyEntry = {
      id: Date.now(),
      title: danceTitle,
      score: `${finalScore}%`,
      rank: rank,
      samples: finalSampleCount,
      date: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
    };

    setScoreHistory((prev) => [historyEntry, ...prev]);

    setLastDanceInfo({
      title: danceTitle,
      score: `${finalScore}%`,
      rank: rank,
    });
    setGameState("pause");
    setPauseCountdown(5);
    setAudioTimeLeft(null);

    let remaining = 5;
    const intervalId = window.setInterval(() => {
      remaining -= 1;
      setPauseCountdown(remaining);
      if (remaining <= 0) {
        window.clearInterval(intervalId);
        setLastDanceInfo(null);
        
        const dancesWithAudio = catalogue.dances.filter((d) => d.audioUrl);
        const unplayedDances = dancesWithAudio.filter((d) => !playedDanceIdsRef.current.includes(d.id));
        if (unplayedDances.length > 0) {
          const nextDance = unplayedDances[Math.floor(Math.random() * unplayedDances.length)];
          setSelectedDanceId(nextDance.id);
          prepareCountdown();
        } else {
          stopMusic();
          setGameState("idle");
          setPerformanceRating({ text: "TERMINE", color: "text-emerald-400 font-extrabold" });
          setCoachComments(["Felicitations ! Vous avez complete toutes les danses de la playlist !"]);
        }
      }
    }, 1000);

    audioTimeoutRef.current = intervalId;
  }, [catalogue, prepareCountdown, stopMusic]);

  const runPreciseAnalysis = useCallback(async () => {
    setGameState("analyzing");
    setPreciseAnalysisProgress(0);
    setPreciseAnalysisStatus("Chargement du modèle de haute précision...");

    // Drain any in-flight worker response from the media loop to avoid stealing our resolver
    inferenceResolverRef.current = null;
    await new Promise((r) => setTimeout(r, 250));

    try {
      // 1. Load yolo26s-pose.onnx
      await loadModelPromise("yolo26s");

      setPreciseAnalysisStatus("Analyse de la chorégraphie...");

      const frames = recordedFramesRef.current;
      const totalSteps = frames.length;
      let currentStep = 0;

      const precisePrecisions = [];
      const liveSequence = [];
      const replayFramesBuffer = []; // Collect ALL frames for animated replay

      // Find the current selected dance
      const selectedDance = selectedDanceId
        ? catalogue?.dances?.find((d) => d.id === selectedDanceId)
        : null;

      const canvas = overlayRef.current;
      const overlayCtx = canvas ? canvas.getContext("2d") : null;

      for (let i = 0; i < frames.length; i++) {
        const frame = frames[i];
        
        // Convert Blob to ImageBitmap
        const bitmap = await createImageBitmap(frame.blob);

        if (overlayCtx && canvas) {
          if (canvas.width !== bitmap.width || canvas.height !== bitmap.height) {
            canvas.width = bitmap.width;
            canvas.height = bitmap.height;
          }
          overlayCtx.clearRect(0, 0, canvas.width, canvas.height);
          overlayCtx.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
        }

        const data = await processFramePromise(bitmap, {
          ...DEFAULT_MODEL_CONFIG,
          model: "yolo26s",
          overlaySize: [bitmap.width, bitmap.height],
        });

        const targetPose = getHighestScorePose(data.results);
        if (targetPose?.keypoints) {
          const sample = createPoseSample(targetPose, frame.timestamp, {
            mirror: activeFeatureRef.current === "camera",
          });
          if (sample) {
            liveSequence.push(sample);
          }

          if (overlayCtx) {
            renderOverlay(
              [targetPose],
              null,
              overlayCtx,
              "pose",
              DEFAULT_MODEL_CONFIG.classes,
            );
          }
        }

        const currentLiveSeq = liveSequence.filter(
          (entry) => frame.timestamp - entry.timestamp <= SEQUENCE_WINDOW_SECONDS
        );

        if (catalogue) {
          const activeCatalogue = selectedDance
            ? { ...catalogue, dances: [selectedDance] }
            : catalogue;

          const matchOptions = {
            sequenceWindowSeconds: SEQUENCE_WINDOW_SECONDS,
            syncToTimeline: true,
            speedFactors: [1],
            maxTimeGapSeconds: 0.22,
          };

          const instantMatch = matchPoseSequenceToCatalogue(
            currentLiveSeq,
            activeCatalogue,
            matchOptions,
          );

          const precision = instantMatch.best ? Math.round(instantMatch.best.score) : 0;

          precisePrecisions.push(precision);

          const currentBestDance = selectedDance || (instantMatch.best
            ? catalogue?.dances?.find((d) => d.id === instantMatch.best.id)
            : null);

          // Collect this frame's user pose + matching catalogue pose for replay
          if (currentBestDance && targetPose && instantMatch.best) {
            const isTimelineSynced = Boolean(
              (activeFeatureRef.current === "camera" && currentAudioRef.current) ||
              activeFeatureRef.current === "video"
            );
            const targetTimestamp = isTimelineSynced
              ? frame.timestamp
              : (
                instantMatch.best.startTimestamp !== null && instantMatch.best.startTimestamp !== undefined
                  ? instantMatch.best.startTimestamp + (frame.timestamp - (liveSequence[0]?.timestamp ?? 0)) * (instantMatch.best.speedFactor ?? 1)
                  : frame.timestamp
              );
            const syncFps = currentBestDance.sampledFps || CATALOGUE_SYNC_FPS;
            const targetFrameIndex = isTimelineSynced
              ? Math.round(Math.max(0, targetTimestamp) * syncFps)
              : null;
            const catalogueFrame = findNearestCatalogueFrame(
              currentBestDance.frames,
              Math.max(0, targetTimestamp),
              targetFrameIndex,
            );

            if (catalogueFrame?.keypoints) {
              const userSample = createPoseSample(targetPose, frame.timestamp, {
                mirror: activeFeatureRef.current === "camera",
              });
              if (userSample) {
                replayFramesBuffer.push({
                  userPose: userSample.keypoints,
                  targetPose: catalogueFrame.keypoints,
                  score: precision,
                });
              }
            }
          }

          if (overlayCtx && currentBestDance && targetPose) {
            const isTimelineSynced = Boolean(
              (activeFeatureRef.current === "camera" && currentAudioRef.current) ||
              activeFeatureRef.current === "video"
            );

            const debugPrediction = createDebugTargetPrediction(
              currentBestDance,
              instantMatch.best,
              liveSequence,
              targetPose,
              frame.timestamp,
              canvas,
              { syncToTimeline: isTimelineSynced },
            );

            if (debugPrediction) {
              renderOverlay(
                [debugPrediction],
                null,
                overlayCtx,
                "pose",
                DEFAULT_MODEL_CONFIG.classes,
                { pose: DEBUG_TARGET_COLORS },
              );
            }
          }
        }

        currentStep++;
        setPreciseAnalysisProgress(Math.min(99, Math.round((currentStep / totalSteps) * 100)));
        
        // Yield thread execution for rendering UI smoothly and pacing playback
        await new Promise((r) => setTimeout(r, 15));
      }

      setPreciseAnalysisStatus("Calcul du score final...");
      setPreciseAnalysisProgress(100);

      const finalScore = precisePrecisions.length > 0
        ? Math.round(precisePrecisions.reduce((a, b) => a + b, 0) / precisePrecisions.length)
        : 0;

      // Store all replay frames for animated overlay display
      if (replayFramesBuffer.length > 0) {
        setPoseReplayFrames(replayFramesBuffer);
      }

      await new Promise((r) => setTimeout(r, 1000));

      // 4. Load back standard yolo26n model
      setPreciseAnalysisStatus("Rechargement du modèle de jeu...");
      await loadModelPromise("yolo26n");

      // Clear recorded frames
      recordedFramesRef.current = [];

      // 5. Start transition countdown with the precise score
      startPauseCountdown(finalScore, liveSequence.length, currentDanceTitleRef.current);

    } catch (err) {
      console.error("Precise analysis failed:", err);
      setPreciseAnalysisStatus("Erreur d'analyse. Rechargement du modèle standard...");
      try {
        await loadModelPromise("yolo26n");
      } catch (e) {
        console.error("Failed to reload model:", e);
      }
      recordedFramesRef.current = [];
      startPauseCountdown(danceScoreRef.current, liveSequenceRef.current.length, currentDanceTitleRef.current);
    }
  }, [catalogue, selectedDanceId, startPauseCountdown, loadModelPromise, processFramePromise]);

  const playMusicForCamera = useCallback(() => {
    stopMusic();

    if (!catalogue?.dances || catalogue.dances.length === 0) return;

    let selectedDance = null;
    if (selectedDanceId) {
      selectedDance = catalogue.dances.find((d) => d.id === selectedDanceId);
    }

    if (!selectedDance || !selectedDance.audioUrl) {
      const dancesWithAudio = catalogue.dances.filter((d) => d.audioUrl);
      if (dancesWithAudio.length > 0) {
        const randomIndex = Math.floor(Math.random() * dancesWithAudio.length);
        selectedDance = dancesWithAudio[randomIndex];
        setSelectedDanceId(selectedDance.id);
        return;
      }
    }

    if (selectedDance && selectedDance.audioUrl) {
      if (!playedDanceIdsRef.current.includes(selectedDance.id)) {
        playedDanceIdsRef.current.push(selectedDance.id);
      }
      const baseUrl = import.meta.env.BASE_URL || "/";
      const audioPath = `${baseUrl.replace(/\/+$/, "")}/${selectedDance.audioUrl.replace(/^\/+/, "")}`;
      const audio = new Audio(audioPath);
      audio.loop = false;
      audio.volume = 0.4;
 
      audio.addEventListener("timeupdate", () => {
        if (!isNaN(audio.duration)) {
          const remaining = Math.max(0, audio.duration - audio.currentTime);
          setAudioTimeLeft(Math.ceil(remaining));
        }
      });

      audio.addEventListener("play", () => {
        resetLiveComparison();
      }, { once: true });

      audio.addEventListener("ended", () => {
        stopMediaLoop();
        stopRecording();
        
        if (activeFeatureRef.current === "camera" && recordedFramesRef.current.length > 0) {
          currentDanceTitleRef.current = selectedDance.title;
          runPreciseAnalysis();
        } else {
          startPauseCountdown(danceScoreRef.current, liveSequenceRef.current.length, selectedDance.title);
        }
      });
 
      currentAudioRef.current = audio;
      audio.play().catch((err) => {
        console.warn("Autoplay block or music play failed:", err);
      });
    }
  }, [catalogue, selectedDanceId, stopMusic, setSelectedDanceId, resetLiveComparison, startPauseCountdown, stopMediaLoop, stopRecording, runPreciseAnalysis]);

  useEffect(() => {
    queueMicrotask(() => {
      if (activeFeature === "camera") {
        if (gameStateRef.current === "detecting") {
          playMusicForCamera();
        }
      } else {
        stopMusic();
      }
    });
    return () => {
      stopMusic();
    };
  }, [activeFeature, selectedDanceId, playMusicForCamera, stopMusic]);

  useEffect(() => {
    if (gameState === "detecting" && (activeFeature === "camera" || activeFeature === "video")) {
      recordedFramesRef.current = [];
      shouldAnalyzeRef.current = true;
      lastRecordedTimeRef.current = 0;
    }
  }, [gameState, activeFeature]);

  useEffect(() => {
    if (gameState !== "countdown") return undefined;

    const interval = window.setInterval(() => {
      setCountdown((current) => {
        if (current <= 1) {
          window.clearInterval(interval);
          sequenceStartRef.current = performance.now();
          liveSequenceRef.current = [];
          matchHistoryRef.current = [];
          setSequenceSampleCount(0);
          setGameState("detecting");
          setPerformanceRating({
            text: "CHERCHE",
            color: "text-violet-300 font-bold animate-pulse",
          });
          setCoachComments([
            "C'est parti. Je compare maintenant ton mouvement dans le temps.",
          ]);
          if (activeFeatureRef.current === "camera") {
            playMusicForCamera();
          }
          return 0;
        }

        return current - 1;
      });
    }, 1000);

    return () => window.clearInterval(interval);
  }, [gameState, playMusicForCamera]);

  const handleInferenceResult = useCallback(
    (data) => {
      if (inferenceResolverRef.current) {
        const resolve = inferenceResolverRef.current;
        inferenceResolverRef.current = null;
        resolve(data);
        return;
      }

      const overlayCtx = overlayRef.current?.getContext("2d");
      if (!overlayCtx) {
        isProcessingRef.current = false;
        return;
      }

      try {
        const targetPose = getHighestScorePose(data.results);
        const displayedResults = targetPose ? [targetPose] : [];

        overlayCtx.clearRect(0, 0, overlayCtx.canvas.width, overlayCtx.canvas.height);
        renderOverlay(
          displayedResults,
          data.maskImageData,
          overlayCtx,
          DEFAULT_MODEL_CONFIG.task,
          DEFAULT_MODEL_CONFIG.classes,
        );

        setProcessingStatus((previous) => ({
          ...previous,
          inferenceTime: data.inferenceTime,
        }));

        const now = performance.now();
        const state = gameStateRef.current;

        if (state === "waiting_for_person" && targetPose?.keypoints) {
          prepareCountdown();
          return;
        }

        if (state === "countdown") {
          setDancePrecision(0);
          setSequenceSampleCount(0);
          setPerformanceRating({
            text: `${countdown}`,
            color: "text-cyan-300 font-black",
          });
        } else if (targetPose?.keypoints && catalogue && state === "detecting") {
          const elapsedSeconds = getSyncTimeSeconds();
          const selectedDance = selectedDanceId
            ? catalogue.dances.find((dance) => dance.id === selectedDanceId)
            : null;
          const isTimelineSynced = Boolean(
            selectedDance &&
            (
              (activeFeatureRef.current === "camera" && currentAudioRef.current) ||
              activeFeatureRef.current === "video"
            ),
          );
          const sample = createPoseSample(targetPose, elapsedSeconds, {
            mirror: activeFeatureRef.current === "camera",
          });

          if (sample) {
            liveSequenceRef.current = liveSequenceRef.current
              .concat(sample)
              .filter(
                (entry) => elapsedSeconds - entry.timestamp <= SEQUENCE_WINDOW_SECONDS,
              );
            setSequenceSampleCount(liveSequenceRef.current.length);
          }

          const activeCatalogue = selectedDance
            ? { ...catalogue, dances: [selectedDance] }
            : catalogue;

          const matchOptions = {
            sequenceWindowSeconds: SEQUENCE_WINDOW_SECONDS,
          };
          if (isTimelineSynced) {
            matchOptions.syncToTimeline = true;
            matchOptions.speedFactors = [1];
            matchOptions.maxTimeGapSeconds = 0.22;
          }

          const instantMatch = matchPoseSequenceToCatalogue(
            liveSequenceRef.current,
            activeCatalogue,
            matchOptions,
          );
          const stabilized = stabilizeMatches(
            matchHistoryRef.current,
            instantMatch,
            now,
            2600,
          );

          matchHistoryRef.current = stabilized.history;
          setCurrentMatch(instantMatch);
          setStableMatch(stabilized.stable);

          const displayMatch = stabilized.stable ?? instantMatch.best;
          const precision = displayMatch ? Math.round(displayMatch.score) : 0;

          if (debugTargetOverlay) {
            const debugDanceId = selectedDance?.id ?? displayMatch?.id ?? selectedDanceId;
            const debugDance = catalogue.dances.find((dance) => dance.id === debugDanceId);
            const debugPrediction = createDebugTargetPrediction(
              debugDance,
              displayMatch,
              liveSequenceRef.current,
              targetPose,
              elapsedSeconds,
              overlayCtx.canvas,
              { syncToTimeline: isTimelineSynced },
            );

            if (debugPrediction) {
              renderOverlay(
                [debugPrediction],
                null,
                overlayCtx,
                "pose",
                DEFAULT_MODEL_CONFIG.classes,
                { pose: DEBUG_TARGET_COLORS },
              );
            }
          }

          setDancePrecision(precision);

          if (displayMatch) {
            sessionPrecisionsRef.current.push(precision);
          } else {
            sessionPrecisionsRef.current.push(0);
          }

          const runningAvg = sessionPrecisionsRef.current.length > 0
            ? Math.round(
                sessionPrecisionsRef.current.reduce((a, b) => a + b, 0) /
                  sessionPrecisionsRef.current.length
              )
            : 0;
          setDanceScore(runningAvg);

          if (displayMatch && precision > 60) {
            setPerformanceRating({
              text: "RANG A",
              color: "text-emerald-400 font-extrabold text-violet-neon",
            });
            pushCoachComment(
              `Excellent ! Ressemblance > 60% (${precision}%).`,
            );
          } else if (displayMatch && precision >= 45) {
            setPerformanceRating({
              text: "RANG B",
              color: "text-amber-300 font-bold",
            });
            pushCoachComment(`Bien ! Ressemblance entre 45% et 60% (${precision}%).`);
          } else {
            setPerformanceRating({
              text: "RANG C",
              color: "text-violet-300 font-bold animate-pulse",
            });
          }
        } else {
          if (state !== "detecting" && state !== "waiting_for_person" && state !== "countdown") {
            setDancePrecision(0);
            setCurrentMatch({ best: null, candidates: [], detected: false, margin: 0 });
            setStableMatch(null);
            matchHistoryRef.current = [];
          }

          if (state === "detecting") {
            setPerformanceRating({
              text: "EN ATTENTE",
              color: "text-slate-500",
            });
          } else if (state === "waiting_for_person") {
            setPerformanceRating({
              text: "ATTENTE",
              color: "text-amber-500 font-bold animate-pulse",
            });
          }
        }
      } finally {
        isProcessingRef.current = false;
      }
    },
    [
      catalogue,
      countdown,
      debugTargetOverlay,
      getSyncTimeSeconds,
      prepareCountdown,
      pushCoachComment,
      selectedDanceId,
    ],
  );

  const handleModelLoaded = useCallback((data) => {
    setProcessingStatus((previous) => ({
      ...previous,
      statusMsg: data.msg,
      statusColor: "green",
      warnUpTime: data.loadTime,
    }));
    if (modelLoadedResolverRef.current) {
      const resolve = modelLoadedResolverRef.current;
      modelLoadedResolverRef.current = null;
      resolve();
      return;
    }
    setActiveFeature(null);
  }, []);

  const handleModelLoadError = useCallback((data) => {
    setProcessingStatus((previous) => ({
      ...previous,
      statusMsg: data.msg,
      statusColor: "red",
    }));
    if (modelLoadedResolverRef.current) {
      const resolve = modelLoadedResolverRef.current;
      modelLoadedResolverRef.current = null;
      resolve();
      return;
    }
    setActiveFeature(null);
  }, []);

  const { postMessage: postInferenceMessage } = useInferenceWorker({
    onModelLoaded: handleModelLoaded,
    onResult: handleInferenceResult,
    onError: handleModelLoadError,
  });

  useEffect(() => {
    postInferenceMessageRef.current = postInferenceMessage;
  }, [postInferenceMessage]);

  const { cameras, getCameras, openCamera, closeCamera, cameraStatus } =
    useWebcam(cameraRef);

  useEffect(() => {
    if (cameraStatus.msg) {
      queueMicrotask(() => {
        setProcessingStatus((previous) => ({
          ...previous,
          statusMsg: cameraStatus.msg,
          statusColor: cameraStatus.color,
        }));
      });
    }
  }, [cameraStatus]);

  const loadModel = useCallback(async () => {
    setProcessingStatus((previous) => ({
      ...previous,
      statusMsg: "Loading model...",
      statusColor: "red",
    }));
    setActiveFeature("loading");

    DEFAULT_MODEL_CONFIG.modelPath = `${import.meta.env.BASE_URL}models/${DEFAULT_MODEL_CONFIG.model}-${DEFAULT_MODEL_CONFIG.task}.onnx`;
    postInferenceMessage({
      type: "LOAD_MODEL",
      config: DEFAULT_MODEL_CONFIG,
    });
  }, [postInferenceMessage]);

  useEffect(() => {
    queueMicrotask(() => {
      loadModel();
    });
  }, [loadModel]);

  const processMediaFrame = useCallback(async (mediaElement) => {
    if (
      isProcessingRef.current ||
      !mediaElement ||
      mediaElement.readyState < 2 ||
      !mediaElement.videoWidth ||
      !mediaElement.videoHeight ||
      !overlayRef.current
    ) {
      return;
    }

    isProcessingRef.current = true;

    try {
      const vw = mediaElement.videoWidth;
      const vh = mediaElement.videoHeight;
      const scale = Math.min(1, MAX_INFER_DIM / Math.max(vw, vh));
      const inferW = Math.round(vw * scale);
      const inferH = Math.round(vh * scale);

      let bitmap;
      if (activeFeatureRef.current === "camera") {
        if (!mirrorCanvas) {
          mirrorCanvas = document.createElement("canvas");
          mirrorCtx = mirrorCanvas.getContext("2d");
        }
        if (mirrorCanvas.width !== inferW || mirrorCanvas.height !== inferH) {
          mirrorCanvas.width = inferW;
          mirrorCanvas.height = inferH;
        }

        mirrorCtx.clearRect(0, 0, inferW, inferH);
        mirrorCtx.save();
        mirrorCtx.translate(inferW, 0);
        mirrorCtx.scale(-1, 1);
        mirrorCtx.drawImage(mediaElement, 0, 0, inferW, inferH);
        mirrorCtx.restore();

        // Capture frame to JPEG blob in memory for post-analysis
        const nowMs = performance.now();
        if (shouldAnalyzeRef.current && nowMs - lastRecordedTimeRef.current >= 66) { // ~15 FPS
          lastRecordedTimeRef.current = nowMs;
          const timestamp = getSyncTimeSeconds();
          mirrorCanvas.toBlob((blob) => {
            if (shouldAnalyzeRef.current && blob) {
              recordedFramesRef.current.push({
                timestamp,
                blob
              });
            }
          }, "image/jpeg", 0.65);
        }

        bitmap = await createImageBitmap(mirrorCanvas);
      } else {
        // Capture frame to JPEG blob for video post-analysis as well
        const nowMs = performance.now();
        if (shouldAnalyzeRef.current && nowMs - lastRecordedTimeRef.current >= 66) { // ~15 FPS
          lastRecordedTimeRef.current = nowMs;
          const timestamp = getSyncTimeSeconds();
          if (!mirrorCanvas) {
            mirrorCanvas = document.createElement("canvas");
            mirrorCtx = mirrorCanvas.getContext("2d");
          }
          if (mirrorCanvas.width !== inferW || mirrorCanvas.height !== inferH) {
            mirrorCanvas.width = inferW;
            mirrorCanvas.height = inferH;
          }
          mirrorCtx.clearRect(0, 0, inferW, inferH);
          mirrorCtx.drawImage(mediaElement, 0, 0, inferW, inferH);
          mirrorCanvas.toBlob((blob) => {
            if (shouldAnalyzeRef.current && blob) {
              recordedFramesRef.current.push({
                timestamp,
                blob
              });
            }
          }, "image/jpeg", 0.65);
        }

        bitmap = await createImageBitmap(mediaElement, {
          resizeWidth: inferW,
          resizeHeight: inferH,
          resizeQuality: "low",
        });
      }

      if (overlayRef.current.width !== inferW || overlayRef.current.height !== inferH) {
        overlayRef.current.width = inferW;
        overlayRef.current.height = inferH;
      }

      DEFAULT_MODEL_CONFIG.overlaySize = [inferW, inferH];

      postInferenceMessage(
        {
          type: "INFERENCE",
          config: DEFAULT_MODEL_CONFIG,
          bitmap,
        },
        [bitmap],
      );
    } catch (error) {
      console.error("Frame capture error:", error);
      isProcessingRef.current = false;
    }
  }, [postInferenceMessage, getSyncTimeSeconds]);

  const startMediaLoop = useCallback((featureName, mediaRef) => {
    isProcessingRef.current = false;
    mediaLoopTokenRef.current += 1;
    const loopToken = mediaLoopTokenRef.current;

    const loop = async () => {
      if (
        mediaLoopTokenRef.current !== loopToken ||
        activeFeatureRef.current !== featureName
      ) {
        return;
      }

      const mediaElement = mediaRef.current;
      if (!mediaElement) return;

      if (featureName === "video" && (mediaElement.paused || mediaElement.ended)) {
        return;
      }

      await processMediaFrame(mediaElement);
      requestAnimationFrame(loop);
    };

    requestAnimationFrame(loop);
  }, [processMediaFrame]);

  const startCameraLoop = useCallback(() => {
    startMediaLoop("camera", cameraRef);
  }, [startMediaLoop]);

  const waitForCameraFrame = useCallback(async () => {
    const video = cameraRef.current;
    if (!video) return false;

    if (video.readyState < 2 || !video.videoWidth || !video.videoHeight) {
      await new Promise((resolve) => {
        const done = () => {
          video.removeEventListener("loadedmetadata", done);
          video.removeEventListener("loadeddata", done);
          resolve();
        };
        video.addEventListener("loadedmetadata", done, { once: true });
        video.addEventListener("loadeddata", done, { once: true });
        window.setTimeout(done, 2000);
      });
    }

    await video.play().catch(() => {});
    return video.readyState >= 2 && video.videoWidth > 0 && video.videoHeight > 0;
  }, []);

  useEffect(() => {
    if (activeFeature === "camera" && (gameState === "countdown" || gameState === "detecting" || gameState === "waiting_for_person")) {
      startCameraLoop();
    }
  }, [activeFeature, gameState, startCameraLoop]);

  const startVideoLoop = useCallback(() => {
    startMediaLoop("video", videoRef);
  }, [startMediaLoop]);

  const handleVideoFile = useCallback((file) => {
    if (!file) return;

    if (activeFeatureRef.current === "camera") {
      closeCamera();
    }

    abortRecording();
    stopMediaLoop();
    resetLiveComparison();
    setDanceScore(0);
    setCountdown(0);
    setVideoName(file.name);
    setVideoSrc(URL.createObjectURL(file));
    setActiveFeature("video");
    setGameState("detecting");
    setPerformanceRating({
      text: "VIDEO",
      color: "text-cyan-300 font-black",
    });
    setCoachComments([
      `Video chargee: ${file.name}. Lance la lecture pour voir la detection realtime.`,
    ]);
  }, [closeCamera, resetLiveComparison, stopMediaLoop, abortRecording]);

  const stopVideo = useCallback(() => {
    abortRecording();
    stopMediaLoop();
    videoRef.current?.pause();
    setVideoSrc("");
    setVideoName("");
    if (overlayRef.current) {
      overlayRef.current.getContext("2d")?.clearRect(
        0,
        0,
        overlayRef.current.width,
        overlayRef.current.height,
      );
      overlayRef.current.width = 0;
      overlayRef.current.height = 0;
    }
    if (activeFeatureRef.current === "video") {
      setActiveFeature(null);
    }
    setGameState("idle");
    resetLiveComparison();
    setPerformanceRating({ text: "PRET", color: "text-slate-400" });
    setCoachComments(["Video fermee. Tu peux relancer la camera ou ouvrir une autre video."]);
  }, [resetLiveComparison, stopMediaLoop, abortRecording]);

  const handleVideoLoad = useCallback(() => {
    processMediaFrame(videoRef.current);
  }, [processMediaFrame]);

  const handleVideoPlay = useCallback(() => {
    if (activeFeatureRef.current !== "video") return;
    setGameState("detecting");
    setPerformanceRating({
      text: "DETECTE",
      color: "text-cyan-300 font-black",
    });
    startVideoLoop();
  }, [startVideoLoop]);

  const handleVideoPause = useCallback(() => {
    processMediaFrame(videoRef.current);
  }, [processMediaFrame]);

  const handleVideoSeeked = useCallback(() => {
    resetLiveComparison();
    processMediaFrame(videoRef.current);
  }, [processMediaFrame, resetLiveComparison]);

  const handleVideoEnded = useCallback(() => {
    stopMediaLoop();
    stopRecording();
    setPerformanceRating({
      text: "FIN",
      color: "text-slate-400 font-black",
    });

    if (recordedFramesRef.current.length > 0) {
      currentDanceTitleRef.current = videoName || "Video Test";
      runPreciseAnalysis();
    }
  }, [stopMediaLoop, stopRecording, runPreciseAnalysis, videoName]);

  const toggleCamera = useCallback(async () => {
    if (activeFeature === "camera") {
      stopMediaLoop();
      closeCamera();
      if (overlayRef.current) {
        overlayRef.current.width = 0;
        overlayRef.current.height = 0;
      }
      setActiveFeature(null);
      activeFeatureRef.current = null;
      setGameState("idle");
      gameStateRef.current = "idle";
      setCountdown(COUNTDOWN_SECONDS);
      setSequenceSampleCount(0);
      liveSequenceRef.current = [];
      setPerformanceRating({ text: "PRET", color: "text-slate-400" });
      if (document.fullscreenElement === cameraContainerRef.current) {
        document.exitFullscreen().catch((err) => {
          console.warn("Could not exit fullscreen:", err);
        });
      }
      return;
    }

    if (activeFeature === "video") {
      stopVideo();
    }

    const camerasList = await getCameras();
    if (camerasList.length === 0) {
      setProcessingStatus((previous) => ({
        ...previous,
        statusMsg: "No cameras found",
        statusColor: "red",
      }));
      return;
    }

    const selectedDeviceId = cameraSelectorRef.current
      ? cameraSelectorRef.current.value
      : camerasList[0].deviceId;
    const success = await openCamera(selectedDeviceId);

    if (success) {
      activeFeatureRef.current = "camera";
      gameStateRef.current = "waiting_for_person";
      setActiveFeature("camera");
      setGameState("waiting_for_person");
      setPerformanceRating({ text: "ATTENTE", color: "text-amber-500 font-bold animate-pulse" });
      setCoachComments(["Caméra activée. Présentez-vous devant l'écran pour lancer la détection."]);
      if (cameraContainerRef.current?.requestFullscreen) {
        cameraContainerRef.current.requestFullscreen().catch((err) => {
          console.warn("Could not enter fullscreen:", err);
        });
      }
      void waitForCameraFrame().then((ready) => {
        if (ready && activeFeatureRef.current === "camera") {
          startCameraLoop();
        }
      });
    }
  }, [activeFeature, closeCamera, getCameras, openCamera, stopMediaLoop, stopVideo, cameraContainerRef, startCameraLoop, waitForCameraFrame]);

  const handleCameraLoad = useCallback(() => {
    startCameraLoop();
  }, [startCameraLoop]);

  const imageLoad = useCallback(() => {}, []);

  const instantCandidates = (currentMatch?.candidates ?? []).slice(0, 2);
  const shownMatch = stableMatch ?? currentMatch.best;

  return (
    <div className="max-w-7xl mx-auto px-4 py-6 text-white min-h-screen flex flex-col gap-6">
      <header className="text-center">
        <h1 className="text-4xl md:text-6xl font-extrabold tracking-wider uppercase">
          <RainbowText text="Just Dance Captor" />
        </h1>
        <p className="text-slate-400 text-sm mt-2">
          Compare les mouvements camera avec le catalogue extrait de tes videos.
        </p>
      </header>

      <main className="grid grid-cols-1 xl:grid-cols-[minmax(0,2fr)_minmax(320px,1fr)] gap-6 items-start">
        <div className="flex flex-col gap-4">
          <section className="relative">
            <ImageDisplay
              cameraRef={cameraRef}
              videoRef={videoRef}
              imgRef={imgRef}
              overlayRef={overlayRef}
              imgSrc={imgSrc}
              videoSrc={videoSrc}
              onCameraLoad={handleCameraLoad}
              onVideoLoad={handleVideoLoad}
              onVideoPlay={handleVideoPlay}
              onVideoPause={handleVideoPause}
              onVideoSeeked={handleVideoSeeked}
              onVideoEnded={handleVideoEnded}
              onImageLoad={imageLoad}
              activeFeature={activeFeature}
              audioTimeLeft={audioTimeLeft}
              cameraContainerRef={cameraContainerRef}
              gameState={gameState}
              countdown={countdown}
              selectedDanceId={selectedDanceId}
              catalogue={catalogue}
              dancePrecision={dancePrecision}
              danceScore={danceScore}
              sessionPrecisions={sessionPrecisionsRef.current}
            >
              {/* Floating remaining time overlay (visible in both normal and fullscreen modes) */}
              {audioTimeLeft !== null && audioTimeLeft !== undefined && (
                <div className="absolute top-4 left-1/2 transform -translate-x-1/2 z-45 flex items-center gap-1.5 px-4 py-1.5 rounded-full bg-black/60 backdrop-blur-md border border-fuchsia-500/30 text-fuchsia-300 font-mono text-sm font-bold shadow-[0_0_15px_rgba(217,70,239,0.25)] select-none">
                  <span className="w-2 h-2 rounded-full bg-fuchsia-500 animate-ping inline-block mr-0.5"></span>
                  🎵 {formatTime(audioTimeLeft)} restant
                </div>
              )}

              {/* Countdown HUD overlay inside the camera screen */}
              {activeFeature === "camera" && gameState === "countdown" && (
                <div 
                  className={`absolute inset-0 flex items-center justify-center flex-col z-50 transition-all duration-1000 ${
                    countdown <= 2 
                      ? "bg-[#050414]/0 opacity-0 pointer-events-none" 
                      : "bg-[#050414] opacity-100"
                  }`}
                >
                  <span className={`text-9xl md:text-11xl font-black text-transparent bg-clip-text bg-gradient-to-r from-yellow-300 via-pink-500 to-cyan-400 animate-bounce font-display transition-all duration-1000 ${
                    countdown <= 2 ? "opacity-0 scale-75" : "opacity-100 scale-100"
                  }`}>
                    {countdown}
                  </span>
                  <span className={`text-2xl uppercase font-black text-white tracking-widest mt-4 font-display transition-all duration-1000 ${
                    countdown <= 2 ? "opacity-0" : "opacity-100"
                  }`}>
                    Préparez-vous !
                  </span>
                </div>
              )}

              {/* Loading spinner */}
              {activeFeature === "loading" && (
                <div className="absolute inset-0 bg-black/70 flex items-center justify-center flex-col gap-3 rounded-2xl z-50">
                  <div className="w-12 h-12 border-4 border-violet-500 border-t-transparent rounded-full animate-spin"></div>
                  <span className="text-violet-300 font-bold">Chargement du modele...</span>
                </div>
              )}

              {/* High precision analysis banner */}
              {gameState === "analyzing" && (
                <div className="absolute bottom-4 left-4 right-4 bg-black/85 backdrop-blur-md flex items-center justify-between gap-4 rounded-xl p-4 border border-violet-500/30 shadow-[0_0_30px_rgba(139,92,246,0.3)] z-50 animate-fade-in">
                  <div className="flex items-center gap-3">
                    <div className="w-8 h-8 border-4 border-t-fuchsia-500 border-r-fuchsia-500 border-b-violet-500 border-l-violet-500 rounded-full animate-spin flex-shrink-0"></div>
                    <div className="text-left">
                      <h2 className="text-sm font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-violet-400 to-fuchsia-500 uppercase tracking-wider">
                        Analyse Haute Précision
                      </h2>
                      <p className="text-slate-300 text-xs mt-0.5 font-semibold line-clamp-1">{preciseAnalysisStatus}</p>
                    </div>
                  </div>
                  <div className="flex items-center gap-3 flex-1 max-w-xs justify-end">
                    <div className="w-full bg-[#050818]/80 rounded-full h-2 overflow-hidden border border-violet-500/20 shadow-inner">
                      <div 
                        className="bg-gradient-to-r from-violet-500 to-fuchsia-500 h-full rounded-full transition-all duration-300"
                        style={{ width: `${preciseAnalysisProgress}%` }}
                      ></div>
                    </div>
                    <span className="text-xs text-violet-300 font-bold tracking-wider flex-shrink-0">{preciseAnalysisProgress}%</span>
                  </div>
                </div>
              )}

              {/* Pause screen with silhouette replay */}
              {gameState === "pause" && lastDanceInfo && (
                <div className="absolute inset-0 bg-black/95 backdrop-blur-md flex flex-col md:flex-row items-center justify-center gap-8 rounded-2xl p-6 text-center animate-fade-in border border-violet-500/30 shadow-[0_0_50px_rgba(139,92,246,0.3)] z-50 overflow-y-auto">
                  <div className="flex flex-col items-center gap-4">
                    <div className="animate-bounce">
                      <span className="text-5xl">🏆</span>
                    </div>
                    <div>
                      <h2 className="text-2xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-violet-400 to-fuchsia-500 uppercase tracking-widest text-violet-neon">
                        Musique Terminée !
                      </h2>
                      <p className="text-slate-300 text-sm mt-1 font-semibold max-w-[285px] truncate">{lastDanceInfo.title}</p>
                    </div>

                    <div className="bg-[#050818]/60 border border-violet-500/30 rounded-2xl px-6 py-4 shadow-inner flex flex-col items-center gap-1 min-w-[200px]">
                      <span className="text-[10px] uppercase tracking-wider text-violet-400 font-bold">Votre Score</span>
                      <span className="text-5xl font-black text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-emerald-400 animate-pulse font-display">
                        {lastDanceInfo.score}
                      </span>
                      <span className="text-xs font-semibold text-slate-400 mt-1">
                        {lastDanceInfo.rank === "A" ? "🔥 RANG A - SUPER STAR !" :
                         lastDanceInfo.rank === "B" ? "✨ RANG B - TRÈS BIEN !" :
                                                      "👍 RANG C - BIEN JOUÉ !"}
                      </span>
                    </div>

                    <div className="flex flex-col items-center gap-1 mt-1">
                      <span className="text-[10px] uppercase tracking-wider text-slate-400 font-semibold">Prochaine danse dans</span>
                      <div className="flex items-center justify-center w-10 h-10 rounded-full border-2 border-fuchsia-500 text-fuchsia-400 font-black text-lg bg-fuchsia-500/10 shadow-[0_0_15px_rgba(217,70,239,0.4)]">
                        {pauseCountdown}
                      </div>
                    </div>
                  </div>

                  {poseReplayFrames.length > 0 && (
                    <PoseReplayCanvas frames={poseReplayFrames} />
                  )}
                </div>
              )}

              {/* Immersive Score HUD overlay inside the camera screen */}
              {activeFeature === "camera" && (gameState === "detecting" || gameState === "countdown" || gameState === "waiting_for_person") && (
                <div className="absolute bottom-8 left-8 right-8 z-40 flex items-center justify-between bg-black/65 backdrop-blur-md border border-fuchsia-500/30 rounded-2xl p-6 md:p-8 shadow-[0_0_40px_rgba(217,70,239,0.35)] gap-8 transition-all duration-300">
                  <div className="flex flex-col min-w-0 flex-1 text-left">
                    <span className="text-sm uppercase font-black text-slate-400 tracking-wider">
                      Chanson
                    </span>
                    <span className="font-display text-xl md:text-3.5xl text-cyan-neon truncate font-black drop-shadow-[0_0_10px_rgba(34,211,238,0.3)]">
                      {selectedDanceId
                        ? catalogue?.dances?.find((d) => d.id === selectedDanceId)?.title ?? "Détection..."
                        : "Recherche..."}
                    </span>
                  </div>

                  <div className="flex-1 max-w-xs flex flex-col gap-2 items-center">
                    <div className="flex items-center gap-2 justify-center">
                      <span className="text-xs uppercase font-extrabold text-slate-400 tracking-wider">
                        Durée :
                      </span>
                      {audioTimeLeft !== null ? (
                        <span className="text-2xl md:text-3xl font-black text-fuchsia-400 font-display animate-pulse drop-shadow-[0_0_10px_rgba(217,70,239,0.6)]">
                          {formatTime(audioTimeLeft)}
                        </span>
                      ) : (
                        <span className="text-slate-500 font-display text-sm">--:--</span>
                      )}
                    </div>
                    <div className="flex items-end gap-0.5 h-14 w-full bg-black/40 rounded-lg p-1.5 overflow-hidden justify-center shadow-inner">
                      {sessionPrecisionsRef.current.slice(-25).map((p, idx) => (
                        <div
                          key={idx}
                          className="w-1.5 rounded-t-sm"
                          style={{
                            height: `${p}%`,
                            backgroundColor: p > 60 ? '#10b981' : p >= 45 ? '#f59e0b' : '#ef4444',
                            boxShadow: `0 0 4px ${p > 60 ? '#10b981' : p >= 45 ? '#f59e0b' : '#ef4444'}`
                          }}
                        />
                      ))}
                    </div>
                  </div>

                  <div className="flex items-center gap-3.5 shrink-0">
                    <button
                      onClick={() => {
                        if (document.fullscreenElement === cameraContainerRef.current) {
                          document.exitFullscreen().catch((err) => console.warn(err));
                        } else {
                          cameraContainerRef.current?.requestFullscreen().catch((err) => console.warn(err));
                        }
                      }}
                      className="bg-violet-600/30 hover:bg-violet-600/60 border border-violet-500/40 hover:border-violet-500/80 rounded-xl p-3 flex items-center justify-center transition-all cursor-pointer shadow-md text-white select-none mr-1.5"
                      title={isFullscreen ? "Quitter le plein écran" : "Plein écran"}
                    >
                      {isFullscreen ? (
                        /* Exit fullscreen icon */
                        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth="2.5">
                          <path strokeLinecap="round" strokeLinejoin="round" d="M9 9h6m0 0v6m0-6L9 15m6-6l-6 6" />
                        </svg>
                      ) : (
                        /* Enter fullscreen icon */
                        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth="2.5">
                          <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 3.75v4.5m0-4.5h4.5m-4.5 0L9 9M3.75 20.25v-4.5m0 4.5h4.5m-4.5 0L9 15M20.25 3.75v4.5m0-4.5h-4.5m4.5 0L15 9m5.25 11.25v-4.5m0 4.5h-4.5m4.5 0L15 15" />
                        </svg>
                      )}
                    </button>
                    <div className="bg-[#050818]/80 border border-violet-500/20 rounded-lg px-5 py-2.5 flex flex-col items-center min-w-[110px]">
                      <span className="text-[10px] text-slate-400 uppercase font-bold tracking-wider">Précision</span>
                      <span className="text-xl md:text-2xl font-black text-cyan-400 font-display">
                        {Math.round(dancePrecision)}%
                      </span>
                    </div>
                    <div className="bg-[#050818]/80 border border-violet-500/20 rounded-lg px-5 py-2.5 flex flex-col items-center min-w-[110px]">
                      <span className="text-[10px] text-slate-400 uppercase font-bold tracking-wider">Score</span>
                      <span className="text-xl md:text-2xl font-black text-fuchsia-400 font-display">
                        {danceScore}%
                      </span>
                    </div>
                  </div>
                </div>
              )}
            </ImageDisplay>
          </section>

          {activeFeature && (
            <div className="mt-3 px-1">
              <div className="relative w-full h-1.5 bg-[#050818] rounded-full overflow-hidden border border-violet-500/20 shadow-inner">
                <div
                  className={`h-full rounded-full transition-all duration-500 ${
                    activeFeature === "loading"
                      ? "bg-gradient-to-r from-violet-500 via-fuchsia-500 to-violet-500 animate-pulse w-full"
                      : gameState === "detecting"
                        ? "bg-gradient-to-r from-violet-600 via-fuchsia-500 to-violet-600 animate-scanner-slow w-full"
                        : gameState === "waiting_for_person"
                          ? "bg-amber-600/30 w-full animate-pulse"
                          : "bg-slate-700/50 w-full"
                  }`}
                  style={{ backgroundSize: "200% 100%" }}
                ></div>
              </div>
              <div className="flex justify-between items-center mt-1.5 text-[10px] uppercase tracking-wider font-extrabold text-violet-300">
                <span className="flex items-center gap-1.5">
                  <span
                    className={`w-2 h-2 rounded-full ${
                      activeFeature === "loading" || gameState === "detecting" || gameState === "waiting_for_person"
                        ? "bg-fuchsia-500 animate-ping"
                        : "bg-slate-600"
                    }`}
                  ></span>
                  IA : {activeFeature === "loading"
                    ? "Initialisation"
                    : activeFeature === "video"
                      ? "Detection realtime sur video locale"
                      : gameState === "waiting_for_person"
                      ? "En attente d'un joueur..."
                      : gameState === "countdown"
                        ? "Compte a rebours"
                        : "Comparaison temporelle locale"}
                </span>
                {(gameState === "detecting" || gameState === "countdown") && (
                  <span className="text-cyan-400">Precision : {Math.round(dancePrecision)}%</span>
                )}
              </div>
            </div>
          )}

        <div className="card-violet flex flex-col gap-3">
          <h2 className="text-lg font-bold border-b border-violet-500/20 pb-2">
            <RainbowText text="Commandes" />
          </h2>
          <input
            ref={fileVideoRef}
            type="file"
            accept="video/*"
            hidden
            onChange={(event) => {
              handleVideoFile(event.target.files?.[0]);
              event.target.value = "";
            }}
          />
          <div className="flex flex-wrap gap-3">
            <button
              onClick={toggleCamera}
              className={`px-5 py-2.5 rounded-lg font-bold shadow-md transition-all cursor-pointer ${
                activeFeature === "camera"
                  ? "bg-red-600 hover:bg-red-500 text-white"
                  : "bg-violet-600 hover:bg-violet-500 text-white"
              }`}
              disabled={activeFeature === "loading"}
            >
              {activeFeature === "camera" ? "Arreter la camera" : "Activer la camera"}
            </button>
            <button
              onClick={() => {
                if (activeFeature === "video") {
                  stopVideo();
                } else {
                  fileVideoRef.current?.click();
                }
              }}
              className={`px-5 py-2.5 rounded-lg font-bold shadow-md transition-all cursor-pointer ${
                activeFeature === "video"
                  ? "bg-red-600 hover:bg-red-500 text-white"
                  : "bg-cyan-600 hover:bg-cyan-500 text-white"
              }`}
              disabled={activeFeature === "loading" || activeFeature === "camera"}
            >
              {activeFeature === "video" ? "Fermer la video" : "Tester une video"}
            </button>
            {activeFeature === "video" && (
              <button
                onClick={() => {
                  resetLiveComparison();
                  processMediaFrame(videoRef.current);
                }}
                className="px-5 py-2.5 rounded-lg bg-[#050818] hover:bg-violet-950 text-violet-200 border border-violet-500/30 font-bold transition-all cursor-pointer"
              >
                Reset analyse
              </button>
            )}
          </div>
          <label className="flex items-center justify-between gap-3 rounded-lg border border-cyan-500/20 bg-[#050818]/70 px-3 py-2">
            <span className="flex flex-col">
              <span className="text-sm font-bold text-slate-200">Debug pose cible</span>
              <span className="text-xs text-slate-500">Superpose la pose attendue en cyan/rose.</span>
            </span>
            <input
              type="checkbox"
              className="h-5 w-5 accent-cyan-400"
              checked={debugTargetOverlay}
              onChange={(event) => setDebugTargetOverlay(event.target.checked)}
            />
          </label>
          {activeFeature === "video" && (
            <p className="text-xs text-slate-400 truncate">
              Source video : {videoName || "video locale"}
            </p>
          )}
        </div>

        <div className="card-violet flex flex-col gap-3">
          <div className="flex items-center justify-between border-b border-violet-500/20 pb-2">
            <h2 className="text-lg font-bold">
              <RainbowText text="Historique des Scores" />
            </h2>
            {scoreHistory.length > 0 && (
              <button
                onClick={() => setScoreHistory([])}
                className="text-xs text-red-400 hover:text-red-300 transition-colors font-semibold uppercase tracking-wider cursor-pointer"
              >
                Effacer
              </button>
            )}
          </div>
          
          {scoreHistory.length === 0 ? (
            <p className="text-sm text-slate-400 text-center py-4 italic">
              Aucun score enregistré pour le moment. Complétez une danse pour commencer !
            </p>
          ) : (
            <div className="flex flex-col gap-2.5 max-h-[280px] overflow-y-auto pr-1">
              {scoreHistory.map((entry) => {
                let rankColor = "text-slate-400";
                let rankBg = "bg-slate-500/10 border-slate-500/20";
                if (entry.rank === "S") {
                  rankColor = "text-cyan-300 font-extrabold text-violet-neon";
                  rankBg = "bg-cyan-950/20 border-cyan-500/30";
                } else if (entry.rank === "A") {
                  rankColor = "text-emerald-400 font-extrabold";
                  rankBg = "bg-emerald-950/20 border-emerald-500/30";
                } else if (entry.rank === "B") {
                  rankColor = "text-amber-300 font-bold";
                  rankBg = "bg-amber-950/10 border-amber-500/25";
                } else if (entry.rank === "C") {
                  rankColor = "text-violet-300 font-semibold";
                  rankBg = "bg-violet-950/10 border-violet-500/20";
                }

                return (
                  <div
                    key={entry.id}
                    className="flex items-center justify-between gap-4 bg-[#050818]/60 border border-violet-500/10 rounded-xl p-3 hover:border-violet-500/30 transition-all duration-200"
                  >
                    <div className="flex flex-col min-w-0 flex-1">
                      <span className="font-semibold text-slate-100 truncate text-sm">
                        {entry.title}
                      </span>
                      <div className="flex items-center gap-3 text-xs text-slate-400 mt-1">
                        <span>⏱️ {entry.date}</span>
                        <span>•</span>
                        <span>🕺 {entry.samples} séquences</span>
                      </div>
                    </div>
                    
                    <div className="flex items-center gap-3 shrink-0">
                      <div className="flex flex-col items-end">
                        <span className="text-xs text-slate-400 uppercase font-bold">Score</span>
                        <span className="text-lg font-black text-violet-300 font-mono">
                          {entry.score}
                        </span>
                      </div>
                      
                      <div className={`w-10 h-10 rounded-lg border flex items-center justify-center text-lg font-black ${rankBg} ${rankColor}`}>
                        {entry.rank}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>

        <aside className="flex flex-col gap-4">
          <div className="card-violet">
            <div className="flex items-start justify-between gap-3">
              <div>
                <p className="text-xs uppercase tracking-wider text-slate-400 font-bold">
                  Detection temporelle
                </p>
                <h2 className="text-2xl font-black text-white mt-1">
                  {gameState === "waiting_for_person"
                    ? "Présentez-vous devant l'IA"
                    : gameState === "countdown"
                      ? `Depart dans ${countdown}`
                      : shownMatch
                        ? shownMatch.title
                        : "Aucune danse"}
                </h2>
              </div>
              <span className={`text-xl font-black ${performanceRating.color}`}>
                {performanceRating.text}
              </span>
            </div>

            <div className="mt-5">
              <div className="flex justify-between text-xs uppercase tracking-wider text-slate-400 font-bold mb-2">
                <span>Correspondance</span>
                <span>{dancePrecision}%</span>
              </div>
              <div className="w-full h-3 rounded-full overflow-hidden bg-[#050818] border border-violet-500/20">
                <div
                  className="h-full bg-gradient-to-r from-violet-500 to-emerald-400 transition-all duration-300"
                  style={{ width: `${dancePrecision}%` }}
                ></div>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-3 mt-5">
              <div className="bg-[#050818]/70 border border-violet-500/20 rounded-lg p-3">
                <p className="text-xs text-slate-400 uppercase font-bold">Score Moyen</p>
                <p className="text-2xl font-black text-violet-300">{danceScore}%</p>
              </div>
              <div className="bg-[#050818]/70 border border-violet-500/20 rounded-lg p-3">
                <p className="text-xs text-slate-400 uppercase font-bold">Sequence</p>
                <p className="text-2xl font-black text-violet-300">{sequenceSampleCount}</p>
              </div>
            </div>
          </div>

          <div className="card-violet">
            <h2 className="text-lg font-bold border-b border-violet-500/20 pb-2">
              <RainbowText text="Catalogue local" />
            </h2>
            <p
              className={`text-sm mt-3 ${
                catalogueStatus.state === "error" ? "text-red-300" : "text-slate-300"
              }`}
            >
              {catalogueStatus.message}
            </p>
            <div className="mt-4 flex flex-col gap-2">
              {(catalogue?.dances ?? []).map((dance) => {
                const isSelected = dance.id === selectedDanceId;
                return (
                  <div
                    key={dance.id}
                    onClick={() => setSelectedDanceId(isSelected ? null : dance.id)}
                    className={`flex items-center justify-between gap-3 bg-[#050818]/70 border rounded-lg px-3 py-2 cursor-pointer transition-all duration-200 ${
                      isSelected
                        ? "border-fuchsia-500 shadow-[0_0_12px_rgba(217,70,239,0.35)] bg-fuchsia-950/20"
                        : "border-violet-500/20 hover:border-violet-500/50"
                    }`}
                  >
                    <div className="flex flex-col min-w-0">
                      <span className="font-semibold text-slate-100 truncate">{dance.title}</span>
                      {dance.audioUrl && (
                        <span className="text-[10px] text-fuchsia-400 font-medium flex items-center gap-1 mt-0.5">
                          🎵 Musique liée
                        </span>
                      )}
                    </div>
                    <span className="text-xs text-slate-400 shrink-0">
                      {dance.detectedFrames}/{dance.sampledFrames}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>

          <div className="card-violet">
            <h2 className="text-lg font-bold border-b border-violet-500/20 pb-2">
              <RainbowText text="Candidats instantanes" />
            </h2>
            <div className="mt-4 flex flex-col gap-3">
              {instantCandidates.length === 0 ? (
                <p className="text-sm text-slate-400">
                  {gameState === "countdown"
                    ? "La comparaison commence apres le compte a rebours."
                    : "Pas encore assez de poses dans la sequence live."}
                </p>
              ) : (
                instantCandidates.map((candidate) => (
                  <div key={candidate.id} className="flex flex-col gap-2">
                    <div className="flex justify-between gap-3 text-sm">
                      <span className="font-semibold text-slate-100 truncate">
                        {candidate.title}
                      </span>
                      <span className="font-mono text-violet-300">
                        {candidate.score.toFixed(1)}%
                      </span>
                    </div>
                    <div className="h-2 rounded-full overflow-hidden bg-[#050818]">
                      <div
                        className="h-full bg-violet-500 transition-all duration-300"
                        style={{ width: `${candidate.score}%` }}
                      ></div>
                    </div>
                    <p className="text-xs text-slate-500">
                      segment {candidate.startTimestamp ?? "-"}s - {candidate.matchedSamples ?? 0} poses
                    </p>
                  </div>
                ))
              )}
            </div>
          </div>
        </aside>
      </main>

      <PoseOverlayTool catalogue={catalogue} />

      <footer className="mt-2 border-t border-violet-500/10 pt-4 flex flex-col items-center">
        <button
          onClick={() => setShowAdvancedSettings((value) => !value)}
          className="text-xs text-violet-400 hover:text-violet-300 transition-colors underline cursor-pointer"
        >
          {showAdvancedSettings ? "Hide advanced settings" : "Show advanced settings"}
        </button>

        {showAdvancedSettings && (
          <div className="w-full mt-4 animate-details-show">
            <SettingsPanel
              cameraSelectorRef={cameraSelectorRef}
              imgszTypeSelectorRef={imgszTypeSelectorRef}
              modelConfigRef={modelConfigRef}
              defaultModelConfig={DEFAULT_MODEL_CONFIG}
              customClasses={customClasses}
              cameras={cameras}
              activeFeature={activeFeature}
              defaultClasses={classes}
              loadModel={loadModel}
            />
            <ModelStatus
              warnUpTime={processingStatus.warnUpTime}
              inferenceTime={processingStatus.inferenceTime}
              statusMsg={processingStatus.statusMsg}
              statusColor={processingStatus.statusColor}
            />
          </div>
        )}
      </footer>
    </div>
  );
}

export default App;
