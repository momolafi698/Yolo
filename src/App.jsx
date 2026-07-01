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
  const activeFeatureRef = useRef(activeFeature);
  const gameStateRef = useRef(gameState);
  const isProcessingRef = useRef(false);
  const mediaLoopTokenRef = useRef(0);
  const matchHistoryRef = useRef([]);
  const liveSequenceRef = useRef([]);
  const sequenceStartRef = useRef(0);
  const lastScoreTimeRef = useRef(0);
  const lastCommentRef = useRef("");
  const lastCommentTimeRef = useRef(0);
  const currentAudioRef = useRef(null);
  const audioTimeoutRef = useRef(null);

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
          return 0;
        }

        return current - 1;
      });
    }, 1000);

    return () => window.clearInterval(interval);
  }, [gameState]);

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

  const resetLiveComparison = useCallback(() => {
    matchHistoryRef.current = [];
    liveSequenceRef.current = [];
    sequenceStartRef.current = performance.now();
    setCurrentMatch({ best: null, candidates: [], detected: false, margin: 0 });
    setStableMatch(null);
    setDancePrecision(0);
    setSequenceSampleCount(0);
  }, []);

  const prepareCountdown = useCallback(() => {
    resetLiveComparison();
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
  }, [resetLiveComparison]);

  const stopMusic = useCallback(() => {
    if (audioTimeoutRef.current) {
      clearTimeout(audioTimeoutRef.current);
      audioTimeoutRef.current = null;
    }
    if (currentAudioRef.current) {
      currentAudioRef.current.pause();
      currentAudioRef.current.currentTime = 0;
      currentAudioRef.current = null;
    }
  }, []);

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
      }
    }

    if (selectedDance && selectedDance.audioUrl) {
      const baseUrl = import.meta.env.BASE_URL || "/";
      const audioPath = `${baseUrl.replace(/\/+$/, "")}/${selectedDance.audioUrl.replace(/^\/+/, "")}`;
      const audio = new Audio(audioPath);
      audio.loop = false;
      audio.volume = 0.4;

      audio.addEventListener("ended", () => {
        audioTimeoutRef.current = setTimeout(() => {
          const dancesWithAudio = catalogue.dances.filter((d) => d.audioUrl);
          const currentIndex = dancesWithAudio.findIndex((d) => d.id === selectedDance.id);
          if (currentIndex !== -1 && currentIndex < dancesWithAudio.length - 1) {
            const nextDance = dancesWithAudio[currentIndex + 1];
            setSelectedDanceId(nextDance.id);
          } else {
            stopMusic();
          }
        }, 3000);
      });

      audio.play().catch((err) => {
        console.warn("Autoplay block or music play failed:", err);
      });
      currentAudioRef.current = audio;
    }
  }, [catalogue, selectedDanceId, stopMusic, setSelectedDanceId]);

  useEffect(() => {
    if (activeFeature === "camera") {
      playMusicForCamera();
    } else {
      stopMusic();
    }
    return () => {
      stopMusic();
    };
  }, [activeFeature, selectedDanceId, playMusicForCamera, stopMusic]);

  const handleInferenceResult = useCallback(
    (data) => {
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
          const elapsedSeconds = activeFeatureRef.current === "video"
            ? (videoRef.current?.currentTime ?? 0)
            : (now - sequenceStartRef.current) / 1000;
          const sample = createPoseSample(targetPose, elapsedSeconds);

          if (sample) {
            liveSequenceRef.current = liveSequenceRef.current
              .concat(sample)
              .filter(
                (entry) => elapsedSeconds - entry.timestamp <= SEQUENCE_WINDOW_SECONDS,
              );
            setSequenceSampleCount(liveSequenceRef.current.length);
          }

          const instantMatch = matchPoseSequenceToCatalogue(
            liveSequenceRef.current,
            catalogue,
            {
              sequenceWindowSeconds: SEQUENCE_WINDOW_SECONDS,
            },
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
          setDancePrecision(precision);

          if (displayMatch && precision >= 70) {
            setPerformanceRating({
              text: "MATCH",
              color: "text-emerald-400 font-extrabold",
            });
            pushCoachComment(
              `Sequence reconnue: ${displayMatch.title} (${precision}%, ${displayMatch.matchedSamples} poses).`,
            );

            if (now - lastScoreTimeRef.current > 1200) {
              setDanceScore((previous) => previous + precision);
              lastScoreTimeRef.current = now;
            }
          } else if (displayMatch && precision >= 52) {
            setPerformanceRating({
              text: "PROBABLE",
              color: "text-amber-300 font-bold",
            });
            pushCoachComment(`Le mouvement ressemble a ${displayMatch.title}, continue dans le tempo.`);
          } else {
            setPerformanceRating({
              text: "CHERCHE",
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
    [catalogue, countdown, prepareCountdown, pushCoachComment],
  );

  const handleModelLoaded = useCallback((data) => {
    setProcessingStatus((previous) => ({
      ...previous,
      statusMsg: data.msg,
      statusColor: "green",
      warnUpTime: data.loadTime,
    }));
    setActiveFeature(null);
  }, []);

  const handleModelLoadError = useCallback((data) => {
    setProcessingStatus((previous) => ({
      ...previous,
      statusMsg: data.msg,
      statusColor: "red",
    }));
    setActiveFeature(null);
  }, []);

  const { postMessage: postInferenceMessage } = useInferenceWorker({
    onModelLoaded: handleModelLoaded,
    onResult: handleInferenceResult,
    onError: handleModelLoadError,
  });

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

      const bitmap = await createImageBitmap(mediaElement, {
        resizeWidth: inferW,
        resizeHeight: inferH,
        resizeQuality: "low",
      });

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
  }, [postInferenceMessage]);

  const stopMediaLoop = useCallback(() => {
    mediaLoopTokenRef.current += 1;
  }, []);

  const startMediaLoop = useCallback((featureName, mediaRef) => {
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

  const startVideoLoop = useCallback(() => {
    startMediaLoop("video", videoRef);
  }, [startMediaLoop]);

  const handleVideoFile = useCallback((file) => {
    if (!file) return;

    if (activeFeatureRef.current === "camera") {
      closeCamera();
    }

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
  }, [closeCamera, resetLiveComparison, stopMediaLoop]);

  const stopVideo = useCallback(() => {
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
  }, [resetLiveComparison, stopMediaLoop]);

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
    setPerformanceRating({
      text: "FIN",
      color: "text-slate-400 font-black",
    });
  }, [stopMediaLoop]);

  const toggleCamera = useCallback(async () => {
    if (activeFeature === "camera") {
      stopMediaLoop();
      closeCamera();
      if (overlayRef.current) {
        overlayRef.current.width = 0;
        overlayRef.current.height = 0;
      }
      setActiveFeature(null);
      setGameState("idle");
      setCountdown(COUNTDOWN_SECONDS);
      setSequenceSampleCount(0);
      liveSequenceRef.current = [];
      setPerformanceRating({ text: "PRET", color: "text-slate-400" });
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
      setActiveFeature("camera");
      setGameState("waiting_for_person");
      setPerformanceRating({ text: "ATTENTE", color: "text-amber-500 font-bold animate-pulse" });
      setCoachComments(["Caméra activée. Présentez-vous devant l'écran pour lancer la détection."]);
    }
  }, [activeFeature, closeCamera, getCameras, openCamera, stopMediaLoop, stopVideo]);

  const handleCameraLoad = useCallback(() => {
    startCameraLoop();
  }, [startCameraLoop]);

  const imageLoad = useCallback(() => {}, []);

  const candidates = currentMatch.candidates.slice(0, 2);
  const shownMatch = stableMatch ?? currentMatch.best;

  return (
    <div className="max-w-7xl mx-auto px-4 py-6 text-white min-h-screen flex flex-col gap-6">
      <header className="text-center">
        <h1 className="text-4xl md:text-5xl font-extrabold tracking-wider uppercase bg-gradient-to-r from-violet-400 via-fuchsia-500 to-violet-600 bg-clip-text text-transparent text-violet-neon">
          Just Dance Captor
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
          />

          {activeFeature === "loading" && (
            <div className="absolute inset-0 bg-black/70 flex items-center justify-center flex-col gap-3 rounded-2xl">
              <div className="w-12 h-12 border-4 border-violet-500 border-t-transparent rounded-full animate-spin"></div>
              <span className="text-violet-300 font-bold">Chargement du modele...</span>
            </div>
          )}

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
        </section>

        <div className="card-violet flex flex-col gap-3">
          <h2 className="text-lg font-bold text-violet-300 border-b border-violet-500/20 pb-2">
            Commandes
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
            <button
              onClick={() => {
                if (activeFeature === "video") {
                  resetLiveComparison();
                  processMediaFrame(videoRef.current);
                } else {
                  prepareCountdown();
                }
              }}
              className="px-5 py-2.5 rounded-lg bg-[#050818] hover:bg-violet-950 text-violet-200 border border-violet-500/30 font-bold transition-all cursor-pointer"
              disabled={activeFeature !== "camera" && activeFeature !== "video"}
            >
              {activeFeature === "video" ? "Reset analyse" : "Relancer 5s"}
            </button>
          </div>
          {activeFeature === "video" && (
            <p className="text-xs text-slate-400 truncate">
              Source video : {videoName || "video locale"}
            </p>
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
                <p className="text-xs text-slate-400 uppercase font-bold">Score</p>
                <p className="text-2xl font-black text-violet-300">{danceScore}</p>
              </div>
              <div className="bg-[#050818]/70 border border-violet-500/20 rounded-lg p-3">
                <p className="text-xs text-slate-400 uppercase font-bold">Sequence</p>
                <p className="text-2xl font-black text-violet-300">{sequenceSampleCount}</p>
              </div>
            </div>
          </div>

          <div className="card-violet">
            <h2 className="text-lg font-bold text-violet-300 border-b border-violet-500/20 pb-2">
              Catalogue local
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
            <h2 className="text-lg font-bold text-violet-300 border-b border-violet-500/20 pb-2">
              Candidats instantanes
            </h2>
            <div className="mt-4 flex flex-col gap-3">
              {candidates.length === 0 ? (
                <p className="text-sm text-slate-400">
                  {gameState === "countdown"
                    ? "La comparaison commence apres le compte a rebours."
                    : "Pas encore assez de poses dans la sequence live."}
                </p>
              ) : (
                candidates.map((candidate) => (
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
