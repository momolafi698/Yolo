import "./assets/App.css";
import { useCallback, useEffect, useRef, useState } from "react";
import classes from "./utils/yolo_classes.json";
import { renderOverlay } from "./utils/render-overlay";
import {
  loadPoseCatalogue,
  matchPoseToCatalogue,
  stabilizeMatches,
} from "./utils/catalogue-matcher";

import { useInferenceWorker } from "./hooks/useInferenceWorker";
import { useWebcam } from "./hooks/useWebcam";

import ImageDisplay from "./components/ImageDisplay";
import ModelStatus from "./components/ModelStatus";
import SettingsPanel from "./components/SettingsPanel";

const DEFAULT_MODEL_CONFIG = {
  inputShape: [1, 3, 640, 640],
  overlaySize: [640, 640],
  iouThreshold: 0.35,
  scoreThreshold: 0.45,
  backend: "wasm",
  numThreads: 1,
  enableNMS: true,
  model: "yolo11n",
  modelPath: "",
  task: "pose",
  imgszType: "dynamic",
  classes,
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
  const [customModels] = useState([]);
  const [customClasses] = useState([]);
  const [imgSrc] = useState(null);
  const [details, setDetails] = useState([]);
  const [activeFeature, setActiveFeature] = useState(null);
  const [gameState, setGameState] = useState("idle");
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
  const [coachComments, setCoachComments] = useState([
    "Active la camera et reproduis un mouvement extrait d'une des deux videos.",
  ]);
  const [showAdvancedSettings, setShowAdvancedSettings] = useState(false);

  const modelConfigRef = useRef(DEFAULT_MODEL_CONFIG);
  const cameraSelectorRef = useRef(null);
  const imgszTypeSelectorRef = useRef(null);
  const imgRef = useRef(null);
  const overlayRef = useRef(null);
  const cameraRef = useRef(null);
  const activeFeatureRef = useRef(activeFeature);
  const isProcessingRef = useRef(false);
  const matchHistoryRef = useRef([]);
  const lastScoreTimeRef = useRef(0);
  const lastCommentRef = useRef("");
  const lastCommentTimeRef = useRef(0);

  useEffect(() => {
    activeFeatureRef.current = activeFeature;
  }, [activeFeature]);

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

  const handleInferenceResult = useCallback(
    (data) => {
      const overlayCtx = overlayRef.current?.getContext("2d");
      if (!overlayCtx) {
        isProcessingRef.current = false;
        return;
      }

      overlayCtx.clearRect(0, 0, overlayCtx.canvas.width, overlayCtx.canvas.height);
      renderOverlay(
        data.results,
        data.maskImageData,
        overlayCtx,
        DEFAULT_MODEL_CONFIG.task,
        DEFAULT_MODEL_CONFIG.classes,
      );

      setDetails(data.results);
      setProcessingStatus((previous) => ({
        ...previous,
        inferenceTime: data.inferenceTime,
      }));

      const firstPose = data.results?.[0];
      if (firstPose?.keypoints && catalogue) {
        const instantMatch = matchPoseToCatalogue(firstPose, catalogue);
        const now = performance.now();
        const stabilized = stabilizeMatches(
          matchHistoryRef.current,
          instantMatch,
          now,
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
          pushCoachComment(`Mouvement reconnu: ${displayMatch.title} (${precision}%).`);

          if (now - lastScoreTimeRef.current > 1200) {
            setDanceScore((previous) => previous + precision);
            lastScoreTimeRef.current = now;
          }
        } else if (displayMatch && precision >= 52) {
          setPerformanceRating({
            text: "PROBABLE",
            color: "text-amber-300 font-bold",
          });
          pushCoachComment(`Le plus proche semble etre ${displayMatch.title}, continue le mouvement.`);
        } else {
          setPerformanceRating({
            text: "CHERCHE",
            color: "text-violet-300 font-bold animate-pulse",
          });
        }
      } else {
        setDancePrecision(0);
        setCurrentMatch({ best: null, candidates: [], detected: false, margin: 0 });
        setStableMatch(null);
        matchHistoryRef.current = [];
        if (gameState === "detecting") {
          setPerformanceRating({
            text: "EN ATTENTE",
            color: "text-slate-500",
          });
        }
      }

      isProcessingRef.current = false;
    },
    [catalogue, gameState, pushCoachComment],
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

  const startCameraLoop = useCallback(() => {
    const loop = async () => {
      if (activeFeatureRef.current !== "camera") return;

      if (
        !isProcessingRef.current &&
        cameraRef.current &&
        cameraRef.current.readyState >= 2
      ) {
        isProcessingRef.current = true;

        try {
          const bitmap = await createImageBitmap(cameraRef.current);

          if (
            overlayRef.current.width !== cameraRef.current.videoWidth ||
            overlayRef.current.height !== cameraRef.current.videoHeight
          ) {
            overlayRef.current.width = cameraRef.current.videoWidth;
            overlayRef.current.height = cameraRef.current.videoHeight;
          }

          DEFAULT_MODEL_CONFIG.overlaySize = [
            overlayRef.current.width,
            overlayRef.current.height,
          ];

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
      }

      requestAnimationFrame(loop);
    };

    loop();
  }, [postInferenceMessage]);

  const resetDetection = useCallback(() => {
    matchHistoryRef.current = [];
    setCurrentMatch({ best: null, candidates: [], detected: false, margin: 0 });
    setStableMatch(null);
    setDanceScore(0);
    setDancePrecision(0);
    setPerformanceRating({ text: "CHERCHE", color: "text-violet-300 font-bold" });
    setCoachComments([
      "Detection remise a zero. Reproduis un mouvement d'une video du catalogue.",
    ]);
  }, []);

  const toggleCamera = useCallback(async () => {
    if (activeFeature === "camera") {
      closeCamera();
      if (overlayRef.current) {
        overlayRef.current.width = 0;
        overlayRef.current.height = 0;
      }
      setActiveFeature(null);
      setDetails([]);
      setGameState("idle");
      setPerformanceRating({ text: "PRET", color: "text-slate-400" });
      return;
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
      resetDetection();
      setActiveFeature("camera");
      setGameState("detecting");
    }
  }, [activeFeature, closeCamera, getCameras, openCamera, resetDetection]);

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
        <section className="relative">
          <ImageDisplay
            cameraRef={cameraRef}
            imgRef={imgRef}
            overlayRef={overlayRef}
            imgSrc={imgSrc}
            onCameraLoad={handleCameraLoad}
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
                        : "bg-slate-700/50 w-full"
                  }`}
                  style={{ backgroundSize: "200% 100%" }}
                ></div>
              </div>
              <div className="flex justify-between items-center mt-1.5 text-[10px] uppercase tracking-wider font-extrabold text-violet-300">
                <span className="flex items-center gap-1.5">
                  <span
                    className={`w-2 h-2 rounded-full ${
                      activeFeature === "loading" || gameState === "detecting"
                        ? "bg-fuchsia-500 animate-ping"
                        : "bg-slate-600"
                    }`}
                  ></span>
                  IA : {activeFeature === "loading" ? "Initialisation" : "Comparaison catalogue local"}
                </span>
                {gameState === "detecting" && (
                  <span className="text-cyan-400">Precision : {Math.round(dancePrecision)}%</span>
                )}
              </div>
            </div>
          )}
        </section>

        <aside className="flex flex-col gap-4">
          <div className="card-violet">
            <div className="flex items-start justify-between gap-3">
              <div>
                <p className="text-xs uppercase tracking-wider text-slate-400 font-bold">
                  Detection stable
                </p>
                <h2 className="text-2xl font-black text-white mt-1">
                  {shownMatch ? shownMatch.title : "Aucune danse"}
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
                <p className="text-xs text-slate-400 uppercase font-bold">Squelette</p>
                <p className="text-2xl font-black text-violet-300">{details.length}</p>
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
              {(catalogue?.dances ?? []).map((dance) => (
                <div
                  key={dance.id}
                  className="flex items-center justify-between gap-3 bg-[#050818]/70 border border-violet-500/20 rounded-lg px-3 py-2"
                >
                  <span className="font-semibold text-slate-100 truncate">{dance.title}</span>
                  <span className="text-xs text-slate-400 shrink-0">
                    {dance.detectedFrames}/{dance.sampledFrames}
                  </span>
                </div>
              ))}
            </div>
          </div>

          <div className="card-violet">
            <h2 className="text-lg font-bold text-violet-300 border-b border-violet-500/20 pb-2">
              Candidats instantanes
            </h2>
            <div className="mt-4 flex flex-col gap-3">
              {candidates.length === 0 ? (
                <p className="text-sm text-slate-400">
                  Aucune pose camera comparable pour le moment.
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
                      frame {candidate.frameIndex ?? "-"} - {candidate.comparableKeypoints} points
                    </p>
                  </div>
                ))
              )}
            </div>
          </div>
        </aside>
      </main>

      <section className="captor-grid">
        <div className="card-violet flex flex-col gap-3">
          <h2 className="text-lg font-bold text-violet-300 border-b border-violet-500/20 pb-2">
            Commandes
          </h2>
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
              onClick={resetDetection}
              className="px-5 py-2.5 rounded-lg bg-[#050818] hover:bg-violet-950 text-violet-200 border border-violet-500/30 font-bold transition-all cursor-pointer"
              disabled={activeFeature !== "camera"}
            >
              Reinitialiser
            </button>
          </div>
        </div>

        <div className="card-violet flex flex-col gap-3 xl:col-span-2">
          <h2 className="text-lg font-bold text-violet-300 border-b border-violet-500/20 pb-2">
            Journal
          </h2>
          <div className="flex flex-col gap-2">
            {coachComments.map((comment, index) => (
              <div
                key={`${comment}-${index}`}
                className="bg-[#050818]/60 border-l-2 border-violet-500 px-3 py-2 text-sm text-slate-300 rounded-r-md animate-details-show"
              >
                {comment}
              </div>
            ))}
          </div>
        </div>
      </section>

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
              customClasses={customClasses}
              cameras={cameras}
              customModels={customModels}
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
