import "./assets/App.css";
import { useEffect, useRef, useState, useCallback } from "react";
import classes from "./utils/yolo_classes.json";
import { renderOverlay } from "./utils/render-overlay";
import { evaluateUserPose, detectDanceStyle, searchDanceVideoOnWeb } from "./utils/pose-comparison";

// Hooks
import { useInferenceWorker } from "./hooks/useInferenceWorker";
import { useWebcam } from "./hooks/useWebcam";
import { useVideoProcessWorker } from "./hooks/useVideoProcessWorker";

// Components
import SettingsPanel from "./components/SettingsPanel";
import ImageDisplay from "./components/ImageDisplay";
import ControlButtons from "./components/ControlButtons";
import ModelStatus from "./components/ModelStatus";
import ResultsTable from "./components/ResultsTable";

const DEFAULT_MODEL_CONFIG = {
  inputShape: [1, 3, 640, 640],
  overlaySize: [640, 640],
  iouThreshold: 0.35,
  scoreThreshold: 0.45,
  backend: "webgpu",
  numThreads: 1,
  enableNMS: false,
  model: "yolo26s",
  modelPath: "",
  task: "pose",
  imgszType: "dynamic",
  classes: classes,
};

function App() {
  // --- State ---
  const [processingStatus, setProcessingStatus] = useState({
    warnUpTime: 0,
    inferenceTime: 0,
    statusMsg: "Model not loaded",
    statusColor: "inherit",
  });

  const [customModels, setCustomModels] = useState([]);
  const [customClasses, setCustomClasses] = useState([]);
  const [imgSrc, setImgSrc] = useState(null);
  const [details, setDetails] = useState([]);
  const [activeFeature, setActiveFeature] = useState(null); // null, 'video', 'image', 'camera', 'loading'
  
  // --- States pour Just Dance Captor ---
  const [danceScore, setDanceScore] = useState(0);
  const [dancePrecision, setDancePrecision] = useState(0);
  const [performanceRating, setPerformanceRating] = useState({ text: "PRÊT", color: "text-gray-400" });
  const [coachComments, setCoachComments] = useState(["Faites des mouvements pour que l'IA détecte votre danse !"]);
  const [referenceVideoId, setReferenceVideoId] = useState(""); // Vide au départ, recherché en direct
  
  // Nouveaux states de détection automatique
  const [isSearchingWeb, setIsSearchingWeb] = useState(false);
  const [detectedDanceName, setDetectedDanceName] = useState(null);
  
  // Cycle de jeu Just Dance Captor : 'idle', 'detecting', 'playing', 'ia_lost'
  const [gameState, setGameState] = useState("idle");
  const [detectionTimeLeft, setDetectionTimeLeft] = useState(30);

  // --- Refs ---
  const modelConfigRef = useRef(DEFAULT_MODEL_CONFIG);
  const cameraSelectorRef = useRef(null);
  const imgszTypeSelectorRef = useRef(null);

  const imgRef = useRef(null);
  const overlayRef = useRef(null);
  const cameraRef = useRef(null);
  const fileImageRef = useRef(null);
  const fileVideoRef = useRef(null);
  const activeFeatureRef = useRef(activeFeature);
  const isProcessingRef = useRef(false);
  const lastScoreTimeRef = useRef(0);
  
  const referenceVideoIdRef = useRef(referenceVideoId);
  const isSearchingWebRef = useRef(false);
  const gameStateRef = useRef(gameState);
  
  // Plateforme sélectionnée ('youtube' ou 'tiktok')
  const [danceSourcePlatform, setDanceSourcePlatform] = useState("youtube");
  const danceSourcePlatformRef = useRef(danceSourcePlatform);

  useEffect(() => {
    referenceVideoIdRef.current = referenceVideoId;
  }, [referenceVideoId]);

  useEffect(() => {
    gameStateRef.current = gameState;
  }, [gameState]);

  useEffect(() => {
    danceSourcePlatformRef.current = danceSourcePlatform;
  }, [danceSourcePlatform]);

  // Sync activeFeatureRef
  useEffect(() => {
    activeFeatureRef.current = activeFeature;
  }, [activeFeature]);

  // --- Callbacks for Workers ---

  const handleInferenceResult = useCallback((data) => {
    // Determine context for drawing
    const overlayCtx = overlayRef.current?.getContext("2d");
    if (!overlayCtx) return;

    // Clear and draw
    overlayCtx.clearRect(
      0,
      0,
      overlayCtx.canvas.width,
      overlayCtx.canvas.height,
    );

    renderOverlay(
      data.results,
      data.maskImageData,
      overlayCtx,
      DEFAULT_MODEL_CONFIG.task,
      DEFAULT_MODEL_CONFIG.classes,
    );

    setDetails(data.results);
    setProcessingStatus((prev) => ({
      ...prev,
      inferenceTime: data.inferenceTime,
    }));

    // --- Évaluation de Danse (Just Dance Captor avec Détection & Recherche Web) ---
    if (data.results && data.results.length > 0) {
      const firstResult = data.results[0];
      if (firstResult.keypoints) {
        
        // A. Si la danse n'est pas encore verrouillée, l'IA cherche le style sur vos mouvements
        if (gameStateRef.current === "detecting") {
          const styleName = detectDanceStyle(firstResult.keypoints);
          
          if (styleName && !isSearchingWebRef.current) {
            // Verrouiller la recherche réseau pour éviter des requêtes multiples
            isSearchingWebRef.current = true;
            setIsSearchingWeb(true);
            setDetectedDanceName(styleName);
            
            setCoachComments((prev) => [
              `🔍 IA : Style "${styleName}" détecté ! Recherche de la vidéo sur le web...`,
              ...prev
            ].slice(0, 5));
            
            // Lancer la recherche web
            searchDanceVideoOnWeb(styleName, danceSourcePlatformRef.current).then((result) => {
              if (result && result.videoId) {
                setReferenceVideoId(result.videoId);
                setGameState("playing");
                setDanceScore(0);
                setDancePrecision(0);
                setCoachComments((prev) => [
                  `🎵 Vidéo trouvée sur le Web : "${result.title}" ! Début de la danse !`,
                  ...prev
                ].slice(0, 5));
              } else {
                setCoachComments((prev) => [
                  `❌ Impossible de trouver une vidéo pour "${styleName}" sur le web.`,
                  ...prev
                ].slice(0, 5));
              }
              setIsSearchingWeb(false);
              isSearchingWebRef.current = false;
            }).catch((err) => {
              console.error("Erreur de recherche web :", err);
              setCoachComments((prev) => [
                `❌ Erreur réseau lors de la recherche : ${err.message}`,
                ...prev
              ].slice(0, 5));
              setIsSearchingWeb(false);
              isSearchingWebRef.current = false;
            });
          } else if (!isSearchingWebRef.current) {
            setPerformanceRating({ text: "RECHERCHE", color: "text-violet-400 font-bold animate-pulse" });
            setDancePrecision(0);
          }
        } 
        // B. Si la danse est détectée et verrouillée, l'IA évalue vos mouvements et compte les points
        else if (gameStateRef.current === "playing") {
          const evaluation = evaluateUserPose(firstResult.keypoints, referenceVideoIdRef.current);
          
          // Mettre à jour le taux de précision et le badge
          setDancePrecision(evaluation.precision);
          setPerformanceRating(evaluation.rating);
          
          // Mettre à jour les commentaires du coach
          if (evaluation.comment) {
            setCoachComments((prev) => {
              if (prev[0] === evaluation.comment) return prev;
              return [evaluation.comment, ...prev].slice(0, 5);
            });
          }
          
          // Ajouter les points de score de manière temporisée (toutes les 1.2s max)
          const now = performance.now();
          if (evaluation.scoreGained > 0 && now - lastScoreTimeRef.current > 1200) {
            setDanceScore((prev) => prev + evaluation.scoreGained);
            lastScoreTimeRef.current = now;
          }
        }
      }
    } else {
      setDancePrecision(0);
      if (gameStateRef.current === "playing") {
        setPerformanceRating({ text: "EN ATTENTE", color: "text-slate-500" });
      } else if (gameStateRef.current === "detecting") {
        setPerformanceRating({ text: "RECHERCHE", color: "text-violet-400 font-bold animate-pulse" });
      } else {
        setPerformanceRating({ text: "PRÊT", color: "text-slate-500" });
      }
    }

    // Mark processing as done so loop can continue
    isProcessingRef.current = false;
  }, []);

  const handleModelLoaded = useCallback((data) => {
    setProcessingStatus((prev) => ({
      ...prev,
      statusMsg: data.msg,
      statusColor: "green",
      warnUpTime: data.loadTime,
    }));
    setActiveFeature(null);
  }, []);

  const handleModelLoadError = useCallback((data) => {
    setProcessingStatus((prev) => ({
      ...prev,
      statusMsg: data.msg,
      statusColor: "red",
    }));
    setActiveFeature(null);
  }, []);

  const handleVideoStatusKey = useCallback((msg) => {
    setProcessingStatus((prev) => ({ ...prev, statusMsg: msg }));
  }, []);

  const handleVideoComplete = useCallback((blob) => {
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "processed_video.mp4";
    a.click();
    URL.revokeObjectURL(url);
    setActiveFeature(null);
  }, []);

  // --- Hooks ---

  const { postMessage: postInferenceMessage } = useInferenceWorker({
    onModelLoaded: handleModelLoaded,
    onResult: handleInferenceResult,
    onError: handleModelLoadError,
  });

  const { cameras, getCameras, openCamera, closeCamera, cameraStatus } =
    useWebcam(cameraRef);

  // Sync camera status to processing status
  useEffect(() => {
    if (cameraStatus.msg) {
      setProcessingStatus((prev) => ({
        ...prev,
        statusMsg: cameraStatus.msg,
        statusColor: cameraStatus.color,
      }));
    }
  }, [cameraStatus]);

  // Gestion du chronomètre de détection de l'IA
  useEffect(() => {
    if (gameState !== "detecting") return;

    const interval = setInterval(() => {
      setDetectionTimeLeft((prev) => {
        if (prev <= 1) {
          clearInterval(interval);
          setGameState("ia_lost");
          closeCamera(); // Éteindre la caméra à la défaite
          if (overlayRef.current) {
            overlayRef.current.width = 0;
            overlayRef.current.height = 0;
          }
          setActiveFeature(null);
          setDetails([]);
          return 0;
        }
        return prev - 1;
      });
    }, 1000);

    return () => clearInterval(interval);
  }, [gameState, closeCamera]);

  const { processVideo } = useVideoProcessWorker({
    onStatusUpdate: handleVideoStatusKey,
    onComplete: handleVideoComplete,
  });

  // --- Logic ---

  const loadModel = useCallback(async () => {
    setProcessingStatus((prev) => ({
      ...prev,
      statusMsg: "Loading model...",
      statusColor: "red",
    }));
    setActiveFeature("loading");

    const customModel = customModels.find(
      (model) => model.url === DEFAULT_MODEL_CONFIG.model,
    );

    const modelPath = customModel
      ? customModel.url
      : `${import.meta.env.BASE_URL}models/${DEFAULT_MODEL_CONFIG.model}-${DEFAULT_MODEL_CONFIG.task}.onnx`;

    DEFAULT_MODEL_CONFIG.modelPath = modelPath;

    postInferenceMessage({
      type: "LOAD_MODEL",
      config: DEFAULT_MODEL_CONFIG,
    });
  }, [customModels, postInferenceMessage]);

  const imageLoad = useCallback(async () => {
    if (!imgRef.current) return;
    overlayRef.current.width = imgRef.current.naturalWidth;
    overlayRef.current.height = imgRef.current.naturalHeight;

    DEFAULT_MODEL_CONFIG.overlaySize = [
      overlayRef.current.width,
      overlayRef.current.height,
    ];

    const bitmap = await createImageBitmap(imgRef.current);
    postInferenceMessage(
      {
        type: "INFERENCE",
        config: DEFAULT_MODEL_CONFIG,
        bitmap: bitmap,
      },
      [bitmap],
    );
  }, [postInferenceMessage]);

  // Initial load
  useEffect(() => {
    loadModel();
  }, []);

  // Camera Loop
  const startCameraLoop = useCallback(() => {
    const loop = async () => {
      if (activeFeatureRef.current !== "camera") return;

      if (
        !isProcessingRef.current &&
        cameraRef.current &&
        cameraRef.current.readyState >= 2
      ) {
        isProcessingRef.current = true;

        // Create bitmap from camera
        try {
          const bitmap = await createImageBitmap(cameraRef.current);

          // Adjust overlay size if needed
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
              bitmap: bitmap,
            },
            [bitmap],
          );
        } catch (e) {
          console.error("Frame capture error:", e);
          isProcessingRef.current = false;
        }
      }
      requestAnimationFrame(loop);
    };
    loop();
  }, [postInferenceMessage]);

  // Handlers
  const handleAddModel = useCallback((event) => {
    const file = event.target.files[0];
    if (file) {
      const fileName = file.name.replace(".onnx", "");
      const fileUrl = URL.createObjectURL(file);
      setCustomModels((prev) => [...prev, { name: fileName, url: fileUrl }]);
    }
  }, []);

  const handleAddClassesFile = useCallback((event) => {
    const file = event.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const jsonData = JSON.parse(e.target.result);
        const fileName = file.name.replace(/\.json$/i, "");
        setCustomClasses((prev) => [
          ...prev,
          { name: fileName, data: jsonData },
        ]);
        setProcessingStatus((prev) => ({
          ...prev,
          statusMsg: `Classes file "${fileName}" loaded successfully`,
          statusColor: "green",
        }));
      } catch (error) {
        setProcessingStatus((prev) => ({
          ...prev,
          statusMsg: error.message || "Error parsing JSON file",
          statusColor: "red",
        }));
      }
    };
    reader.readAsText(file);
  }, []);

  const handleOpenImage = useCallback(
    (imgUrl = null) => {
      if (imgUrl) {
        setImgSrc(imgUrl);
        setActiveFeature("image");
      } else if (imgSrc) {
        if (imgSrc.startsWith("blob:")) {
          URL.revokeObjectURL(imgSrc);
        }
        if (overlayRef.current) {
          overlayRef.current.width = 0;
          overlayRef.current.height = 0;
        }
        setImgSrc(null);
        setDetails([]);
        setActiveFeature(null);
      }
    },
    [imgSrc],
  );

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
    } else {
      const camerasList = await getCameras();
      if (camerasList.length > 0) {
        const selectedDeviceId = cameraSelectorRef.current
          ? cameraSelectorRef.current.value
          : camerasList[0].deviceId;
        const success = await openCamera(selectedDeviceId);
        if (success) {
          setActiveFeature("camera");
          setGameState("detecting");
          setDetectionTimeLeft(30);
          setCoachComments(["L'IA analyse votre danse ! Bougez face à l'écran."]);
        }
      } else {
        setProcessingStatus((prev) => ({
          ...prev,
          statusMsg: "No cameras found",
          statusColor: "red",
        }));
      }
    }
  }, [activeFeature, closeCamera, getCameras, openCamera]);

  const handleCameraLoad = useCallback(() => {
    startCameraLoop();
  }, [startCameraLoop]);

  const handleOpenVideo = useCallback(
    (file) => {
      if (file) {
        processVideo(file, DEFAULT_MODEL_CONFIG);
        setActiveFeature("video");
      }
    },
    [processVideo],
  );

  // Variable pour déplier/replier les paramètres avancés
  const [showAdvancedSettings, setShowAdvancedSettings] = useState(false);

  return (
    <div className="max-w-7xl mx-auto px-4 py-6 text-white min-h-screen flex flex-col gap-6">
      {/* ZONE HAUTE : Titre centré */}
      <header className="text-center">
        <h1 className="text-4xl md:text-5xl font-extrabold tracking-wider uppercase bg-gradient-to-r from-violet-400 via-fuchsia-500 to-violet-600 bg-clip-text text-transparent text-violet-neon">
          Just dance captor
        </h1>
        <p className="text-slate-400 text-sm mt-2">
          Dansez face à la caméra et comparez votre style en temps réel !
        </p>
      </header>

      {/* ZONE CENTRALE : Webcam et contrôles rapides */}
      <main className="flex flex-col items-center gap-4 bg-[#0a0f26]/80 border border-violet-500/20 rounded-2xl p-6 shadow-xl shadow-violet-950/10">
        <div className="w-full max-w-3xl aspect-video overflow-hidden rounded-lg bg-black border border-violet-500/30 shadow-inner flex items-center justify-center relative">
          <ImageDisplay
            cameraRef={cameraRef}
            imgRef={imgRef}
            overlayRef={overlayRef}
            imgSrc={imgSrc}
            onCameraLoad={handleCameraLoad}
            onImageLoad={imageLoad}
            activeFeature={activeFeature}
          />
          
          {/* Badge Chrono de détection IA */}
          {gameState === "detecting" && (
            <div className="absolute top-4 right-4 bg-red-950/90 border border-red-500 px-3 py-1.5 rounded-full shadow-lg flex items-center gap-2 animate-pulse z-10">
              <span className="w-2 h-2 bg-red-500 rounded-full"></span>
              <span className="text-xs font-black tracking-wide text-white">
                IA CHERCHE : {detectionTimeLeft}s
              </span>
            </div>
          )}

          {/* Badge IA active en jeu */}
          {gameState === "playing" && (
            <div className="absolute top-4 right-4 bg-emerald-950/90 border border-emerald-500 px-3 py-1.5 rounded-full shadow-lg flex items-center gap-2 z-10">
              <span className="w-2 h-2 bg-emerald-500 rounded-full animate-ping"></span>
              <span className="text-xs font-black tracking-wide text-white">
                IA EN JEU
              </span>
            </div>
          )}

          {/* Écran de défaite de l'IA (Game Over) */}
          {gameState === "ia_lost" && (
            <div className="absolute inset-0 flex flex-col items-center justify-center bg-black/95 p-6 text-center animate-details-show z-20">
              <span className="text-6xl mb-2">💀</span>
              <h3 className="text-3xl font-black uppercase text-red-500 tracking-wider text-violet-neon mb-2">
                L'IA A PERDU !
              </h3>
              <p className="text-slate-300 mb-6 max-w-md text-xs leading-relaxed">
                Le chrono de 30s s'est écoulé. L'IA n'a pas été capable de reconnaître vos mouvements de danse. Vous êtes trop fort pour elle !
              </p>
              <button
                onClick={async () => {
                  setDanceScore(0);
                  setDancePrecision(0);
                  setGameState("detecting");
                  setDetectionTimeLeft(30);
                  setReferenceVideoId("");
                  setDetectedDanceName(null);
                  setCoachComments(["Nouvelle partie ! Dansez pour que l'IA tente de vous reconnaître."]);
                  
                  const camerasList = await getCameras();
                  if (camerasList.length > 0) {
                    const selectedDeviceId = cameraSelectorRef.current
                      ? cameraSelectorRef.current.value
                      : camerasList[0].deviceId;
                    const success = await openCamera(selectedDeviceId);
                    if (success) {
                      setActiveFeature("camera");
                    }
                  }
                }}
                className="px-6 py-3 bg-violet-600 hover:bg-violet-500 active:bg-violet-700 text-white font-bold rounded-lg shadow-lg shadow-violet-600/30 transition-all duration-300 flex items-center gap-2 cursor-pointer"
              >
                🎮 Prendre ma revanche (Défier l'IA)
              </button>
            </div>
          )}

          {activeFeature === "loading" && (
            <div className="absolute inset-0 bg-black/70 flex items-center justify-center flex-col gap-3">
              <div className="w-12 h-12 border-4 border-violet-500 border-t-transparent rounded-full animate-spin"></div>
              <span className="text-violet-400 font-bold animate-text-loading">Chargement du modèle d'IA...</span>
            </div>
          )}
          {!activeFeature && gameState !== "ia_lost" && (
            <div className="absolute inset-0 flex flex-col items-center justify-center bg-[#050818]/60 p-4 text-center">
              <p className="text-slate-300 mb-4 max-w-md">
                Prêt à relever le défi ? Activez votre caméra pour commencer à danser.
              </p>
              <button
                onClick={toggleCamera}
                className="px-6 py-3 bg-violet-600 hover:bg-violet-500 active:bg-violet-700 text-white font-bold rounded-lg shadow-lg shadow-violet-600/30 transition-all duration-300 flex items-center gap-2 cursor-pointer"
              >
                🎥 Activer ma caméra
              </button>
            </div>
          )}
        </div>

        {/* Barre de réflexion de l'IA juste sous l'écran de la caméra */}
        {activeFeature && (
          <div className="w-full max-w-3xl -mt-2 mb-2 px-1">
            <div className="relative w-full h-1.5 bg-[#050818] rounded-full overflow-hidden border border-violet-500/20 shadow-inner">
              <div
                className={`h-full rounded-full transition-all duration-500 ${
                  activeFeature === "loading"
                    ? "bg-gradient-to-r from-violet-500 via-fuchsia-500 to-violet-500 animate-pulse w-full"
                    : gameState === "detecting"
                    ? "bg-gradient-to-r from-violet-600 via-fuchsia-500 to-violet-600 animate-scanner-slow w-full"
                    : gameState === "playing"
                    ? "bg-gradient-to-r from-violet-500 via-cyan-400 to-fuchsia-500 animate-scanner-fast w-full"
                    : "bg-slate-700/50 w-full"
                }`}
                style={{
                  backgroundSize: "200% 100%"
                }}
              ></div>
            </div>
            <div className="flex justify-between items-center mt-1.5 text-[10px] uppercase tracking-wider font-extrabold text-violet-300">
              <span className="flex items-center gap-1.5">
                <span className={`w-2 h-2 rounded-full ${activeFeature === "loading" || gameState === "detecting" || gameState === "playing" ? "bg-fuchsia-500 animate-ping" : "bg-slate-600"}`}></span>
                IA : {
                  activeFeature === "loading"
                    ? "Initialisation & Chargement..."
                    : gameState === "detecting"
                    ? "Recherche de la chorégraphie..."
                    : gameState === "playing"
                    ? "Analyse de danse active..."
                    : "En veille"
                }
              </span>
              {(gameState === "playing" || gameState === "detecting") && (
                <span className="text-cyan-400">
                  Précision : {Math.round(dancePrecision)}%
                </span>
              )}
            </div>
          </div>
        )}

        {/* Contrôles simples */}
        {activeFeature && activeFeature !== "loading" && (
          <div className="flex gap-4">
            <button
              onClick={toggleCamera}
              className="px-6 py-2.5 bg-red-600 hover:bg-red-500 active:bg-red-700 text-white font-bold rounded-lg shadow-md transition-all cursor-pointer"
            >
              🛑 Arrêter la caméra
            </button>
            <button
              onClick={() => {
                setDanceScore(0);
                setDancePrecision(0);
                setGameState("detecting");
                setDetectionTimeLeft(30);
                setReferenceVideoId("");
                setDetectedDanceName(null);
                setCoachComments(["Chronomètre réinitialisé ! L'IA cherche de nouveau votre danse."]);
              }}
              className="px-6 py-2.5 bg-violet-700 hover:bg-violet-600 active:bg-violet-800 text-white font-bold rounded-lg shadow-md transition-all cursor-pointer"
            >
              🔄 Réinitialiser la détection
            </button>
          </div>
        )}
      </main>


      {/* PARAMÈTRES AVANCÉS REPLIABLES (Pour garder l'accès au backend et aux stats) */}
      <footer className="mt-4 border-t border-violet-500/10 pt-4 flex flex-col items-center">
        <button
          onClick={() => setShowAdvancedSettings(!showAdvancedSettings)}
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
