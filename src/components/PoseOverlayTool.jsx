import { memo, useCallback, useEffect, useMemo, useRef, useState } from "react";
import classes from "../utils/yolo_classes.json";
import { renderOverlay } from "../utils/render-overlay";

const DEFAULT_PLACEMENT = {
  xPercent: 50,
  yPercent: 58,
  scalePercent: 16,
  speed: 1,
  poseStart: 0,
  loop: true,
};

function findNearestFrame(frames, timestamp) {
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

function createPrediction(frame, canvas, placement) {
  if (!frame?.keypoints?.length || canvas.width === 0 || canvas.height === 0) {
    return null;
  }

  const shortSide = Math.min(canvas.width, canvas.height);
  const origin = {
    x: canvas.width * (placement.xPercent / 100),
    y: canvas.height * (placement.yPercent / 100),
  };
  const scale = shortSide * (placement.scalePercent / 100);
  const keypoints = frame.keypoints.map((point) => ({
    name: point.name,
    x: origin.x + point.x * scale,
    y: origin.y + point.y * scale,
    score: point.score ?? 1,
  }));

  const visiblePoints = keypoints.filter(
    (point) => Number.isFinite(point.x) && Number.isFinite(point.y) && point.score > 0.5,
  );

  if (visiblePoints.length === 0) return null;

  const minX = Math.min(...visiblePoints.map((point) => point.x));
  const minY = Math.min(...visiblePoints.map((point) => point.y));
  const maxX = Math.max(...visiblePoints.map((point) => point.x));
  const maxY = Math.max(...visiblePoints.map((point) => point.y));
  const padding = Math.max(8, shortSide * 0.03);

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

const PoseOverlayTool = memo(function PoseOverlayTool({ catalogue }) {
  const fileInputRef = useRef(null);
  const videoRef = useRef(null);
  const overlayRef = useRef(null);
  const animationRef = useRef(null);

  const [videoUrl, setVideoUrl] = useState("");
  const [videoName, setVideoName] = useState("");
  const [selectedDanceId, setSelectedDanceId] = useState("");
  const [placement, setPlacement] = useState(DEFAULT_PLACEMENT);

  const dances = useMemo(() => catalogue?.dances ?? [], [catalogue]);
  const selectedDance = useMemo(
    () => dances.find((dance) => dance.id === selectedDanceId) ?? dances[0] ?? null,
    [dances, selectedDanceId],
  );

  useEffect(() => () => {
    if (videoUrl) URL.revokeObjectURL(videoUrl);
  }, [videoUrl]);

  const updatePlacement = useCallback((key, value) => {
    setPlacement((current) => ({
      ...current,
      [key]: value,
    }));
  }, []);

  const resizeOverlay = useCallback(() => {
    const video = videoRef.current;
    const canvas = overlayRef.current;
    if (!video || !canvas || !video.videoWidth || !video.videoHeight) return false;

    if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
    }
    return true;
  }, []);

  const drawOverlayFrame = useCallback(() => {
    const video = videoRef.current;
    const canvas = overlayRef.current;
    const context = canvas?.getContext("2d");

    if (video && canvas && context && resizeOverlay()) {
      context.clearRect(0, 0, canvas.width, canvas.height);

      if (selectedDance?.frames?.length) {
        const lastFrameTimestamp = selectedDance.frames.at(-1)?.timestamp ?? 0;
        let poseTime = placement.poseStart + video.currentTime * placement.speed;

        if (placement.loop && lastFrameTimestamp > 0) {
          poseTime %= lastFrameTimestamp;
        }

        const frame = poseTime <= lastFrameTimestamp + 1 / 30
          ? findNearestFrame(selectedDance.frames, poseTime)
          : null;
        const prediction = createPrediction(frame, canvas, placement);

        if (prediction) {
          void renderOverlay([prediction], null, context, "pose", classes);
        }
      }
    }
  }, [placement, resizeOverlay, selectedDance]);

  useEffect(() => {
    const tick = () => {
      drawOverlayFrame();
      animationRef.current = window.requestAnimationFrame(tick);
    };

    animationRef.current = window.requestAnimationFrame(tick);
    return () => {
      if (animationRef.current) window.cancelAnimationFrame(animationRef.current);
    };
  }, [drawOverlayFrame]);

  const handleVideoFile = useCallback((file) => {
    if (!file) return;
    const nextUrl = URL.createObjectURL(file);
    setVideoUrl((currentUrl) => {
      if (currentUrl) URL.revokeObjectURL(currentUrl);
      return nextUrl;
    });
    setVideoName(file.name);
  }, []);

  const resetPlacement = useCallback(() => {
    setPlacement(DEFAULT_PLACEMENT);
  }, []);

  const controlClass =
    "w-full rounded-lg border border-violet-500/30 bg-[#050818] px-3 py-2 text-sm text-slate-100 outline-none focus:border-cyan-400";
  const labelClass = "text-xs uppercase tracking-wider text-slate-400 font-bold";

  return (
    <section className="card-violet">
      <div className="flex flex-col gap-4">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-3 border-b border-violet-500/20 pb-3">
          <div>
            <p className="text-xs uppercase tracking-wider text-cyan-300 font-black">
              Overlay navigateur
            </p>
            <h2 className="text-xl font-black text-white">Video + pose catalogue</h2>
          </div>
          <div className="flex flex-wrap gap-2">
            <input
              ref={fileInputRef}
              type="file"
              accept="video/*"
              hidden
              onChange={(event) => {
                handleVideoFile(event.target.files?.[0]);
                event.target.value = "";
              }}
            />
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              className="px-4 py-2 rounded-lg bg-cyan-600 hover:bg-cyan-500 text-white font-bold transition-colors"
            >
              Choisir video
            </button>
            <button
              type="button"
              onClick={resetPlacement}
              className="px-4 py-2 rounded-lg bg-[#050818] hover:bg-violet-950 text-violet-200 border border-violet-500/30 font-bold transition-colors"
            >
              Recentrer
            </button>
          </div>
        </div>

        <div className="grid grid-cols-1 xl:grid-cols-[minmax(0,2fr)_minmax(280px,1fr)] gap-4 items-start">
          <div className="relative overflow-hidden rounded-lg border border-violet-500/20 bg-black min-h-[260px] flex items-center justify-center">
            {videoUrl ? (
              <div className="relative w-full">
                <video
                  ref={videoRef}
                  src={videoUrl}
                  className="block w-full max-h-[620px] bg-black"
                  controls
                  playsInline
                  onLoadedMetadata={resizeOverlay}
                  onLoadedData={resizeOverlay}
                />
                <canvas
                  ref={overlayRef}
                  className="absolute inset-0 h-full w-full pointer-events-none"
                />
              </div>
            ) : (
              <p className="text-sm text-slate-500 font-semibold">Aucune video selectionnee</p>
            )}
          </div>

          <div className="flex flex-col gap-4">
            <div className="flex flex-col gap-2">
              <label className={labelClass} htmlFor="pose-overlay-dance">
                Pose
              </label>
              <select
                id="pose-overlay-dance"
                className={controlClass}
                value={selectedDanceId || dances[0]?.id || ""}
                disabled={dances.length === 0}
                onChange={(event) => setSelectedDanceId(event.target.value)}
              >
                {dances.length === 0 ? (
                  <option value="">Catalogue indisponible</option>
                ) : (
                  dances.map((dance) => (
                    <option key={dance.id} value={dance.id}>
                      {dance.title}
                    </option>
                  ))
                )}
              </select>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <label className="flex flex-col gap-2">
                <span className={labelClass}>X</span>
                <input
                  className="accent-cyan-400"
                  type="range"
                  min="0"
                  max="100"
                  value={placement.xPercent}
                  onChange={(event) => updatePlacement("xPercent", Number(event.target.value))}
                />
              </label>
              <label className="flex flex-col gap-2">
                <span className={labelClass}>Y</span>
                <input
                  className="accent-cyan-400"
                  type="range"
                  min="0"
                  max="100"
                  value={placement.yPercent}
                  onChange={(event) => updatePlacement("yPercent", Number(event.target.value))}
                />
              </label>
            </div>

            <label className="flex flex-col gap-2">
              <span className={labelClass}>Taille</span>
              <input
                className="accent-cyan-400"
                type="range"
                min="6"
                max="36"
                value={placement.scalePercent}
                onChange={(event) => updatePlacement("scalePercent", Number(event.target.value))}
              />
            </label>

            <div className="grid grid-cols-2 gap-3">
              <label className="flex flex-col gap-2">
                <span className={labelClass}>Depart pose</span>
                <input
                  className={controlClass}
                  type="number"
                  min="0"
                  step="0.1"
                  value={placement.poseStart}
                  onChange={(event) => updatePlacement("poseStart", Number(event.target.value) || 0)}
                />
              </label>
              <label className="flex flex-col gap-2">
                <span className={labelClass}>Vitesse</span>
                <input
                  className={controlClass}
                  type="number"
                  min="0.1"
                  step="0.1"
                  value={placement.speed}
                  onChange={(event) => updatePlacement("speed", Number(event.target.value) || 1)}
                />
              </label>
            </div>

            <label className="flex items-center justify-between gap-3 rounded-lg border border-violet-500/20 bg-[#050818]/70 px-3 py-2">
              <span className="text-sm font-bold text-slate-200">Boucler la pose</span>
              <input
                type="checkbox"
                className="h-5 w-5 accent-cyan-400"
                checked={placement.loop}
                onChange={(event) => updatePlacement("loop", event.target.checked)}
              />
            </label>

            <div className="flex items-center justify-between gap-3 text-xs text-slate-400">
              <span className="truncate">{videoName || "Video locale"}</span>
              <span className="shrink-0">
                {selectedDance ? `${selectedDance.frames.length} poses` : "0 pose"}
              </span>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
});

export default PoseOverlayTool;
