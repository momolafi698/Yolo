import React, { memo } from "react";

const ImageDisplay = memo(function ImageDisplay({
  cameraRef,
  videoRef,
  imgRef,
  overlayRef,
  imgSrc,
  videoSrc,
  onCameraLoad,
  onVideoLoad,
  onVideoPlay,
  onVideoPause,
  onVideoSeeked,
  onVideoEnded,
  onImageLoad,
  activeFeature,
  cameraContainerRef,
  children,
}) {
  return (
    <div className="container p-5 mb-6 relative min-h-[400px] flex flex-col">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-4">
          <h2 className="text-xl font-bold text-white tracking-tight flex items-center gap-2 font-display text-violet-neon">
            <span className="w-2.5 h-6 bg-gradient-to-b from-fuchsia-500 to-pink-500 rounded-full inline-block"></span>
            Preview
          </h2>
        </div>

        {activeFeature && (
          <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-gray-700/50 border border-gray-600/50">
            <span
              className={`w-2 h-2 rounded-full ${activeFeature === "camera" ? "bg-red-500 animate-pulse" : "bg-violet-500"}`}
            ></span>
            <span className="text-xs font-semibold uppercase tracking-wider text-gray-300">
              {activeFeature === "camera"
                ? "Live Stream"
                : activeFeature === "video"
                  ? "Video Source"
                  : "Static Image"}
            </span>
          </div>
        )}
      </div>

      <div className="flex-1 rounded-xl bg-gray-900/50 border-2 border-dashed border-gray-700 relative overflow-hidden flex items-center justify-center">
        {activeFeature === null && (
          <div className="text-center p-8">
            <div className="w-20 h-20 bg-gray-800 rounded-full flex items-center justify-center mx-auto mb-4 border border-gray-700 shadow-inner">
              <svg
                className="w-10 h-10 text-gray-600"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={1.5}
                  d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
                />
              </svg>
            </div>
            <h3 className="text-lg font-bold text-gray-300 mb-1">
              No Media Selected
            </h3>
            <p className="text-gray-500 text-sm max-w-xs mx-auto">
              Choose a source from the control panel below to start detection
            </p>
          </div>
        )}

        <div className="relative w-full h-full flex items-center justify-center" hidden={activeFeature === null}>
          <div ref={cameraContainerRef} className="relative w-full max-h-[600px] flex items-center justify-center bg-[#050414] rounded-xl overflow-hidden">
            <video
              className="block max-h-[600px] w-full object-contain"
              style={activeFeature === "camera" ? { transform: "scaleX(-1)" } : undefined}
              ref={cameraRef}
              onLoadedMetadata={onCameraLoad}
              hidden={activeFeature !== "camera"}
              autoPlay
              playsInline
              muted
            />
            <video
              className="block max-h-[600px] w-full object-contain"
              ref={videoRef}
              src={videoSrc}
              onLoadedMetadata={onVideoLoad}
              onLoadedData={onVideoLoad}
              onPlay={onVideoPlay}
              onPause={onVideoPause}
              onSeeked={onVideoSeeked}
              onEnded={onVideoEnded}
              hidden={activeFeature !== "video"}
              controls
              playsInline
            />
            <img
              id="img"
              ref={imgRef}
              src={imgSrc}
              onLoad={onImageLoad}
              hidden={activeFeature !== "image"}
              className="block max-h-[600px] w-full object-contain"
              alt="Source"
            />
            <canvas
              ref={overlayRef}
              hidden={activeFeature === null}
              className="absolute inset-0 w-full h-full pointer-events-none"
              style={{ objectFit: "contain" }}
            ></canvas>
            {children}
          </div>
        </div>
      </div>
    </div>
  );
});

export default ImageDisplay;
