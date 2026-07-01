import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { mkdir, readdir } from "node:fs/promises";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";
import ffmpegPath from "ffmpeg-static";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PROJECT_ROOT = path.resolve(__dirname, "..");
const INPUT_DIR = path.resolve(PROJECT_ROOT, "videos");
const OUTPUT_DIR = path.resolve(PROJECT_ROOT, "public/audio");
const VIDEO_EXTENSIONS = new Set([".mp4", ".mov", ".m4v", ".webm", ".avi"]);

function slugify(name) {
  return name
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLowerCase()
    .replace(/['"]/g, "")
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

async function extractAudio(videoPath, audioPath) {
  return new Promise((resolve, reject) => {
    const ffmpeg = spawn(ffmpegPath, [
      "-y",
      "-hide_banner",
      "-loglevel",
      "error",
      "-i",
      videoPath,
      "-vn",
      "-acodec",
      "libmp3lame",
      "-q:a",
      "2",
      audioPath
    ], { stdio: "inherit" });

    ffmpeg.on("error", reject);
    ffmpeg.on("close", (code) => {
      if (code === 0) resolve();
      else reject(new Error(`ffmpeg exited with code ${code}`));
    });
  });
}

async function main() {
  if (!existsSync(INPUT_DIR)) {
    console.error(`Input directory does not exist: ${INPUT_DIR}`);
    return;
  }

  await mkdir(OUTPUT_DIR, { recursive: true });

  const files = await readdir(INPUT_DIR, { withFileTypes: true });
  const videos = files
    .filter((file) => file.isFile() && VIDEO_EXTENSIONS.has(path.extname(file.name).toLowerCase()))
    .map((file) => path.join(INPUT_DIR, file.name));

  if (videos.length === 0) {
    console.log("No videos found in videos/ directory.");
    return;
  }

  console.log(`Extracting audio from ${videos.length} video(s)...`);

  for (const video of videos) {
    const filename = path.basename(video, path.extname(video));
    const slug = slugify(filename);
    const audioPath = path.join(OUTPUT_DIR, `${slug}.mp3`);

    console.log(`\nExtracting audio from "${path.basename(video)}" -> "${slug}.mp3"`);
    try {
      await extractAudio(video, audioPath);
      console.log(`  Done.`);
    } catch (error) {
      console.error(`  Extraction failed: ${error.message}`);
    }
  }
}

main().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
