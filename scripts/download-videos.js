import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { mkdir, readFile } from "node:fs/promises";
import { createRequire } from "node:module";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

const require = createRequire(import.meta.url);
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PROJECT_ROOT = path.resolve(__dirname, "..");

let ffmpegPath = null;
try {
  ffmpegPath = require("ffmpeg-static");
} catch {
  // ignore
}

const DEFAULTS = {
  input: "video-urls.txt",
  output: "videos",
  archive: ".yt-dlp-archive.txt",
};

const FORMAT_480P_FIRST = [
  "bv*[height<=480][ext=mp4]+ba[ext=m4a]",
  "bv*[height<=480]+ba",
  "b[height<=480]",
  "bv*[height<=720][ext=mp4]+ba[ext=m4a]",
  "bv*[height<=720]+ba",
  "best",
].join("/");

function printHelp() {
  console.log(`
Download videos listed in a text file into the local videos directory.

Usage:
  npm run download:videos
  npm run download:videos -- --input urls.txt
  node scripts/download-videos.js --input urls.txt --output videos

Text file format:
  https://example.com/video-one
  # Blank lines and lines beginning with # are ignored.
  https://example.com/video-two

Options:
  --input <path>       Text file containing one URL per line. Default: ${DEFAULTS.input}
  --output <dir>       Destination directory. Default: ${DEFAULTS.output}
  --help               Show this help.

Environment:
  YT_DLP_BIN           yt-dlp executable name or path. Default: yt-dlp
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
      case "--help":
      case "-h":
        options.help = true;
        break;
      default:
        throw new Error(`Unknown option: ${arg}`);
    }
  }

  options.input = path.resolve(PROJECT_ROOT, options.input);
  options.output = path.resolve(PROJECT_ROOT, options.output);
  options.archive = path.join(options.output, DEFAULTS.archive);

  return options;
}

async function readUrls(inputPath) {
  if (!existsSync(inputPath)) {
    throw new Error(`URL file does not exist: ${inputPath}`);
  }

  const contents = await readFile(inputPath, "utf8");
  const urls = [];
  const seen = new Set();

  for (const line of contents.split(/\r?\n/)) {
    const value = line.trim();
    if (!value || value.startsWith("#")) continue;
    if (seen.has(value)) continue;
    seen.add(value);
    urls.push(value);
  }

  return urls;
}

function runYtDlp(url, options) {
  const executable = process.env.YT_DLP_BIN || "yt-dlp";
  const args = [
    "--format",
    FORMAT_480P_FIRST,
    "--merge-output-format",
    "mp4",
    "--paths",
    options.output,
    "--output",
    "%(title).120B [%(id)s].%(ext)s",
    "--download-archive",
    options.archive,
    "--no-overwrites",
    "--continue",
  ];

  if (ffmpegPath) {
    args.push("--ffmpeg-location", ffmpegPath);
  }

  args.push(url);

  return new Promise((resolve, reject) => {
    const child = spawn(executable, args, { stdio: "inherit" });

    child.on("error", (error) => {
      if (error.code === "ENOENT") {
        reject(new Error(`Could not find ${executable}. Install yt-dlp or set YT_DLP_BIN.`));
        return;
      }
      reject(error);
    });
    child.on("close", (code) => {
      if (code === 0) {
        resolve();
        return;
      }
      reject(new Error(`${executable} exited with code ${code}`));
    });
  });
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  if (options.help) {
    printHelp();
    return;
  }

  const urls = await readUrls(options.input);
  if (urls.length === 0) {
    throw new Error(`No URLs found in ${options.input}`);
  }

  await mkdir(options.output, { recursive: true });

  let failures = 0;
  console.log(`Downloading ${urls.length} video(s) into ${path.relative(PROJECT_ROOT, options.output)}`);
  console.log("Format preference: best MP4 video/audio at 480p or below, then close fallbacks.");

  for (const [index, url] of urls.entries()) {
    console.log(`\n[${index + 1}/${urls.length}] ${url}`);
    try {
      await runYtDlp(url, options);
    } catch (error) {
      failures++;
      console.error(`Download failed: ${error.message}`);
    }
  }

  if (failures > 0) {
    throw new Error(`${failures} download(s) failed`);
  }
}

main().catch((error) => {
  console.error(`\nVideo download failed: ${error.message}`);
  process.exitCode = 1;
});
