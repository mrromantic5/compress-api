import os
import io
import uuid
import tempfile
import subprocess
import shutil
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

# ─────────────────────────────────────────
#  App setup
# ─────────────────────────────────────────
app = FastAPI(
    title="FileSquish API",
    description="Compress images, audio, and video — free, fast, no subscriptions.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path(tempfile.gettempdir()) / "filesquish"
UPLOAD_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────

MIME_GROUPS = {
    "image": ["image/jpeg", "image/png", "image/webp", "image/gif", "image/bmp", "image/tiff"],
    "audio": ["audio/mpeg", "audio/mp3", "audio/wav", "audio/ogg", "audio/flac", "audio/aac", "audio/m4a"],
    "video": ["video/mp4", "video/webm", "video/avi", "video/mov", "video/mkv", "video/quicktime"],
}

EXT_MAP = {
    "image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp",
    "image/gif": ".gif", "image/bmp": ".bmp", "image/tiff": ".tiff",
    "audio/mpeg": ".mp3", "audio/mp3": ".mp3", "audio/wav": ".wav",
    "audio/ogg": ".ogg", "audio/flac": ".flac", "audio/aac": ".aac",
    "audio/m4a": ".m4a", "video/mp4": ".mp4", "video/webm": ".webm",
    "video/avi": ".avi", "video/mov": ".mov", "video/mkv": ".mkv",
    "video/quicktime": ".mov",
}

def get_file_group(content_type: str) -> Optional[str]:
    for group, types in MIME_GROUPS.items():
        if content_type in types:
            return group
    return None

def human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"

def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None

# ─────────────────────────────────────────
#  Compression functions
# ─────────────────────────────────────────

def compress_image(data: bytes, content_type: str, quality: int = 75, max_width: Optional[int] = None) -> tuple[bytes, str]:
    img = Image.open(io.BytesIO(data))

    # Convert RGBA / palette → RGB for JPEG output
    if img.mode in ("RGBA", "P", "LA"):
        out_format = "PNG"
        out_ext = ".png"
    else:
        out_format = "JPEG"
        out_ext = ".jpg"

    # Optional downscale
    if max_width and img.width > max_width:
        ratio = max_width / img.width
        img = img.resize((max_width, int(img.height * ratio)), Image.LANCZOS)

    buf = io.BytesIO()
    save_kwargs = {"optimize": True}
    if out_format == "JPEG":
        save_kwargs["quality"] = quality
        save_kwargs["progressive"] = True
    elif out_format == "PNG":
        # PNG compress level 0-9
        save_kwargs["compress_level"] = max(0, min(9, (100 - quality) // 11))

    img.save(buf, format=out_format, **save_kwargs)
    return buf.getvalue(), out_ext


def compress_audio(src_path: Path, quality: int = 75) -> tuple[Path, str]:
    """Re-encode audio with ffmpeg to a lower bitrate MP3."""
    # Map quality (0-100) → bitrate (32k-320k)
    bitrate = int(32 + (quality / 100) * 288)
    bitrate = max(32, min(320, bitrate))
    out_path = src_path.with_suffix(".out.mp3")
    cmd = [
        "ffmpeg", "-y", "-i", str(src_path),
        "-codec:a", "libmp3lame",
        "-b:a", f"{bitrate}k",
        "-q:a", "2",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode())
    return out_path, ".mp3"


def compress_video(src_path: Path, quality: int = 75) -> tuple[Path, str]:
    """Re-encode video with ffmpeg using CRF (lower = better quality)."""
    # Map quality (0-100) → CRF (51=worst → 18=best)
    crf = int(51 - (quality / 100) * 33)
    crf = max(18, min(51, crf))
    out_path = src_path.with_suffix(".out.mp4")

    # Probe input resolution
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height",
         "-of", "csv=p=0", str(src_path)],
        capture_output=True, text=True
    )
    scale_filter = []
    try:
        w, h = map(int, probe.stdout.strip().split(","))
        # Downscale to max 720p to save CPU time
        if h > 720:
            scale_filter = ["-vf", "scale=-2:720"]
        elif w > 1280:
            scale_filter = ["-vf", "scale=1280:-2"]
    except Exception:
        pass  # If probe fails, skip scaling

    cmd = [
        "ffmpeg", "-y", "-i", str(src_path),
        "-vcodec", "libx264",
        "-crf", str(crf),
        "-preset", "ultrafast",   # Much faster encoding on low CPU
        "-tune", "fastdecode",    # Optimise for speed
        *scale_filter,            # Cap at 720p if needed
        "-acodec", "aac",
        "-b:a", "96k",            # Slightly lower audio bitrate
        "-movflags", "+faststart",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode())
    return out_path, ".mp4"


# ─────────────────────────────────────────
#  Routes
# ─────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name": "FileSquish API",
        "status": "online",
        "endpoints": {
            "compress": "POST /compress",
            "health": "GET /health",
            "info": "GET /info",
        },
    }


@app.api_route("/health", methods=["GET", "HEAD"])
def health():
    return {"status": "ok", "ffmpeg": ffmpeg_available()}


@app.get("/info")
def info():
    return {
        "supported_types": MIME_GROUPS,
        "ffmpeg_available": ffmpeg_available(),
        "max_file_size_mb": 200,
        "quality_range": "1-100 (default 75)",
    }


@app.post("/compress")
async def compress(
    file: UploadFile = File(...),
    quality: int = Query(default=75, ge=1, le=100, description="Compression quality 1-100"),
    max_width: Optional[int] = Query(default=None, ge=100, le=7680, description="Max image width (images only)"),
):
    content_type = file.content_type or ""
    file_group = get_file_group(content_type)

    if not file_group:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {content_type}. Supported: images, audio, video.",
        )

    data = await file.read()
    original_size = len(data)

    # Guard: 200 MB max
    if original_size > 200 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large. Max 200 MB.")

    uid = uuid.uuid4().hex
    original_name = Path(file.filename or "file").stem

    try:
        # ── Images ─────────────────────────────────────
        if file_group == "image":
            compressed, ext = compress_image(data, content_type, quality, max_width)
            out_path = UPLOAD_DIR / f"{uid}{ext}"
            out_path.write_bytes(compressed)

        # ── Audio / Video (needs ffmpeg) ────────────────
        else:
            if not ffmpeg_available():
                raise HTTPException(
                    status_code=501,
                    detail="FFmpeg not installed on this server. Audio/video compression unavailable.",
                )
            in_ext = EXT_MAP.get(content_type, ".bin")
            in_path = UPLOAD_DIR / f"{uid}_in{in_ext}"
            in_path.write_bytes(data)

            if file_group == "audio":
                out_path, ext = compress_audio(in_path, quality)
            else:
                out_path, ext = compress_video(in_path, quality)

            in_path.unlink(missing_ok=True)

        compressed_size = out_path.stat().st_size
        saved_pct = round((1 - compressed_size / original_size) * 100, 1)

        download_name = f"{original_name}_compressed{out_path.suffix}"

        response = FileResponse(
            path=str(out_path),
            filename=download_name,
            media_type="application/octet-stream",
            headers={
                "X-Original-Size": str(original_size),
                "X-Compressed-Size": str(compressed_size),
                "X-Size-Reduction": f"{saved_pct}%",
                "X-Original-Size-Human": human_size(original_size),
                "X-Compressed-Size-Human": human_size(compressed_size),
            },
        )
        return response

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Compression failed: {str(e)}")
