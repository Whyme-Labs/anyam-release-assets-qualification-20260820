from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "base.mp4"
RENDERED = ROOT / "mathfix" / "rendered"
OUTDIR = ROOT / "mathfix" / "out"
PARTS = OUTDIR / "parts"
OUTDIR.mkdir(parents=True, exist_ok=True)
FINAL = OUTDIR / "latent-variable-models-kokoro-mathfix.mp4"
VIDEO_ONLY = OUTDIR / "video_only.mp4"
FPS = 30

SCENES = [
    ("AEObjective", 42.749, 55.505),
    ("VAEReparameterization", 119.865, 130.480),
    ("ELBO", 141.413, 149.325),
    ("RateDistortion", 157.816, 169.249),
    ("VQNearest", 190.389, 205.350),
    ("StraightThrough", 219.901, 234.651),
    ("RQResidual", 289.264, 298.343),
    ("SAESparsity", 360.034, 366.406),
]


def sh(cmd: list[str], capture: bool = False) -> str:
    print("+", " ".join(cmd), flush=True)
    p = subprocess.run(cmd, check=True, text=True, capture_output=capture)
    return p.stdout.strip() if capture else ""


def ffprobe_duration(path: Path) -> float:
    return float(sh([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path)
    ], capture=True))


def video_frame_count(path: Path) -> int:
    # MP4/H.264 exposes nb_frames reliably for this CFR 30 fps master. Fall
    # back to counting decoded frames if the container omits it.
    out = sh([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=nb_frames", "-of", "json", str(path)
    ], capture=True)
    data = json.loads(out)
    raw = data.get("streams", [{}])[0].get("nb_frames")
    if raw and raw != "N/A":
        return int(raw)
    counted = sh([
        "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=nb_read_frames",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path)
    ], capture=True)
    return int(counted)


def audio_hash(path: Path) -> str:
    p = subprocess.run([
        "ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0", "-c", "copy",
        "-f", "hash", "-hash", "sha256", "-"
    ], check=True, text=True, capture_output=True)
    return p.stdout.strip().split("=", 1)[-1]


def encode_gap(part: Path, start_frame: int, end_frame: int) -> None:
    n = end_frame - start_frame
    if n <= 0:
        return
    vf = (
        f"[0:v]trim=start_frame={start_frame}:end_frame={end_frame},"
        "setpts=PTS-STARTPTS,setsar=1,format=yuv420p[v]"
    )
    sh([
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(BASE),
        "-filter_complex", vf, "-map", "[v]", "-an",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p", "-r", str(FPS), "-video_track_timescale", "90000",
        "-g", "60", "-keyint_min", "60", "-sc_threshold", "0",
        str(part),
    ])
    got = video_frame_count(part)
    if got != n:
        raise RuntimeError(f"gap frame mismatch {part.name}: expected {n}, got {got}")


def encode_math(part: Path, scene_file: Path, start_frame: int, end_frame: int) -> None:
    n = end_frame - start_frame
    if n <= 0:
        raise RuntimeError(f"empty math interval: {part}")
    filters = (
        f"[0:v]trim=start_frame={start_frame}:end_frame={end_frame},"
        "setpts=PTS-STARTPTS,setsar=1,format=yuv420p[base];"
        f"[1:v]fps={FPS},setpts=PTS-STARTPTS,"
        f"tpad=stop_mode=clone:stop_duration=2,trim=start_frame=0:end_frame={n},"
        "setpts=PTS-STARTPTS,setsar=1,format=rgba[ov];"
        "[base][ov]overlay=0:0:shortest=1:eof_action=repeat:format=auto,"
        "format=yuv420p[v]"
    )
    sh([
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(BASE), "-i", str(scene_file),
        "-filter_complex", filters, "-map", "[v]", "-an",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p", "-r", str(FPS), "-video_track_timescale", "90000",
        "-g", "60", "-keyint_min", "60", "-sc_threshold", "0",
        str(part),
    ])
    got = video_frame_count(part)
    if got != n:
        raise RuntimeError(f"math frame mismatch {part.name}: expected {n}, got {got}")


def patch_video() -> None:
    if not BASE.exists():
        raise FileNotFoundError(BASE)
    for scene, _, _ in SCENES:
        p = RENDERED / f"{scene}.mov"
        if not p.exists():
            raise FileNotFoundError(p)

    base_duration = ffprobe_duration(BASE)
    base_frames = video_frame_count(BASE)
    if base_frames < 100:
        raise RuntimeError(f"implausible frame count: {base_frames}")

    # Convert narration-derived seconds to the nearest actual 30 fps frame.
    # This makes every boundary contiguous and prevents sub-frame drift across
    # 17 independently encoded segments.
    framed = []
    for scene, start, end in SCENES:
        a = max(0, min(base_frames, round(start * FPS)))
        b = max(a + 1, min(base_frames, round(end * FPS)))
        framed.append((scene, start, end, a, b))

    if PARTS.exists():
        shutil.rmtree(PARTS)
    PARTS.mkdir(parents=True)

    encoded_parts: list[Path] = []
    cursor = 0
    part_idx = 0
    for scene, start, end, a, b in framed:
        if a < cursor:
            raise RuntimeError(f"overlapping scene boundary at {scene}")
        if a > cursor:
            p = PARTS / f"{part_idx:02d}_base.mp4"
            encode_gap(p, cursor, a)
            encoded_parts.append(p)
            part_idx += 1
        p = PARTS / f"{part_idx:02d}_{scene}.mp4"
        encode_math(p, RENDERED / f"{scene}.mov", a, b)
        encoded_parts.append(p)
        part_idx += 1
        cursor = b

    if cursor < base_frames:
        p = PARTS / f"{part_idx:02d}_base.mp4"
        encode_gap(p, cursor, base_frames)
        encoded_parts.append(p)

    total_part_frames = sum(video_frame_count(p) for p in encoded_parts)
    if total_part_frames != base_frames:
        raise RuntimeError(
            f"assembled frame budget mismatch: parts={total_part_frames}, base={base_frames}"
        )

    concat_file = PARTS / "concat.txt"
    concat_file.write_text(
        "".join(f"file '{p.resolve().as_posix()}'\n" for p in encoded_parts),
        encoding="utf-8",
    )
    sh([
        "ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
        "-i", str(concat_file), "-c", "copy", "-movflags", "+faststart", str(VIDEO_ONLY)
    ])

    joined_frames = video_frame_count(VIDEO_ONLY)
    if joined_frames != base_frames:
        raise RuntimeError(f"concat lost frames: expected {base_frames}, got {joined_frames}")

    # Remux the original AAC packet stream exactly. No speech or music is
    # decoded or recompressed, so Kokoro narration stays bit-identical.
    sh([
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(VIDEO_ONLY), "-i", str(BASE),
        "-map", "0:v:0", "-map", "1:a:0", "-map_metadata", "1",
        "-c:v", "copy", "-c:a", "copy", "-movflags", "+faststart", str(FINAL)
    ])

    final_duration = ffprobe_duration(FINAL)
    final_frames = video_frame_count(FINAL)
    if final_frames != base_frames:
        raise RuntimeError(f"final frame mismatch: base={base_frames}, final={final_frames}")
    # Container duration is normally governed by the copied AAC stream, so it
    # should equal the source master to within one video frame.
    if abs(base_duration - final_duration) > (1 / FPS + 0.01):
        raise RuntimeError(
            f"duration mismatch base={base_duration:.3f}, final={final_duration:.3f}"
        )

    base_ah = audio_hash(BASE)
    final_ah = audio_hash(FINAL)
    if base_ah != final_ah:
        raise RuntimeError(f"audio changed: {base_ah} != {final_ah}")

    sh(["ffmpeg", "-v", "error", "-i", str(FINAL), "-f", "null", "-"])

    qc = {
        "base_duration_seconds": base_duration,
        "final_duration_seconds": final_duration,
        "duration_delta_seconds": round(final_duration - base_duration, 6),
        "base_video_frames": base_frames,
        "final_video_frames": final_frames,
        "fps": FPS,
        "audio_sha256": final_ah,
        "audio_bit_identical_to_base": base_ah == final_ah,
        "output_sha256": hashlib.sha256(FINAL.read_bytes()).hexdigest(),
        "formula_overlays": [
            {
                "scene": scene,
                "requested_start": start,
                "requested_end": end,
                "start_frame": a,
                "end_frame": b,
                "actual_start": round(a / FPS, 6),
                "actual_end": round(b / FPS, 6),
            }
            for scene, start, end, a, b in framed
        ],
    }
    (OUTDIR / "qc.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
    print(json.dumps(qc, indent=2))


def make_contact_sheet() -> None:
    frames = OUTDIR / "frames"
    frames.mkdir(exist_ok=True)
    paths = []
    for i, (scene, a, b) in enumerate(SCENES, start=1):
        t = (a + b) / 2
        p = frames / f"{i:02d}_{scene}.jpg"
        sh([
            "ffmpeg", "-y", "-loglevel", "error", "-ss", f"{t:.3f}", "-i", str(FINAL),
            "-frames:v", "1", "-q:v", "2", str(p)
        ])
        paths.append(p)
    sh([
        "montage", *map(str, paths), "-tile", "4x2", "-geometry", "480x270+5+5",
        str(OUTDIR / "math_formula_contact_sheet.jpg")
    ])


def publish_here() -> dict:
    index = OUTDIR / "index.html"
    index.write_text("""<!doctype html>
<html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">
<title>Latent variable models — math-fixed Kokoro cut</title>
<style>body{margin:0;background:#08080b;color:#fff;font:16px system-ui,sans-serif;display:grid;place-items:center;min-height:100vh}.wrap{width:min(1100px,94vw)}video{width:100%;border:1px solid #34343d;border-radius:14px;background:#000}h1{font-size:24px;margin:0 0 12px}.meta{color:#aaa;margin:10px 0 18px}a{color:#a175f1}</style></head>
<body><main class=\"wrap\"><h1>Latent variable models: AE, VAE, VQ-VAE, RQ-VAE, SAE, RAE</h1>
<div class=\"meta\">Kokoro am_liam narration · corrected time-proportional section rail · Manim-rendered mathematical scenes</div>
<video controls preload=\"metadata\" src=\"latent-variable-models-kokoro-mathfix.mp4\"></video>
<p><a href=\"latent-variable-models-kokoro-mathfix.mp4\" download>Download MP4</a></p></main></body></html>""", encoding="utf-8")

    files = [
        (index, "text/html; charset=utf-8"),
        (FINAL, "video/mp4"),
        (OUTDIR / "math_formula_contact_sheet.jpg", "image/jpeg"),
        (OUTDIR / "qc.json", "application/json"),
    ]
    manifest = [
        {
            "path": p.name,
            "size": p.stat().st_size,
            "contentType": ctype,
            "hash": hashlib.sha256(p.read_bytes()).hexdigest(),
        }
        for p, ctype in files
    ]
    payload = {
        "files": manifest,
        "displayName": "Latent variable models — Manim math fix",
        "displayDescription": "Kokoro-narrated latent variable model explainer with Manim-rendered formulas and corrected chapter progress rail.",
    }
    req = urllib.request.Request(
        "https://here.now/api/v1/publish",
        data=json.dumps(payload).encode(),
        method="POST",
        headers={"Content-Type": "application/json", "X-HereNow-Client": "chatgpt/mathfix"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        created = json.load(r)

    upload_map = {u["path"]: u["url"] for u in created["upload"]["uploads"]}
    for p, ctype in files:
        if p.name not in upload_map:
            continue
        req = urllib.request.Request(
            upload_map[p.name], data=p.read_bytes(), method="PUT", headers={"Content-Type": ctype}
        )
        with urllib.request.urlopen(req, timeout=300) as r:
            if r.status not in (200, 201, 204):
                raise RuntimeError(f"upload failed {p.name}: {r.status}")

    req = urllib.request.Request(
        created["finalizeUrl"],
        data=json.dumps({"versionId": created["versionId"]}).encode(),
        method="POST",
        headers={"Content-Type": "application/json", "X-HereNow-Client": "chatgpt/mathfix"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        final = json.load(r)

    site = final["siteUrl"]
    result = {
        "siteUrl": site,
        "videoUrl": site.rstrip("/") + "/" + FINAL.name,
        "contactSheetUrl": site.rstrip("/") + "/math_formula_contact_sheet.jpg",
        "expiresAt": final.get("publishStatus", {}).get("expiresAt"),
    }
    (OUTDIR / "publish_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("PUBLISHED_RESULT=" + json.dumps(result, separators=(",", ":")))
    return result


if __name__ == "__main__":
    patch_video()
    make_contact_sheet()
    if "--publish" in sys.argv:
        publish_here()
