from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "base.mp4"
RENDERED = ROOT / "mathfix" / "rendered"
OUTDIR = ROOT / "mathfix" / "out"
OUTDIR.mkdir(parents=True, exist_ok=True)
FINAL = OUTDIR / "latent-variable-models-kokoro-mathfix.mp4"

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
    out = sh([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path)
    ], capture=True)
    return float(out)


def audio_hash(path: Path) -> str:
    p = subprocess.run([
        "ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0", "-c", "copy",
        "-f", "hash", "-hash", "sha256", "-"
    ], check=True, text=True, capture_output=True)
    line = p.stdout.strip()
    return line.split("=", 1)[-1]


def patch_video() -> None:
    if not BASE.exists():
        raise FileNotFoundError(BASE)
    for scene, _, _ in SCENES:
        p = RENDERED / f"{scene}.mov"
        if not p.exists():
            raise FileNotFoundError(p)

    base_d = ffprobe_duration(BASE)
    cmd = ["ffmpeg", "-y", "-i", str(BASE)]
    for scene, _, _ in SCENES:
        cmd += ["-i", str(RENDERED / f"{scene}.mov")]

    # Absolute PTS on transparent qtrle clips is fragile in FFmpeg's overlay
    # filter. Instead, split the base into local-time segments, overlay each
    # Manim clip at t=0, then concatenate the visual segments. The original
    # Kokoro AAC stream is mapped directly from input 0 and never re-encoded.
    segment_count = len(SCENES) * 2 + 1
    base_labels = [f"base{i}" for i in range(segment_count)]
    filters: list[str] = [
        f"[0:v]split={segment_count}" + "".join(f"[{label}]" for label in base_labels)
    ]
    concat_labels: list[str] = []
    prev = 0.0
    split_idx = 0

    for input_idx, (scene, start, end) in enumerate(SCENES, start=1):
        if start <= prev or end <= start:
            raise RuntimeError(f"invalid scene range {scene}: {start}..{end}")

        pre = f"pre{input_idx}"
        filters.append(
            f"[{base_labels[split_idx]}]trim=start={prev:.3f}:end={start:.3f},"
            f"setpts=PTS-STARTPTS,format=yuv420p[{pre}]"
        )
        concat_labels.append(f"[{pre}]")
        split_idx += 1

        duration = end - start
        under = f"under{input_idx}"
        over = f"over{input_idx}"
        math = f"math{input_idx}"
        filters.append(
            f"[{base_labels[split_idx]}]trim=start={start:.3f}:end={end:.3f},"
            f"setpts=PTS-STARTPTS,format=yuv420p[{under}]"
        )
        # Pad the last transparent frame so the overlay is guaranteed to be
        # longer than the exact base interval. shortest=1 then makes the base
        # interval the timing authority and prevents accumulated drift.
        filters.append(
            f"[{input_idx}:v]trim=start=0:duration={duration:.3f},setpts=PTS-STARTPTS,"
            f"tpad=stop_mode=clone:stop_duration=1,format=argb[{over}]"
        )
        filters.append(
            f"[{under}][{over}]overlay=0:0:eof_action=pass:shortest=1:format=auto,"
            f"format=yuv420p[{math}]"
        )
        concat_labels.append(f"[{math}]")
        split_idx += 1
        prev = end

    tail = "tail"
    filters.append(
        f"[{base_labels[split_idx]}]trim=start={prev:.3f}:end={base_d:.6f},"
        f"setpts=PTS-STARTPTS,format=yuv420p[{tail}]"
    )
    concat_labels.append(f"[{tail}]")
    filters.append(
        "".join(concat_labels) + f"concat=n={len(concat_labels)}:v=1:a=0[vout]"
    )

    cmd += [
        "-filter_complex", ";".join(filters),
        "-map", "[vout]",
        "-map", "0:a:0?",
        "-map_metadata", "0",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        "-movflags", "+faststart",
        str(FINAL),
    ]
    sh(cmd)

    final_d = ffprobe_duration(FINAL)
    if abs(base_d - final_d) > 0.08:
        raise RuntimeError(f"duration mismatch base={base_d:.3f}, final={final_d:.3f}")

    base_ah = audio_hash(BASE)
    final_ah = audio_hash(FINAL)
    if base_ah != final_ah:
        raise RuntimeError(f"audio changed: {base_ah} != {final_ah}")

    # Decode the entire output. Any corrupt packet makes this command fail.
    sh(["ffmpeg", "-v", "error", "-i", str(FINAL), "-f", "null", "-"])

    qc = {
        "base_duration_seconds": base_d,
        "final_duration_seconds": final_d,
        "duration_delta_seconds": round(final_d - base_d, 6),
        "audio_sha256": final_ah,
        "audio_bit_identical_to_base": base_ah == final_ah,
        "output_sha256": hashlib.sha256(FINAL.read_bytes()).hexdigest(),
        "formula_overlays": [
            {"scene": s, "start": a, "end": b, "duration": round(b-a, 3)} for s, a, b in SCENES
        ],
    }
    (OUTDIR / "qc.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
    print(json.dumps(qc, indent=2))


def make_contact_sheet() -> None:
    frames = OUTDIR / "frames"
    frames.mkdir(exist_ok=True)
    mids = [(s, (a+b)/2) for s, a, b in SCENES]
    paths = []
    for i, (scene, t) in enumerate(mids, start=1):
        p = frames / f"{i:02d}_{scene}.jpg"
        sh(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{t:.3f}", "-i", str(FINAL), "-frames:v", "1", "-q:v", "2", str(p)])
        paths.append(p)
    sh(["montage", *map(str, paths), "-tile", "4x2", "-geometry", "480x270+5+5", str(OUTDIR / "math_formula_contact_sheet.jpg")])


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
    manifest = []
    for p, ctype in files:
        manifest.append({
            "path": p.name,
            "size": p.stat().st_size,
            "contentType": ctype,
            "hash": hashlib.sha256(p.read_bytes()).hexdigest(),
        })
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
        data = p.read_bytes()
        ureq = urllib.request.Request(upload_map[p.name], data=data, method="PUT", headers={"Content-Type": ctype})
        with urllib.request.urlopen(ureq, timeout=300) as r:
            if r.status not in (200, 201, 204):
                raise RuntimeError(f"upload failed {p.name}: {r.status}")

    fin_req = urllib.request.Request(
        created["finalizeUrl"],
        data=json.dumps({"versionId": created["versionId"]}).encode(),
        method="POST",
        headers={"Content-Type": "application/json", "X-HereNow-Client": "chatgpt/mathfix"},
    )
    with urllib.request.urlopen(fin_req, timeout=60) as r:
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
