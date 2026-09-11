#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFont

SCENES = [
    {"class": "AEObjective", "start": 42.20, "end": 57.20, "topic": "AE encoder, decoder and reconstruction objective"},
    {"class": "VAEReparameterization", "start": 109.00, "end": 131.00, "topic": "VAE posterior and reparameterization trick"},
    {"class": "VAEELBO", "start": 131.20, "end": 153.70, "topic": "ELBO decomposition and minimization form"},
    {"class": "RateDistortion", "start": 158.00, "end": 171.50, "topic": "Rate-distortion operating curve"},
    {"class": "VQQuantization", "start": 190.00, "end": 208.00, "topic": "Nearest-code quantization and VQ-VAE loss terms"},
    {"class": "StraightThrough", "start": 220.00, "end": 237.00, "topic": "Straight-through estimator, forward and backward paths"},
    {"class": "RQResidual", "start": 269.40, "end": 300.40, "topic": "Residual quantization recursion and additive code"},
    {"class": "SAESparsity", "start": 337.80, "end": 368.80, "topic": "Sparse-autoencoder objective, L1 and Top-K"},
]


def run(cmd: list[str], *, capture: bool = False) -> str:
    print("+", " ".join(cmd), flush=True)
    p = subprocess.run(cmd, check=True, text=True, stdout=subprocess.PIPE if capture else None, stderr=subprocess.STDOUT if capture else None)
    return p.stdout if capture else ""


def ffprobe(path: Path) -> dict:
    raw = run([
        "ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)
    ], capture=True)
    return json.loads(raw)


def duration(meta: dict) -> float:
    return float(meta["format"]["duration"])


def video_stream(meta: dict) -> dict:
    return next(s for s in meta["streams"] if s.get("codec_type") == "video")


def audio_hash(path: Path) -> str:
    out = run([
        "ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0", "-c", "copy", "-f", "hash", "-hash", "sha256", "-"
    ], capture=True)
    return out.strip().split("=", 1)[-1]


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def locate_scene(media: Path, class_name: str) -> Path:
    matches = sorted(media.rglob(f"{class_name}.mp4"))
    if not matches:
        raise FileNotFoundError(f"Could not locate rendered scene {class_name}.mp4 under {media}")
    return matches[-1]


def compose(base: Path, media: Path, output: Path) -> None:
    scene_paths = [locate_scene(media, s["class"]) for s in SCENES]
    cmd = ["ffmpeg", "-y", "-i", str(base)]
    for scene in scene_paths:
        cmd += ["-i", str(scene)]

    filters: list[str] = []
    previous = "[0:v]"
    for idx, spec in enumerate(SCENES, start=1):
        d = spec["end"] - spec["start"]
        fade_out_start = max(0.0, d - 0.35)
        filters.append(
            f"[{idx}:v]fps=30,scale=1280:540:flags=lanczos,"
            f"tpad=stop_mode=clone:stop_duration={d + 2:.3f},trim=duration={d:.3f},"
            f"format=yuva420p,fade=t=in:st=0:d=0.35:alpha=1,"
            f"fade=t=out:st={fade_out_start:.3f}:d=0.35:alpha=1,"
            f"setpts=PTS-STARTPTS+{spec['start']:.3f}/TB[ov{idx}]"
        )
        out = f"[v{idx}]"
        filters.append(
            f"{previous}[ov{idx}]overlay=x=0:y=80:eof_action=pass:"
            f"enable='between(t,{spec['start']:.3f},{spec['end']:.3f})'{out}"
        )
        previous = out

    output.parent.mkdir(parents=True, exist_ok=True)
    cmd += [
        "-filter_complex", ";".join(filters),
        "-map", previous,
        "-map", "0:a:0?",
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        "-movflags", "+faststart",
        "-max_muxing_queue_size", "4096",
        str(output),
    ]
    run(cmd)


def extract_frame(video: Path, t: float, dest: Path) -> None:
    run([
        "ffmpeg", "-y", "-v", "error", "-ss", f"{t:.3f}", "-i", str(video),
        "-frames:v", "1", "-q:v", "2", str(dest)
    ])


def image_metrics(base_img: Path, final_img: Path) -> tuple[float, float, float]:
    a = Image.open(base_img).convert("RGB")
    b = Image.open(final_img).convert("RGB")
    content_a = a.crop((0, 80, 1280, 620))
    content_b = b.crop((0, 80, 1280, 620))
    bottom_a = a.crop((0, 620, 1280, 720))
    bottom_b = b.crop((0, 620, 1280, 720))

    def mean_abs(x: Image.Image, y: Image.Image) -> float:
        hist = ImageChops.difference(x, y).histogram()
        count = x.width * x.height * 3
        return sum(value * n for value, n in enumerate(hist)) / count

    content_mad = mean_abs(content_a, content_b)
    bottom_mad = mean_abs(bottom_a, bottom_b)
    luma = content_b.convert("L")
    mean_luma = sum(i * n for i, n in enumerate(luma.histogram())) / (luma.width * luma.height)
    return content_mad, bottom_mad, mean_luma


def contact_sheet(images: list[Path], dest: Path) -> None:
    thumbs: list[Image.Image] = []
    for p in images:
        im = Image.open(p).convert("RGB")
        im.thumbnail((620, 349), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (640, 390), "#050509")
        canvas.paste(im, ((640 - im.width) // 2, 8))
        draw = ImageDraw.Draw(canvas)
        draw.text((18, 360), p.stem.replace("final_", "Scene "), fill="#F4F1FF")
        thumbs.append(canvas)
    rows = math.ceil(len(thumbs) / 2)
    sheet = Image.new("RGB", (1280, rows * 390), "#050509")
    for i, im in enumerate(thumbs):
        sheet.paste(im, ((i % 2) * 640, (i // 2) * 390))
    dest.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(dest, quality=92)


def qc(base: Path, final: Path, qc_dir: Path) -> Path:
    qc_dir.mkdir(parents=True, exist_ok=True)
    run(["ffmpeg", "-v", "error", "-i", str(final), "-f", "null", "-"])

    base_meta = ffprobe(base)
    final_meta = ffprobe(final)
    base_d = duration(base_meta)
    final_d = duration(final_meta)
    vs = video_stream(final_meta)
    base_audio = audio_hash(base)
    final_audio = audio_hash(final)

    if abs(base_d - final_d) > 0.12:
        raise RuntimeError(f"Duration drift too large: base={base_d}, final={final_d}")
    if int(vs["width"]) != 1280 or int(vs["height"]) != 720:
        raise RuntimeError(f"Unexpected output geometry: {vs['width']}x{vs['height']}")
    if base_audio != final_audio:
        raise RuntimeError("Audio packet hash changed; narration must remain untouched")

    metrics = []
    final_stills = []
    for i, spec in enumerate(SCENES, start=1):
        t = (spec["start"] + spec["end"]) / 2
        b = qc_dir / f"base_{i:02d}.png"
        f = qc_dir / f"final_{i:02d}.png"
        extract_frame(base, t, b)
        extract_frame(final, t, f)
        cmad, bmad, luma = image_metrics(b, f)
        if cmad < 4.0:
            raise RuntimeError(f"Scene {i} did not materially alter the content region: MAD={cmad:.3f}")
        if bmad > 4.0:
            raise RuntimeError(f"Scene {i} altered subtitle/progress region too much: MAD={bmad:.3f}")
        if luma < 7.0:
            raise RuntimeError(f"Scene {i} content region appears blank: luma={luma:.3f}")
        metrics.append((i, spec, t, cmad, bmad, luma))
        final_stills.append(f)

    sheet = qc_dir / "manim_math_contact_sheet.jpg"
    contact_sheet(final_stills, sheet)

    report = qc_dir / "MANIM_MATH_PASS_QC.md"
    lines = [
        "# Manim math pass QC",
        "",
        "## Media checks",
        "",
        f"- Base duration: {base_d:.3f} s",
        f"- Final duration: {final_d:.3f} s",
        f"- Duration delta: {final_d - base_d:+.3f} s",
        f"- Final video: {vs['width']}x{vs['height']} at {vs.get('r_frame_rate', 'unknown')}",
        f"- Base audio packet SHA-256: `{base_audio}`",
        f"- Final audio packet SHA-256: `{final_audio}`",
        f"- Final MP4 SHA-256: `{file_sha256(final)}`",
        "- Full FFmpeg decode: passed",
        "- Audio identity: passed",
        "",
        "## Formula-scene checks",
        "",
        "| # | Interval | Topic | Content-region MAD | Preserved-bottom MAD | Mean luma |",
        "|---:|---:|---|---:|---:|---:|",
    ]
    for i, spec, t, cmad, bmad, luma in metrics:
        lines.append(
            f"| {i} | {spec['start']:.2f}-{spec['end']:.2f}s | {spec['topic']} | {cmad:.2f} | {bmad:.2f} | {luma:.2f} |"
        )
    lines += [
        "",
        "The Manim panel occupies y=80..619 only. The existing top HUD, subtitles, chapter rail and corrected time-proportional progress bar remain outside the replacement region.",
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def build_site(final: Path, qc_dir: Path, site: Path) -> None:
    site.mkdir(parents=True, exist_ok=True)
    out_video = site / "latent-variable-models-manim-math-pass.mp4"
    shutil.copy2(final, out_video)
    shutil.copy2(qc_dir / "manim_math_contact_sheet.jpg", site / "manim-math-contact-sheet.jpg")
    shutil.copy2(qc_dir / "MANIM_MATH_PASS_QC.md", site / "MANIM_MATH_PASS_QC.md")
    intervals = "\n".join(
        f"<li><strong>{s['start']:.1f}-{s['end']:.1f}s</strong><span>{s['topic']}</span></li>" for s in SCENES
    )
    html = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Latent variable models: Manim math pass</title>
<style>
:root{{--bg:#08080d;--panel:#101018;--text:#f4f1ff;--muted:#aaa8b8;--purple:#8b5cf6;--orange:#f97352}}
*{{box-sizing:border-box}} body{{margin:0;background:radial-gradient(circle at 50% 0,#181126 0,#08080d 45%);color:var(--text);font:16px/1.5 system-ui,sans-serif}}
main{{max-width:1120px;margin:auto;padding:32px 20px 60px}} h1{{font-size:clamp(30px,5vw,58px);line-height:1.02;margin:0 0 12px}} p{{color:var(--muted);max-width:850px}}
video{{width:100%;border:1px solid #302547;border-radius:18px;background:#000;box-shadow:0 20px 70px #0008}}
.actions{{display:flex;gap:12px;flex-wrap:wrap;margin:18px 0 30px}} a{{color:var(--text)}} .button{{background:var(--purple);padding:11px 16px;border-radius:999px;text-decoration:none;font-weight:700}} .secondary{{background:#24212d}}
ul{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:10px;padding:0;list-style:none}} li{{background:var(--panel);border:1px solid #292534;border-radius:12px;padding:13px 15px}} li strong{{color:var(--orange);display:block;font-variant-numeric:tabular-nums}} li span{{color:var(--muted)}} img{{max-width:100%;border-radius:14px;border:1px solid #292534}}
</style></head><body><main>
<p style="color:#b89cff;font-weight:800;letter-spacing:.08em">REVISED TECHNICAL CUT</p>
<h1>Latent variable models<br>with structured Manim mathematics</h1>
<p>Eight formula-heavy intervals were rebuilt as staged mathematical explanations. The Kokoro narration, subtitles, section timing and corrected time-proportional progress rail remain unchanged.</p>
<video controls preload="metadata"><source src="latent-variable-models-manim-math-pass.mp4" type="video/mp4"></video>
<div class="actions"><a class="button" href="latent-variable-models-manim-math-pass.mp4" download>Download MP4</a><a class="button secondary" href="MANIM_MATH_PASS_QC.md">Read QC report</a></div>
<h2>Rebuilt intervals</h2><ul>{intervals}</ul>
<h2>Contact sheet</h2><img src="manim-math-contact-sheet.jpg" alt="Contact sheet of eight Manim formula scenes">
</main></body></html>'''
    (site / "index.html").write_text(html, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--media", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--qc", type=Path, required=True)
    ap.add_argument("--site", type=Path, required=True)
    args = ap.parse_args()
    compose(args.base, args.media, args.output)
    qc(args.base, args.output, args.qc)
    build_site(args.output, args.qc, args.site)


if __name__ == "__main__":
    main()
