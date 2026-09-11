#!/usr/bin/env python3
from __future__ import annotations

import bisect
import hashlib
import html
import json
import math
import os
import re
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import requests
import soundfile as sf
from kokoro import KPipeline
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H = 1280, 720
RENDER_FPS = 20
OUTPUT_FPS = 30
SR = 24_000
OUT = Path("output")
OUT.mkdir(parents=True, exist_ok=True)

BG = (5, 6, 11)
BG2 = (17, 18, 29)
WHITE = (246, 247, 252)
GREY = (156, 159, 176)
GREY_DARK = (51, 54, 68)
PURPLE = (116, 63, 246)
PURPLE_LIGHT = (181, 151, 255)
ORANGE = (246, 116, 72)
GREEN = (142, 242, 112)
RED = (242, 74, 91)
CYAN = (99, 231, 239)
BLACK = (0, 0, 0)

CHAPTERS = [
    "Foundations",
    "AE",
    "VAE",
    "VQ-VAE",
    "RQ-VAE",
    "SAE",
    "RAE",
    "Choosing a latent space",
]

SOURCES = [
    {
        "label": "Reducing the Dimensionality of Data with Neural Networks",
        "authors": "G. E. Hinton and R. R. Salakhutdinov, 2006",
        "url": "https://www.science.org/doi/10.1126/science.1127647",
    },
    {
        "label": "Auto-Encoding Variational Bayes",
        "authors": "D. P. Kingma and M. Welling, 2013",
        "url": "https://arxiv.org/abs/1312.6114",
    },
    {
        "label": "Neural Discrete Representation Learning",
        "authors": "A. van den Oord, O. Vinyals and K. Kavukcuoglu, 2017",
        "url": "https://arxiv.org/abs/1711.00937",
    },
    {
        "label": "Autoregressive Image Generation using Residual Quantization",
        "authors": "D. Lee et al., 2022",
        "url": "https://arxiv.org/abs/2203.01941",
    },
    {
        "label": "Sparse Autoencoders Find Highly Interpretable Features in Language Models",
        "authors": "H. Cunningham et al., 2023",
        "url": "https://arxiv.org/abs/2309.08600",
    },
    {
        "label": "Diffusion Transformers with Representation Autoencoders",
        "authors": "B. Zheng et al., 2025",
        "url": "https://arxiv.org/abs/2510.11690",
    },
]


@dataclass
class Segment:
    chapter: str
    headline: str
    narration: str
    visual: str


SEGMENTS = [
    Segment(
        "Foundations",
        "Pixels are observations, not explanations",
        "A single image can contain millions of measured values. Yet most of those values move together. The umbrella shape, the robot pose, the direction of the rain, and the lighting explain far more than a raw list of pixels.",
        "observation",
    ),
    Segment(
        "Foundations",
        "A latent variable is a hidden coordinate",
        "A latent variable is an internal coordinate used to explain or reconstruct an observation. It may correspond to a meaningful factor, but the model is not required to name it. The useful question is what structure the coordinate preserves.",
        "latent",
    ),
    Segment(
        "Foundations",
        "One family, different constraints",
        "Autoencoders, variational autoencoders, vector quantized models, residual quantizers, sparse autoencoders, and representation autoencoders all create internal representations. They differ in what they force those representations to become.",
        "family",
    ),
    Segment(
        "AE",
        "The deterministic autoencoder",
        "An autoencoder has two learned maps. The encoder turns an input x into a latent vector z. The decoder maps z back into a reconstruction, x hat. Training adjusts both maps to reduce reconstruction error.",
        "ae",
    ),
    Segment(
        "AE",
        "The bottleneck forces selection",
        "When the latent vector is narrower than the input, the network cannot copy every number independently. It must retain regularities that help the decoder recover the input. That bottleneck is a learned form of nonlinear compression.",
        "bottleneck",
    ),
    Segment(
        "AE",
        "A point, not a probability distribution",
        "For a standard autoencoder, the same input produces the same latent point. A common objective is the squared distance between x and x hat, although perceptual or task-specific reconstruction losses can also be used.",
        "aeloss",
    ),
    Segment(
        "AE",
        "Compression is a rate-distortion choice",
        "A smaller latent rate usually throws away more detail. A larger latent can reconstruct more accurately, but it may learn an identity-like shortcut. Architecture, noise, regularization, and the downstream task decide which information survives.",
        "tradeoff",
    ),
    Segment(
        "AE",
        "Good reconstruction does not guarantee sampling",
        "A plain autoencoder only learns where encoded training examples happen to land. Empty regions between those points may decode to nonsense. To generate by sampling, we need either a learned prior or a latent space constrained to have a usable distribution.",
        "manifold",
    ),
    Segment(
        "VAE",
        "Encode a distribution instead of one point",
        "A variational autoencoder replaces the deterministic latent point with an approximate posterior distribution. For a diagonal Gaussian, the encoder predicts a mean vector and a scale vector for every input.",
        "vae",
    ),
    Segment(
        "VAE",
        "Reparameterization keeps gradients usable",
        "Sampling is written as z equals mu plus sigma multiplied by epsilon, where epsilon comes from a standard normal distribution. Randomness is isolated in epsilon, while gradients still flow through mu and sigma.",
        "reparam",
    ),
    Segment(
        "VAE",
        "The ELBO balances two jobs",
        "The variational objective contains a reconstruction term and a K L divergence term. Reconstruction asks the decoder to preserve the example. The K L term asks each approximate posterior to stay compatible with the chosen prior.",
        "elbo",
    ),
    Segment(
        "VAE",
        "Regularization fills the latent space",
        "Without regularization, encoded examples can form isolated islands. Pressure toward a shared prior makes nearby samples more likely to decode into plausible outputs. The latent space becomes easier to interpolate and sample, at a cost in reconstruction fidelity.",
        "prior",
    ),
    Segment(
        "VAE",
        "Generation becomes a defined operation",
        "At generation time, sample z from the prior and pass it through the decoder. The model is probabilistic because it connects a prior, a likelihood, and an approximate posterior, not merely because noise was added somewhere in a network.",
        "sampling",
    ),
    Segment(
        "VAE",
        "Posterior collapse is a real failure mode",
        "If the decoder can model the data without using z, the approximate posterior may collapse toward the prior and carry little information. Capacity schedules, decoder design, richer priors, and objective changes are common ways to fight this.",
        "collapse",
    ),
    Segment(
        "VQ-VAE",
        "Replace coordinates with codebook entries",
        "A vector quantized variational autoencoder first produces continuous encoder vectors, then replaces each vector with its nearest entry in a learned codebook. The decoder receives the selected embeddings rather than the original encoder outputs.",
        "vq",
    ),
    Segment(
        "VQ-VAE",
        "Nearest-neighbor lookup creates discrete states",
        "Each location chooses an integer code. If the codebook contains K entries, one location carries at most log base two of K bits before considering entropy coding. The representation is now a grid or sequence of symbols.",
        "nearest",
    ),
    Segment(
        "VQ-VAE",
        "Images become token maps",
        "The silver robot is no longer represented by one smooth coordinate cloud. It becomes a map of code indices. Repeated visual patterns can reuse the same symbols, much like recurring subwords reuse tokens in a language model.",
        "tokens",
    ),
    Segment(
        "VQ-VAE",
        "Three pressures train the bottleneck",
        "The usual objective combines reconstruction, a codebook update term, and a commitment term. Stop-gradient operations route learning signals so the encoder commits to selected entries while the codebook moves toward encoder outputs.",
        "vqloss",
    ),
    Segment(
        "VQ-VAE",
        "A separate prior models token sequences",
        "The autoencoder learns useful discrete codes. A second model can then learn the probability of those code sequences. Generation samples tokens from that prior and decodes them into pixels, audio, or another observation space.",
        "tokenprior",
    ),
    Segment(
        "VQ-VAE",
        "Codebooks can be used badly",
        "Some entries may never be selected, while a few entries absorb too much traffic. This codebook collapse reduces effective capacity. Usage statistics, initialization, replacement rules, and quantizer design matter as much as nominal codebook size.",
        "deadcodes",
    ),
    Segment(
        "RQ-VAE",
        "Quantize the residual, then do it again",
        "Residual-quantized V A E uses several quantization stages. The first codeword approximates the encoder vector. The next stage quantizes what remains, and later stages keep correcting the residual.",
        "rq",
    ),
    Segment(
        "RQ-VAE",
        "A latent vector becomes a sum of codewords",
        "Instead of selecting one enormous codebook entry, the model represents a vector as the sum of several entries. The first code can capture a coarse pattern, while later codes add smaller corrections.",
        "rqsum",
    ),
    Segment(
        "RQ-VAE",
        "Coarse structure arrives before fine detail",
        "For our robot image, an early code can identify the broad silhouette. Another adds the umbrella. Later residual codes recover rain direction, highlights, and edge details. This is a useful visual analogy, not a guarantee that each level receives a human-readable meaning.",
        "coarsefine",
    ),
    Segment(
        "RQ-VAE",
        "Depth expands the representational rate",
        "With M stages and K choices per stage, a fixed position can express up to M times log base two of K nominal bits. Residual quantization trades a deeper code stack for better approximation without requiring one exponentially large codebook.",
        "rate",
    ),
    Segment(
        "RQ-VAE",
        "The sequence model sees stacked codes",
        "The downstream prior must predict several codes for each spatial position. That changes sequence length, factorization, and sampling cost. R Q V A E is therefore a systems trade between spatial compression, quantization depth, and prior complexity.",
        "stacked",
    ),
    Segment(
        "SAE",
        "Sparse autoencoders use a wide dictionary",
        "A sparse autoencoder often has more latent units than input dimensions, but activates only a small subset for each example. The representation is overcomplete in width and selective in activity.",
        "sae",
    ),
    Segment(
        "SAE",
        "Reconstruction plus a sparsity constraint",
        "One training objective adds an L one penalty to the reconstruction loss. Other variants keep only the top K activations. The pressure is simple: explain the input using as few active features as practical.",
        "sparsity",
    ),
    Segment(
        "SAE",
        "Sparse features can separate superposed signals",
        "When many concepts share a smaller dense activation space, a sparse dictionary can sometimes recover directions that activate more selectively. Researchers use this idea to inspect neural network activations and search for interpretable features.",
        "features",
    ),
    Segment(
        "SAE",
        "Interpretability is measured, not assumed",
        "A sparse unit is not automatically a clean concept. Features may split, merge, stay polysemantic, or depend on the dataset used to train the autoencoder. Causal interventions and held-out evaluations are needed before assigning a meaning.",
        "saecaveat",
    ),
    Segment(
        "RAE",
        "Representation autoencoders start elsewhere",
        "Here, R A E means the representation autoencoder proposed for diffusion transformers. Instead of training the encoder only to reconstruct pixels, it begins with a pretrained representation encoder such as DINO, SigLIP, or M A E, then trains a decoder back to images.",
        "rae",
    ),
    Segment(
        "RAE",
        "The encoder can remain frozen",
        "The pretrained encoder already organizes images by semantic structure. Keeping it frozen preserves that geometry while the new decoder learns to reconstruct pixels from the representation tokens.",
        "frozen",
    ),
    Segment(
        "RAE",
        "Rich latents are deliberately high-dimensional",
        "Classical latent diffusion compresses aggressively because denoising cost grows with latent size. Representation autoencoders accept wider, semantically rich tokens. The goal is not the smallest bottleneck. It is a useful space for a generative transformer.",
        "rich",
    ),
    Segment(
        "RAE",
        "The generator must adapt to the new geometry",
        "A diffusion transformer now denoises in this high-dimensional representation space, then the decoder returns to pixels. The cited R A E work adds architectural changes to make optimization in that space practical. The autoencoder itself can still be deterministic.",
        "dit",
    ),
    Segment(
        "RAE",
        "Not every latent model is trying to compress",
        "The autoencoder family is better understood as representation design. Some models remove bits. Some impose a probability distribution. Some create discrete symbols, sparse features, or rich semantic tokens. Compression is only one possible objective.",
        "notcompression",
    ),
    Segment(
        "Choosing a latent space",
        "Four design axes clarify the family",
        "Ask whether the representation is deterministic or probabilistic, continuous or discrete, dense or sparse, and compact or rich. These axes describe the bottleneck more precisely than the word autoencoder alone.",
        "axes",
    ),
    Segment(
        "Choosing a latent space",
        "Choose by the operation you need",
        "Use an A E when reconstruction or dimensionality reduction is central. Use a V A E when a smooth probabilistic latent model matters. Use V Q or residual quantization when symbols and learned token priors matter. Use an S A E to seek sparse explanatory directions. Use an R A E when semantic representation quality is the starting point for generation.",
        "chooser",
    ),
    Segment(
        "Choosing a latent space",
        "The constraint defines the representation",
        "The encoder and decoder are only the outer shell. The real model is the constraint placed between them. A point, a distribution, a code, a residual stack, a sparse dictionary, or a rich semantic token grid. Change that constraint, and you change what the model can learn.",
        "final",
    ),
]


BOLD_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/opentype/noto/NotoSans-Bold.ttf",
]
REG_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/opentype/noto/NotoSans-Regular.ttf",
]
MONO_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansMono-Regular.ttf",
]


def find_font(paths: list[str]) -> str:
    for path in paths:
        if Path(path).exists():
            return path
    raise FileNotFoundError(f"No suitable font found in {paths}")


BOLD_PATH = find_font(BOLD_PATHS)
REG_PATH = find_font(REG_PATHS)
MONO_PATH = find_font(MONO_PATHS)
FONT_CACHE: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}


def font(size: int, kind: str = "regular") -> ImageFont.FreeTypeFont:
    path = {"bold": BOLD_PATH, "regular": REG_PATH, "mono": MONO_PATH}[kind]
    key = (path, int(size))
    if key not in FONT_CACHE:
        FONT_CACHE[key] = ImageFont.truetype(path, int(size))
    return FONT_CACHE[key]


def clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def ease_out(v: float) -> float:
    v = clamp01(v)
    return 1.0 - (1.0 - v) ** 3


def speakify(text: str) -> str:
    replacements = [
        (r"RQ-VAE", "R Q V A E"),
        (r"VQ-VAE", "V Q V A E"),
        (r"\bVAE\b", "V A E"),
        (r"\bSAE\b", "S A E"),
        (r"\bRAE\b", "R A E"),
        (r"\bAE\b", "A E"),
        (r"\bKL\b", "K L"),
        (r"\bELBO\b", "elbo"),
        (r"\bDiT\b", "D I T"),
        (r"\bDINO\b", "Dino"),
        (r"\bMAE\b", "M A E"),
        (r"L1", "L one"),
        (r"top-k", "top K"),
        (r"log2", "log base two"),
        (r"x-hat", "x hat"),
    ]
    result = text
    for pattern, replacement in replacements:
        result = re.sub(pattern, replacement, result)
    return result


def synthesize(pipeline: KPipeline, text: str, speed: float = 1.04) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for _graphemes, _phonemes, audio in pipeline(speakify(text), voice="am_liam", speed=speed):
        if audio is None:
            continue
        if hasattr(audio, "detach"):
            arr = audio.detach().cpu().numpy().astype(np.float32)
        else:
            arr = np.asarray(audio, dtype=np.float32)
        chunks.append(arr.reshape(-1))
    if not chunks:
        raise RuntimeError(f"Kokoro produced no audio for: {text[:80]}")
    data = np.concatenate(chunks)
    nz = np.where(np.abs(data) > 2e-4)[0]
    if nz.size:
        left = max(0, int(nz[0]) - int(0.035 * SR))
        right = min(len(data), int(nz[-1]) + int(0.08 * SR))
        data = data[left:right]
    fade = min(int(0.012 * SR), len(data) // 4)
    if fade > 1:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        data[:fade] *= ramp
        data[-fade:] *= ramp[::-1]
    peak = float(np.max(np.abs(data))) if data.size else 1.0
    if peak > 0:
        data = data * min(1.0, 0.91 / peak)
    return data.astype(np.float32)


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(round(seconds * SR)), dtype=np.float32)


def split_caption(text: str, max_chars: int = 67) -> list[str]:
    words = text.split()
    chunks: list[str] = []
    current: list[str] = []
    for word in words:
        candidate = " ".join(current + [word])
        if current and len(candidate) > max_chars:
            chunks.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        chunks.append(" ".join(current))
    return chunks


def sec_to_vtt(t: float) -> str:
    t = max(0.0, t)
    ms = int(round(t * 1000))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def sec_to_clock(t: float) -> str:
    sec = max(0, int(t + 0.5))
    return f"{sec // 60}:{sec % 60:02d}"


def build_timeline() -> tuple[list[dict[str, Any]], list[dict[str, Any]], np.ndarray, float]:
    print("Loading Kokoro-82M / am_liam...", flush=True)
    pipeline = KPipeline(lang_code="a", device="cpu")
    items: list[dict[str, Any]] = []
    cues: list[dict[str, Any]] = []
    audio_blocks: list[np.ndarray] = []
    cursor = 0.0

    opening_duration = 2.15
    items.append({
        "kind": "opening",
        "chapter": "Foundations",
        "headline": "Latent variable models",
        "start": cursor,
        "voice_end": cursor,
        "end": cursor + opening_duration,
    })
    audio_blocks.append(silence(opening_duration))
    cursor += opening_duration

    previous_chapter: str | None = None
    for index, seg in enumerate(SEGMENTS):
        if previous_chapter is not None and seg.chapter != previous_chapter:
            chapter_duration = 1.28
            items.append({
                "kind": "chapter",
                "chapter": seg.chapter,
                "headline": seg.chapter,
                "start": cursor,
                "voice_end": cursor,
                "end": cursor + chapter_duration,
            })
            audio_blocks.append(silence(chapter_duration))
            cursor += chapter_duration

        print(f"TTS {index + 1:02d}/{len(SEGMENTS)}: {seg.headline}", flush=True)
        audio = synthesize(pipeline, seg.narration)
        start = cursor
        voice_end = start + len(audio) / SR
        next_chapter = SEGMENTS[index + 1].chapter if index + 1 < len(SEGMENTS) else None
        gap = 0.48 if next_chapter != seg.chapter else 0.26
        end = voice_end + gap
        item = {
            "kind": "segment",
            "chapter": seg.chapter,
            "headline": seg.headline,
            "narration": seg.narration,
            "visual": seg.visual,
            "start": start,
            "voice_end": voice_end,
            "end": end,
            "index": index,
        }
        items.append(item)
        audio_blocks.append(audio)
        audio_blocks.append(silence(gap))

        cap_chunks = split_caption(seg.narration)
        weights = [max(1, len(chunk.split())) for chunk in cap_chunks]
        total_weight = sum(weights)
        cap_cursor = start
        for chunk, weight in zip(cap_chunks, weights):
            duration = (voice_end - start) * weight / total_weight
            cues.append({"start": cap_cursor, "end": cap_cursor + duration, "text": chunk})
            cap_cursor += duration

        cursor = end
        previous_chapter = seg.chapter

    outro_duration = 2.4
    items.append({
        "kind": "outro",
        "chapter": "Choosing a latent space",
        "headline": "The bottleneck is the model",
        "start": cursor,
        "voice_end": cursor,
        "end": cursor + outro_duration,
    })
    audio_blocks.append(silence(outro_duration))
    cursor += outro_duration

    speech = np.concatenate(audio_blocks)
    total_duration = len(speech) / SR
    n = len(speech)
    t = np.arange(n, dtype=np.float32) / SR
    ambient = (
        0.0080 * np.sin(2 * np.pi * 55.0 * t)
        + 0.0045 * np.sin(2 * np.pi * 82.41 * t + 0.4)
        + 0.0032 * np.sin(2 * np.pi * 110.0 * t + 1.1)
    ).astype(np.float32)
    slow = (0.72 + 0.28 * np.sin(2 * np.pi * 0.018 * t + 0.6)).astype(np.float32)
    ambient *= slow
    fade_len = min(int(1.8 * SR), n // 3)
    if fade_len > 1:
        ramp = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
        ambient[:fade_len] *= ramp
        ambient[-fade_len:] *= ramp[::-1]
    mixed = speech + ambient
    peak = float(np.max(np.abs(mixed)))
    if peak > 0.96:
        mixed *= 0.96 / peak

    sf.write(OUT / "latent-models-kokoro.wav", mixed, SR, subtype="PCM_16")
    with (OUT / "captions.vtt").open("w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for cue in cues:
            f.write(f"{sec_to_vtt(cue['start'])} --> {sec_to_vtt(cue['end'])}\n")
            f.write(cue["text"] + "\n\n")
    with (OUT / "timeline.json").open("w", encoding="utf-8") as f:
        json.dump({"duration": total_duration, "items": items, "cues": cues}, f, indent=2)
    return items, cues, mixed, total_duration


def text_size(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.FreeTypeFont) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=fnt, stroke_width=0)
    return box[2] - box[0], box[3] - box[1]


def fit_font(draw: ImageDraw.ImageDraw, text: str, max_width: int, start_size: int, kind: str = "bold", min_size: int = 18) -> ImageFont.FreeTypeFont:
    size = start_size
    while size > min_size:
        fnt = font(size, kind)
        if text_size(draw, text, fnt)[0] <= max_width:
            return fnt
        size -= 1
    return font(min_size, kind)


def wrap_text(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        candidate = " ".join(current + [word])
        if current and text_size(draw, candidate, fnt)[0] > max_width:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return lines


def rounded(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], fill: tuple[int, int, int] | tuple[int, int, int, int], outline: tuple[int, int, int] | None = None, width: int = 2, radius: int = 18) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def line_arrow(draw: ImageDraw.ImageDraw, start: tuple[float, float], end: tuple[float, float], color: tuple[int, int, int] = WHITE, width: int = 4, progress: float = 1.0) -> None:
    progress = clamp01(progress)
    x0, y0 = start
    x1, y1 = end
    x = x0 + (x1 - x0) * progress
    y = y0 + (y1 - y0) * progress
    draw.line((x0, y0, x, y), fill=color, width=width)
    if progress > 0.92:
        angle = math.atan2(y1 - y0, x1 - x0)
        head = 13
        a1 = angle + math.pi * 0.82
        a2 = angle - math.pi * 0.82
        p1 = (x + head * math.cos(a1), y + head * math.sin(a1))
        p2 = (x + head * math.cos(a2), y + head * math.sin(a2))
        draw.polygon([(x, y), p1, p2], fill=color)


def glow_ellipse(im: Image.Image, box: tuple[int, int, int, int], color: tuple[int, int, int], alpha: int = 150, blur: int = 24) -> None:
    layer = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    d.ellipse(box, fill=(*color, alpha))
    layer = layer.filter(ImageFilter.GaussianBlur(blur))
    im.alpha_composite(layer)


def glow_rect(im: Image.Image, box: tuple[int, int, int, int], color: tuple[int, int, int], alpha: int = 130, blur: int = 24, radius: int = 18) -> None:
    layer = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    d.rounded_rectangle(box, radius=radius, fill=(*color, alpha))
    layer = layer.filter(ImageFilter.GaussianBlur(blur))
    im.alpha_composite(layer)


def glow_text(im: Image.Image, xy: tuple[int, int], text: str, fnt: ImageFont.FreeTypeFont, fill: tuple[int, int, int] = WHITE, glow: tuple[int, int, int] = PURPLE, anchor: str = "mm", stroke_width: int = 0) -> None:
    layer = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    d.text(xy, text, font=fnt, fill=(*glow, 220), anchor=anchor, stroke_width=stroke_width, stroke_fill=(*glow, 220))
    blurred = layer.filter(ImageFilter.GaussianBlur(16))
    im.alpha_composite(blurred)
    d2 = ImageDraw.Draw(im)
    d2.text(xy, text, font=fnt, fill=fill, anchor=anchor, stroke_width=stroke_width, stroke_fill=BLACK)


def draw_robot(draw: ImageDraw.ImageDraw, cx: int, cy: int, scale: float = 1.0, umbrella: bool = True, accent: tuple[int, int, int] = PURPLE) -> None:
    s = scale
    def P(x: float, y: float) -> tuple[int, int]:
        return int(cx + x * s), int(cy + y * s)
    width = max(2, int(4 * s))
    draw.rounded_rectangle((*P(-48, -72), *P(48, 8)), radius=max(6, int(14 * s)), fill=(23, 25, 34), outline=WHITE, width=width)
    draw.ellipse((*P(-27, -47), *P(-11, -31)), fill=CYAN, outline=WHITE, width=max(1, int(2 * s)))
    draw.ellipse((*P(11, -47), *P(27, -31)), fill=CYAN, outline=WHITE, width=max(1, int(2 * s)))
    draw.line((*P(-20, -10), *P(20, -10)), fill=GREY, width=max(2, int(3 * s)))
    draw.rounded_rectangle((*P(-58, 11), *P(58, 100)), radius=max(8, int(13 * s)), fill=(17, 19, 28), outline=WHITE, width=width)
    draw.rounded_rectangle((*P(-31, 30), *P(31, 68)), radius=max(4, int(8 * s)), fill=accent, outline=WHITE, width=max(1, int(2 * s)))
    draw.line((*P(-58, 34), *P(-88, 77)), fill=WHITE, width=width)
    draw.line((*P(58, 34), *P(88, 77)), fill=WHITE, width=width)
    draw.ellipse((*P(-99, 68), *P(-79, 88)), fill=(23, 25, 34), outline=WHITE, width=max(1, int(2 * s)))
    draw.ellipse((*P(79, 68), *P(99, 88)), fill=(23, 25, 34), outline=WHITE, width=max(1, int(2 * s)))
    draw.line((*P(-31, 100), *P(-43, 145)), fill=WHITE, width=width)
    draw.line((*P(31, 100), *P(43, 145)), fill=WHITE, width=width)
    draw.line((*P(-61, 145), *P(-25, 145)), fill=WHITE, width=width)
    draw.line((*P(25, 145), *P(61, 145)), fill=WHITE, width=width)
    draw.line((*P(0, -72), *P(0, -94)), fill=WHITE, width=max(2, int(3 * s)))
    draw.ellipse((*P(-7, -106), *P(7, -92)), fill=accent, outline=WHITE, width=max(1, int(2 * s)))
    if umbrella:
        draw.line((*P(89, 78), *P(89, -123)), fill=WHITE, width=max(2, int(3 * s)))
        draw.arc((*P(-15, -180), *P(193, -62)), 184, 356, fill=accent, width=max(5, int(8 * s)))
        draw.line((*P(-12, -119), *P(190, -119)), fill=WHITE, width=max(2, int(3 * s)))
        draw.arc((*P(72, 58), *P(105, 96)), 0, 180, fill=WHITE, width=max(2, int(3 * s)))


def draw_box_label(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], title: str, subtitle: str = "", active: bool = False, accent: tuple[int, int, int] = PURPLE) -> None:
    fill = (21, 22, 33) if not active else (37, 26, 71)
    outline = accent if active else GREY_DARK
    rounded(draw, box, fill, outline, width=3 if active else 2, radius=20)
    x0, y0, x1, y1 = box
    draw.text(((x0 + x1) // 2, y0 + 31), title, font=font(27, "bold"), fill=WHITE, anchor="mm")
    if subtitle:
        fnt = fit_font(draw, subtitle, x1 - x0 - 24, 18, "regular", 14)
        draw.text(((x0 + x1) // 2, y1 - 27), subtitle, font=fnt, fill=PURPLE_LIGHT if active else GREY, anchor="mm")


def draw_pipeline(draw: ImageDraw.ImageDraw, labels: list[str], y: int = 355, active: int = -1, x0: int = 145, x1: int = 1135) -> None:
    n = len(labels)
    gap = 26
    box_w = int((x1 - x0 - gap * (n - 1)) / n)
    for i, label in enumerate(labels):
        left = x0 + i * (box_w + gap)
        draw_box_label(draw, (left, y - 62, left + box_w, y + 62), label, active=i == active)
        if i + 1 < n:
            line_arrow(draw, (left + box_w + 5, y), (left + box_w + gap - 5, y), PURPLE_LIGHT if i < active else WHITE, 3)


def draw_codebook(draw: ImageDraw.ImageDraw, x: int, y: int, cols: int = 5, rows: int = 4, cell: int = 43, active: int = 7) -> None:
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            bx = x + c * (cell + 8)
            by = y + r * (cell + 8)
            rounded(draw, (bx, by, bx + cell, by + cell), PURPLE if idx == active else (24, 25, 37), WHITE if idx == active else GREY_DARK, width=2, radius=9)
            draw.text((bx + cell // 2, by + cell // 2), str(idx), font=font(15, "mono"), fill=WHITE if idx == active else GREY, anchor="mm")


def draw_token_grid(draw: ImageDraw.ImageDraw, x: int, y: int, values: list[int], cols: int = 6, cell: int = 54) -> None:
    for i, value in enumerate(values):
        r, c = divmod(i, cols)
        bx = x + c * (cell + 8)
        by = y + r * (cell + 8)
        active = i % 5 in (0, 1)
        rounded(draw, (bx, by, bx + cell, by + cell), (43, 30, 83) if active else (22, 23, 34), PURPLE_LIGHT if active else GREY_DARK, width=2, radius=10)
        draw.text((bx + cell // 2, by + cell // 2), str(value), font=font(18, "mono"), fill=WHITE, anchor="mm")


def draw_formula(draw: ImageDraw.ImageDraw, text: str, y: int, color: tuple[int, int, int] = WHITE, size: int = 45) -> None:
    fnt = fit_font(draw, text, 1120, size, "mono", 23)
    rounded(draw, (110, y - 52, 1170, y + 52), (15, 16, 24), GREY_DARK, width=2, radius=18)
    draw.text((640, y), text, font=fnt, fill=color, anchor="mm")


def draw_headline(im: Image.Image, headline: str) -> None:
    d = ImageDraw.Draw(im)
    fnt = fit_font(d, headline, 1110, 49, "bold", 29)
    glow_text(im, (640, 130), headline, fnt, WHITE, PURPLE, "mm")


def scene_opening() -> Image.Image:
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    glow_ellipse(im, (380, 185, 900, 595), PURPLE, 120, 65)
    glow_text(im, (640, 275), "LATENT", font(104, "bold"), WHITE, PURPLE, "mm")
    glow_text(im, (640, 370), "VARIABLE MODELS", font(72, "bold"), WHITE, PURPLE, "mm")
    d = ImageDraw.Draw(im)
    d.text((640, 454), "AE  |  VAE  |  VQ-VAE  |  RQ-VAE  |  SAE  |  RAE", font=font(29, "mono"), fill=PURPLE_LIGHT, anchor="mm")
    d.text((640, 516), "What changes when we change the bottleneck?", font=font(27), fill=GREY, anchor="mm")
    return im


def scene_chapter(chapter: str) -> Image.Image:
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    idx = CHAPTERS.index(chapter) + 1
    glow_text(im, (640, 270), f"{idx:02d}", font(105, "mono"), PURPLE_LIGHT, PURPLE, "mm")
    d = ImageDraw.Draw(im)
    fnt = fit_font(d, chapter, 1080, 76, "bold", 42)
    glow_text(im, (640, 395), chapter, fnt, WHITE, PURPLE, "mm")
    d.line((335, 475, 945, 475), fill=PURPLE, width=5)
    return im


def scene_outro() -> Image.Image:
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    glow_ellipse(im, (420, 180, 860, 580), PURPLE, 110, 58)
    glow_text(im, (640, 305), "THE BOTTLENECK", font(68, "bold"), WHITE, PURPLE, "mm")
    glow_text(im, (640, 390), "IS THE MODEL", font(74, "bold"), WHITE, PURPLE, "mm")
    d = ImageDraw.Draw(im)
    d.text((640, 500), "latent-variable-models.here.now", font=font(22, "mono"), fill=GREY, anchor="mm")
    return im


def scene_for_segment(seg: Segment) -> Image.Image:
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw_headline(im, seg.headline)
    d = ImageDraw.Draw(im)
    v = seg.visual

    if v == "observation":
        glow_ellipse(im, (105, 200, 515, 590), PURPLE, 80, 42)
        draw_robot(d, 310, 385, 1.02)
        line_arrow(d, (510, 370), (660, 370), WHITE, 5)
        rounded(d, (700, 210, 1110, 535), (16, 17, 26), GREY_DARK, 2, 22)
        for r in range(8):
            for c in range(10):
                x = 730 + c * 35
                y = 246 + r * 33
                k = (r * 17 + c * 11) % 100
                fill = PURPLE if k < 18 else (72 + k, 72 + k // 2, 88 + k // 3)
                d.rectangle((x, y, x + 24, y + 22), fill=fill)
        d.text((905, 570), "millions of correlated values", font=font(21), fill=GREY, anchor="mm")

    elif v == "latent":
        draw_robot(d, 245, 385, 0.9)
        line_arrow(d, (420, 360), (555, 360), WHITE, 4)
        glow_rect(im, (570, 210, 1100, 525), PURPLE, 100, 34)
        rounded(d, (590, 225, 1080, 505), (19, 20, 31), PURPLE, 3, 25)
        knobs = [(710, 320, "pose"), (840, 320, "shape"), (970, 320, "light"), (775, 430, "rain"), (905, 430, "style")]
        for x, y, label in knobs:
            d.ellipse((x - 38, y - 38, x + 38, y + 38), fill=(31, 32, 45), outline=WHITE, width=3)
            angle = (x + y) % 200 / 200 * math.pi * 1.5
            d.line((x, y, x + 27 * math.cos(angle), y + 27 * math.sin(angle)), fill=PURPLE_LIGHT, width=4)
            d.text((x, y + 61), label, font=font(18), fill=GREY, anchor="mm")

    elif v == "family":
        labels = ["AE", "VAE", "VQ-VAE", "RQ-VAE", "SAE", "RAE"]
        subtitles = ["point", "distribution", "code", "code stack", "sparse", "semantic"]
        for i, (lab, sub) in enumerate(zip(labels, subtitles)):
            c = i % 3
            r = i // 3
            x0 = 105 + c * 360
            y0 = 230 + r * 165
            active = i in (1, 3, 5)
            draw_box_label(d, (x0, y0, x0 + 320, y0 + 125), lab, sub, active=active)

    elif v in {"ae", "vae", "vq", "rae"}:
        draw_robot(d, 155, 370, 0.62)
        if v == "ae":
            draw_pipeline(d, ["ENCODER", "z", "DECODER"], y=365, active=1, x0=330, x1=1010)
        elif v == "vae":
            draw_pipeline(d, ["ENCODER", "mu, sigma", "sample z", "DECODER"], y=365, active=1, x0=295, x1=1085)
            for j in range(13):
                a = 2 * math.pi * j / 13
                x = 690 + int(math.cos(a) * (31 + (j % 3) * 8))
                y = 365 + int(math.sin(a) * (22 + (j % 4) * 6))
                d.ellipse((x - 4, y - 4, x + 4, y + 4), fill=PURPLE_LIGHT)
        elif v == "vq":
            draw_box_label(d, (315, 295, 515, 435), "ENCODER", "continuous vectors")
            line_arrow(d, (525, 365), (605, 365), WHITE, 4)
            draw_codebook(d, 620, 260, active=7)
            line_arrow(d, (890, 365), (960, 365), WHITE, 4)
            draw_box_label(d, (970, 295, 1145, 435), "DECODER", "selected entries")
        else:
            draw_box_label(d, (310, 285, 565, 445), "FROZEN", "representation encoder", active=True)
            line_arrow(d, (580, 365), (690, 365), WHITE, 4)
            draw_token_grid(d, 710, 280, [8, 3, 11, 4, 8, 15, 2, 9, 7, 7, 12, 4], cols=4, cell=48)
            line_arrow(d, (950, 365), (1020, 365), WHITE, 4)
            draw_box_label(d, (1030, 295, 1170, 435), "DECODER", "pixels")

    elif v == "bottleneck":
        d.text((190, 365), "784", font=font(76, "mono"), fill=WHITE, anchor="mm")
        d.text((1090, 365), "784", font=font(76, "mono"), fill=WHITE, anchor="mm")
        d.polygon([(300, 225), (560, 315), (560, 415), (300, 505)], fill=(25, 26, 38), outline=WHITE)
        glow_rect(im, (565, 290, 715, 440), PURPLE, 130, 28)
        rounded(d, (575, 300, 705, 430), PURPLE, WHITE, 3, 22)
        d.text((640, 365), "32", font=font(62, "mono"), fill=WHITE, anchor="mm")
        d.polygon([(720, 315), (980, 225), (980, 505), (720, 415)], fill=(25, 26, 38), outline=WHITE)
        d.text((640, 505), "the narrow channel forces selection", font=font(24), fill=GREY, anchor="mm")

    elif v == "aeloss":
        draw_robot(d, 250, 385, 0.72)
        draw_robot(d, 1030, 385, 0.72, accent=PURPLE_LIGHT)
        line_arrow(d, (420, 365), (830, 365), PURPLE_LIGHT, 5)
        draw_formula(d, "L_rec = || x - x_hat ||^2", 520, PURPLE_LIGHT, 43)
        d.text((640, 260), "same input  ->  same latent point", font=font(28, "mono"), fill=WHITE, anchor="mm")

    elif v == "tradeoff":
        x0, y0, x1, y1 = 225, 520, 1080, 225
        d.line((x0, y0, x1, y0), fill=WHITE, width=3)
        d.line((x0, y0, x0, y1), fill=WHITE, width=3)
        d.text((1070, 555), "latent rate", font=font(22), fill=GREY, anchor="rm")
        d.text((170, 245), "reconstruction quality", font=font(22), fill=GREY, anchor="lm")
        pts = [(310, 470), (430, 410), (585, 350), (760, 303), (960, 270)]
        d.line(pts, fill=PURPLE_LIGHT, width=5)
        for i, (x, y) in enumerate(pts):
            d.ellipse((x - 11, y - 11, x + 11, y + 11), fill=PURPLE if i == 2 else WHITE, outline=PURPLE, width=3)
        d.text((585, 390), "useful operating point", font=font(22, "bold"), fill=PURPLE_LIGHT, anchor="mm")

    elif v == "manifold":
        rounded(d, (150, 210, 1130, 535), (14, 15, 24), GREY_DARK, 2, 20)
        clusters = [((350, 340), CYAN), ((640, 430), PURPLE_LIGHT), ((920, 300), ORANGE)]
        for (cx, cy), col in clusters:
            for j in range(15):
                a = 2 * math.pi * j / 15
                r = 18 + (j * 13) % 55
                x = cx + int(math.cos(a) * r)
                y = cy + int(math.sin(a) * r * 0.55)
                d.ellipse((x - 6, y - 6, x + 6, y + 6), fill=col)
        d.line((390, 350, 880, 310), fill=RED, width=4)
        d.text((640, 285), "untrained gap", font=font(25, "bold"), fill=RED, anchor="mm")
        d.text((640, 565), "a reconstruction map is not automatically a sampling prior", font=font(23), fill=GREY, anchor="mm")

    elif v == "reparam":
        draw_formula(d, "z = mu + sigma * epsilon", 315, PURPLE_LIGHT, 54)
        draw_box_label(d, (155, 410, 415, 525), "mu, sigma", "learned by encoder", active=True)
        draw_box_label(d, (510, 410, 770, 525), "epsilon", "sampled from N(0, I)")
        draw_box_label(d, (865, 410, 1125, 525), "z", "differentiable path", active=True)
        line_arrow(d, (420, 468), (500, 468), WHITE, 4)
        line_arrow(d, (775, 468), (855, 468), WHITE, 4)

    elif v == "elbo":
        d.line((640, 225, 640, 500), fill=WHITE, width=7)
        d.line((350, 330, 930, 330), fill=WHITE, width=6)
        d.ellipse((625, 315, 655, 345), fill=PURPLE, outline=WHITE, width=3)
        rounded(d, (175, 370, 535, 505), (28, 23, 42), PURPLE, 3, 18)
        rounded(d, (745, 370, 1105, 505), (35, 25, 25), ORANGE, 3, 18)
        d.text((355, 420), "RECONSTRUCTION", font=font(29, "bold"), fill=WHITE, anchor="mm")
        d.text((355, 465), "preserve the example", font=font(21), fill=GREY, anchor="mm")
        d.text((925, 420), "KL TO PRIOR", font=font(29, "bold"), fill=WHITE, anchor="mm")
        d.text((925, 465), "organize the space", font=font(21), fill=GREY, anchor="mm")

    elif v == "prior":
        rounded(d, (120, 220, 555, 530), (14, 15, 24), GREY_DARK, 2, 18)
        rounded(d, (725, 220, 1160, 530), (14, 15, 24), PURPLE, 2, 18)
        d.text((338, 250), "isolated islands", font=font(25, "bold"), fill=GREY, anchor="mm")
        d.text((942, 250), "shared prior", font=font(25, "bold"), fill=PURPLE_LIGHT, anchor="mm")
        for j in range(24):
            cx = 220 + (j % 3) * 115
            cy = 330 + (j // 8) * 62 + (j % 4) * 4
            d.ellipse((cx - 6, cy - 6, cx + 6, cy + 6), fill=[CYAN, ORANGE, PURPLE_LIGHT][j % 3])
            a = 2 * math.pi * j / 24
            r = 35 + (j % 5) * 19
            x = 942 + int(math.cos(a) * r)
            y = 385 + int(math.sin(a) * r * 0.7)
            d.ellipse((x - 6, y - 6, x + 6, y + 6), fill=PURPLE_LIGHT)
        line_arrow(d, (575, 375), (705, 375), WHITE, 5)

    elif v == "sampling":
        glow_ellipse(im, (100, 235, 400, 520), PURPLE, 80, 36)
        d.ellipse((190, 300, 310, 420), fill=(30, 26, 55), outline=PURPLE_LIGHT, width=4)
        d.text((250, 360), "z ~ N(0,I)", font=font(24, "mono"), fill=WHITE, anchor="mm")
        line_arrow(d, (400, 360), (540, 360), WHITE, 5)
        draw_box_label(d, (555, 290, 765, 430), "DECODER", "p(x | z)", active=True)
        line_arrow(d, (775, 360), (880, 360), WHITE, 5)
        for i, cx in enumerate([960, 1080]):
            draw_robot(d, cx, 382, 0.43, accent=PURPLE if i == 0 else ORANGE)

    elif v == "collapse":
        draw_box_label(d, (125, 290, 340, 430), "ENCODER", "q(z | x)")
        draw_box_label(d, (530, 290, 750, 430), "z", "ignored", active=False)
        draw_box_label(d, (930, 255, 1170, 465), "POWERFUL", "decoder shortcut", active=True, accent=RED)
        line_arrow(d, (350, 360), (515, 360), GREY, 4)
        d.line((555, 275, 725, 445), fill=RED, width=9)
        d.line((725, 275, 555, 445), fill=RED, width=9)
        d.arc((300, 175, 1070, 540), 205, 330, fill=ORANGE, width=6)
        d.text((665, 195), "information bypasses z", font=font(27, "bold"), fill=ORANGE, anchor="mm")

    elif v == "nearest":
        rounded(d, (95, 205, 1185, 535), (13, 14, 23), GREY_DARK, 2, 18)
        pts = [(265, 410), (375, 300), (480, 445), (610, 330), (760, 255), (900, 420), (1040, 315)]
        for i, (x, y) in enumerate(pts):
            d.ellipse((x - 17, y - 17, x + 17, y + 17), fill=PURPLE if i == 3 else (30, 31, 45), outline=WHITE, width=3)
            d.text((x, y + 42), f"e{i}", font=font(18, "mono"), fill=GREY, anchor="mm")
        qx, qy = 680, 405
        d.ellipse((qx - 13, qy - 13, qx + 13, qy + 13), fill=ORANGE, outline=WHITE, width=3)
        d.line((qx, qy, 610, 330), fill=ORANGE, width=5)
        d.text((710, 455), "encoder vector", font=font(22), fill=ORANGE, anchor="mm")
        d.text((610, 275), "nearest code", font=font(24, "bold"), fill=PURPLE_LIGHT, anchor="mm")

    elif v == "tokens":
        draw_robot(d, 260, 385, 0.77)
        line_arrow(d, (440, 365), (555, 365), WHITE, 5)
        draw_token_grid(d, 585, 235, [7, 7, 12, 4, 19, 3, 8, 8, 8, 2, 14, 6, 21, 3, 3, 5, 12, 9, 7, 1, 8, 4, 16, 2], cols=6, cell=47)
        d.text((835, 555), "discrete latent token map", font=font(25, "bold"), fill=PURPLE_LIGHT, anchor="mm")

    elif v == "vqloss":
        draw_formula(d, "L = L_rec + L_codebook + beta * L_commit", 300, PURPLE_LIGHT, 37)
        cards = [
            (150, "RECONSTRUCT", "decoder preserves x", PURPLE),
            (465, "MOVE CODES", "entries follow encoder", CYAN),
            (780, "COMMIT", "encoder stays near code", ORANGE),
        ]
        for x, title, sub, col in cards:
            rounded(d, (x, 405, x + 285, 520), (20, 21, 31), col, 3, 18)
            d.text((x + 142, 440), title, font=font(24, "bold"), fill=WHITE, anchor="mm")
            d.text((x + 142, 482), sub, font=font(18), fill=GREY, anchor="mm")

    elif v == "tokenprior":
        draw_token_grid(d, 105, 260, [7, 7, 12, 4, 19, 3, 8, 8, 2, 14, 6, 21], cols=4, cell=48)
        line_arrow(d, (360, 365), (475, 365), WHITE, 5)
        draw_box_label(d, (490, 270, 790, 455), "TOKEN PRIOR", "autoregressive or diffusion", active=True)
        line_arrow(d, (805, 365), (915, 365), WHITE, 5)
        draw_robot(d, 1060, 385, 0.62)

    elif v == "deadcodes":
        d.text((290, 225), "nominal codebook", font=font(26, "bold"), fill=WHITE, anchor="mm")
        draw_codebook(d, 130, 270, cols=5, rows=4, cell=46, active=1)
        line_arrow(d, (450, 365), (560, 365), WHITE, 5)
        d.text((850, 225), "actual usage", font=font(26, "bold"), fill=WHITE, anchor="mm")
        heights = [245, 190, 155, 85, 55, 24, 18, 12, 8, 5]
        for i, h in enumerate(heights):
            x = 610 + i * 50
            col = PURPLE if i < 3 else GREY_DARK
            d.rectangle((x, 510 - h, x + 30, 510), fill=col)
        d.text((850, 555), "many entries receive almost no traffic", font=font(21), fill=RED, anchor="mm")

    elif v == "rq":
        draw_box_label(d, (95, 295, 285, 430), "z_e", "encoder vector")
        stages = [(385, "Q1", "coarse"), (625, "Q2", "residual"), (865, "Q3", "residual")]
        for i, (x, title, sub) in enumerate(stages):
            line_arrow(d, (290 if i == 0 else x - 85, 365), (x - 15, 365), WHITE, 4)
            draw_box_label(d, (x, 295, x + 155, 430), title, sub, active=True)
        line_arrow(d, (1025, 365), (1130, 365), WHITE, 4)
        d.text((640, 515), "r_m = r_(m-1) - e_km", font=font(29, "mono"), fill=PURPLE_LIGHT, anchor="mm")

    elif v == "rqsum":
        draw_formula(d, "z_q ~= e_k1 + e_k2 + e_k3 + ... + e_kM", 310, PURPLE_LIGHT, 39)
        xs = [245, 465, 685, 905]
        colors = [PURPLE, CYAN, ORANGE, GREEN]
        for i, (x, col) in enumerate(zip(xs, colors), 1):
            glow_ellipse(im, (x - 62, 405, x + 62, 529), col, 75, 24)
            d.ellipse((x - 48, 420, x + 48, 516), fill=(22, 23, 35), outline=col, width=4)
            d.text((x, 468), f"e{i}", font=font(30, "mono"), fill=WHITE, anchor="mm")
        d.text((640, 565), "several small choices approximate one vector", font=font(23), fill=GREY, anchor="mm")

    elif v == "coarsefine":
        stages = [
            (185, 0.42, "silhouette"),
            (455, 0.48, "+ umbrella"),
            (735, 0.55, "+ rain"),
            (1030, 0.62, "+ detail"),
        ]
        for i, (x, sc, label) in enumerate(stages):
            draw_robot(d, x, 385, sc, umbrella=i >= 1, accent=[GREY, PURPLE, CYAN, ORANGE][i])
            d.text((x, 555), label, font=font(21, "bold"), fill=[GREY, PURPLE_LIGHT, CYAN, ORANGE][i], anchor="mm")
            if i + 1 < len(stages):
                line_arrow(d, (x + 90, 365), (stages[i + 1][0] - 90, 365), WHITE, 3)

    elif v == "rate":
        draw_formula(d, "rate per position <= M * log2(K) bits", 300, PURPLE_LIGHT, 39)
        d.text((300, 440), "MORE STAGES", font=font(32, "bold"), fill=WHITE, anchor="mm")
        for i in range(5):
            rounded(d, (170 + i * 58, 475 - i * 8, 215 + i * 58, 520 - i * 8), PURPLE if i < 3 else GREY_DARK, WHITE, 2, 8)
        d.text((930, 440), "ONE HUGE CODEBOOK", font=font(32, "bold"), fill=WHITE, anchor="mm")
        draw_codebook(d, 790, 465, cols=6, rows=1, cell=40, active=3)
        d.text((640, 570), "depth trades against codebook width", font=font(23), fill=GREY, anchor="mm")

    elif v == "stacked":
        d.text((270, 225), "spatial positions", font=font(24, "bold"), fill=WHITE, anchor="mm")
        for r in range(4):
            for c in range(4):
                x = 125 + c * 78
                y = 270 + r * 65
                rounded(d, (x, y, x + 54, y + 45), (22, 23, 34), GREY_DARK, 2, 8)
                d.text((x + 27, y + 22), str((r * 4 + c) % 9), font=font(17, "mono"), fill=WHITE, anchor="mm")
        line_arrow(d, (460, 385), (570, 385), WHITE, 5)
        d.text((835, 225), "stack of codes per position", font=font(24, "bold"), fill=PURPLE_LIGHT, anchor="mm")
        for row in range(4):
            y = 285 + row * 68
            for col in range(5):
                x = 650 + col * 88
                rounded(d, (x, y - col * 3, x + 62, y + 45 - col * 3), PURPLE if col == 0 else (27, 28, 42), WHITE if col == 0 else GREY_DARK, 2, 8)
                d.text((x + 31, y + 21 - col * 3), str((row * 7 + col * 3) % 23), font=font(17, "mono"), fill=WHITE, anchor="mm")

    elif v == "sae":
        draw_box_label(d, (90, 300, 285, 430), "ACTIVATION", "dense vector")
        line_arrow(d, (295, 365), (400, 365), WHITE, 4)
        rounded(d, (415, 220, 865, 510), (15, 16, 25), GREY_DARK, 2, 18)
        for i in range(18):
            x = 445 + (i % 9) * 44
            y = 270 + (i // 9) * 125
            active = i in (2, 7, 13)
            h = 72 if active else 20
            d.rectangle((x, y + 75 - h, x + 23, y + 75), fill=PURPLE if active else GREY_DARK)
        d.text((640, 470), "wide dictionary, few active units", font=font(22), fill=PURPLE_LIGHT, anchor="mm")
        line_arrow(d, (880, 365), (985, 365), WHITE, 4)
        draw_box_label(d, (995, 300, 1185, 430), "RECONSTRUCT", "decoder")

    elif v == "sparsity":
        draw_formula(d, "L = ||x - x_hat||^2 + lambda * ||a||_1", 300, PURPLE_LIGHT, 37)
        d.text((355, 420), "dense", font=font(24, "bold"), fill=GREY, anchor="mm")
        for i in range(14):
            h = 35 + (i * 29) % 110
            d.rectangle((180 + i * 25, 520 - h, 196 + i * 25, 520), fill=GREY)
        line_arrow(d, (545, 450), (695, 450), WHITE, 5)
        d.text((925, 420), "sparse", font=font(24, "bold"), fill=PURPLE_LIGHT, anchor="mm")
        for i in range(14):
            active = i in (2, 8, 11)
            h = [118, 90, 105][[2, 8, 11].index(i)] if active else 5
            d.rectangle((750 + i * 25, 520 - h, 766 + i * 25, 520), fill=PURPLE if active else GREY_DARK)

    elif v == "features":
        rounded(d, (105, 215, 1175, 535), (15, 16, 25), GREY_DARK, 2, 18)
        labels = ["curved edge", "metallic highlight", "umbrella", "robot face", "falling rain", "standing pose"]
        positions = [(260, 305), (520, 305), (790, 305), (1030, 305), (390, 445), (820, 445)]
        for i, (label, (x, y)) in enumerate(zip(labels, positions)):
            active = i in (2, 3, 4)
            rounded(d, (x - 110, y - 45, x + 110, y + 45), (43, 30, 83) if active else (24, 25, 37), PURPLE_LIGHT if active else GREY_DARK, 2, 18)
            d.text((x, y), label, font=fit_font(d, label, 190, 21, "bold", 16), fill=WHITE if active else GREY, anchor="mm")
        d.text((640, 570), "candidate explanatory directions", font=font(22), fill=GREY, anchor="mm")

    elif v == "saecaveat":
        cards = [
            (120, "SPLIT", "one concept -> several units", ORANGE),
            (460, "MERGE", "several concepts -> one unit", RED),
            (800, "SHIFT", "meaning changes by context", PURPLE),
        ]
        for x, title, sub, col in cards:
            rounded(d, (x, 260, x + 300, 465), (20, 21, 31), col, 3, 20)
            d.text((x + 150, 320), title, font=font(38, "bold"), fill=col, anchor="mm")
            lines = wrap_text(d, sub, font(21), 250)
            for j, line in enumerate(lines):
                d.text((x + 150, 390 + j * 28), line, font=font(21), fill=WHITE, anchor="mm")
        d.text((640, 525), "test causally before naming a feature", font=font(25, "bold"), fill=PURPLE_LIGHT, anchor="mm")

    elif v == "frozen":
        rounded(d, (100, 230, 555, 520), (17, 18, 27), GREY_DARK, 2, 20)
        d.text((328, 270), "PIXEL AUTOENCODER", font=font(28, "bold"), fill=WHITE, anchor="mm")
        draw_box_label(d, (160, 330, 340, 445), "ENCODER", "trained for pixels")
        draw_box_label(d, (370, 330, 520, 445), "z", "compact")
        rounded(d, (725, 230, 1180, 520), (25, 20, 43), PURPLE, 3, 20)
        d.text((952, 270), "REPRESENTATION AUTOENCODER", font=fit_font(d, "REPRESENTATION AUTOENCODER", 410, 28, "bold", 20), fill=PURPLE_LIGHT, anchor="mm")
        draw_box_label(d, (770, 330, 990, 445), "FROZEN ENCODER", "semantic pretraining", active=True)
        draw_box_label(d, (1020, 330, 1145, 445), "tokens", "rich", active=True)

    elif v == "rich":
        d.text((315, 225), "compact latent", font=font(26, "bold"), fill=GREY, anchor="mm")
        rounded(d, (165, 275, 465, 470), (17, 18, 27), GREY_DARK, 2, 18)
        for i in range(8):
            d.rectangle((205 + i * 30, 350 - ((i * 19) % 55), 224 + i * 30, 430), fill=GREY)
        line_arrow(d, (500, 375), (620, 375), WHITE, 5)
        d.text((930, 225), "rich representation tokens", font=font(26, "bold"), fill=PURPLE_LIGHT, anchor="mm")
        rounded(d, (650, 245, 1160, 500), (26, 21, 46), PURPLE, 3, 18)
        draw_token_grid(d, 700, 300, [2, 8, 13, 5, 21, 3, 9, 7, 14, 4, 17, 6, 11, 1, 19], cols=5, cell=49)

    elif v == "dit":
        draw_box_label(d, (75, 290, 265, 435), "NOISE", "representation space")
        line_arrow(d, (275, 365), (375, 365), WHITE, 4)
        draw_box_label(d, (390, 255, 690, 470), "DIFFUSION", "transformer denoiser", active=True)
        line_arrow(d, (705, 365), (805, 365), WHITE, 4)
        draw_box_label(d, (820, 290, 990, 435), "DECODER", "to pixels")
        line_arrow(d, (1000, 365), (1070, 365), WHITE, 4)
        draw_robot(d, 1150, 390, 0.42)
        d.text((540, 520), "optimize for wide semantic latents", font=font(23), fill=PURPLE_LIGHT, anchor="mm")

    elif v == "notcompression":
        labels = [
            ("REMOVE BITS", GREY),
            ("SHAPE A PRIOR", PURPLE),
            ("CREATE TOKENS", CYAN),
            ("SELECT FEATURES", ORANGE),
            ("KEEP SEMANTICS", GREEN),
        ]
        for i, (label, col) in enumerate(labels):
            x0 = 105 + i * 215
            rounded(d, (x0, 270, x0 + 190, 465), (18, 19, 29), col, 3, 18)
            d.text((x0 + 95, 335), str(i + 1), font=font(42, "mono"), fill=col, anchor="mm")
            lines = wrap_text(d, label, font(20, "bold"), 155)
            for j, line in enumerate(lines):
                d.text((x0 + 95, 405 + j * 26), line, font=font(20, "bold"), fill=WHITE, anchor="mm")

    elif v == "axes":
        axes = [
            ("deterministic", "probabilistic", 0.38),
            ("continuous", "discrete", 0.63),
            ("dense", "sparse", 0.72),
            ("compact", "rich", 0.58),
        ]
        for i, (left, right, val) in enumerate(axes):
            y = 250 + i * 78
            d.text((180, y), left, font=font(22, "bold"), fill=GREY, anchor="lm")
            d.text((1100, y), right, font=font(22, "bold"), fill=GREY, anchor="rm")
            d.line((350, y, 930, y), fill=GREY_DARK, width=13)
            x = int(350 + 580 * val)
            glow_ellipse(im, (x - 31, y - 31, x + 31, y + 31), PURPLE, 120, 18)
            d.ellipse((x - 17, y - 17, x + 17, y + 17), fill=PURPLE, outline=WHITE, width=3)

    elif v == "chooser":
        choices = [
            ("RECONSTRUCT", "AE", GREY),
            ("SAMPLE SMOOTHLY", "VAE", PURPLE),
            ("LEARN TOKENS", "VQ / RQ", CYAN),
            ("ISOLATE FEATURES", "SAE", ORANGE),
            ("START SEMANTIC", "RAE", GREEN),
        ]
        for i, (goal, model, col) in enumerate(choices):
            y = 215 + i * 67
            rounded(d, (155, y, 1125, y + 52), (18, 19, 29), col if i == 1 else GREY_DARK, 2, 13)
            d.text((190, y + 26), goal, font=font(20, "bold"), fill=WHITE, anchor="lm")
            d.text((1080, y + 26), model, font=font(25, "mono"), fill=col, anchor="rm")

    elif v == "final":
        labels = ["POINT", "DISTRIBUTION", "CODE", "RESIDUAL STACK", "SPARSE DICTIONARY", "SEMANTIC GRID"]
        for i, label in enumerate(labels):
            c = i % 3
            r = i // 3
            x0 = 105 + c * 360
            y0 = 225 + r * 160
            active = i in (1, 3, 5)
            rounded(d, (x0, y0, x0 + 320, y0 + 115), (43, 30, 83) if active else (20, 21, 31), PURPLE_LIGHT if active else GREY_DARK, 3 if active else 2, 18)
            d.text((x0 + 160, y0 + 57), label, font=fit_font(d, label, 280, 25, "bold", 17), fill=WHITE, anchor="mm")
        d.text((640, 560), "change the constraint, change the representation", font=font(26, "bold"), fill=PURPLE_LIGHT, anchor="mm")

    else:
        d.text((640, 365), seg.headline, font=font(42, "bold"), fill=WHITE, anchor="mm")

    return im


def make_background() -> Image.Image:
    arr = np.zeros((H, W, 3), dtype=np.uint8)
    for y in range(H):
        k = y / (H - 1)
        arr[y, :, 0] = int(BG[0] * (1 - k) + BG2[0] * k)
        arr[y, :, 1] = int(BG[1] * (1 - k) + BG2[1] * k)
        arr[y, :, 2] = int(BG[2] * (1 - k) + BG2[2] * k)
    im = Image.fromarray(arr, "RGB")
    d = ImageDraw.Draw(im)
    rng = np.random.default_rng(42)
    for _ in range(105):
        x = int(rng.integers(0, W))
        y = int(rng.integers(0, 600))
        r = int(rng.integers(1, 3))
        c = int(rng.integers(65, 165))
        d.ellipse((x - r, y - r, x + r, y + r), fill=(c, c, min(255, c + 25)))
    for x in range(70, W, 64):
        for y in range(105, 600, 64):
            d.ellipse((x - 1, y - 1, x + 1, y + 1), fill=(28, 29, 42))
    return im


def locate_item(items: list[dict[str, Any]], starts: list[float], t: float) -> dict[str, Any] | None:
    i = bisect.bisect_right(starts, t) - 1
    if i < 0 or i >= len(items):
        return None
    item = items[i]
    return item if t < item["end"] else None


def locate_caption(cues: list[dict[str, Any]], starts: list[float], t: float) -> str:
    i = bisect.bisect_right(starts, t) - 1
    if i < 0 or i >= len(cues):
        return ""
    cue = cues[i]
    return cue["text"] if t <= cue["end"] else ""


def draw_hud(frame: Image.Image, chapter: str, elapsed: float, total: float) -> None:
    d = ImageDraw.Draw(frame)
    rounded(d, (35, 25, 276, 72), (92, 47, 205), WHITE, 2, 23)
    name_font = fit_font(d, chapter, 211, 22, "bold", 15)
    d.text((155, 48), chapter, font=name_font, fill=WHITE, anchor="mm")
    d.text((1240, 49), f"{sec_to_clock(elapsed)} / {sec_to_clock(total)}", font=font(18, "mono"), fill=GREY, anchor="rm")


def draw_progress(frame: Image.Image, elapsed: float, total: float, chapter: str) -> None:
    d = ImageDraw.Draw(frame)
    y0 = 690
    d.rectangle((0, y0, W, H), fill=(28, 29, 39))
    chapter_w = W / len(CHAPTERS)
    for i, name in enumerate(CHAPTERS):
        x0 = int(i * chapter_w)
        x1 = int((i + 1) * chapter_w)
        if i:
            d.line((x0, y0, x0, H), fill=(210, 210, 222), width=1)
        current = name == chapter
        if current:
            d.rectangle((x0, y0, x1, H), fill=(72, 48, 124))
        short = name if len(name) <= 13 else name[:12] + "…"
        d.text(((x0 + x1) // 2, 705), short, font=font(13, "bold"), fill=WHITE if current else GREY, anchor="mm")
    d.rectangle((0, y0, int(W * clamp01(elapsed / max(total, 0.1))), y0 + 5), fill=PURPLE_LIGHT)


def draw_caption(frame: Image.Image, caption: str) -> None:
    if not caption:
        return
    d = ImageDraw.Draw(frame)
    d.rounded_rectangle((55, 594, 1225, 681), radius=18, fill=(0, 0, 0, 205))
    fnt = font(29, "bold")
    lines = wrap_text(d, caption, fnt, 1090)
    if len(lines) > 2:
        fnt = font(25, "bold")
        lines = wrap_text(d, caption, fnt, 1090)[:2]
    line_h = 36
    start_y = 637 - (len(lines) - 1) * line_h // 2
    for i, line in enumerate(lines):
        d.text((640, start_y + i * line_h), line, font=fnt, fill=WHITE, anchor="mm", stroke_width=4, stroke_fill=BLACK)


def draw_motion(frame: Image.Image, item: dict[str, Any], local: float) -> None:
    d = ImageDraw.Draw(frame)
    duration = max(0.01, item["end"] - item["start"])
    p = clamp01(local / duration)
    phase = 0.5 + 0.5 * math.sin(local * 3.1)
    kind = item["kind"]
    if kind in {"opening", "chapter", "outro"}:
        for j in range(3):
            x = int(-420 + (W + 840) * ((p + j * 0.13) % 1.0))
            d.rectangle((x, 198 + j * 165, x + 360, 204 + j * 165), fill=PURPLE_LIGHT)
        return
    visual = item.get("visual", "")
    if visual in {"ae", "vae", "vq", "rae", "tokenprior", "rq", "dit"}:
        x = int(285 + (W - 570) * ((local * 0.24) % 1.0))
        glow = Image.new("RGBA", frame.size, (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow)
        gd.ellipse((x - 18, 347, x + 18, 383), fill=(*PURPLE_LIGHT, 180))
        glow = glow.filter(ImageFilter.GaussianBlur(11))
        frame.alpha_composite(glow)
        d = ImageDraw.Draw(frame)
        d.ellipse((x - 7, 358, x + 7, 372), fill=WHITE)
    elif visual in {"nearest", "tokens", "deadcodes", "stacked"}:
        idx = int(local * 2.0) % 12
        x = 600 + (idx % 6) * 56
        y = 230 + (idx // 6) * 70
        d.rounded_rectangle((x, y, x + 48, y + 48), radius=10, outline=PURPLE_LIGHT, width=4)
    elif visual in {"sae", "sparsity", "features", "saecaveat"}:
        x = 470 + (int(local * 2.4) % 8) * 44
        h = int(55 + 35 * phase)
        d.rectangle((x, 505 - h, x + 18, 505), fill=PURPLE_LIGHT)
    elif visual in {"prior", "sampling", "manifold", "reparam"}:
        for j in range(7):
            a = local * (0.55 + j * 0.04) + j
            x = int(640 + math.cos(a) * (85 + j * 17))
            y = int(365 + math.sin(a * 1.17) * (38 + j * 8))
            d.ellipse((x - 4, y - 4, x + 4, y + 4), fill=PURPLE_LIGHT)
    elif visual in {"family", "final", "chooser", "notcompression"}:
        x = int(80 + (W - 160) * ((local * 0.10) % 1.0))
        d.rectangle((x, 570, x + 90, 574), fill=PURPLE_LIGHT)
    else:
        alpha = int(45 + 55 * phase)
        d.line((110, 578, 1170, 578), fill=(*PURPLE, alpha), width=2)


def render_video(items: list[dict[str, Any]], cues: list[dict[str, Any]], total_duration: float) -> Path:
    print(f"Rendering {total_duration:.2f}s at {RENDER_FPS} fps...", flush=True)
    base = make_background().convert("RGB")
    scene_rgbs: list[Image.Image] = []
    for item in items:
        if item["kind"] == "opening":
            overlay = scene_opening()
        elif item["kind"] == "chapter":
            overlay = scene_chapter(item["chapter"])
        elif item["kind"] == "outro":
            overlay = scene_outro()
        else:
            overlay = scene_for_segment(SEGMENTS[item["index"]])
        composed = Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB")
        scene_rgbs.append(composed)

    poster = scene_rgbs[0].copy()
    poster.save(OUT / "poster.jpg", quality=94, subsampling=0)

    silent_path = OUT / "latent-models-silent.mp4"
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
        "-r", str(RENDER_FPS), "-i", "-",
        "-an", "-vf", f"fps={OUTPUT_FPS}",
        "-c:v", "libx264", "-preset", "medium", "-crf", "21",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(silent_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    if proc.stdin is None:
        raise RuntimeError("ffmpeg stdin was not created")

    starts = [item["start"] for item in items]
    cue_starts = [cue["start"] for cue in cues]
    total_frames = int(math.ceil(total_duration * RENDER_FPS))
    current_item_index = 0
    for frame_index in range(total_frames):
        t = frame_index / RENDER_FPS
        while current_item_index + 1 < len(items) and t >= items[current_item_index + 1]["start"]:
            current_item_index += 1
        item = items[current_item_index]
        local = t - item["start"]
        duration = item["end"] - item["start"]
        fade_in = ease_out(local / 0.48)
        fade_out = clamp01((item["end"] - t) / 0.33)
        alpha = min(fade_in, fade_out)
        scene = scene_rgbs[current_item_index]
        frame = Image.blend(base, scene, alpha).convert("RGBA")
        draw_motion(frame, item, local)
        draw_hud(frame, item["chapter"], t, total_duration)
        caption = locate_caption(cues, cue_starts, t)
        draw_caption(frame, caption)
        draw_progress(frame, t, total_duration, item["chapter"])
        try:
            proc.stdin.write(frame.convert("RGB").tobytes())
        except BrokenPipeError as exc:
            raise RuntimeError("ffmpeg stopped while receiving frames") from exc
        if frame_index % max(1, total_frames // 20) == 0:
            print(f"  frames {frame_index}/{total_frames}", flush=True)

    proc.stdin.close()
    code = proc.wait()
    if code != 0:
        raise RuntimeError(f"ffmpeg video encoder exited with {code}")

    final_path = OUT / "latent-variable-models-kokoro.mp4"
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(silent_path), "-i", str(OUT / "latent-models-kokoro.wav"),
        "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-ac", "2",
        "-shortest", "-movflags", "+faststart", str(final_path),
    ], check=True)
    return final_path


def build_sources_html() -> str:
    rows = "\n".join(
        f'<li><a href="{html.escape(src["url"])}" target="_blank" rel="noreferrer">{html.escape(src["label"])}</a><span>{html.escape(src["authors"])}</span></li>'
        for src in SOURCES
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sources | Latent variable models</title>
<style>body{{margin:0;background:#07080d;color:#f6f7fc;font:16px/1.65 system-ui,sans-serif}}main{{max-width:850px;margin:auto;padding:52px 24px}}a{{color:#b597ff}}li{{margin:20px 0}}span{{display:block;color:#9c9fb0}}.back{{display:inline-block;margin-bottom:28px}}</style></head>
<body><main><a class="back" href="./">Back to video</a><h1>Primary sources</h1><ol>{rows}</ol></main></body></html>"""


def build_index_html(duration: float, video_size: int) -> str:
    source_cards = "".join(
        f'<a class="source" href="{html.escape(src["url"])}" target="_blank" rel="noreferrer"><strong>{html.escape(src["label"])}</strong><span>{html.escape(src["authors"])}</span></a>'
        for src in SOURCES
    )
    size_mb = video_size / (1024 * 1024)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="A narrated explainer of AE, VAE, VQ-VAE, RQ-VAE, SAE and representation autoencoders.">
<title>Latent variable models</title>
<style>
:root{{--bg:#07080d;--panel:#11121c;--line:#303247;--ink:#f6f7fc;--muted:#9da0b2;--purple:#743ff6;--purple2:#b597ff}}
*{{box-sizing:border-box}}body{{margin:0;background:radial-gradient(circle at 50% 0,#211442 0,#0a0b12 38%,#06070b 75%);color:var(--ink);font:16px/1.55 system-ui,-apple-system,Segoe UI,sans-serif}}
main{{width:min(1180px,calc(100% - 32px));margin:0 auto;padding:44px 0 72px}}header{{display:flex;align-items:flex-end;justify-content:space-between;gap:24px;margin-bottom:25px}}h1{{font-size:clamp(38px,7vw,82px);line-height:.95;margin:0;letter-spacing:-.055em}}.eyebrow{{color:var(--purple2);font-weight:700;letter-spacing:.16em;text-transform:uppercase;margin-bottom:12px}}.meta{{color:var(--muted);text-align:right;white-space:nowrap}}.player{{padding:10px;background:linear-gradient(135deg,rgba(116,63,246,.8),rgba(181,151,255,.15) 45%,rgba(255,255,255,.08));border-radius:24px;box-shadow:0 34px 90px rgba(0,0,0,.42)}}video{{display:block;width:100%;aspect-ratio:16/9;background:#000;border-radius:16px}}.actions{{display:flex;flex-wrap:wrap;gap:12px;margin:22px 0 38px}}.button{{display:inline-flex;align-items:center;justify-content:center;padding:12px 18px;border-radius:999px;text-decoration:none;font-weight:750;border:1px solid var(--line);color:var(--ink);background:var(--panel)}}.button.primary{{background:var(--purple);border-color:var(--purple)}}.note{{color:var(--muted);max-width:820px}}h2{{font-size:30px;margin-top:48px}}.models{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}.model{{background:rgba(17,18,28,.84);border:1px solid var(--line);border-radius:18px;padding:18px}}.model strong{{display:block;color:var(--purple2);font-size:20px;margin-bottom:5px}}.sources{{display:grid;grid-template-columns:repeat(2,1fr);gap:12px}}.source{{display:block;text-decoration:none;color:var(--ink);background:rgba(17,18,28,.84);border:1px solid var(--line);border-radius:16px;padding:16px;transition:.18s transform,.18s border-color}}.source:hover{{transform:translateY(-2px);border-color:var(--purple)}}.source span{{display:block;color:var(--muted);font-size:14px;margin-top:4px}}footer{{margin-top:48px;color:var(--muted);font-size:14px}}@media(max-width:780px){{header{{display:block}}.meta{{text-align:left;margin-top:16px}}.models,.sources{{grid-template-columns:1fr}}main{{padding-top:28px}}}}
</style>
</head>
<body><main>
<header><div><div class="eyebrow">Neural representation design</div><h1>Latent variable<br>models</h1></div><div class="meta">{sec_to_clock(duration)} · 720p · Kokoro am_liam<br>{size_mb:.1f} MiB MP4</div></header>
<section class="player"><video controls playsinline preload="metadata" poster="poster.jpg"><source src="latent-variable-models-kokoro.mp4" type="video/mp4"><track kind="captions" src="captions.vtt" srclang="en" label="English" default>Your browser cannot play this video.</video></section>
<div class="actions"><a class="button primary" href="latent-variable-models-kokoro.mp4" download>Download MP4</a><a class="button" href="captions.vtt" download>Download captions</a><a class="button" href="sources.html">Read sources</a></div>
<p class="note">A code-drawn explainer of deterministic and probabilistic latent spaces, discrete quantization, residual code stacks, sparse dictionaries, and representation autoencoders. Narration is generated locally with Kokoro-82M using the am_liam voice.</p>
<h2>The models covered</h2><section class="models">
<div class="model"><strong>AE</strong>Deterministic continuous bottleneck optimized for reconstruction.</div>
<div class="model"><strong>VAE</strong>Approximate posterior regularized toward a sampling prior.</div>
<div class="model"><strong>VQ-VAE</strong>Nearest-neighbor codebook turns latent vectors into symbols.</div>
<div class="model"><strong>RQ-VAE</strong>Several codewords successively explain the remaining residual.</div>
<div class="model"><strong>SAE</strong>Wide dictionary with only a small active feature set per example.</div>
<div class="model"><strong>RAE</strong>Pretrained semantic encoder paired with a learned reconstruction decoder.</div>
</section>
<h2>Primary sources</h2><section class="sources">{source_cards}</section>
<footer>Original diagrams and motion graphics rendered in code. Published for direct viewing and download.</footer>
</main></body></html>"""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def publish_here_now(video_path: Path, duration: float) -> dict[str, Any]:
    (OUT / "sources.html").write_text(build_sources_html(), encoding="utf-8")
    (OUT / "index.html").write_text(build_index_html(duration, video_path.stat().st_size), encoding="utf-8")
    publish_files = {
        "index.html": OUT / "index.html",
        "latent-variable-models-kokoro.mp4": video_path,
        "poster.jpg": OUT / "poster.jpg",
        "captions.vtt": OUT / "captions.vtt",
        "sources.html": OUT / "sources.html",
    }
    manifest = []
    for rel, path in publish_files.items():
        content_type = {
            ".html": "text/html; charset=utf-8",
            ".mp4": "video/mp4",
            ".jpg": "image/jpeg",
            ".vtt": "text/vtt; charset=utf-8",
        }[path.suffix]
        manifest.append({
            "path": rel,
            "size": path.stat().st_size,
            "contentType": content_type,
            "hash": sha256_file(path),
        })
    create = requests.post(
        "https://here.now/api/v1/publish",
        headers={"X-HereNow-Client": "chatgpt/latent-video-recovery", "Content-Type": "application/json"},
        json={
            "files": manifest,
            "displayName": "Latent variable models",
            "displayDescription": "AE, VAE, VQ-VAE, RQ-VAE, SAE and RAE explained with code-drawn motion graphics and Kokoro narration.",
            "viewer": {
                "title": "Latent variable models",
                "description": "A narrated visual guide to six autoencoder and latent representation families.",
                "ogImagePath": "poster.jpg",
            },
        },
        timeout=60,
    )
    create.raise_for_status()
    payload = create.json()
    upload_map = {entry["path"]: entry for entry in payload["upload"]["uploads"]}
    for rel, path in publish_files.items():
        entry = upload_map.get(rel)
        if entry is None:
            continue
        headers = entry.get("headers") or {"Content-Type": next(x["contentType"] for x in manifest if x["path"] == rel)}
        print(f"Uploading {rel} ({path.stat().st_size / 1024 / 1024:.1f} MiB)...", flush=True)
        with path.open("rb") as f:
            response = requests.put(entry["url"], headers=headers, data=f, timeout=900)
        response.raise_for_status()
    final = requests.post(
        payload["upload"]["finalizeUrl"],
        headers={"Content-Type": "application/json"},
        json={"versionId": payload["upload"]["versionId"]},
        timeout=90,
    )
    final.raise_for_status()
    final_payload = final.json()
    result = {
        "siteUrl": payload["siteUrl"],
        "videoUrl": payload["siteUrl"].rstrip("/") + "/latent-variable-models-kokoro.mp4",
        "claimUrl": payload.get("claimUrl"),
        "claimToken": payload.get("claimToken"),
        "expiresAt": payload.get("expiresAt"),
        "anonymous": payload.get("anonymous", False),
        "finalize": final_payload,
        "durationSeconds": duration,
        "videoBytes": video_path.stat().st_size,
        "sha256": sha256_file(video_path),
    }
    (OUT / "herenow-result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def verify(video_path: Path, expected_duration: float) -> dict[str, Any]:
    probe = subprocess.run([
        "ffprobe", "-v", "error", "-show_entries",
        "format=duration,size:stream=index,codec_type,codec_name,width,height,r_frame_rate,sample_rate,channels",
        "-of", "json", str(video_path),
    ], check=True, text=True, capture_output=True)
    data = json.loads(probe.stdout)
    actual = float(data["format"]["duration"])
    if abs(actual - expected_duration) > 0.6:
        raise RuntimeError(f"Duration mismatch: video={actual:.3f}, expected={expected_duration:.3f}")
    decode = subprocess.run([
        "ffmpeg", "-v", "error", "-i", str(video_path), "-f", "null", "-",
    ], text=True, capture_output=True)
    if decode.returncode != 0:
        raise RuntimeError(f"Decode verification failed: {decode.stderr[-2000:]}")
    report = {
        "expectedDuration": expected_duration,
        "actualDuration": actual,
        "sizeBytes": video_path.stat().st_size,
        "sha256": sha256_file(video_path),
        "probe": data,
    }
    (OUT / "verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    items, cues, _audio, duration = build_timeline()
    video_path = render_video(items, cues, duration)
    verification = verify(video_path, duration)
    result = publish_here_now(video_path, duration)
    metadata = {
        "title": "Latent variable models",
        "voice": "Kokoro-82M am_liam",
        "durationSeconds": duration,
        "renderFps": RENDER_FPS,
        "outputFps": OUTPUT_FPS,
        "resolution": [W, H],
        "segments": len(SEGMENTS),
        "chapters": CHAPTERS,
        "verification": verification,
        "hereNow": result,
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
