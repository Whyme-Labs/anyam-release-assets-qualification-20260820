"""Compatibility and correctness wrapper for the Manim math-pass compositor.

The complete implementation is pinned to an immutable commit. This wrapper
applies three narrow fixes before executing it:

1. RGB histograms have 256 bins per channel. Histogram indices must therefore
   be reduced modulo 256 when computing mean absolute pixel differences.
2. Reset the base-video timestamps before chaining overlays.
3. Limit complex-filter threading for a deterministic eight-overlay graph.
"""
from __future__ import annotations

from urllib.request import urlopen

SOURCE_URL = (
    "https://raw.githubusercontent.com/Whyme-Labs/"
    "anyam-release-assets-qualification-20260820/"
    "f238ad14a9596103d09180d1c5ed6af83f10619e/compose_math_pass.py"
)

with urlopen(SOURCE_URL, timeout=30) as response:
    source = response.read().decode("utf-8")

replacements = {
    'cmd = ["ffmpeg", "-y", "-i", str(base)]':
        'cmd = ["ffmpeg", "-y", "-filter_complex_threads", "1", "-i", str(base)]',
    'filters: list[str] = []\n    previous = "[0:v]"':
        'filters: list[str] = ["[0:v]setpts=PTS-STARTPTS[base]"]\n    previous = "[base]"',
    'return sum(value * n for value, n in enumerate(hist)) / count':
        'return sum((value % 256) * n for value, n in enumerate(hist)) / count',
}

for old, new in replacements.items():
    if old not in source:
        raise RuntimeError(f"Pinned compositor changed or patch target missing: {old}")
    source = source.replace(old, new, 1)

exec(compile(source, SOURCE_URL, "exec"), globals(), globals())
