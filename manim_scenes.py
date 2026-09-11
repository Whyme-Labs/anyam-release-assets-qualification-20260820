"""Runtime-patched source for the latent-model Manim math pass.

The complete scene implementation is pinned to the immutable commit below.
This loader applies a narrow compatibility fix for Manim Community 0.19.0,
whose Text constructor does not accept the legacy letter_spacing argument.
"""
from __future__ import annotations

from urllib.request import urlopen

SOURCE_URL = (
    "https://raw.githubusercontent.com/Whyme-Labs/"
    "anyam-release-assets-qualification-20260820/"
    "fc0d729a51b3abfa6675d09aad82c10030b6cf24/manim_scenes.py"
)

with urlopen(SOURCE_URL, timeout=30) as response:
    source = response.read().decode("utf-8")

source = source.replace(", letter_spacing=0.6", "")
exec(compile(source, SOURCE_URL, "exec"), globals(), globals())
