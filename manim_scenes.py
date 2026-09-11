"""Runtime-patched source for the latent-model Manim math pass.

The complete scene implementation is pinned to an immutable commit. This
loader applies compatibility and mathematical-correctness fixes before Manim
executes the scenes.
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

replacements = {
    ", letter_spacing=0.6": "",
    r'r"\beta\,\underbrace{D_{\mathrm{KL}}(q_{\phi}(z\mid x)\|p(z))}_{\text{shape latent space}}"':
        r'r"\underbrace{D_{\mathrm{KL}}(q_{\phi}(z\mid x)\|p(z))}_{\text{shape latent space}}"',
    r'r"\min\;\mathcal J"': r'r"\min\;\mathcal J_{\beta}"',
    'self.play(TransformFromCopy(elbo, loss), run_time=1.6)\n        self.play(Indicate(rec_box, color=PURPLE_LIGHT), Indicate(kl_box, color=ORANGE), run_time=1.2)':
        'self.play(TransformFromCopy(elbo, loss), run_time=1.6)\n'
        '        beta_note = label("beta = 1: standard VAE; beta changes the rate-distortion trade-off", 21, GREY).next_to(loss, DOWN, buff=0.12)\n'
        '        self.play(FadeIn(beta_note, shift=UP * 0.06), run_time=0.6)\n'
        '        self.play(Indicate(rec_box, color=PURPLE_LIGHT), Indicate(kl_box, color=ORANGE), run_time=1.2)',
}

for old, new in replacements.items():
    if old not in source:
        raise RuntimeError(f"Pinned Manim source changed or patch target missing: {old}")
    source = source.replace(old, new, 1)

exec(compile(source, SOURCE_URL, "exec"), globals(), globals())
