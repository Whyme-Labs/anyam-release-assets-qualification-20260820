# Manim math pass QC

## Media checks

- Base duration: 513.676 s
- Final duration: 513.676 s
- Duration delta: +0.000 s
- Final video: 1280x720 at 30/1
- Base audio packet SHA-256: `99d349a586bc6edd6a15a31f377c8b0315d6997331451da1451537b6c83c6662`
- Final audio packet SHA-256: `99d349a586bc6edd6a15a31f377c8b0315d6997331451da1451537b6c83c6662`
- Final MP4 SHA-256: `646e59eb1377bab7485771c22db121074b0f254e17a96a5048ba61bbd4e837d9`
- Full FFmpeg decode: passed
- Audio identity: passed

## Formula-scene checks

| # | Interval | Topic | Content-region MAD | Preserved-bottom MAD | Mean luma |
|---:|---:|---|---:|---:|---:|
| 1 | 42.20-57.20s | AE encoder, decoder and reconstruction objective | 19.86 | 0.28 | 11.83 |
| 2 | 109.00-131.00s | VAE posterior and reparameterization trick | 24.16 | 0.16 | 14.88 |
| 3 | 131.20-153.70s | ELBO decomposition and minimization form | 24.15 | 0.27 | 15.15 |
| 4 | 158.00-171.50s | Rate-distortion operating curve | 24.82 | 0.32 | 11.74 |
| 5 | 190.00-208.00s | Nearest-code quantization and VQ-VAE loss terms | 48.75 | 0.22 | 45.89 |
| 6 | 220.00-237.00s | Straight-through estimator, forward and backward paths | 22.51 | 0.32 | 14.65 |
| 7 | 269.40-300.40s | Residual quantization recursion and additive code | 20.37 | 0.38 | 13.60 |
| 8 | 337.80-368.80s | Sparse-autoencoder objective, L1 and Top-K | 26.94 | 0.31 | 14.98 |

The Manim panel occupies y=80..619 only. The existing top HUD, subtitles, chapter rail and corrected time-proportional progress bar remain outside the replacement region.
