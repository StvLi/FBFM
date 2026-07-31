# Licensing and attribution

- The upstream checkout is `Wan-Video/Wan2.2` at the commit recorded in
  `UPSTREAM.lock`. Its Apache-2.0 `LICENSE.txt` and upstream attribution must
  remain in the applied checkout.
- The FBFM overlay files in `overlay/` and the corresponding additions in
  `patches/wan2.2_fbfm.patch` are released under the repository's Apache-2.0
  `LICENSE`. Copyright in those contributions remains with their respective
  authors and contributors.
- FlashAttention, PyTorch, Diffusers, Transformers, ImageIO, OpenCV, and
  other dependencies retain their own licenses; installing them does not
  transfer those licenses to this repository.
- Raw RealSense DB3 recordings, full experiment outputs, and ordinary
  run-generated videos are research artifacts, not code. They are excluded
  from this repository and have no implied redistribution permission; each
  requires a separate data-owner, privacy, and licensing review.
- The only generated media included in this release are the three curated MP4
  files allowlisted, named, and checksummed in `artifacts/README.md`. Their
  inclusion does not grant permission to redistribute the source recordings,
  other generated outputs, or any additional media.
