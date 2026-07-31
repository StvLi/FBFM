# Curated Wan2.2 artifacts

This directory contains small, sanitized JSON summaries and three curated
MP4 files selected for the public release. Paths are deliberately relative,
and no camera serial, USB metadata, or raw capture metadata is retained.

The following MP4 files are the complete allowlist of generated media included
in the repository:

| File | SHA256 |
| --- | --- |
| `robot_arm_ball_stop/base_future.mp4` | `427ca565ddcf3b1aca19ed66ec2457d580d72c0a723d6c249fa6daffe79c50ac` |
| `robot_arm_ball_stop/fbfm_ours_future.mp4` | `fca9fbe3bf013660631181c997353dd8d580bd2d146a7acca80d270429e7b118` |
| `robot_arm_ball_stop/reference_future_121f.mp4` | `94713d1c7b501cab4b2a115c83ac437f7ecf51e0eccaef1164503518bd2c6fe0` |

Raw RealSense DB3 recordings, authoritative raw inputs, complete experiment
outputs, and ordinary run-generated videos remain outside Git. Do not infer
that another MP4 is publishable merely because it was produced by these
scripts. Add media only after an explicit data-owner, privacy, and licensing
review; record its checksum and provenance at the same time.
