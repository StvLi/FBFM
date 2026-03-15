"""
Global constants and configuration factory for the 1D FBFM pre-experiment.

All tunable hyper-parameters and path conventions live here so that every
other module can ``from pre_test_2.config import …`` without circular deps.
"""

from pre_test_2.physics_env import EnvConfig
from pre_test_2.fbfm_processor import FBFMConfig, GuidanceMode

# ── Dimensions ────────────────────────────────────────────────────────
STATE_DIM = 2
ACTION_DIM = 1
HORIZON = 16

# ── Paper's 3-segment action window (H=16) ───────────────────────────
#   [0,  d)      frozen during inference delay   weight = 1.0
#   [d,  H-s)    modifiable, exp-decay guidance
#   [H-s, H)     empty planning tail             weight = 0.0
INFERENCE_DELAY = 4   # d
EMPTY_HORIZON   = 5   # s
REPLAN_INTERVAL = 9   # actions executed per cycle (d + 5)

# ── Simulation ────────────────────────────────────────────────────────
TOTAL_STEPS   = 300
RESULTS_DIR   = "pre_test_2/results_final"
CHECKPOINT    = "pre_test_2/checkpoints/best_model.pt"
DEFAULT_SEEDS = [0, 1, 2, 3, 4]

TRAIN_ENV_CFG = EnvConfig(mass=1.0, damping=0.5, stiffness=0.1, dt=0.02)


# ── Configuration factory ────────────────────────────────────────────
def get_cfg(mode: GuidanceMode, **overrides) -> FBFMConfig:
    """Create FBFMConfig with tuned default weights and paper's 3-segment structure."""
    kwargs = dict(
        mode=mode,
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        horizon=HORIZON,
        max_guidance_weight=1.0,
        state_max_guidance_weight=2.5 if mode in (GuidanceMode.FBFM, GuidanceMode.FBFM_IDENTITY) else 1.0,
        num_denoise_steps=20,
        prefix_schedule="exponential",
        exponential_decay_rate=0.7,
        inference_delay=INFERENCE_DELAY,
        empty_horizon=EMPTY_HORIZON,
        state_feedback_horizon=INFERENCE_DELAY,
    )
    kwargs.update(overrides)
    return FBFMConfig(**kwargs)
