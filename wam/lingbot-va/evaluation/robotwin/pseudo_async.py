class PseudoAsyncHistory:
    """Stage each real history segment exactly once before the next chunk.

    Lingbot's real KV cache must contain only observations and actions that were
    available when the active solver was launched.  Observations produced while
    that solver runs belong to FBFM's dynamic feedback set; they are staged here
    and become real history for the following launch.
    """

    def __init__(self, initial_action):
        if initial_action.ndim != 3 or initial_action.shape[1] < 1:
            raise ValueError(
                "initial_action must have shape (channels, frames, actions_per_frame)"
            )
        self._pending = ([], initial_action[:, :1].copy())

    def take(self):
        """Consume the one history segment assigned to the next KV update."""
        if self._pending is None:
            raise RuntimeError("pseudo-async history has already been consumed")
        pending = self._pending
        self._pending = None
        return pending

    def stage_execution(self, observations, action, *, execution_horizon):
        """Stage observations and aligned action frames from one executed suffix."""
        if self._pending is not None:
            raise RuntimeError("consume pending history before staging another segment")
        if action.ndim != 3 or action.shape[1] < 1 or action.shape[2] < 1:
            raise ValueError(
                "action must have shape (channels, frames, actions_per_frame)"
            )
        if execution_horizon <= 0:
            raise ValueError("execution_horizon must be positive")
        if execution_horizon % action.shape[2] != 0:
            raise ValueError(
                "execution_horizon must align to complete Lingbot action frames"
            )
        executed_frames = execution_horizon // action.shape[2]
        if executed_frames > action.shape[1]:
            raise ValueError("execution_horizon exceeds the action chunk")
        self._pending = (
            list(observations),
            action[:, -executed_frames:].copy(),
        )


def solver_step_grant(
    simulation_step: int,
    *,
    total_simulation_steps: int,
    total_solver_steps: int,
) -> int:
    """Return the deterministic solver budget released by one simulation step."""
    if total_simulation_steps <= 0:
        raise ValueError("total_simulation_steps must be positive")
    if total_solver_steps <= 0:
        raise ValueError("total_solver_steps must be positive")
    if not 1 <= simulation_step <= total_simulation_steps:
        raise ValueError(
            f"simulation_step must be in [1, {total_simulation_steps}]"
        )
    consumed_before = (
        (simulation_step - 1) * total_solver_steps // total_simulation_steps
    )
    consumed_after = simulation_step * total_solver_steps // total_simulation_steps
    return consumed_after - consumed_before
