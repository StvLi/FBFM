import threading
import time

from dreamzero_fbfm.pseudo_clock import SolverClock, solver_grants


def test_integer_grants_are_deterministic_and_complete():
    assert solver_grants(8, 16) == (2, 2, 2, 2, 2, 2, 2, 2)
    assert solver_grants(3, 8) == (2, 3, 3)
    assert solver_grants(8, 8, release_policy="after_feedback") == (
        0, 0, 0, 0, 0, 0, 0, 8
    )


def test_grant_waits_for_solver_completion():
    clock = SolverClock()
    clock.start(enabled=True)
    result = {}

    def worker():
        assert clock.consume()
        time.sleep(0.02)
        clock.complete()
        assert clock.consume()
        time.sleep(0.02)
        clock.complete()

    thread = threading.Thread(target=worker)
    thread.start()
    started = time.perf_counter()
    result.update(clock.grant_and_wait(2, timeout=1))
    elapsed = time.perf_counter() - started
    thread.join()
    assert result["accepted"] is True
    assert result["completed"] == 2
    assert elapsed >= 0.03
