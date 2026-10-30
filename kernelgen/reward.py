"""Reward used by the RL loop and recorded in the benchmark CSV."""

import math


def compute_reward(speedup: float | None, correct: bool) -> float:
    """reward = speedup * correct.

    Anything that did not produce a finite measured speedup, including kernels
    that failed to compile or run, gets 0.
    """
    if not correct or speedup is None or not math.isfinite(speedup):
        return 0.0
    return float(speedup)
