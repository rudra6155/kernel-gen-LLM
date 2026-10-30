import math

from kernelgen.reward import compute_reward


def test_reward_is_speedup_when_correct():
    assert compute_reward(2.5, True) == 2.5


def test_reward_is_zero_when_incorrect():
    assert compute_reward(10.0, False) == 0.0


def test_reward_is_zero_without_a_measurement():
    assert compute_reward(None, True) == 0.0


def test_reward_is_zero_for_non_finite_speedup():
    assert compute_reward(math.inf, True) == 0.0
    assert compute_reward(math.nan, True) == 0.0


def test_slow_correct_kernel_keeps_its_sub_one_reward():
    assert compute_reward(0.4, True) == 0.4
