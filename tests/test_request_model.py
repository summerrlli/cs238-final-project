"""Unit tests for request_model.py. Run: python -m pytest -v"""

import numpy as np
import pytest

import request_model as rm
from request_model import ACT, INV, RED, T_Z

B2 = np.array([0.7225, 0.1750, 0.1025])  # [1,0,0] T^2, computed by hand


# ---- transition matrix ----
def test_rows_sum_to_one_and_nonnegative():
    assert np.allclose(T_Z.sum(axis=1), 1.0)
    assert (T_Z >= 0).all()
    rm.check_transition_matrix(T_Z)


def test_invalid_is_absorbing():
    assert np.allclose(np.linalg.matrix_power(T_Z, 50)[INV], [0, 0, 1])
    rng = np.random.default_rng(0)
    assert all(rm.sample_next_state(INV, T_Z, rng) == INV for _ in range(1000))


@pytest.mark.parametrize("bad", [
    [[0.9, 0.1, 0.1], [0, 0.9, 0.1], [0, 0, 1]],        # row sum != 1
    [[1.1, -0.1, 0], [0, 0.9, 0.1], [0, 0, 1]],         # negative entry
    [[0.85, 0.1, 0.05], [0, 0.9, 0.1], [0.1, 0, 0.9]],  # INV not absorbing
    np.eye(2),                                          # wrong shape
])
def test_bad_matrices_rejected(bad):
    with pytest.raises(ValueError):
        rm.check_transition_matrix(bad)


# ---- belief prediction ----
def test_two_step_prediction():
    b0 = np.array([1.0, 0.0, 0.0])
    assert np.allclose(b0 @ np.linalg.matrix_power(T_Z, 2), B2, atol=1e-12)
    assert np.allclose(rm.predict_belief(b0, T_Z, 2), B2, atol=1e-12)


def test_zero_steps_returns_same_belief():
    b = np.array([0.2, 0.5, 0.3])
    assert np.allclose(rm.predict_belief(b, T_Z, 0), b)


@pytest.mark.parametrize("steps", [1, 2, 3, 10, 25])
def test_stepwise_matches_matrix_power(steps):
    for b in ([1, 0, 0], [0, 1, 0], [0.3, 0.3, 0.4]):
        assert np.allclose(rm.predict_belief(b, T_Z, steps),
                           rm.predict_belief_stepwise(b, T_Z, steps), atol=1e-12)


def test_beliefs_stay_valid_distributions():
    for b in ([1, 0, 0], [0, 1, 0], [0, 0, 1], [0.5, 0.25, 0.25]):
        for k in range(31):
            out = rm.predict_belief(b, T_Z, k)
            assert (out >= 0).all() and np.isclose(out.sum(), 1.0)


def test_bad_belief_inputs_rejected():
    with pytest.raises(ValueError):
        rm.predict_belief([0.5, 0.6, 0.0], T_Z, 1)
    with pytest.raises(ValueError):
        rm.predict_belief([1, 0, 0], T_Z, -1)


def test_sampler_matches_prediction():
    """Monte Carlo: empirical z_2 frequencies should match [1,0,0] T^2."""
    rng = np.random.default_rng(238)
    n = 100_000
    counts = np.bincount([rm.simulate_states(ACT, 2, T_Z, rng)[-1] for _ in range(n)],
                         minlength=3)
    assert np.allclose(counts / n, B2, atol=0.007)  # about 4 standard errors


# ---- delayed reports ----
def test_report_act_at_0_received_at_2():
    report = rm.StatusReport(status=ACT, sample_time=0, receive_time=2)
    b = rm.belief_from_report(report, t=2, T=T_Z)
    assert np.allclose(b, B2, atol=1e-12)
    assert not np.allclose(b, [1, 0, 0])  # NOT "the report describes now"


def test_elapsed_time_counts_from_sample_time():
    report = rm.StatusReport(ACT, sample_time=3, receive_time=5)
    b = rm.belief_from_report(report, t=6, T=T_Z)  # 3 steps since sampling
    assert np.allclose(b, rm.predict_belief([1, 0, 0], T_Z, 3))


def test_zero_delay_and_inv_reports():
    assert np.allclose(rm.belief_from_report(rm.StatusReport(RED, 4, 4), 4, T_Z), [0, 1, 0])
    assert np.allclose(rm.belief_from_report(rm.StatusReport(INV, 0, 3), 10, T_Z), [0, 0, 1])


def test_report_unusable_before_arrival():
    with pytest.raises(ValueError):
        rm.belief_from_report(rm.StatusReport(ACT, 0, 2), t=1, T=T_Z)


def test_conditioning_matches_monte_carlo():
    """Start uncertain at time 0, query at tau=2, keep only runs where the
    report says ACT, and check the true status at t=4 against the belief."""
    rng = np.random.default_rng(1)
    hits = []
    for _ in range(100_000):
        z0 = int(rng.choice(3, p=[0.6, 0.3, 0.1]))
        states = rm.simulate_states(z0, 4, T_Z, rng)
        if states[2] == ACT:
            hits.append(states[4])
    freq = np.bincount(hits, minlength=3) / len(hits)
    expected = rm.belief_from_report(rm.StatusReport(ACT, 2, 4), 4, T_Z)
    assert np.allclose(freq, expected, atol=0.01)


# ---- delays and query channel ----
def test_sample_delay_frequencies():
    rng = np.random.default_rng(2)
    draws = np.array([rm.sample_delay(rng) for _ in range(40_000)])
    for d, p in rm.DELAY_PROBS.items():
        assert abs(np.mean(draws == d) - p) < 0.01
    with pytest.raises(ValueError):
        rm.sample_delay(rng, {0: 0.5, 1: 0.4})  # sums to 0.9


def test_prob_reply_within():
    assert np.isclose(rm.prob_reply_within(0), 0.4)
    assert np.isclose(rm.prob_reply_within(2), 0.9)
    assert np.isclose(rm.prob_reply_within(10), 1.0)


def test_send_query_timestamps():
    r = rm.send_query(RED, tau=7, rng=np.random.default_rng(3), delay_probs={2: 1.0})
    assert r == rm.StatusReport(RED, 7, 9)


def test_only_one_outstanding_query():
    ch = rm.QueryChannel(np.random.default_rng(4), {2: 1.0})
    assert ch.send(ACT, tau=0) == 2
    with pytest.raises(RuntimeError):
        ch.send(ACT, tau=1)
    assert ch.receive(1) is None
    assert ch.receive(2) == rm.StatusReport(ACT, 0, 2)
    assert ch.receive(3) is None  # delivered only once
    ch.send(RED, tau=3)           # new query now allowed
