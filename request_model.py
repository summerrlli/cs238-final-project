"""Hidden request-status model and belief prediction (CS238 project).

Follows the proposal: hidden request status z_t in {Act, Red, Inv}, Markov
transitions T_z, belief b_t(z), and timestamped status reports that arrive
after a random delay D. This is NOT a simulator or a POMDP planner.
"""

from dataclasses import dataclass

import numpy as np

ACT, RED, INV = 0, 1, 2  # active, redirected, invalid (fulfilled or canceled)
STATE_NAMES = ("ACT", "RED", "INV")

# T_Z[i, j] = P(z_{t+1} = j | z_t = i). Synthetic numbers, not hospital data.
# A redirected request can later become invalid; invalid is absorbing.
T_Z = np.array([
    [0.85, 0.10, 0.05],
    [0.00, 0.90, 0.10],
    [0.00, 0.00, 1.00],
])

# Reply delay D in time steps -> probability. Also synthetic.
DELAY_PROBS = {0: 0.4, 1: 0.3, 2: 0.2, 3: 0.1}


def check_transition_matrix(T):
    """Raise ValueError unless T is 3x3, nonnegative, row-stochastic, INV absorbing."""
    T = np.asarray(T, dtype=float)
    if T.shape != (3, 3):
        raise ValueError(f"T must be 3x3, got shape {T.shape}")
    if (T < 0).any():
        raise ValueError("T has a negative probability")
    if not np.allclose(T.sum(axis=1), 1.0):
        raise ValueError("each row of T must sum to 1")
    if not np.allclose(T[INV], [0, 0, 1]):
        raise ValueError("INV must be absorbing")


def check_belief(b):
    """Return b as an array, or raise ValueError if it is not a distribution."""
    b = np.asarray(b, dtype=float)
    if b.shape != (3,) or (b < 0).any() or not np.isclose(b.sum(), 1.0):
        raise ValueError(f"not a probability vector over 3 states: {b}")
    return b


def sample_next_state(z, T, rng):
    """Draw z_{t+1} from row T[z]."""
    return int(rng.choice(3, p=T[z]))


def simulate_states(z0, steps, T, rng):
    """Return a sampled trajectory [z_0, z_1, ..., z_steps]."""
    states = [z0]
    for _ in range(steps):
        states.append(sample_next_state(states[-1], T, rng))
    return states


def predict_belief(b, T, steps=1):
    """Return b_{t+k} = b_t T^k.

    One step is proposal Eq. 1: b'(z') = sum_z T(z'|z) b(z), i.e. b @ T.
    With steps = 0, T^0 = I, so b is returned unchanged.
    """
    if steps < 0:
        raise ValueError("steps must be >= 0")
    return check_belief(b) @ np.linalg.matrix_power(T, steps)


def predict_belief_stepwise(b, T, steps=1):
    """Same result as predict_belief, applying b <- b @ T one step at a time."""
    b = check_belief(b)
    for _ in range(steps):
        b = b @ T
    return b


# ---- Delayed status reports ------------------------------------------------
# Assumptions (first version):
#   1. A query sent at time tau reports z_tau exactly. It is correct about
#      time tau, not necessarily about the time it arrives.
#   2. It arrives at tau + D, with D drawn from DELAY_PROBS independently of z.
#   3. At most one query is outstanding, and no other status info arrives.
#   4. The robot's actions do not change the request status.

@dataclass(frozen=True)
class StatusReport:
    status: int        # z observed at sample_time
    sample_time: int   # tau: when the query was sent
    receive_time: int  # tau + D: when the reply reaches the robot


def sample_delay(rng, delay_probs=DELAY_PROBS):
    """Draw a reply delay D (numpy raises ValueError if probs are invalid)."""
    return int(rng.choice(list(delay_probs), p=list(delay_probs.values())))


def prob_reply_within(k, delay_probs=DELAY_PROBS):
    """P(D <= k): chance a reply arrives within k steps (term in proposal Eq. 2)."""
    return sum(p for d, p in delay_probs.items() if d <= k)


def send_query(z_tau, tau, rng, delay_probs=DELAY_PROBS):
    """Report produced by a query sent at time tau when the true status is z_tau."""
    return StatusReport(z_tau, tau, tau + sample_delay(rng, delay_probs))


def belief_from_report(report, t, T):
    """Current belief b_t after receiving `report` at or before time t.

    The report is exact about time tau, so the belief at tau is one-hot on
    report.status (by the Markov property, older history no longer matters).
    The request kept changing while the reply was in transit, so we predict
    forward t - tau steps, NOT t - receive_time:
        b_t = one_hot(status) T^(t - tau)
    """
    if t < report.receive_time:
        raise ValueError("report has not arrived yet")
    b_tau = np.zeros(3)
    b_tau[report.status] = 1.0
    return predict_belief(b_tau, T, t - report.sample_time)


class QueryChannel:
    """Holds at most one outstanding status query (assumption 3)."""

    def __init__(self, rng, delay_probs=DELAY_PROBS):
        self.rng = rng
        self.delay_probs = delay_probs
        self.pending = None

    def send(self, z_tau, tau):
        """Send a query at time tau; return its arrival time."""
        if self.pending is not None:
            raise RuntimeError("a query is already outstanding")
        self.pending = send_query(z_tau, tau, self.rng, self.delay_probs)
        return self.pending.receive_time

    def receive(self, t):
        """Return the pending report if it has arrived by time t, else None."""
        if self.pending is None or t < self.pending.receive_time:
            return None
        report, self.pending = self.pending, None
        return report


def format_belief(b):
    """E.g. 'ACT=0.7225 RED=0.1750 INV=0.1025'."""
    return " ".join(f"{name}={p:.4f}" for name, p in zip(STATE_NAMES, b))
