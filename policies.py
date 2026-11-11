"""Baseline query policies (CS238 project).

Every policy sees only an Observation (no true status) and uses the same
navigation rule: head for the destination implied by the NEWEST RECEIVED
report (the assignment at t=0 counts as a report of ACT):
    ACT -> MOVE toward ED,  RED -> REROUTE (toward ALT),  INV -> RETURN.
The policies differ only in WHEN they send a QUERY.

The commitment-aware rule (proposal Eq. 2) and a POMDP planner are NOT
implemented here.
"""

from request_model import INV, RED
from simulator import ED, MOVE, QUERY, RETURN, REROUTE, Action


def navigate(obs, graph):
    status = obs.last_report.status
    if status == INV:
        return Action(RETURN)
    if status == RED:
        return Action(REROUTE)
    return Action(MOVE, graph.next_hop(obs.position, ED))


class Policy:
    name = "policy"

    def reset(self):
        pass

    def wants_query(self, obs):
        return False

    def act(self, obs, graph):
        if not obs.query_pending and self.wants_query(obs):
            return Action(QUERY)
        return navigate(obs, graph)


class NeverQuery(Policy):
    """Never queries after departure; always follows the original assignment."""
    name = "never_query"


class PeriodicQuery(Policy):
    """Query when no query is pending and at least `period` steps have passed
    since the last query was sent (or since departure at t=0)."""

    def __init__(self, period=3):
        self.period = period
        self.name = f"periodic_{period}"

    def wants_query(self, obs):
        last = obs.last_query_step if obs.last_query_step is not None else 0
        return obs.t - last >= self.period


class InfoAgeQuery(Policy):
    """Query when no query is pending and the age of the newest information,
    t - (sample time of the newest received report), exceeds `threshold`."""

    def __init__(self, threshold=3):
        self.threshold = threshold
        self.name = f"info_age_{threshold}"

    def wants_query(self, obs):
        return obs.t - obs.last_report.sample_time > self.threshold


class Scripted(Policy):
    """Plays a fixed list of actions (for tests and hand-checked traces);
    falls back to `navigate` when the list runs out."""
    name = "scripted"

    def __init__(self, actions):
        self.actions = list(actions)

    def reset(self):
        self._i = 0

    def act(self, obs, graph):
        if self._i < len(self.actions):
            self._i += 1
            return self.actions[self._i - 1]
        return navigate(obs, graph)
