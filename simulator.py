"""Discrete-time hospital delivery simulator (CS238 project).

A robot carries supplies from SUPPLY toward the emergency department (ED).
While it travels, the hidden request status z_t in {ACT, RED, INV} evolves by
the Markov chain in request_model.py. The robot only learns about z through
timestamped, delayed status reports.

The graph is SYNTHETIC and intentionally tiny. It does not represent any real
hospital layout.

Timing conventions (one decision = one time step, t -> t+1):
  * MOVE / RETURN / REROUTE: advance one unit of distance along an edge.
    An edge of length L takes L steps; between nodes the robot is "in
    transit" (Position with v != None). It may continue or turn around.
  * WAIT: stay in place (also allowed mid-edge) for one step.
  * QUERY: the robot stops for one step to submit a status query. The
    query is submitted at the END of that step, time tau = t+1, and the
    report describes z_{tau}. It arrives at tau + D. With D = 0 the robot
    can use it at its next decision (t+1). Only one query may be pending;
    the robot may keep moving while it waits.

Costs (all configurable in RewardConfig, applied once per step, no overlap):
  * move_cost per MOVE/RETURN/REROUTE step,
  * wait_cost per WAIT step,
  * query_cost per QUERY step (the initiation cost; it replaces, not adds
    to, the wait cost of that stationary step). The response delay D costs
    nothing by itself; any time lost is paid through the steps that elapse.
  * reversal_cost: see HospitalGraph.is_reversal.
"""

from dataclasses import dataclass, field

import numpy as np

import request_model as rm
from request_model import ACT, INV, RED, STATE_NAMES, T_Z

# ---- graph -------------------------------------------------------------------
SUPPLY, J1, J2, J3, J4, ED, ALT = "SUPPLY", "J1", "J2", "J3", "J4", "ED", "ALT"
NODES = (SUPPLY, J1, J2, J3, J4, ED, ALT)

# Undirected edges with integer travel times (in steps). Synthetic numbers.
EDGES = {
    (SUPPLY, J1): 1,
    (J1, J2): 1,
    (J2, J3): 1,
    (J3, J4): 3,   # commitment segment sigma (forward direction J3 -> J4)
    (J4, ED): 1,
    (J2, ALT): 2,
    (J4, ALT): 4,
}
COMMIT_EDGE = (J3, J4)
TERMINAL_NODES = (SUPPLY, ED, ALT)  # arriving at any of these ends the episode


@dataclass(frozen=True)
class Position:
    """Robot location. At a node: Position(u). In transit: Position(u, v, p),
    meaning it left u, is heading to v, and has travelled p of len(u,v) steps
    (0 < p < len)."""
    u: str
    v: str | None = None
    progress: int = 0

    @property
    def at_node(self):
        return self.v is None

    def __str__(self):
        return self.u if self.at_node else f"{self.u}->{self.v}({self.progress})"


class HospitalGraph:
    def __init__(self, edges=EDGES, commit_edge=COMMIT_EDGE, terminals=TERMINAL_NODES):
        self.length, self.adj = {}, {}
        for (a, b), L in edges.items():
            if int(L) != L or L < 1:
                raise ValueError(f"edge {a}-{b} needs an integer length >= 1")
            self.length[(a, b)] = self.length[(b, a)] = int(L)
            self.adj.setdefault(a, []).append(b)
            self.adj.setdefault(b, []).append(a)
        for n in self.adj:
            self.adj[n].sort()
        if commit_edge not in self.length:
            raise ValueError("commitment edge must be an edge of the graph")
        self.commit_edge = commit_edge
        self.terminals = tuple(terminals)

    def neighbors(self, n):
        return list(self.adj[n])

    def edge_length(self, a, b):
        return self.length[(a, b)]

    def is_reversal(self, a, b):
        """REVERSAL RULE: c_rev is charged once whenever the robot STARTS
        travelling backward (J4 -> J3) on the commitment edge, i.e. either
          (1) it leaves node J4 heading to J3, or
          (2) it is in transit J3 -> J4 and turns around toward J3.
        Entering the segment forward, continuing, and waiting on it are free
        (apart from normal step costs). Continuing backward after the turn is
        not charged again."""
        return (b, a) == self.commit_edge

    # Shortest paths. Terminal nodes may be a start or an end but never an
    # intermediate stop (arriving there would end the episode).
    def _dist_to(self, dst, move_cost, rev_cost):
        """Cheapest cost from every node to dst; an edge costs
        len * move_cost, plus rev_cost if it is a reversal."""
        dist = {n: np.inf for n in self.adj}
        dist[dst] = 0.0
        for _ in range(len(self.adj)):  # Bellman-Ford; the graph is tiny
            for (a, b), L in self.length.items():
                if b != dst and b in self.terminals:
                    continue
                c = L * move_cost + (rev_cost if self.is_reversal(a, b) else 0.0) + dist[b]
                if c < dist[a]:
                    dist[a] = c
        return dist

    def step_cost_to(self, pos, dst, move_cost=1.0, rev_cost=0.0):
        """Return {next_target: cost-to-go} for each way out of `pos`."""
        d = self._dist_to(dst, move_cost, rev_cost)
        options = {}
        if pos.at_node:
            for n in self.neighbors(pos.u):
                if n != dst and n in self.terminals:
                    continue
                rev = rev_cost if self.is_reversal(pos.u, n) else 0.0
                options[n] = self.edge_length(pos.u, n) * move_cost + rev + d[n]
        else:
            L = self.edge_length(pos.u, pos.v)
            if pos.v == dst or pos.v not in self.terminals:
                options[pos.v] = (L - pos.progress) * move_cost + d[pos.v]
            if pos.u == dst or pos.u not in self.terminals:
                rev = rev_cost if self.is_reversal(pos.v, pos.u) else 0.0
                options[pos.u] = pos.progress * move_cost + rev + d[pos.u]
        return options

    def next_hop(self, pos, dst, move_cost=1.0, rev_cost=0.0):
        """Neighbor (or edge endpoint) to MOVE toward on a cheapest route to
        dst. Ties prefer continuing forward, then alphabetical order."""
        if pos.at_node and pos.u == dst:
            raise ValueError(f"already at {dst}")
        opts = self.step_cost_to(pos, dst, move_cost, rev_cost)
        if not pos.at_node:  # prefer forward on ties
            order = [pos.v, pos.u]
        else:
            order = sorted(opts)
        best = min((o for o in order if o in opts), key=lambda o: opts[o])
        if not np.isfinite(opts[best]):
            raise ValueError(f"{dst} unreachable from {pos}")
        return best

    def time_to(self, pos, dst):
        """Fewest travel steps from pos to dst (ignoring reversal cost)."""
        if pos.at_node and pos.u == dst:
            return 0.0
        return min(self.step_cost_to(pos, dst).values())


# ---- rewards, scenarios, actions, observations --------------------------------
@dataclass(frozen=True)
class RewardConfig:
    """Synthetic rewards (positive) and costs (subtracted). Not hospital data."""
    correct_delivery: float = 100.0
    wrong_destination: float = -50.0   # delivered to the other unit
    invalid_delivery: float = -20.0    # delivered after the request became INV
    abandon_valid: float = -60.0       # returned to SUPPLY while ACT or RED
    return_invalid: float = 0.0        # returned to SUPPLY while INV (correct)
    timeout: float = -40.0             # horizon reached without finishing
    move_cost: float = 1.0
    wait_cost: float = 1.0
    query_cost: float = 2.0
    reversal_cost: float = 10.0


@dataclass(frozen=True)
class Scenario:
    """Pre-generated environment randomness, shared across policies.
    z[t] is the true status at time t (t = 0..horizon); delays[k] is the
    reply delay of the k-th query the robot sends."""
    z: tuple
    delays: tuple

    @property
    def horizon(self):
        return len(self.z) - 1


def make_scenario(seed, horizon=20, T=T_Z, delay_probs=rm.DELAY_PROBS, z0=ACT):
    """Reproducible scenario. Status and delays use independent child streams
    so changing one never perturbs the other."""
    rz, rd = (np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(2))
    z = rm.simulate_states(z0, horizon, T, rz)
    delays = [rm.sample_delay(rd, delay_probs) for _ in range(horizon)]
    return Scenario(tuple(z), tuple(delays))


MOVE, QUERY, WAIT, RETURN, REROUTE = "MOVE", "QUERY", "WAIT", "RETURN", "REROUTE"


@dataclass(frozen=True)
class Action:
    kind: str
    target: str | None = None  # only for MOVE

    def __str__(self):
        return f"MOVE({self.target})" if self.kind == MOVE else self.kind


@dataclass(frozen=True)
class Observation:
    """Everything a policy may use. Deliberately has NO true status field."""
    t: int
    position: Position
    belief: np.ndarray            # robot's b_t over (ACT, RED, INV)
    last_report: rm.StatusReport  # newest report received (t=0 assignment counts)
    query_pending: bool
    pending_sample_time: int | None
    last_query_step: int | None   # decision time of the robot's last QUERY
    queries_sent: int


# ---- environment ----------------------------------------------------------------
OUTCOMES = ("delivered_correct_ED", "delivered_correct_ALT", "wrong_destination",
            "delivered_invalid", "returned_after_invalid", "abandoned_valid", "timeout")
SUCCESS = ("delivered_correct_ED", "delivered_correct_ALT")
INCORRECT_DELIVERY = ("wrong_destination", "delivered_invalid")


class DeliveryEnv:
    def __init__(self, scenario, graph=None, rewards=None, T=T_Z):
        rm.check_transition_matrix(T)
        if scenario.z[0] != ACT:
            raise ValueError("episodes start with an active request")
        self.scn = scenario
        self.graph = graph or HospitalGraph()
        self.R = rewards or RewardConfig()
        self.T = T
        self.horizon = scenario.horizon
        self.t = 0
        self.position = Position(SUPPLY)
        self.channel = rm.QueryChannel(rng=None)  # delays come from the scenario
        self.belief = np.array([1.0, 0.0, 0.0])
        self.last_report = rm.StatusReport(ACT, 0, 0)  # the assignment itself
        self.last_query_step = None
        self.queries_sent = 0
        self.done = False
        self.outcome = None
        self.totals = {"move": 0.0, "wait": 0.0, "query": 0.0, "reversal": 0.0, "terminal": 0.0}
        self.travel_steps = 0
        self.reversals = 0
        self.unnecessary_travel = 0
        self.trace = []

    @property
    def true_status(self):
        """Hidden. For logging/tests only; never passed to policies."""
        return self.scn.z[self.t]

    @property
    def total_reward(self):
        return sum(self.totals.values())

    def observe(self):
        p = self.channel.pending
        return Observation(self.t, self.position, self.belief.copy(), self.last_report,
                           p is not None, p.sample_time if p else None,
                           self.last_query_step, self.queries_sent)

    def resolve(self, action):
        """Turn RETURN / REROUTE into a concrete MOVE target."""
        if action.kind in (RETURN, REROUTE):
            dst = SUPPLY if action.kind == RETURN else ALT
            return Action(MOVE, self.graph.next_hop(
                self.position, dst, self.R.move_cost, self.R.reversal_cost))
        return action

    def _advance(self, target):
        """Move one step toward `target`. Returns (new_position, reversed?)."""
        pos, g = self.position, self.graph
        if pos.at_node:
            if target not in g.adj[pos.u]:
                raise ValueError(f"{target} is not adjacent to {pos.u}")
            u, v, p, rev = pos.u, target, 1, g.is_reversal(pos.u, target)
        elif target == pos.v:
            u, v, p, rev = pos.u, pos.v, pos.progress + 1, False
        elif target == pos.u:  # turn around mid-edge
            L = g.edge_length(pos.u, pos.v)
            u, v, p, rev = pos.v, pos.u, L - pos.progress + 1, g.is_reversal(pos.v, pos.u)
        else:
            raise ValueError(f"in transit {pos}: can only MOVE to {pos.u} or {pos.v}")
        new = Position(v) if p >= g.edge_length(u, v) else Position(u, v, p)
        return new, rev

    def step(self, action):
        if self.done:
            raise RuntimeError("episode is over")
        t, z_t, before = self.t, self.true_status, self.position
        concrete = self.resolve(action)
        reward = {}
        if concrete.kind == QUERY:
            if self.channel.pending is not None:
                raise RuntimeError("a query is already pending")
            reward["query"] = -self.R.query_cost
        elif concrete.kind == WAIT:
            reward["wait"] = -self.R.wait_cost
        elif concrete.kind == MOVE:
            self.position, rev = self._advance(concrete.target)
            reward["move"] = -self.R.move_cost
            self.travel_steps += 1
            if rev:
                reward["reversal"] = -self.R.reversal_cost
                self.reversals += 1
            # Unnecessary travel: a move made while the request is truly INV
            # that does not bring the robot closer (in steps) to SUPPLY.
            if z_t == INV and self.graph.time_to(self.position, SUPPLY) >= \
                    self.graph.time_to(before, SUPPLY):
                self.unnecessary_travel += 1
        else:
            raise ValueError(f"unknown action {action}")

        # Time advances; the hidden status moves to z_{t+1}.
        self.t = t + 1
        z_next = self.true_status
        if concrete.kind == QUERY:
            delay = self.scn.delays[self.queries_sent]
            self.channel.send(z_next, tau=self.t, delay=delay)
            self.last_query_step = t
            self.queries_sent += 1

        # Robot-side belief: predict one step, then condition on any arrival.
        self.belief = rm.predict_belief(self.belief, self.T, 1)
        report = self.channel.receive(self.t)
        if report is not None:
            self.belief = rm.belief_from_report(report, self.t, self.T)
            self.last_report = report

        # Termination.
        # Arrival only counts after a movement step (WAIT/QUERY at SUPPLY
        # before departure is not a "return").
        arrived = concrete.kind == MOVE and self.position.at_node
        if arrived and self.position.u in TERMINAL_NODES:
            self.outcome = self._arrival_outcome(self.position.u, z_next)
        elif self.t >= self.horizon:
            self.outcome = "timeout"
        if self.outcome:
            self.done = True
            reward["terminal"] = self._terminal_reward(self.outcome)
        for k, v in reward.items():
            self.totals[k] += v

        self.trace.append({
            "t": t, "action": str(action), "resolved": str(concrete),
            "pos_before": str(before), "pos_after": str(self.position),
            "z_t": STATE_NAMES[z_t], "z_next": STATE_NAMES[z_next],
            "report": None if report is None else
                f"{STATE_NAMES[report.status]}@tau={report.sample_time} (arrived t={report.receive_time})",
            "pending": self.channel.pending is not None,
            "belief": [round(float(x), 4) for x in self.belief],
            "reward": sum(reward.values()), "outcome": self.outcome,
        })
        return self.observe(), sum(reward.values()), self.done

    @staticmethod
    def _arrival_outcome(node, z):
        if node == SUPPLY:
            return "returned_after_invalid" if z == INV else "abandoned_valid"
        if z == INV:
            return "delivered_invalid"
        if (node, z) == (ED, ACT):
            return "delivered_correct_ED"
        if (node, z) == (ALT, RED):
            return "delivered_correct_ALT"
        return "wrong_destination"

    def _terminal_reward(self, outcome):
        R = self.R
        return {
            "delivered_correct_ED": R.correct_delivery,
            "delivered_correct_ALT": R.correct_delivery,
            "wrong_destination": R.wrong_destination,
            "delivered_invalid": R.invalid_delivery,
            "returned_after_invalid": R.return_invalid,
            "abandoned_valid": R.abandon_valid,
            "timeout": R.timeout,
        }[outcome]


@dataclass
class EpisodeResult:
    policy: str
    outcome: str
    total_reward: float
    travel_steps: int
    query_count: int
    reversals: int
    unnecessary_travel_after_invalid: int
    steps: int
    trace: list = field(repr=False, default_factory=list)

    @property
    def success(self):
        return self.outcome in SUCCESS

    @property
    def incorrect_delivery(self):
        return self.outcome in INCORRECT_DELIVERY


def run_episode(policy, scenario, graph=None, rewards=None, T=T_Z):
    env = DeliveryEnv(scenario, graph, rewards, T)
    policy.reset()
    obs = env.observe()
    while not env.done:
        obs, _, _ = env.step(policy.act(obs, env.graph))
    return EpisodeResult(policy.name, env.outcome, env.total_reward, env.travel_steps,
                         env.queries_sent, env.reversals, env.unnecessary_travel,
                         env.t, env.trace)


def format_trace(trace):
    lines = [f"{'t':>2}  {'action':<12} {'resolved':<12} {'pos':<13} {'z_t->z_t+1':<10} "
             f"{'report received':<32} {'belief[ACT,RED,INV]':<24} {'r':>6}"]
    for s in trace:
        lines.append(
            f"{s['t']:>2}  {s['action']:<12} {s['resolved']:<12} {s['pos_after']:<13} "
            f"{s['z_t'] + '->' + s['z_next']:<10} {s['report'] or '-':<32} "
            f"{str(s['belief']):<24} {s['reward']:>6.1f}" + (f"  END: {s['outcome']}" if s['outcome'] else ""))
    return "\n".join(lines)
