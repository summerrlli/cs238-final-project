"""Reproducible demo: graph, belief model, and hand-checkable episode traces.

Run: python demo.py
"""

import numpy as np

import request_model as rm
from policies import InfoAgeQuery, NeverQuery, PeriodicQuery
from request_model import ACT, INV, RED, STATE_NAMES, T_Z, format_belief
from simulator import (COMMIT_EDGE, EDGES, HospitalGraph, RewardConfig, Scenario,
                       format_trace, make_scenario, run_episode)

H = 20


def fixed_scenario(change_t=None, to=None, delay=0):
    z = [ACT] * (H + 1)
    if change_t is not None:
        z[change_t:] = [to] * (H + 1 - change_t)
    return Scenario(tuple(z), tuple([delay] * H))


# Hand-checkable traces (all use the periodic-3 baseline).
TRACES = {
    "A. unchanged active request (delay 0)": fixed_scenario(),
    "B. redirected to ALT at t=2 (delay 0)": fixed_scenario(2, RED),
    "C. invalidated at t=2 (delay 0)": fixed_scenario(2, INV),
    "D. redirected at t=2, report delay 2 (robot already committed)": fixed_scenario(2, RED, 2),
}


def main():
    print("=== Synthetic hospital graph (not a real layout) ===")
    for (a, b), L in EDGES.items():
        tag = "   <- commitment segment (reversal cost applies J4->J3)" if (a, b) == COMMIT_EDGE else ""
        print(f"  {a:>6} -- {b:<6} length {L}{tag}")
    print(f"  rewards: {RewardConfig()}")

    print("\n=== Request model ===")
    print(T_Z)
    print("  [1,0,0] T^2 =", format_belief(rm.predict_belief([1, 0, 0], T_Z, 2)))

    graph = HospitalGraph()
    for title, scn in TRACES.items():
        r = run_episode(PeriodicQuery(3), scn, graph)
        print(f"\n=== Trace {title} ===")
        print(f"  true z: {''.join(STATE_NAMES[z][0] for z in scn.z)}  (A=ACT R=RED I=INV, t=0..{H})")
        print(format_trace(r.trace))
        print(f"  -> outcome={r.outcome} reward={r.total_reward:.1f} travel={r.travel_steps} "
              f"queries={r.query_count} "
              f"reversals={r.reversals} unnecessary_after_INV={r.unnecessary_travel_after_invalid}")

    print("\n=== Same random scenario (seed 3), three baselines ===")
    scn = make_scenario(3, H)
    print(f"  true z: {''.join(STATE_NAMES[z][0] for z in scn.z)}   delays: {scn.delays[:6]}...")
    for p in (NeverQuery(), PeriodicQuery(3), InfoAgeQuery(3)):
        r = run_episode(p, scn, graph)
        print(f"  {p.name:<12} outcome={r.outcome:<24} reward={r.total_reward:7.1f} "
              f"travel={r.travel_steps} queries={r.query_count}")


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    main()
