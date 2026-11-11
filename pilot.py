"""Small paired pilot of the three baselines.

Design: 20 scenarios from seeds 0..19 (make_scenario). Each scenario fixes the
whole true-status trajectory z_0..z_H and the delay of the 1st, 2nd, ... query.
Every policy is run on every scenario, so request changes and the k-th
query's delay are identical across policies (paired design).

This is a pilot for debugging and sanity checks. 20 episodes are far too
few for significance claims, and none are made.

Run: python pilot.py   -> results/pilot_episodes.csv, results/pilot_summary.json
"""

import csv
import json
from pathlib import Path

import numpy as np

from policies import InfoAgeQuery, NeverQuery, PeriodicQuery
from request_model import INV, RED
from simulator import make_scenario, run_episode

SEEDS = range(20)
HORIZON = 20
OUT = Path(__file__).parent / "results"


def first_change(z, state):
    return next((t for t, s in enumerate(z) if s == state), None)


def main():
    OUT.mkdir(exist_ok=True)
    policies = [NeverQuery(), PeriodicQuery(3), InfoAgeQuery(3)]
    scenarios = {s: make_scenario(s, HORIZON) for s in SEEDS}
    rows = []
    for seed, scn in scenarios.items():
        for p in policies:
            r = run_episode(p, scn)
            rows.append({
                "seed": seed, "policy": p.name, "outcome": r.outcome,
                "total_reward": r.total_reward, "travel_steps": r.travel_steps,
                "query_count": r.query_count, "reversals": r.reversals,
                "success": int(r.success), "incorrect_delivery": int(r.incorrect_delivery),
                "unnecessary_travel_after_invalid": r.unnecessary_travel_after_invalid,
                "episode_steps": r.steps,
                "t_first_RED": first_change(scn.z, RED), "t_first_INV": first_change(scn.z, INV),
            })
    with open(OUT / "pilot_episodes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    summary = {"design": "paired; seeds 0-19; horizon 20; same scenarios for all policies",
               "n_episodes_per_policy": len(SEEDS), "policies": {}}
    print(f"{'policy':<12} {'reward mean (sd)':>18} {'travel':>7} {'queries':>8} "
          f"{'success':>8} {'incorrect':>9} {'unnec.INV':>9}")
    for p in policies:
        rs = [r for r in rows if r["policy"] == p.name]
        col = lambda k: np.array([r[k] for r in rs], dtype=float)
        rew = col("total_reward")
        s = {
            "total_reward_mean": rew.mean(), "total_reward_sd": rew.std(ddof=1),
            "total_reward_min": rew.min(), "total_reward_max": rew.max(),
            "travel_steps_mean": col("travel_steps").mean(),
            "query_count_mean": col("query_count").mean(),
            "correct_delivery_rate": col("success").mean(),
            "incorrect_delivery_rate": col("incorrect_delivery").mean(),
            "unnecessary_travel_after_invalid_mean": col("unnecessary_travel_after_invalid").mean(),
            "outcome_counts": {o: sum(r["outcome"] == o for r in rs)
                               for o in sorted({r["outcome"] for r in rs})},
        }
        summary["policies"][p.name] = {k: (round(float(v), 4) if not isinstance(v, dict) else v)
                                       for k, v in s.items()}
        print(f"{p.name:<12} {s['total_reward_mean']:8.1f} ({s['total_reward_sd']:6.1f}) "
              f"{s['travel_steps_mean']:7.2f} {s['query_count_mean']:8.2f} "
              f"{s['correct_delivery_rate']:8.2f} {s['incorrect_delivery_rate']:9.2f} "
              f"{s['unnecessary_travel_after_invalid_mean']:9.2f}")
    for p in policies:
        print(f"  {p.name}: {summary['policies'][p.name]['outcome_counts']}")
    (OUT / "pilot_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {OUT/'pilot_episodes.csv'} and {OUT/'pilot_summary.json'}")


if __name__ == "__main__":
    main()
