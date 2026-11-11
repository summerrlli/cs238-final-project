# CS238 Final Project

## To Check or Not to Check? Commitment-Aware Task Verification for Hospital Delivery Robots

This project studies when a hospital delivery robot should verify that its task is still valid. A request may stay active, be redirected to another unit, or become invalid while the robot travels. Each verification costs something, and its reply arrives after a delay. The reply reports the request's status at the time the query was sent. The central question is whether verification decisions improve when they account for an upcoming commitment segment, where turning back is expensive.

## Current status (Nov 10)

**Implemented and tested:**

- The request-status model.
- A discrete-time delivery simulator on a small synthetic graph.
- Three simple baselines: never query, periodic query, and an information-age threshold.
- A 20-scenario paired pilot.

**Not implemented:** the commitment-aware query rule (proposal Eq. 2) and the belief-state reference planner.

See `progress_evidence.md` for test output and pilot numbers.

```bash
python -m pytest -v     # 68 tests
python demo.py          # graph, belief example, 4 hand-checkable traces
python pilot.py         # writes results/pilot_episodes.csv, results/pilot_summary.json
python plot_graph.py    # writes figures/hospital_graph.png
```

Requirements: Python 3, NumPy, pytest, and Matplotlib (only for the figure). Everything runs on a CPU.

## Files

| File | Contents |
| ---- | -------- |
| `request_model.py` | `T_Z`, belief prediction, timestamped delayed reports, one-query channel |
| `simulator.py` | Graph, reversal rule, actions, rewards, `DeliveryEnv`, `run_episode` |
| `policies.py` | `NeverQuery`, `PeriodicQuery`, `InfoAgeQuery`, and `Scripted` (for tests) |
| `tests/` | pytest suites for the model and the simulator |
| `demo.py`, `pilot.py`, `plot_graph.py` | Reproducible scripts |
| `results/`, `figures/` | Pilot output and graph figure |

## Hospital graph (synthetic, not a real layout)

```
SUPPLY -1- J1 -1- J2 -1- J3 ===3=== J4 -1- ED
                   \                /
                    2-- ALT ---4---
```

Edge labels are travel steps. `J3 === J4` is the commitment segment σ.

**Reversal rule:** `c_rev` (default 10) is charged once each time the robot *starts* traveling backward (J4 → J3) on σ. That happens in two cases:

1. The robot leaves J4 toward J3.
2. The robot is partway along J3 → J4 and turns around.

Entering σ forward, continuing, and waiting on it cost nothing extra. SUPPLY, ED, and ALT are terminal: arriving at one ends the episode, so routes never pass through them.

## Time, actions, and costs

Each decision takes one time step. The hidden status then advances `z_t → z_{t+1}` using `T_Z`.

| Action | Effect | Cost |
| ------ | ------ | ---- |
| `MOVE(n)` | One unit along an edge. An edge of length L takes L steps, with a transit position `u->v(p)` between nodes. Mid-edge, the robot can continue or turn back. | `move_cost` (1), plus `c_rev` if the step is a reversal |
| `RETURN` / `REROUTE` | `MOVE` along the cheapest route to SUPPLY / ALT. Route cost includes `c_rev`. | Same as `MOVE` |
| `WAIT` | Stay in place. | `wait_cost` (1) |
| `QUERY` | Stop for one step and submit a query at time τ = t+1. The report is the true status `z_τ`, and it arrives at τ + D. At most one query can be pending, and the robot may move while it waits. | `query_cost` (2). This is the initiation cost and replaces the wait cost for that step. The delay itself costs nothing directly; any steps that pass are charged as usual. |

Delays come from `D ~ {0:.4, 1:.3, 2:.2, 3:.1}`. With D = 0, the report is usable at the very next decision.

The robot's belief is updated in two ways:

- Each step: `b ← b T_Z`.
- When a report arrives: `b_t = one_hot(z_τ) T_Z^(t-τ)`. The exponent counts from τ, not from the arrival time.

## Terminal outcomes and rewards (synthetic, configurable in `RewardConfig`)

| Outcome | Condition (status at the arrival time) | Reward |
| ------- | -------------------------------------- | ------ |
| `delivered_correct_ED` | Arrive at ED while ACT | +100 |
| `delivered_correct_ALT` | Arrive at ALT while RED | +100 |
| `wrong_destination` | ED while RED, or ALT while ACT | −50 |
| `delivered_invalid` | ED or ALT while INV (never counted as success) | −20 |
| `returned_after_invalid` | Arrive at SUPPLY while INV | 0 |
| `abandoned_valid` | Arrive at SUPPLY while ACT or RED | −60 |
| `timeout` | Horizon (20 steps) reached first | −40 |

## Request model

```
T_Z = [[0.85, 0.10, 0.05],    # ACT can stay, be redirected, or become invalid
       [0.00, 0.90, 0.10],    # RED can later become invalid
       [0.00, 0.00, 1.00]]    # INV is absorbing
```

These numbers are synthetic, not hospital data. `[1,0,0] T_Z^2 = [0.7225, 0.175, 0.1025]`.

Reports are exact about the status at τ, and delays are independent of `z`. The robot's actions do not affect the request.
