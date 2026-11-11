"""Tests for simulator.py and policies.py. Run: python -m pytest -v"""

import dataclasses

import numpy as np
import pytest

import request_model as rm
from policies import InfoAgeQuery, NeverQuery, PeriodicQuery, Scripted
from request_model import ACT, INV, RED, T_Z
from simulator import (ALT, ED, J1, J2, J3, J4, MOVE, QUERY, RETURN, REROUTE, SUPPLY,
                       WAIT, Action, DeliveryEnv, HospitalGraph, Observation, Position,
                       RewardConfig, Scenario, make_scenario, run_episode)

H = 20


def scn(change_t=None, to=None, delay=0, horizon=H):
    """Deterministic scenario: ACT until change_t, then `to` forever."""
    z = [ACT] * (horizon + 1)
    if change_t is not None:
        z[change_t:] = [to] * (horizon + 1 - change_t)
    delays = delay if isinstance(delay, (list, tuple)) else [delay] * horizon
    return Scenario(tuple(z), tuple(delays))


def M(target):
    return Action(MOVE, target)


def play(env, actions):
    for a in actions:
        env.step(a)
    return env


TO_J3 = [M(J1), M(J2), M(J3)]
TO_ED = TO_J3 + [M(J4)] * 3 + [M(ED)]  # 7 steps


# ---- graph movement -----------------------------------------------------------
def test_graph_is_small_and_has_required_locations():
    g = HospitalGraph()
    assert 6 <= len(g.adj) <= 8
    assert {SUPPLY, ED, ALT} <= set(g.adj)
    assert g.edge_length(J3, J4) == g.edge_length(J4, J3) == 3


def test_unit_edge_takes_one_step():
    env = play(DeliveryEnv(scn()), [M(J1)])
    assert env.t == 1 and env.position == Position(J1)


def test_long_edge_takes_multiple_steps_with_transit_state():
    env = play(DeliveryEnv(scn()), TO_J3)
    seen = []
    for _ in range(3):
        env.step(M(J4))
        seen.append(env.position)
    assert seen == [Position(J3, J4, 1), Position(J3, J4, 2), Position(J4)]
    assert env.t == 6


def test_movement_is_deterministic_across_runs():
    a = play(DeliveryEnv(scn()), TO_ED)
    b = play(DeliveryEnv(scn()), TO_ED)
    assert a.trace == b.trace


def test_illegal_moves_rejected():
    env = DeliveryEnv(scn())
    with pytest.raises(ValueError):
        env.step(M(J2))  # not adjacent to SUPPLY
    env = play(DeliveryEnv(scn()), TO_J3 + [M(J4)])
    with pytest.raises(ValueError):
        env.step(M(ED))  # mid-edge, can only go to J3 or J4


# ---- reversal rule --------------------------------------------------------------
def test_entering_commitment_forward_is_free():
    env = play(DeliveryEnv(scn()), TO_J3 + [M(J4)] * 3)
    assert env.totals["reversal"] == 0 and env.reversals == 0
    assert env.totals["move"] == -6


def test_turning_around_inside_commitment_costs_c_rev_once():
    env = play(DeliveryEnv(scn()), TO_J3 + [M(J4)])          # at J3->J4(1)
    env.step(M(J3))                                          # turn around
    assert env.position == Position(J3)                      # 1 step back
    assert env.reversals == 1 and env.totals["reversal"] == -10.0


def test_turning_around_deeper_then_continuing_back_charged_once():
    env = play(DeliveryEnv(scn()), TO_J3 + [M(J4), M(J4)])   # J3->J4(2)
    env.step(M(J3))
    assert env.position == Position(J4, J3, 2)
    env.step(M(J3))
    assert env.position == Position(J3)
    assert env.reversals == 1 and env.totals["reversal"] == -10.0


def test_leaving_J4_toward_J3_costs_c_rev():
    env = play(DeliveryEnv(scn()), TO_J3 + [M(J4)] * 3 + [M(J3)])
    assert env.reversals == 1 and env.totals["reversal"] == -10.0


def test_turning_around_on_ordinary_edge_is_free():
    env = play(DeliveryEnv(scn()), [M(J1), M(J2), M(ALT), M(J2)])
    assert env.position == Position(J2) and env.reversals == 0


def test_reversal_cost_is_configurable():
    env = DeliveryEnv(scn(), rewards=RewardConfig(reversal_cost=3.5))
    play(env, TO_J3 + [M(J4), M(J3)])
    assert env.totals["reversal"] == -3.5


def test_reroute_inside_commitment_prefers_forward_when_c_rev_large():
    env = play(DeliveryEnv(scn()), TO_J3 + [M(J4)])
    assert env.resolve(Action(REROUTE)) == M(J4)   # forward 2+4=6 < back 1+10+3=14
    cheap = DeliveryEnv(scn(), rewards=RewardConfig(reversal_cost=0.0))
    play(cheap, TO_J3 + [M(J4)])
    assert cheap.resolve(Action(REROUTE)) == M(J3)  # back 1+0+3=4 < 6


# ---- delivery outcomes ----------------------------------------------------------
def test_correct_delivery_to_ED_while_active():
    env = play(DeliveryEnv(scn()), TO_ED)
    assert env.outcome == "delivered_correct_ED"
    assert env.total_reward == pytest.approx(100 - 7)


def test_correct_delivery_to_ALT_while_redirected():
    env = play(DeliveryEnv(scn(1, RED)), [M(J1), M(J2), M(ALT), M(ALT)])
    assert env.outcome == "delivered_correct_ALT"
    assert env.total_reward == pytest.approx(100 - 4)


def test_ED_while_redirected_is_wrong_destination():
    env = play(DeliveryEnv(scn(2, RED)), TO_ED)
    assert env.outcome == "wrong_destination"
    assert env.total_reward == pytest.approx(-50 - 7)


def test_ALT_while_active_is_wrong_destination():
    env = play(DeliveryEnv(scn()), [M(J1), M(J2), M(ALT), M(ALT)])
    assert env.outcome == "wrong_destination"


def test_delivery_of_invalid_request_is_not_success():
    for dest_path in (TO_ED, [M(J1), M(J2), M(ALT), M(ALT)]):
        r = run_episode(Scripted(dest_path), scn(2, INV))
        assert r.outcome == "delivered_invalid"
        assert not r.success and r.incorrect_delivery


def test_outcome_uses_status_at_arrival_time():
    # Request becomes RED exactly at the arrival time t=7.
    env = play(DeliveryEnv(scn(7, RED)), TO_ED)
    assert env.t == 7 and env.outcome == "wrong_destination"
    env = play(DeliveryEnv(scn(8, RED)), TO_ED)
    assert env.outcome == "delivered_correct_ED"


def test_return_outcomes():
    back = [M(J1), M(J2), Action(RETURN), Action(RETURN)]
    assert run_episode(Scripted(back), scn(1, INV)).outcome == "returned_after_invalid"
    r = run_episode(Scripted(back), scn())
    assert r.outcome == "abandoned_valid" and r.total_reward == pytest.approx(-60 - 4)


# ---- redirection / invalidation via queries ---------------------------------------
def test_redirected_request_leads_periodic_to_ALT():
    r = run_episode(PeriodicQuery(3), scn(2, RED))
    assert r.outcome == "delivered_correct_ALT"


def test_invalidated_request_leads_periodic_back_to_supply():
    r = run_episode(PeriodicQuery(3), scn(2, INV))
    assert r.outcome == "returned_after_invalid"
    assert r.unnecessary_travel_after_invalid == 1  # the J2->J3 move at t=2


def test_never_query_ignores_changes():
    assert run_episode(NeverQuery(), scn(2, RED)).outcome == "wrong_destination"
    assert run_episode(NeverQuery(), scn(2, INV)).outcome == "delivered_invalid"


# ---- queries and delays -------------------------------------------------------------
def test_query_with_zero_delay_usable_next_step():
    env = DeliveryEnv(scn(1, RED, delay=0))
    env.step(Action(QUERY))                       # t=0 -> 1
    obs = env.observe()
    assert obs.last_report == rm.StatusReport(RED, 1, 1)
    assert not obs.query_pending
    assert np.allclose(obs.belief, [0, 1, 0])
    assert env.totals["query"] == -2.0 and env.totals["wait"] == 0.0  # no double count


@pytest.mark.parametrize("d", [1, 2, 3])
def test_query_with_multi_step_delay(d):
    env = DeliveryEnv(scn(1, RED, delay=d))
    env.step(Action(QUERY))                       # sampled at tau=1
    for _ in range(d - 1):
        env.step(Action(WAIT))
        assert env.observe().query_pending
        assert env.observe().last_report.sample_time == 0
    env.step(Action(WAIT))                        # now t = 1 + d
    obs = env.observe()
    assert obs.t == 1 + d and not obs.query_pending
    assert obs.last_report == rm.StatusReport(RED, 1, 1 + d)


def test_at_most_one_pending_query():
    env = DeliveryEnv(scn(delay=3))
    env.step(Action(QUERY))
    with pytest.raises(RuntimeError):
        env.step(Action(QUERY))
    for _ in range(3):
        env.step(Action(WAIT))
    env.step(Action(QUERY))                       # allowed after arrival
    assert env.queries_sent == 2


def test_policies_never_double_query():
    for seed in range(30):
        s = make_scenario(seed, delay_probs={3: 1.0})
        for p in (PeriodicQuery(1), InfoAgeQuery(0)):
            run_episode(p, s)  # would raise RuntimeError on a second pending query


def test_robot_keeps_moving_while_query_pending():
    env = DeliveryEnv(scn(delay=3))
    env.step(Action(QUERY))
    positions = []
    for _ in range(3):
        assert env.observe().query_pending
        env.step(M(J1) if env.position == Position(SUPPLY) else M(J2) if env.position == Position(J1) else M(J3))
        positions.append(env.position)
    assert positions == [Position(J1), Position(J2), Position(J3)]
    assert not env.observe().query_pending


def test_query_delays_come_from_scenario_in_order():
    env = DeliveryEnv(scn(delay=[2, 0] + [0] * (H - 2)))
    env.step(Action(QUERY))
    assert env.channel.pending.receive_time == 3
    play(env, [Action(WAIT)] * 2)
    env.step(Action(QUERY))
    assert env.last_report == rm.StatusReport(ACT, 4, 4)


# ---- belief updates -------------------------------------------------------------
def test_belief_predicts_between_reports():
    env = play(DeliveryEnv(scn()), [Action(WAIT)] * 2)
    assert np.allclose(env.belief, [0.7225, 0.175, 0.1025])


def test_belief_after_delayed_report_counts_from_sample_time():
    # Report samples ACT at tau=1, arrives t=4; request turned RED at t=2.
    env = DeliveryEnv(scn(2, RED, delay=3))
    env.step(Action(QUERY))
    play(env, [Action(WAIT)] * 3)
    expected = rm.predict_belief([1, 0, 0], T_Z, 4 - 1)
    assert env.last_report == rm.StatusReport(ACT, 1, 4)
    assert np.allclose(env.belief, expected)
    assert not np.allclose(env.belief, [1, 0, 0])
    assert env.true_status == RED  # stale but correctly time-stamped report
    env.step(Action(WAIT))         # keeps predicting afterwards
    assert np.allclose(env.belief, rm.predict_belief([1, 0, 0], T_Z, 4))


def test_observation_has_no_true_status():
    names = {f.name for f in dataclasses.fields(Observation)}
    assert not names & {"z", "status", "true_status", "z_t"}
    obs = DeliveryEnv(scn(1, INV)).observe()
    assert obs.last_report.status == ACT  # only the original assignment


# ---- horizon / bookkeeping -----------------------------------------------------------
def test_episode_ends_at_horizon():
    env = DeliveryEnv(scn(horizon=5))
    play(env, [Action(WAIT)] * 5)
    assert env.done and env.outcome == "timeout" and env.t == 5
    assert env.total_reward == pytest.approx(-5 - 40)
    with pytest.raises(RuntimeError):
        env.step(Action(WAIT))


def test_reward_total_equals_sum_of_trace_rewards():
    for seed in range(10):
        for p in (NeverQuery(), PeriodicQuery(3), InfoAgeQuery(3)):
            r = run_episode(p, make_scenario(seed))
            assert sum(s["reward"] for s in r.trace) == pytest.approx(r.total_reward)
            assert r.travel_steps + r.query_count <= r.steps


def test_scenarios_reproducible_and_shared():
    assert make_scenario(5) == make_scenario(5)
    assert make_scenario(5) != make_scenario(6)
    a = run_episode(PeriodicQuery(3), make_scenario(5))
    b = run_episode(PeriodicQuery(3), make_scenario(5))
    assert a.trace == b.trace


def test_policy_query_timing():
    r = run_episode(PeriodicQuery(3), scn(delay=0))
    assert [s["t"] for s in r.trace if s["action"] == QUERY] == [3, 6]
    r = run_episode(InfoAgeQuery(3), scn(delay=0))
    # info age at t is t - 0 until the first report; query when age > 3.
    assert [s["t"] for s in r.trace if s["action"] == QUERY][0] == 4
    assert run_episode(NeverQuery(), scn()).query_count == 0


def test_wait_or_query_at_supply_before_departure_is_not_a_return():
    env = play(DeliveryEnv(scn(delay=0)), [Action(WAIT), Action(QUERY)])
    assert not env.done and env.position == Position(SUPPLY)
