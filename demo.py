"""Demo of the request-status model. Run: python demo.py"""

import numpy as np

import request_model as rm
from request_model import ACT, STATE_NAMES, T_Z, format_belief

rm.check_transition_matrix(T_Z)
print("T_z (rows = from ACT/RED/INV, cols = to ACT/RED/INV):")
print(T_Z)

print("\n1) Prediction from b0 = [1, 0, 0]:  b_k = b0 T^k")
for k in range(5):
    print(f"   k={k}: {format_belief(rm.predict_belief([1, 0, 0], T_Z, k))}")

print("\n2) Report says ACT at tau=0, arrives at t=2")
b = rm.belief_from_report(rm.StatusReport(ACT, sample_time=0, receive_time=2), 2, T_Z)
print(f"   correct belief : {format_belief(b)}")
print(f"   naive (wrong)  : {format_belief([1, 0, 0])}  <- treats old report as current")

print("\n3) Sampled episode (seed 7), query sent at tau=1")
rng = np.random.default_rng(7)
states = rm.simulate_states(ACT, 8, T_Z, rng)
channel = rm.QueryChannel(rng)
arrival = channel.send(states[1], tau=1)
print(f"   true z_0..z_8: {[STATE_NAMES[z] for z in states]}")
print(f"   query sees z_1={STATE_NAMES[states[1]]}, reply arrives at t={arrival}")
report = channel.receive(arrival)
b = rm.belief_from_report(report, arrival, T_Z)
print(f"   t={arrival}: belief {format_belief(b)}  (true z_{arrival}={STATE_NAMES[states[arrival]]})")

print(f"\n4) P(D <= k) for k=0..3: {[round(rm.prob_reply_within(k), 2) for k in range(4)]}")
