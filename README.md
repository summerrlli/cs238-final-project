# CS238 Final Project

## To Check or Not to Check? Commitment-Aware Task Verification for Hospital Delivery Robots

This project studies when a hospital delivery robot should verify that its task is still valid. A request may remain active, be redirected to another unit, or become invalid while the robot travels. Verification has a cost, and replies arrive after a delay, reporting the request's status when the query was sent.

The central question is whether verification decisions improve when they account for an approaching commitment segment, where changing course becomes expensive.

## Current Status

The project proposal is complete. The simulator has not been implemented, and no experiments have been run.

## Planned Approach

We plan to model delivery on a small hospital graph with hidden request status and delayed verification reports. We will compare:

- Periodic verification.
- Verification based on time since the last check.
- A commitment-aware querying heuristic.
- An approximate belief-state planner.

Evaluation will measure unnecessary travel, query count, valid delivery completion, total reward, and computation time.

## Next Steps

1. Finalize the graph, request dynamics, costs, and delayed-report semantics.
2. Implement the simulator and validate hand-checkable cases.
3. Implement the comparison methods.
4. Evaluate performance across query delays and reversal costs using consistent scenarios.
