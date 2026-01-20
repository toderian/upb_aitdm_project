# FL + DP Simulation Results

**Date:** 2026-01-20 15:12:29

## Configuration

- **Number of Clients:** 3
- **Federated Rounds:** 5
- **Target Epsilon (ε):** 10.0
- **Delta (δ):** 1e-05
- **Samples per Client:** 1000

## Results Summary

- **Final Global Accuracy:** 0.5000
- **Final Max Client ε:** 9.99

## Round-by-Round Progress

| Round | Global Accuracy | Max ε |
|-------|-----------------|-------|
| 1 | 0.5000 | 9.99 |
| 2 | 0.5000 | 9.99 |
| 3 | 0.5000 | 9.99 |
| 4 | 0.5000 | 9.99 |
| 5 | 0.5000 | 9.99 |

## Privacy Analysis

Each client achieved (ε=10.0, δ=1e-05)-differential privacy per round.
With composition across rounds, the total privacy budget increases.