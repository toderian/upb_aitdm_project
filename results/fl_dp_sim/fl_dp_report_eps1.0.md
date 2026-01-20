# FL + DP Simulation Results

**Date:** 2026-01-20 15:09:06

## Configuration

- **Number of Clients:** 3
- **Federated Rounds:** 5
- **Target Epsilon (ε):** 1.0
- **Delta (δ):** 1e-05
- **Samples per Client:** 1000

## Results Summary

- **Final Global Accuracy:** 0.5060
- **Final Max Client ε:** 0.99

## Round-by-Round Progress

| Round | Global Accuracy | Max ε |
|-------|-----------------|-------|
| 1 | 0.5000 | 0.99 |
| 2 | 0.5000 | 0.99 |
| 3 | 0.5000 | 0.99 |
| 4 | 0.5000 | 0.99 |
| 5 | 0.5060 | 0.99 |

## Privacy Analysis

Each client achieved (ε=1.0, δ=1e-05)-differential privacy per round.
With composition across rounds, the total privacy budget increases.