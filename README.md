# Self-Evolving Federated KAN for Non-Stationary LEO-UAV Edge Signal Intelligence

Code for the paper targeting **IEEE JSTSP** Special Issue *"Autonomous and
Evolutive Optimization in Networked AI"*.

A federated Kolmogorov-Arnold Network (KAN) performs automatic modulation
classification (AMC) across UAV/IoT edge nodes served by a passing LEO
constellation. As satellite elevation sweeps, the receive SNR drifts, so each
node's data distribution is **non-stationary**. An online drift detector triggers
**self-evolution** (KAN grid-extension that adds spline knots without overwriting
old ones) plus pseudo-label refresh, while a contextual-bandit controller trades
model growth and update sparsity against a per-slot uplink budget. This unifies a
supervised learner (the KAN) with an RL controller, per the call.

## Layout
```
src/sefedkan/   kan, data, orbit, drift, controller, fed, metrics, experiment, methods
scripts/        smoke.py (CPU wiring check), run_main.py (server)
tests/          correctness tests (grid extension, sparse FedAvg, drift)
run.md          exact server commands + expected outputs
```

## Quick start
```bash
pip install -r requirements.txt
python scripts/smoke.py        # SMOKE PASS
python -m pytest tests/ -q     # 6 passed
```
See `run.md` for the RadioML 2016.10a / 2018.01A runs on the GPU server.

## Key design choices
- KAN is a compact efficient-KAN B-spline layer in pure PyTorch (no pykan
  dependency); `extend_grid` is the self-evolution operator.
- Heterogeneous grids are aggregated in a unified grid space per round
  (function-preserving extension makes this lossless).
- All methods (ours + baselines + ablations) share one experiment loop for a
  fair comparison; baselines include a param-matched MLP and a static-grid KAN
  that isolates the evolution mechanism.

Authors: Do Phuc Hao (DAU), Truong Duy Dinh (PTIT).
