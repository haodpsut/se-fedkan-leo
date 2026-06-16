"""Self-Evolving Federated KAN for non-stationary LEO-UAV edge signal intelligence.

Package layout:
  kan.py         - efficient-KAN B-spline layer + grid-extension (self-evolution),
                   param-matched MLP baseline, classifier wrappers.
  data.py        - RadioML loaders (2016.10a / 2018.01A) + synthetic AMC generator
                   + multimodal featurization (raw I/Q view + spectral-moment view).
  orbit.py       - lightweight analytic LEO pass: elevation(t), SNR(elevation),
                   visibility gate. Optional SGP4/TLE path on the server.
  drift.py       - online drift detectors (Page-Hinkley, ADWIN-lite) on loss stream.
  controller.py  - evolve/compress/participate controllers: Fixed, DualThreshold,
                   contextual Bandit (LinUCB). The RL piece of the SL+RL union.
  fed.py         - client local update, top-k sparsification + comm-bit accounting,
                   FedAvg / FedProx aggregation.
  metrics.py     - accuracy under drift, forgetting / backward transfer, comm, energy.
  experiment.py  - the slotted federated loop tying everything together.
"""
__version__ = "0.1.0"
