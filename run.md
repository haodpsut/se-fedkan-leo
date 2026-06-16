# run.md — exact commands for the GPU server (S5 handshake)

Workflow: I (Claude) build + smoke-test locally and push. You pull to the RTX 4090
server, run, and push `results/` back. Then I pull and analyze (S6).

## 0. Environment bootstrap on the RTX 4090 server (conda-only host)
```bash
# clone
git clone https://github.com/haodpsut/se-fedkan-leo.git
cd se-fedkan-leo
mkdir -p data results/logs

# create the conda env (numpy/scipy/sklearn/h5py/pyyaml/matplotlib/tqdm + skyfield/sgp4)
conda env create -f environment.yml
conda activate sefedkan

# install CUDA-enabled PyTorch INSIDE the env via the cu121 index
# (lesson from satellite-KAN: use --index-url, NOT --extra-index-url)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# verify GPU is visible
python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```
If `conda env create` is slow or conflicts, the minimal manual path also works:
```bash
conda create -n sefedkan python=3.11 -y && conda activate sefedkan
pip install numpy scipy scikit-learn pyyaml h5py matplotlib tqdm
pip install torch --index-url https://download.pytorch.org/whl/cu121
# optional, only for real-TLE orbits: pip install skyfield sgp4
```

## 1. Smoke (verify wiring on the server, ~10s, CPU is fine)
```bash
python scripts/smoke.py        # expect: SMOKE PASS
python -m pytest tests/ -q     # expect: 6 passed
```

## 1b. Run everything inside tmux (recommended for the long jobs)
`run_all.sh` runs smoke -> pytest -> synthetic -> rml2016 -> rml2018 (skips a
RadioML run if its file is missing) and tees all output to
`results/run_all_<timestamp>.log`.
```bash
tmux new -s sefedkan                 # start a detachable session
conda activate sefedkan
bash scripts/run_all.sh              # or: SEEDS=3 DEVICE=cuda bash scripts/run_all.sh
# detach: press Ctrl-b then d   (job keeps running)
```
Reattach / monitor later:
```bash
tmux attach -t sefedkan             # reattach
tmux ls                             # list sessions
tail -f results/run_all_*.log       # follow progress without attaching
```
When it prints DONE, push the results:
```bash
git add results/ && git commit -m "results: server run" && git push
```

## 2. Synthetic full run (no download needed, larger than smoke)
```bash
python scripts/run_main.py --source synthetic --seeds 5 --out results/synth.csv
```

## 3. RadioML 2016.10a (dev + ablation, light ~600 MB)
Download `RML2016.10a_dict.pkl` (DeepSig) into `data/`.
```bash
python scripts/run_main.py --source rml2016 \
    --data-path data/RML2016.10a_dict.pkl \
    --device cuda --seeds 5 --n-nodes 12 --out results/rml2016.csv
```

## 4. RadioML 2018.01A (headline, ~20 GB HDF5)
Download `GOLD_XYZ_OSC.0001_1024.hdf5` (DeepSig) into `data/`.
```bash
python scripts/run_main.py --source rml2018 \
    --data-path data/GOLD_XYZ_OSC.0001_1024.hdf5 \
    --device cuda --seeds 5 --n-nodes 16 --per-combo 400 \
    --out results/rml2018.csv
```

## Expected outputs (push these back to git)
- `results/*.csv`         one row per (method, seed): avg_acc, bwt, forgetting,
                          total_bits, final_grid, params, time_s
- `results/logs/*.npz`    per-slot log per run (acc_now, bits, grid, drift_flags,
                          snr_med, n_part) for the time-series figures
- console "summary (mean +/- std over seeds)" block — paste into `results/summary.txt`

## Default setup (since the pivot)
- `--arch conv` (1D-CNN front-end on raw I/Q) and `--drift-mode incremental`
  (new modulation classes appear over the mission) are the DEFAULTS now.
  The old flat-feature / snr_sweep paths remain for the negative-control ablation
  (`--arch flat`, `--drift-mode snr_sweep`).
- Headline metrics are now `final_acc_all` (accuracy on ALL classes at the end)
  and `forgetting` (continual-learning), plus `total_bits` and `detection_delay`.

## What ours should show (the story to verify, NOT to assume)
- `sefedkan` highest `final_acc_all` and lowest `forgetting` vs all baselines
  -> self-evolution + drift-aware pseudo retains old classes best
- `sefedkan` total_bits ~2x lower than MLP / static baselines -> controller compresses
- ablations: full > no_evolve (evolution helps final acc);
  full forgets less than no_pseudo (drift-aware pseudo helps);
  bandit beats static/dual on the acc-vs-bits Pareto
- snr_sweep run is the NEGATIVE control: evolution should NOT help there (cyclic
  drift gives no growing complexity) — report this honestly as the boundary.
- Wilcoxon over seeds for the headline gaps; report mean +/- std.
- NOTE: baselines may win instantaneous `avg_acc` early (few classes); ours wins
  the continual metrics (final_acc_all, forgetting). State this honestly.

## Methods
sefedkan (ours), fedkan_static, fedkan_dual, fedkan_no_pseudo, fedkan_no_evolve,
fedavg_mlp, fedprox_mlp. Select a subset with `--methods a b c`.
