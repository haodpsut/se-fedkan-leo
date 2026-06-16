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

## What ours should show (the story to verify, NOT to assume)
- `sefedkan` >= `fedkan_static` on avg_acc  -> the *evolution* helps, not just KAN
- `sefedkan` total_bits <= MLP baselines    -> controller compresses
- `sefedkan` final_grid grows over passes   -> drift triggered self-evolution
- ablations rank: full > {no_evolve, no_pseudo, dual, static}  (some may not — report honestly)
- Wilcoxon over seeds for the headline acc gap; report mean +/- std.

## Methods
sefedkan (ours), fedkan_static, fedkan_dual, fedkan_no_pseudo, fedkan_no_evolve,
fedavg_mlp, fedprox_mlp. Select a subset with `--methods a b c`.
