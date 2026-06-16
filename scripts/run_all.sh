#!/usr/bin/env bash
# Run the whole experiment battery on the GPU server, tee-ing all output to a
# timestamped log. Safe to launch inside tmux and detach.
#
#   conda activate sefedkan
#   bash scripts/run_all.sh
#
# Env knobs (override on the command line, e.g. SEEDS=3 bash scripts/run_all.sh):
SEEDS="${SEEDS:-5}"
DEVICE="${DEVICE:-cuda}"
RML2016="${RML2016:-data/RML2016.10a_dict.pkl}"
RML2018="${RML2018:-data/GOLD_XYZ_OSC.0001_1024.hdf5}"

set -u
cd "$(dirname "$0")/.." || exit 1
mkdir -p results/logs
TS="$(date +%Y%m%d_%H%M%S)"
LOG="results/run_all_${TS}.log"

# tee everything (stdout+stderr) to the log AND the terminal
exec > >(tee -a "$LOG") 2>&1

echo "================ SE-FedKAN run_all @ ${TS} ================"
echo "SEEDS=${SEEDS} DEVICE=${DEVICE}"
python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())"
echo

echo "---- [1/4] smoke + unit tests ----"
python scripts/smoke.py || { echo "SMOKE FAILED — aborting"; exit 1; }
if python -c "import pytest" 2>/dev/null; then
  python -m pytest tests/ -q || { echo "PYTEST FAILED — aborting"; exit 1; }
else
  echo "pytest not installed -> skipping unit tests (run 'pip install pytest' to enable)"
fi

echo "---- [2/4] synthetic full (no download needed) ----"
python scripts/run_main.py --source synthetic --seeds "${SEEDS}" \
    --device "${DEVICE}" --out results/synth.csv

if [ -f "${RML2016}" ]; then
  echo "---- [3/4] RadioML 2016.10a ----"
  python scripts/run_main.py --source rml2016 --data-path "${RML2016}" \
      --device "${DEVICE}" --seeds "${SEEDS}" --n-nodes 12 --out results/rml2016.csv
else
  echo "---- [3/4] SKIP rml2016 (missing ${RML2016}) ----"
fi

if [ -f "${RML2018}" ]; then
  echo "---- [4/4] RadioML 2018.01A (headline) ----"
  python scripts/run_main.py --source rml2018 --data-path "${RML2018}" \
      --device "${DEVICE}" --seeds "${SEEDS}" --n-nodes 16 --per-combo 400 \
      --out results/rml2018.csv
else
  echo "---- [4/4] SKIP rml2018 (missing ${RML2018}) ----"
fi

echo
echo "================ DONE @ $(date +%H:%M:%S) ================"
echo "Results: results/*.csv ; per-slot logs: results/logs/*.npz ; this log: ${LOG}"
echo "Now: git add results/ && git commit -m 'results: server run ${TS}' && git push"
