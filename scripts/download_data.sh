#!/usr/bin/env bash
# Download RadioML datasets into data/ from DeepSig opendata (direct links).
# Default: only the SMALL set (RML2016.10a, ~600 MB). Pass --2018 to also fetch
# the large headline set (~20 GB compressed).
#
#   bash scripts/download_data.sh           # 2016.10a only (recommended first)
#   bash scripts/download_data.sh --2018    # also the big one
set -e
cd "$(dirname "$0")/.." || exit 1
mkdir -p data && cd data

URL2016="https://opendata.deepsig.io/datasets/2016.10/RML2016.10a.tar.bz2"
URL2018="https://opendata.deepsig.io/datasets/2018.01/2018.01.OSC.0001_1024x2M.h5.tar.gz"

echo "==> [1] RML2016.10a (~600 MB)"
if [ ! -f RML2016.10a_dict.pkl ]; then
  wget -c -O RML2016.10a.tar.bz2 "$URL2016"
  tar xjf RML2016.10a.tar.bz2
  # the tarball may extract a file literally named RML2016.10a_dict.pkl,
  # or RML2016.10a.pkl — normalize to the expected name:
  [ -f RML2016.10a.pkl ] && mv -f RML2016.10a.pkl RML2016.10a_dict.pkl || true
  ls -la RML2016.10a_dict.pkl
else
  echo "    already present, skipping"
fi

if [ "${1:-}" = "--2018" ]; then
  echo "==> [2] RML2018.01a (~20 GB) — this is large and slow"
  if [ ! -f GOLD_XYZ_OSC.0001_1024.hdf5 ]; then
    wget -c -O RML2018.01a.tar.gz "$URL2018"
    tar xzf RML2018.01a.tar.gz
    ls -la *.hdf5 *.h5 2>/dev/null || true
    echo "NOTE: if the extracted .hdf5/.h5 name differs from GOLD_XYZ_OSC.0001_1024.hdf5,"
    echo "      pass the real name to run_main with --data-path data/<name>"
  else
    echo "    already present, skipping"
  fi
fi
echo "DONE. Files in $(pwd):"; ls -la
