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

KAGGLE_SLUG="nolasthitnotomorrow/radioml2016-deepsigcom"

echo "==> [1] RML2016.10a (~600 MB)"
if [ -f RML2016.10a_dict.pkl ]; then
  echo "    already present, skipping"
else
  ok=0
  # try DeepSig opendata up to 3x (it sometimes returns 502 / expired cert)
  for try in 1 2 3; do
    echo "    [deepsig attempt $try]"
    wget -c --no-check-certificate -O RML2016.10a.tar.bz2 "$URL2016" || true
    if [ -s RML2016.10a.tar.bz2 ] && tar xjf RML2016.10a.tar.bz2 2>/dev/null; then
      ok=1; break
    fi
    echo "    deepsig failed, retrying..."; sleep 3
  done
  # fallback: Kaggle mirror (needs kaggle CLI + ~/.config/kaggle/kaggle.json)
  if [ "$ok" -ne 1 ]; then
    echo "    deepsig unavailable -> trying Kaggle mirror $KAGGLE_SLUG"
    if command -v kaggle >/dev/null 2>&1; then
      kaggle datasets download -d "$KAGGLE_SLUG" -p . --unzip && ok=1
    else
      echo "    !! kaggle CLI not found. Install + add token, then re-run:"
      echo "       pip install kaggle"
      echo "       mkdir -p ~/.config/kaggle && mv ~/Downloads/kaggle.json ~/.config/kaggle/ && chmod 600 ~/.config/kaggle/kaggle.json"
      echo "       kaggle datasets download -d $KAGGLE_SLUG -p data --unzip"
    fi
  fi
  # normalize the extracted name (some mirrors name it RML2016.10a.pkl)
  [ -f RML2016.10a.pkl ] && mv -f RML2016.10a.pkl RML2016.10a_dict.pkl || true
  ls -la RML2016.10a_dict.pkl 2>/dev/null || echo "    NOTE: check the extracted .pkl name with 'ls' and pass it via --data-path"
fi

if [ "${1:-}" = "--2018" ]; then
  echo "==> [2] RML2018.01a (~20 GB) — this is large and slow"
  if [ ! -f GOLD_XYZ_OSC.0001_1024.hdf5 ]; then
    wget -c --no-check-certificate -O RML2018.01a.tar.gz "$URL2018"
    tar xzf RML2018.01a.tar.gz
    ls -la *.hdf5 *.h5 2>/dev/null || true
    echo "NOTE: if the extracted .hdf5/.h5 name differs from GOLD_XYZ_OSC.0001_1024.hdf5,"
    echo "      pass the real name to run_main with --data-path data/<name>"
  else
    echo "    already present, skipping"
  fi
fi
echo "DONE. Files in $(pwd):"; ls -la
