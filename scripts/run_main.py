"""Full experiment runner for the GPU server.

Loops methods x seeds, writes a results CSV (one row per method/seed) and a
per-slot log NPZ per run. Use the full config for the headline; the synthetic
source lets the server reproduce wiring before the real RadioML download lands.

Examples:
  # quick full synthetic (server sanity, larger than smoke):
  python scripts/run_main.py --source synthetic --seeds 5 --out results/synth.csv

  # RadioML 2016.10a (dev / ablation):
  python scripts/run_main.py --source rml2016 \
      --data-path data/RML2016.10a_dict.pkl --seeds 5 --out results/rml2016.csv

  # RadioML 2018.01A (headline):
  python scripts/run_main.py --source rml2018 \
      --data-path data/GOLD_XYZ_OSC.0001_1024.hdf5 --seeds 5 \
      --device cuda --out results/rml2018.csv
"""
import sys, os, csv, json, argparse, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from sefedkan.methods import make_config, METHODS
from sefedkan.experiment import run_experiment


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="synthetic", choices=["synthetic", "rml2016", "rml2018"])
    p.add_argument("--data-path", default="")
    p.add_argument("--methods", nargs="*", default=METHODS)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--feature", default="multimodal", choices=["iq", "stat", "multimodal"])
    p.add_argument("--device", default="cpu")
    p.add_argument("--n-nodes", type=int, default=12)
    p.add_argument("--slots-per-pass", type=int, default=30)
    p.add_argument("--n-passes", type=int, default=4)
    p.add_argument("--grid-size", type=int, default=5)
    p.add_argument("--grid-max", type=int, default=13)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--per-combo", type=int, default=200)
    p.add_argument("--out", default="results/results.csv")
    p.add_argument("--logdir", default="results/logs")
    return p.parse_args()


def main():
    a = parse()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    os.makedirs(a.logdir, exist_ok=True)
    rows = []
    common = dict(source=a.source, data_path=a.data_path, feature=a.feature,
                  device=a.device, n_nodes=a.n_nodes, slots_per_pass=a.slots_per_pass,
                  n_passes=a.n_passes, grid_size=a.grid_size, grid_max=a.grid_max,
                  epochs=a.epochs, per_combo=a.per_combo)
    def flush_csv():
        if not rows:
            return
        keys = list(rows[0].keys())
        with open(a.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)

    for method in a.methods:
        for seed in range(a.seeds):
            cfg = make_config(method, seed=seed, **common)
            t0 = time.time()
            try:
                res, log = run_experiment(cfg)
            except Exception as e:
                import traceback
                print(f"!! {method} seed={seed} FAILED: {e}")
                traceback.print_exc()
                continue
            dt = time.time() - t0
            row = {"method": method, "seed": seed, "time_s": round(dt, 1), **{
                k: res[k] for k in ["avg_acc", "bwt", "forgetting", "total_bits",
                                    "final_grid", "params", "n_classes", "in_dim", "T"]}}
            rows.append(row)
            np.savez(os.path.join(a.logdir, f"{method}_s{seed}.npz"),
                     **{k: np.array(v, dtype=object) for k, v in log.items()})
            flush_csv()   # write incrementally so a later crash never loses earlier runs
            print(f"{method:16s} seed={seed} acc={res['avg_acc']:.4f} "
                  f"bits={res['total_bits']:.3e} grid={res['final_grid']} ({dt:.1f}s)")

    if not rows:
        print("no successful runs"); return

    # aggregate mean+-std per method
    agg = {}
    for r in rows:
        agg.setdefault(r["method"], []).append(r)
    print("\n=== summary (mean +/- std over seeds) ===")
    for m, rs in agg.items():
        acc = np.array([x["avg_acc"] for x in rs])
        bits = np.array([x["total_bits"] for x in rs])
        print(f"{m:16s} acc={acc.mean():.4f}+/-{acc.std():.4f} "
              f"bits={bits.mean():.3e} grid={rs[-1]['final_grid']}")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
