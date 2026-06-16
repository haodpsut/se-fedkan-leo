"""Local CPU smoke test: prove the whole pipeline runs end-to-end on synthetic
AMC data, in a few seconds, for ours + key baselines. Not a result, a wiring check.

Run:  python scripts/smoke.py
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sefedkan.methods import make_config
from sefedkan.experiment import run_experiment

TINY = dict(source="synthetic", per_combo=40, n_nodes=4, slots_per_pass=10,
            n_passes=2, gap_slots=2, slot_samples=48, epochs=1,
            grid_size=4, grid_max=8, hidden=(12,))


def main():
    print("=== SEFedKAN smoke test (synthetic AMC, CPU) ===")
    ok = True
    for method in ["sefedkan", "fedkan_static", "fedavg_mlp"]:
        cfg = make_config(method, **TINY, seed=0)
        t0 = time.time()
        res, log = run_experiment(cfg)
        dt = time.time() - t0
        print(f"\n[{method}]  ({dt:.1f}s)")
        for k in ["avg_acc", "bwt", "forgetting", "total_bits", "final_grid", "params"]:
            print(f"   {k:12s} = {res[k]}")
        # sanity
        if res["avg_acc"] <= 1.0 / res["n_classes"]:
            print(f"   WARN: avg_acc not above chance ({1/res['n_classes']:.3f})")
        if res["total_bits"] <= 0:
            print("   FAIL: no uplink bits accounted"); ok = False
        if method == "sefedkan" and cfg.model_type == "kan":
            if res["final_grid"] < cfg.grid_size:
                print("   FAIL: grid never tracked"); ok = False
    print("\nSMOKE", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
