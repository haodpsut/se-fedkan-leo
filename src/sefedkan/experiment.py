"""The slotted federated loop over a non-stationary LEO edge.

One experiment = one method config run for a fixed seed. Method variants
(baselines vs ours) are pure config combos so they share this exact loop
(fair comparison):

  sefedkan        : kan + bandit controller + evolve-on-drift + pseudo-labels
  fedavg_kan_static: kan + fixed controller + NO evolve   (isolates evolution)
  fedavg_mlp      : mlp + fixed controller
  fedprox_mlp     : mlp + fixed controller + prox (mu>0)
  fedkan_dual     : kan + dual-threshold controller + evolve-on-drift

See run_main.py / scripts for how configs map to these names.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import torch
from torch.nn.utils import parameters_to_vector

from . import data as datamod
from . import orbit
from .kan import KANClassifier, MLPClassifier, mlp_matching_kan
from .drift import make_detector
from .controller import make_controller
from . import fed
from . import metrics


@dataclass
class Config:
    # data
    source: str = "synthetic"        # synthetic | rml2016 | rml2018
    data_path: str = ""
    feature: str = "multimodal"      # iq | stat | multimodal
    per_combo: int = 120             # synthetic frames per (mod,snr)
    # topology / orbit
    n_nodes: int = 6
    slots_per_pass: int = 24
    n_passes: int = 3
    gap_slots: int = 4
    theta_min: float = 10.0
    snr_zenith: float = 18.0
    dirichlet_alpha: float = 0.3     # non-IID class skew per node
    # model
    model_type: str = "kan"          # kan | mlp
    hidden: tuple = (16,)
    grid_size: int = 5
    grid_max: int = 11
    spline_order: int = 3
    # learning / FL
    epochs: int = 2
    lr: float = 0.01
    mu: float = 0.0                  # FedProx
    l1: float = 0.0
    weight_decay: float = 1e-4
    slot_samples: int = 96
    max_eval: int = 256              # cap held-out eval per SNR regime (speed)
    label_frac: float = 0.3
    pseudo_conf: float = 0.9
    pseudo_enabled: bool = True
    # control
    controller: str = "bandit"
    detector: str = "page_hinkley"
    evolve_enabled: bool = True
    budget_bits: float = 2.0e5       # per-slot uplink budget B(t)
    comm_lambda: float = 0.3         # reward penalty weight
    bits_per_param: int = 32
    # misc
    seed: int = 0
    device: str = "cpu"


def _build_nodes(cfg, X, y, snr, n_classes, rng):
    """Assign training frames to nodes (non-IID via Dirichlet), grouped by SNR."""
    snr_bin = np.round(snr).astype(int)
    # global stratified train/eval split
    idx = np.arange(len(y))
    rng.shuffle(idx)
    cut = int(0.7 * len(idx))
    train_idx, eval_idx = idx[:cut], idx[cut:]

    # per-node Dirichlet class proportions
    props = rng.dirichlet([cfg.dirichlet_alpha] * n_classes, cfg.n_nodes)
    by_class = {c: train_idx[y[train_idx] == c] for c in range(n_classes)}
    for c in by_class:
        rng.shuffle(by_class[c])
    nodes = [{"pool": {}, "max_elev": 0.0} for _ in range(cfg.n_nodes)]
    ptr = {c: 0 for c in range(n_classes)}
    for c in range(n_classes):
        pool_c = by_class[c]
        counts = (props[:, c] / props[:, c].sum() * len(pool_c)).astype(int)
        for nidx in range(cfg.n_nodes):
            take = pool_c[ptr[c]:ptr[c] + counts[nidx]]
            ptr[c] += counts[nidx]
            for j in take:
                nodes[nidx]["pool"].setdefault(int(snr_bin[j]), []).append(j)
    # heterogeneous pass geometry per node
    for nidx in range(cfg.n_nodes):
        nodes[nidx]["max_elev"] = float(rng.uniform(25, 85))
        for s in nodes[nidx]["pool"]:
            nodes[nidx]["pool"][s] = np.array(nodes[nidx]["pool"][s])

    # regime eval sets by SNR bin (global, held-out), capped for speed
    eval_by_snr = {}
    for s in np.unique(snr_bin[eval_idx]):
        ids = eval_idx[snr_bin[eval_idx] == s]
        if len(ids) > cfg.max_eval:
            ids = rng.choice(ids, cfg.max_eval, replace=False)
        eval_by_snr[int(s)] = ids
    return nodes, eval_by_snr, train_idx, eval_idx


def _nearest_snr_pool(pool, target):
    if not pool:
        return None
    keys = np.array(list(pool.keys()))
    return int(keys[np.argmin(np.abs(keys - target))])


def run_experiment(cfg: Config, verbose=False):
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    # --- data ---
    if cfg.source == "synthetic":
        bank = datamod.synthetic_amc(per_combo=cfg.per_combo, seed=cfg.seed)
    elif cfg.source == "rml2016":
        bank = datamod.load_radioml2016(cfg.data_path)
    else:
        bank = datamod.load_radioml2018(cfg.data_path, max_per_combo=cfg.per_combo * 4)
    X, dims = datamod.featurize(bank, cfg.feature)
    y, snr = bank.labels, bank.snr
    in_dim, n_classes = X.shape[1], bank.n_classes

    nodes, eval_by_snr, _, _ = _build_nodes(cfg, X, y, snr, n_classes, rng)

    # --- global model ---
    def new_model():
        if cfg.model_type == "kan":
            return KANClassifier(in_dim, cfg.hidden, n_classes,
                                 grid_size=cfg.grid_size, spline_order=cfg.spline_order)
        return MLPClassifier(in_dim, cfg.hidden, n_classes)
    global_model = new_model()
    if cfg.model_type == "mlp":  # report matched size info
        ref_kan = KANClassifier(in_dim, cfg.hidden, n_classes, grid_size=cfg.grid_size)
        matched, gap = mlp_matching_kan(in_dim, n_classes, ref_kan)
        global_model = matched

    detectors = [make_detector(cfg.detector) for _ in range(cfg.n_nodes)]
    controllers = [make_controller(cfg.controller) for _ in range(cfg.n_nodes)]

    # per-node SNR streams over the whole horizon
    streams = []
    for nd in nodes:
        s, v = orbit.node_snr_stream(cfg.slots_per_pass, nd["max_elev"], cfg.theta_min,
                                     cfg.snr_zenith, cfg.n_passes, cfg.gap_slots,
                                     rng=np.random.default_rng(cfg.seed + 1))
        streams.append((s, v))
    T = len(streams[0][0])

    # --- logs ---
    log = {"slot": [], "acc_now": [], "bits": [], "grid": [], "n_part": [],
           "drift_flags": [], "snr_med": []}
    regime_acc_log = {}
    budget_price = 0.0

    def eval_regime(s_bin):
        idx = eval_by_snr.get(s_bin)
        if idx is None or len(idx) == 0:
            return None
        return fed.evaluate(global_model, X[idx], y[idx], cfg.device)

    for t in range(T):
        participants = []
        slot_drift = 0
        snrs_now = []
        for nidx in range(cfg.n_nodes):
            s_db, vis = streams[nidx]
            if not vis[t]:
                continue
            cur_snr = float(s_db[t])
            snrs_now.append(cur_snr)
            pool = nodes[nidx]["pool"]
            sb = _nearest_snr_pool(pool, cur_snr)
            if sb is None or len(pool[sb]) < 8:
                continue
            ids = rng.choice(pool[sb], min(cfg.slot_samples, len(pool[sb])), replace=False)
            Xi, yi = X[ids], y[ids]

            # local model = clone of global
            local = fed.clone_model(global_model)

            # drift signal = local error before update
            pre_err = 1.0 - fed.evaluate(local, Xi, yi, cfg.device)
            flag = detectors[nidx].update(pre_err)
            slot_drift = max(slot_drift, int(flag))

            ctx = {
                "drift": float(flag),
                "snr_norm": np.clip((cur_snr + 20) / 40.0, 0, 1),
                "grid_norm": (getattr(local, "grid_size", cfg.grid_size) - cfg.grid_size)
                             / max(1, cfg.grid_max - cfg.grid_size),
                "budget_price": budget_price,
                "acc": 1.0 - pre_err,
            }
            evolve_step, keep = controllers[nidx].act(ctx)

            if cfg.evolve_enabled and cfg.model_type == "kan" and evolve_step > 0 and flag:
                tgt = min(cfg.grid_max, local.grid_size + evolve_step)
                fed.align_to_grid(local, tgt)

            # pseudo-labels for the unlabeled portion of the slot
            n_lab = max(1, int(cfg.label_frac * len(Xi)))
            Xtr, ytr = list(Xi[:n_lab]), list(yi[:n_lab])
            if cfg.pseudo_enabled and len(Xi) > n_lab:
                Xu = Xi[n_lab:]
                plab, keepm = fed.pseudo_label(local, Xu, cfg.pseudo_conf, cfg.device)
                if keepm.any():
                    Xtr += list(Xu[keepm]); ytr += list(plab[keepm])
            Xtr, ytr = np.asarray(Xtr, np.float32), np.asarray(ytr)

            gvec = None
            if cfg.mu > 0:
                # FedProx anchor = current global, aligned to the local grid so
                # the proximal term matches the local parameter vector dimension.
                g_clone = fed.clone_model(global_model)
                if hasattr(local, "evolve"):
                    fed.align_to_grid(g_clone, local.grid_size)
                gvec = parameters_to_vector(g_clone.parameters()).detach()
            fed.local_train(local, Xtr, ytr, cfg.epochs, cfg.lr, cfg.mu, gvec,
                            cfg.l1, cfg.weight_decay, device=cfg.device)

            participants.append({"model": local, "weight": len(Xtr), "keep": keep,
                                 "nidx": nidx, "ctx": ctx, "action": (evolve_step, keep),
                                 "pre_acc": 1.0 - pre_err})

        snr_med = float(np.median(snrs_now)) if snrs_now else None
        if not participants:
            log["slot"].append(t); log["acc_now"].append(None); log["bits"].append(0)
            log["grid"].append(getattr(global_model, "grid_size", 0))
            log["n_part"].append(0); log["drift_flags"].append(0)
            log["snr_med"].append(snr_med)
            continue

        cur_bin = int(round(snr_med)) if snr_med is not None else 0
        acc_before = eval_regime(cur_bin) or 0.0
        total_bits, _ = fed.fed_aggregate(global_model, participants,
                                          cfg.bits_per_param, cfg.mu)
        acc_now = eval_regime(cur_bin) or 0.0

        # reward & controller update
        reward_base = (acc_now - acc_before) - cfg.comm_lambda * (total_bits / cfg.budget_bits)
        for p in participants:
            controllers[p["nidx"]].update(p["ctx"], p["action"], reward_base)
        budget_price = float(np.clip(total_bits / cfg.budget_bits, 0, 1))

        # regime accuracy log for forgetting / BWT
        for s_bin in eval_by_snr:
            a = eval_regime(s_bin)
            if a is not None:
                regime_acc_log.setdefault(s_bin, []).append((t, a))

        log["slot"].append(t); log["acc_now"].append(acc_now)
        log["bits"].append(total_bits)
        log["grid"].append(getattr(global_model, "grid_size", 0))
        log["n_part"].append(len(participants)); log["drift_flags"].append(slot_drift)
        log["snr_med"].append(snr_med)
        if verbose:
            print(f"t={t:3d} part={len(participants)} snr={snr_med} "
                  f"acc={acc_now:.3f} grid={log['grid'][-1]} bits={total_bits} "
                  f"drift={slot_drift}")

    result = {
        "avg_acc": metrics.average_accuracy(log["acc_now"]),
        "bwt": metrics.backward_transfer(regime_acc_log),
        "forgetting": metrics.forgetting(regime_acc_log),
        "total_bits": float(np.sum(log["bits"])),
        "final_grid": log["grid"][-1] if log["grid"] else 0,
        "params": global_model.num_params(),
        "n_classes": n_classes, "in_dim": in_dim, "dims": dims, "T": T,
    }
    return result, log
