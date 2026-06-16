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
from .models_conv import ConvKANClassifier, ConvMLPClassifier, conv_mlp_matching
from .drift import make_detector
from .controller import make_controller
from . import fed
from . import metrics


@dataclass
class Config:
    # data
    source: str = "synthetic"        # synthetic | rml2016 | rml2018
    data_path: str = ""
    arch: str = "conv"               # conv (raw I/Q + 1D-CNN front-end) | flat
    feature: str = "multimodal"      # used only when arch=="flat": iq|stat|multimodal
    conv_channels: tuple = (16, 32)
    conv_kernel: int = 5
    conv_feat_dim: int = 24
    per_combo: int = 120             # synthetic frames per (mod,snr)
    # topology / orbit
    n_nodes: int = 6
    slots_per_pass: int = 24
    n_passes: int = 3
    gap_slots: int = 4
    theta_min: float = 10.0
    snr_zenith: float = 18.0
    dirichlet_alpha: float = 0.3     # non-IID class skew per node
    drift_mode: str = "incremental"  # incremental (new classes appear) | snr_sweep
    n_init_classes: int = 3          # classes active at mission start (incremental)
    intro_frac: float = 0.7          # introduce all classes within this frac of horizon
    # model
    model_type: str = "kan"          # kan | mlp
    hidden: tuple = (16,)
    grid_size: int = 5
    grid_max: int = 9               # conservative cap: large grids overfit small slots
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

    # held-out eval set per CLASS (for incremental / continual-learning metrics)
    eval_by_class = {}
    for c in range(n_classes):
        ids = eval_idx[y[eval_idx] == c]
        if len(ids) > cfg.max_eval:
            ids = rng.choice(ids, cfg.max_eval, replace=False)
        eval_by_class[int(c)] = ids
    return nodes, eval_by_snr, eval_by_class, train_idx, eval_idx


def _class_intro_schedule(n_classes, T, cfg, rng):
    """Slot at which each class becomes active. The first n_init_classes are
    active from t=0; the rest are introduced, in random order, spread uniformly
    across the first intro_frac of the horizon."""
    order = list(range(n_classes))
    rng.shuffle(order)
    init = order[:cfg.n_init_classes]
    rest = order[cfg.n_init_classes:]
    intro = {c: 0 for c in init}
    if rest:
        last = max(1, int(cfg.intro_frac * T))
        for i, c in enumerate(rest):
            intro[c] = int((i + 1) / (len(rest) + 1) * last)
    return intro


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
    feat_mode = "raw" if cfg.arch == "conv" else cfg.feature
    X, dims = datamod.featurize(bank, feat_mode)
    y, snr = bank.labels, bank.snr
    n_classes = bank.n_classes
    conv_kw = dict(channels=cfg.conv_channels, kernel=cfg.conv_kernel,
                   feat_dim=cfg.conv_feat_dim)

    nodes, eval_by_snr, eval_by_class, _, _ = _build_nodes(cfg, X, y, snr, n_classes, rng)

    # --- global model ---
    if cfg.arch == "conv":
        in_ch, L = X.shape[1], X.shape[2]      # raw I/Q: (N, 2, L)
        in_dim = in_ch * L                     # for reporting only
        if cfg.model_type == "kan":
            global_model = ConvKANClassifier(in_ch, L, n_classes, cfg.hidden,
                                             grid_size=cfg.grid_size,
                                             spline_order=cfg.spline_order, **conv_kw)
        else:
            ref = ConvKANClassifier(in_ch, L, n_classes, cfg.hidden,
                                    grid_size=cfg.grid_size, **conv_kw)
            global_model, _ = conv_mlp_matching(in_ch, L, n_classes, ref, **conv_kw)
    else:
        in_dim = X.shape[1]
        if cfg.model_type == "kan":
            global_model = KANClassifier(in_dim, cfg.hidden, n_classes,
                                         grid_size=cfg.grid_size, spline_order=cfg.spline_order)
        else:
            ref_kan = KANClassifier(in_dim, cfg.hidden, n_classes, grid_size=cfg.grid_size)
            global_model, _ = mlp_matching_kan(in_dim, n_classes, ref_kan)

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

    # class-introduction schedule (incremental drift) + true drift points
    class_intro = _class_intro_schedule(n_classes, T, cfg,
                                        np.random.default_rng(cfg.seed + 7))
    drift_truth = [False] * T
    if cfg.drift_mode == "incremental":
        for c, t0 in class_intro.items():
            if 0 < t0 < T:
                drift_truth[t0] = True

    def active_classes(t):
        if cfg.drift_mode != "incremental":
            return set(range(n_classes))
        return {c for c, t0 in class_intro.items() if t0 <= t}

    # --- logs ---
    log = {"slot": [], "acc_now": [], "bits": [], "grid": [], "n_part": [],
           "drift_flags": [], "snr_med": [], "n_active_classes": []}
    regime_acc_log = {}
    budget_price = 0.0

    def eval_regime(s_bin):
        idx = eval_by_snr.get(s_bin)
        if idx is None or len(idx) == 0:
            return None
        return fed.evaluate(global_model, X[idx], y[idx], cfg.device)

    def eval_class(c):
        idx = eval_by_class.get(c)
        if idx is None or len(idx) == 0:
            return None
        return fed.evaluate(global_model, X[idx], y[idx], cfg.device)

    def eval_cumulative(active):
        """Accuracy on all classes seen so far (the incremental headline)."""
        if not active:
            return None
        ids = np.concatenate([eval_by_class[c] for c in active if len(eval_by_class[c])])
        if len(ids) > cfg.max_eval:
            ids = rng.choice(ids, cfg.max_eval, replace=False)
        return fed.evaluate(global_model, X[ids], y[ids], cfg.device)

    for t in range(T):
        participants = []
        slot_drift = 0
        snrs_now = []
        active = active_classes(t)
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
            cand = pool[sb]
            if cfg.drift_mode == "incremental":     # only currently-active classes
                cand = cand[np.isin(y[cand], list(active))]
            if len(cand) < 8:
                continue
            ids = rng.choice(cand, min(cfg.slot_samples, len(cand)), replace=False)
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

            # drift-aware pseudo-labels: only trust self-labels in STABLE periods.
            # When drift is flagged a new class/regime is likely present, so the
            # model would confidently mislabel it -> we skip pseudo-labeling then.
            n_lab = max(1, int(cfg.label_frac * len(Xi)))
            Xtr, ytr = list(Xi[:n_lab]), list(yi[:n_lab])
            if cfg.pseudo_enabled and not flag and len(Xi) > n_lab:
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
        incremental = cfg.drift_mode == "incremental"

        def regime_acc_before_after():
            if incremental:
                return eval_cumulative(active)
            return eval_regime(int(round(snr_med)) if snr_med is not None else 0)

        if not participants:
            log["slot"].append(t); log["acc_now"].append(None); log["bits"].append(0)
            log["grid"].append(getattr(global_model, "grid_size", 0))
            log["n_part"].append(0); log["drift_flags"].append(0)
            log["snr_med"].append(snr_med)
            log["n_active_classes"].append(len(active))
            continue

        acc_before = regime_acc_before_after() or 0.0
        total_bits, _ = fed.fed_aggregate(global_model, participants,
                                          cfg.bits_per_param, cfg.mu)
        acc_now = regime_acc_before_after() or 0.0

        # reward & controller update
        reward_base = (acc_now - acc_before) - cfg.comm_lambda * (total_bits / cfg.budget_bits)
        for p in participants:
            controllers[p["nidx"]].update(p["ctx"], p["action"], reward_base)
        budget_price = float(np.clip(total_bits / cfg.budget_bits, 0, 1))

        # per-unit accuracy log for forgetting / BWT (per-class if incremental)
        if incremental:
            for c in active:
                a = eval_class(c)
                if a is not None:
                    regime_acc_log.setdefault(c, []).append((t, a))
        else:
            for s_bin in eval_by_snr:
                a = eval_regime(s_bin)
                if a is not None:
                    regime_acc_log.setdefault(s_bin, []).append((t, a))

        log["slot"].append(t); log["acc_now"].append(acc_now)
        log["bits"].append(total_bits)
        log["grid"].append(getattr(global_model, "grid_size", 0))
        log["n_part"].append(len(participants)); log["drift_flags"].append(slot_drift)
        log["snr_med"].append(snr_med)
        log["n_active_classes"].append(len(active))
        if verbose:
            print(f"t={t:3d} part={len(participants)} act={len(active)} "
                  f"acc={acc_now:.3f} grid={log['grid'][-1]} bits={total_bits} "
                  f"drift={slot_drift}")

    final_acc_all = eval_cumulative(set(range(n_classes))) or 0.0
    result = {
        "avg_acc": metrics.average_accuracy(log["acc_now"]),
        "final_acc_all": float(final_acc_all),
        "bwt": metrics.backward_transfer(regime_acc_log),
        "forgetting": metrics.forgetting(regime_acc_log),
        "detection_delay": metrics.detection_delay(drift_truth, log_drift_series(log, T)),
        "total_bits": float(np.sum(log["bits"])),
        "final_grid": log["grid"][-1] if log["grid"] else 0,
        "params": global_model.num_params(),
        "n_classes": n_classes, "in_dim": in_dim, "dims": dims, "T": T,
    }
    return result, log


def log_drift_series(log, T):
    """Dense per-slot drift-flag series (0/1) aligned to slot index, for delay."""
    series = [0] * T
    for s, f in zip(log["slot"], log["drift_flags"]):
        series[s] = int(f)
    return series
