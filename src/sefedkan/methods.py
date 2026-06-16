"""Method presets: each maps a name to Config field overrides. All methods share
the same experiment loop so comparisons are fair (empirical-verification playbook).
"""
from .experiment import Config


def make_config(method: str, **overrides) -> Config:
    base = dict(
        model_type="kan", controller="bandit", evolve_enabled=True,
        pseudo_enabled=True, mu=0.0,
    )
    presets = {
        # ours
        "sefedkan":          dict(),
        # ablations (isolate one mechanism each)
        "fedkan_static":     dict(controller="fixed", evolve_enabled=False),
        "fedkan_dual":       dict(controller="dual_threshold"),
        "fedkan_no_pseudo":  dict(pseudo_enabled=False),
        "fedkan_no_evolve":  dict(evolve_enabled=False),
        # baselines
        "fedavg_mlp":        dict(model_type="mlp", controller="fixed",
                                  evolve_enabled=False, pseudo_enabled=False),
        "fedprox_mlp":       dict(model_type="mlp", controller="fixed",
                                  evolve_enabled=False, pseudo_enabled=False, mu=0.01),
    }
    if method not in presets:
        raise ValueError(f"unknown method {method}; choices={list(presets)}")
    cfg_kwargs = {**base, **presets[method], **overrides}
    return Config(**cfg_kwargs)


METHODS = ["sefedkan", "fedkan_static", "fedkan_dual", "fedkan_no_pseudo",
           "fedkan_no_evolve", "fedavg_mlp", "fedprox_mlp"]
