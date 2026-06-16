"""Minimal correctness tests for the self-evolution and federated mechanics."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import torch
from sefedkan.kan import KANClassifier, MLPClassifier, mlp_matching_kan
from sefedkan import fed
from sefedkan.drift import PageHinkley


def test_grid_extension_preserves_function():
    torch.manual_seed(0)
    m = KANClassifier(6, (8,), 4, grid_size=5)
    x = torch.randn(32, 6) * 0.5
    before = m(x).detach()
    p0 = m.num_params()
    m.evolve(9)
    after = m(x).detach()
    assert m.num_params() > p0                       # actually grew
    assert torch.allclose(before, after, atol=1e-2)  # function preserved


def test_sparsify_bit_accounting():
    d = torch.randn(1000)
    sd, nz = fed.sparsify(d, 0.1)
    assert 90 <= nz <= 110
    assert (sd != 0).sum().item() == nz


def test_fedavg_moves_global():
    torch.manual_seed(0)
    g = KANClassifier(6, (8,), 4, grid_size=5)
    v0 = torch.nn.utils.parameters_to_vector(g.parameters()).detach().clone()
    clients = []
    for _ in range(3):
        c = fed.clone_model(g)
        X = np.random.randn(64, 6).astype("float32"); y = np.random.randint(0, 4, 64)
        fed.local_train(c, X, y, epochs=2)
        clients.append({"model": c, "weight": 64, "keep": 1.0})
    bits, nz = fed.fed_aggregate(g, clients)
    v1 = torch.nn.utils.parameters_to_vector(g.parameters()).detach()
    assert bits > 0
    assert not torch.allclose(v0, v1)


def test_fedavg_handles_heterogeneous_grids():
    torch.manual_seed(0)
    g = KANClassifier(6, (8,), 4, grid_size=5)
    c1 = fed.clone_model(g)
    c2 = fed.clone_model(g); c2.evolve(9)          # one client self-evolved
    for c in (c1, c2):
        X = np.random.randn(48, 6).astype("float32"); y = np.random.randint(0, 4, 48)
        fed.local_train(c, X, y, epochs=1)
    bits, _ = fed.fed_aggregate(g, [{"model": c1, "weight": 48, "keep": 1.0},
                                    {"model": c2, "weight": 48, "keep": 1.0}])
    assert g.grid_size == 9                          # global raised to max
    assert bits > 0


def test_param_matched_mlp():
    k = KANClassifier(20, (16,), 8, grid_size=5)
    m, gap = mlp_matching_kan(20, 8, k)
    assert gap < 0.15


def test_page_hinkley_flags_shift():
    ph = PageHinkley(delta=0.005, lambda_=0.2)
    flags = [ph.update(0.1) for _ in range(50)]
    flags += [ph.update(0.6) for _ in range(50)]     # mean jumps
    assert any(flags[50:])
