import numpy as np
import pytest
import warp as wp

from phost_coupledchol import CoupledCholesky
from helpers import DEVICE, Arrays, cuda_available, dense_from_upper, make_batch, relative_residual

pytestmark = pytest.mark.skipif(not cuda_available(), reason="needs CUDA")
GATE = 1e-4


def solve(chol, arrays, **kw):
    chol.enqueue(*arrays.args(), **kw)
    return arrays.out()


@pytest.mark.parametrize("nv,nvpad,coupled,worlds,tile", [
    (40, 48, 10, 3, 16), (40, 48, 10, 3, 32), (100, 112, 30, 5, 32), (1, 16, 0, 2, 16), (17, 32, 0, 2, 16),
    (33, 48, 1, 2, 16), (200, 208, 64, 2, 32), (330, 336, 90, 2, 32)])
def test_matches_float64_reference(nv, nvpad, coupled, worlds, tile):
    h, g, dense = make_batch(worlds, nv, nvpad, coupled, seed=nv)
    a = Arrays(h, g, nv)
    chol = CoupledCholesky(nv, worlds, DEVICE, tile_size=tile)
    search, sdot, dec = solve(chol, a, fallback=lambda m: None)
    assert (relative_residual(dense, g, search, nv) < GATE).all()
    x = -search.astype(np.float64)
    ref = np.linalg.solve(dense[0], g[0, :nv].astype(np.float64))
    assert np.linalg.norm(x[0] - ref) / np.linalg.norm(ref) < 1e-3
    assert np.allclose(sdot, (search.astype(np.float64) ** 2).sum(1), rtol=1e-4)
    assert np.allclose(dec, [g[w, :nv] @ x[w] for w in range(worlds)], rtol=1e-3)
    assert chol.metrics()["status"] == 0


@pytest.mark.parametrize("name", ["trampoline12_impact", "trampoline12_rest", "trampoline14_impact", "trampoline14_rest"])
def test_real_hessians_captured_from_a_host(name):
    z = np.load(f"tests/data/{name}.npz")
    nv = {"12": 330, "14": 462}[name[10:12]]
    for call in range(z["H"].shape[0]):
        h, g = np.ascontiguousarray(z["H"][call]), np.ascontiguousarray(z["grad"][call])
        dense = dense_from_upper(h, nv)
        a = Arrays(h, g, nv)
        chol = CoupledCholesky(nv, h.shape[0], DEVICE)
        def no_overflow(mask):
            assert mask.numpy().max() == 0, "a real Hessian overflowed max_coupled"
        search, _, _ = solve(chol, a, fallback=no_overflow)
        assert relative_residual(dense, g, search, nv).max() < GATE
        m = chol.metrics()
        assert m["status"] == 0 and m["fallbacks"] == 0 and 0 < m["coupled_last_max"] <= 128


def test_inactive_worlds_are_untouched_and_inputs_unmodified():
    h, g, dense = make_batch(4, 40, 48, 8)
    a = Arrays(h, g, 40, active=[1, 0, 1, 0])
    chol = CoupledCholesky(40, 4, DEVICE)
    search, sdot, dec = solve(chol, a)
    assert (search[[1, 3]] == 123.0).all() and (sdot[[1, 3]] == 123.0).all() and (dec[[1, 3]] == 123.0).all()
    assert (relative_residual(dense[[0, 2]], g[[0, 2]], search[[0, 2]], 40) < GATE).all()
    assert np.array_equal(a.h.numpy(), h) and np.array_equal(a.grad.numpy(), g)


def test_cached_factor_is_reused_only_when_refactor_is_zero():
    # Only the coupled block is cached; diagonal-only DOFs are read live. So change just the block.
    h, g, dense = make_batch(2, 40, 48, 8, seed=1)
    other = make_batch(2, 40, 48, 8, seed=2)[0]
    h2 = h.copy()
    h2[:, :8, :8] = other[:, :8, :8]
    dense2 = dense_from_upper(h2, 40)
    chol = CoupledCholesky(40, 2, DEVICE)
    solve(chol, Arrays(h, g, 40))
    # The caller says H did not change: the cached block factor is used, so the answer is the OLD system's.
    search, _, _ = solve(chol, Arrays(h2, g, 40, refactor=[0, 0]))
    assert (relative_residual(dense, g, search, 40) < GATE).all()
    assert (relative_residual(dense2, g, search, 40) > 1e-2).all()
    # refactor=1 picks up the new block.
    search, _, _ = solve(chol, Arrays(h2, g, 40, refactor=[1, 1]))
    assert (relative_residual(dense2, g, search, 40) < GATE).all()


def test_a_world_with_no_valid_cache_is_refactored_even_if_the_caller_says_reuse():
    h, g, dense = make_batch(2, 40, 48, 8, seed=3)
    chol = CoupledCholesky(40, 2, DEVICE)
    a = Arrays(h, g, 40, refactor=[0, 0])               # first call ever, caller claims reuse
    search, _, _ = solve(chol, a)
    assert (relative_residual(dense, g, search, 40) < GATE).all()


def test_overflowing_worlds_are_handed_to_the_fallback_only_for_active_worlds():
    h, g, dense = make_batch(4, 100, 112, 40, seed=4)
    h[1, :100, :100] = make_batch(1, 100, 112, 10, seed=5)[0][0, :100, :100]      # world 1 stays small
    dense[1] = dense_from_upper(h[[1]], 100)[0]
    seen = []

    def fallback(mask):
        mask = mask.numpy().copy()
        seen.append(mask)
        for w in np.flatnonzero(mask):
            x = np.linalg.solve(dense[w], g[w, :100].astype(np.float64))
            a.search.numpy()                                     # (device write below)
            host = a.search.numpy()
            host[w] = -x
            a.search.assign(host)

    a = Arrays(h, g, 100, active=[1, 1, 1, 0])
    chol = CoupledCholesky(100, 4, DEVICE, max_coupled=32)
    search, _, _ = solve(chol, a, fallback=fallback)
    assert len(seen) == 1 and seen[0].tolist() == [1, 0, 1, 0]       # overflowed AND active
    assert (relative_residual(dense[[0, 1, 2]], g[[0, 1, 2]], search[[0, 1, 2]], 100) < GATE).all()
    assert (search[3] == 123.0).all()
    assert chol.metrics()["fallbacks"] == 2


def test_overflow_without_a_fallback_is_refused():
    h, g, _ = make_batch(1, 100, 112, 10)
    chol = CoupledCholesky(100, 1, DEVICE, max_coupled=32)
    with pytest.raises(ValueError, match="fallback is required"):
        chol.enqueue(*Arrays(h, g, 100).args())


def test_no_fallback_launch_when_overflow_is_impossible():
    h, g, _ = make_batch(1, 40, 48, 8)
    chol = CoupledCholesky(40, 1, DEVICE, max_coupled=128)

    def boom(mask):
        raise AssertionError("fallback called although the set can never overflow")

    solve(chol, Arrays(h, g, 40), fallback=boom)


def hint_from(dense, extra=(), drop=()):
    """A host-like hint: flags the DOFs with off-diagonal entries (plus extra, minus drop)."""
    def hint(flags, refactor):
        host = flags.numpy().copy()
        for w in range(dense.shape[0]):
            off = np.abs(dense[w] - np.diag(np.diag(dense[w]))).sum(1) > 0
            host[w, :off.size] = off
            for i in extra:
                host[w, i] = 1
            for i in drop:
                host[w, i] = 0
        flags.assign(host)
    return hint


def test_hint_gives_the_same_answer_as_scan():
    h, g, dense = make_batch(3, 100, 112, 20, seed=6)
    scan = solve(CoupledCholesky(100, 3, DEVICE), Arrays(h, g, 100))[0]
    hinted = solve(CoupledCholesky(100, 3, DEVICE, rescan_above=1000), Arrays(h, g, 100), hint=hint_from(dense))[0]
    assert np.allclose(scan, hinted, rtol=1e-5, atol=1e-6)


def test_a_loose_superset_hint_is_still_exact():
    h, g, dense = make_batch(2, 100, 112, 10, seed=7)
    a = Arrays(h, g, 100)
    solve(CoupledCholesky(100, 2, DEVICE, rescan_above=1000), a, hint=hint_from(dense, extra=range(50, 60)))
    assert (relative_residual(dense, g, a.out()[0], 100) < GATE).all()


def test_a_loose_hint_above_the_threshold_is_rescanned_to_the_exact_set():
    h, g, dense = make_batch(2, 100, 112, 10, seed=8)
    chol = CoupledCholesky(100, 2, DEVICE, rescan_above=16)
    a = Arrays(h, g, 100)
    solve(chol, a, hint=hint_from(dense, extra=range(50, 80)))
    assert chol.metrics()["rescans"] == 2
    assert (relative_residual(dense, g, a.out()[0], 100) < GATE).all()


def test_verify_structure_flags_a_hint_that_misses_a_coupled_dof():
    h, g, dense = make_batch(2, 100, 112, 10, seed=9)
    chol = CoupledCholesky(100, 2, DEVICE, verify_structure=True, rescan_above=1000)
    solve(chol, Arrays(h, g, 100), hint=hint_from(dense, drop=(3,)))
    assert chol.metrics()["status"] & 4
    ok = CoupledCholesky(100, 2, DEVICE, verify_structure=True, rescan_above=1000)
    solve(ok, Arrays(h, g, 100), hint=hint_from(dense))
    assert ok.metrics()["status"] == 0
    with pytest.raises(ValueError, match="needs a structure hint"):
        ok.enqueue(*Arrays(h, g, 100).args())


def test_a_non_positive_definite_system_is_flagged_not_hidden():
    h, g, _ = make_batch(2, 40, 48, 8, seed=10)
    h[0, 3, 3] = -5.0                                               # indefinite diagonal in a coupled row
    chol = CoupledCholesky(40, 2, DEVICE)
    solve(chol, Arrays(h, g, 40))
    assert chol.metrics()["status"] & 3


def test_metrics_reset():
    h, g, _ = make_batch(1, 40, 48, 8)
    h[0, 3, 3] = -5.0
    chol = CoupledCholesky(40, 1, DEVICE)
    solve(chol, Arrays(h, g, 40))
    assert chol.metrics()["status"]
    chol.reset_metrics()
    assert chol.metrics()["status"] == 0


def test_same_inputs_give_identical_outputs_twice():
    h, g, _ = make_batch(3, 100, 112, 20, seed=11)
    runs = [solve(CoupledCholesky(100, 3, DEVICE), Arrays(h, g, 100))[0] for _ in range(2)]
    assert np.array_equal(runs[0], runs[1])


@pytest.mark.parametrize("bad,match", [
    (dict(tile_size=8), "tile_size"), (dict(max_coupled=0), "max_coupled"), (dict(max_coupled=True), "max_coupled")])
def test_constructor_validation(bad, match):
    with pytest.raises(ValueError, match=match):
        CoupledCholesky(40, 1, DEVICE, **bad)
    for nv in (0, 513):
        with pytest.raises(ValueError, match="nv"):
            CoupledCholesky(nv, 1, DEVICE)
    with pytest.raises(ValueError, match="nworld"):
        CoupledCholesky(40, 0, DEVICE)
    with pytest.raises(ValueError, match="CUDA"):
        CoupledCholesky(40, 1, "cpu")


def test_array_validation():
    h, g, _ = make_batch(2, 40, 48, 8)
    chol = CoupledCholesky(40, 2, DEVICE)
    a = Arrays(h, g, 40)
    bad = list(a.args())
    bad[1] = wp.zeros((2, 48), dtype=wp.float64, device=DEVICE)
    with pytest.raises(ValueError, match="grad"):
        chol.enqueue(*bad, fallback=lambda m: None)
    bad = list(a.args())
    bad[4] = wp.zeros((2, 41), dtype=wp.float32, device=DEVICE)
    with pytest.raises(ValueError, match="search|dimensions"):
        chol.enqueue(*bad, fallback=lambda m: None)
    bad = list(a.args())
    bad[0] = wp.zeros((3, 48, 48), dtype=wp.float32, device=DEVICE)
    with pytest.raises(ValueError, match="h must"):
        chol.enqueue(*bad, fallback=lambda m: None)
