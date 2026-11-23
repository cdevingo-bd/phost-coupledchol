"""The coupled-block Cholesky without Warp: the NumPy backend, the stage check on a CPU host, and applicability by array type."""

import subprocess
import sys

import numpy as np
import pytest

import phost
import phost_coupledchol.phost_impl as impl
from phost import ModelFacts, Registry, Selection
from phost.hosts.chain import ChainHost, make_model
from phost.stages import effective_solve as es
from phost.testing.contract import check_impl
from phost_coupledchol.numpy_solve import NumpyCoupledCholesky


def system(nv=12, worlds=3, coupled=(2, 3, 7), seed=0):
    """H = diagonal + a dense block on ``coupled``; lower triangle NaN (undefined), padded to nv + 2."""
    rng = np.random.default_rng(seed)
    pad = nv + 2
    h = np.full((worlds, pad, pad), np.nan, np.float32)
    rhs = rng.normal(size=(worlds, pad)).astype(np.float32)
    full = []
    for w in range(worlds):
        a = np.diag(rng.uniform(2.0, 4.0, nv))
        block = rng.normal(size=(len(coupled), len(coupled))) * 0.3
        a[np.ix_(coupled, coupled)] += block @ block.T
        full.append(a)
        upper = np.triu(a).astype(np.float32)
        upper[np.tril_indices(nv, -1)] = np.nan          # the lower triangle is undefined
        h[w, :nv, :nv] = upper
    return np.asarray(full), h, rhs


def test_it_solves_a_diagonal_plus_block_system_exactly():
    full, h, rhs = system()
    out = np.zeros((3, 12), np.float32)
    chol = NumpyCoupledCholesky(12, 3)
    chol.solve(h, rhs, out)
    expected = np.stack([np.linalg.solve(full[w].astype(np.float32).astype(np.float64), rhs[w, :12].astype(np.float64)) for w in range(3)])
    assert np.allclose(out, expected, rtol=1e-5, atol=1e-6)
    metrics = chol.metrics()
    assert metrics["status"] == 0 and metrics["coupled_last_max"] == 3 and metrics["max_relative_residual"] < 1e-6


def test_the_undefined_lower_triangle_is_never_read():
    full, h, rhs = system()
    assert np.isnan(h[:, 5, 1]).all()
    out = np.zeros((3, 12), np.float32)
    NumpyCoupledCholesky(12, 3).solve(h, rhs, out)
    assert np.isfinite(out).all()


def test_a_matrix_that_is_not_positive_definite_sets_the_pivot_bit_instead_of_hiding_it():
    _, h, rhs = system()
    h[1, 2, 2] = -5.0                                                 # a coupled DOF whose pivot cannot be positive
    chol = NumpyCoupledCholesky(12, 3)
    chol.solve(h, rhs, np.zeros((3, 12), np.float32))
    assert chol.metrics()["status"] & 1


def test_an_answer_too_coarse_to_meet_the_gate_sets_the_residual_bit_instead_of_hiding_it():
    _, h, rhs = system()
    chol = NumpyCoupledCholesky(12, 3)
    chol.solve(h, rhs, np.zeros((3, 12), np.float16))                  # half precision: about 1e-3 relative, past the 1e-4 gate
    assert chol.metrics()["status"] & 2 and chol.metrics()["max_relative_residual"] > 1e-4
    chol.reset_metrics()
    assert chol.metrics() == dict(status=0, max_relative_residual=0.0, coupled_last_max=0)


def registry():
    reg = Registry()
    impl.register(reg)
    return reg


def test_the_stage_check_passes_on_a_cpu_host_that_has_no_warp_arrays():
    host = ChainHost(make_model(24, 3), worlds=3)
    assert host.facts().backend == "numpy"
    verdict = check_impl(host, Selection(es.NAME, impl.EFFECTIVE, {}), registry())
    assert verdict.ok, verdict.failures
    assert verdict.metrics["max_relative_residual"] < 1e-4 and verdict.metrics["max_native_difference"] < 1e-3


def test_it_is_applicable_by_the_arrays_a_host_hands_over():
    reg = registry()
    numpy_host = ModelFacts(nv=24, world_count=3)
    warp_host = ModelFacts(nv=24, world_count=3, device_kind="cuda", solver="newton", backend="warp", extra={"device": "cuda:0"})
    jax_host = ModelFacts(nv=24, world_count=3, backend="jax")
    effective, scan = reg.impl(impl.EFFECTIVE), reg.impl(impl.SCAN)
    assert effective.applicable(numpy_host) is None and effective.applicable(warp_host) is None
    assert "no coupled-block solver for 'jax' arrays" in effective.applicable(jax_host)
    assert scan.applicable(warp_host) is None
    assert "Warp arrays" in scan.applicable(numpy_host) and "Warp arrays" in scan.applicable(jax_host)
    assert "CUDA" in effective.applicable(ModelFacts(nv=24, world_count=3, backend="warp", extra={"device": "cpu"}))


def test_the_package_and_its_registration_import_without_warp():
    code = ("import sys; sys.modules['warp'] = None\n"                # an import of warp now raises ImportError
            "import phost_coupledchol, phost_coupledchol.phost_impl as m\n"
            "import phost; r = phost.Registry(); m.register(r)\n"
            "assert phost_coupledchol.MAX_NV == 512 and r.impl(m.EFFECTIVE)\n"
            "try:\n    phost_coupledchol.CoupledCholesky\n    raise SystemExit('the Warp class loaded without warp')\n"
            "except ImportError:\n    pass\n"
            "print('ok')")
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert done.returncode == 0 and done.stdout.strip() == "ok", done.stderr
