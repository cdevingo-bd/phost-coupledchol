import numpy as np
import pytest

import phost_coupledchol.phost_impl as impl
from helpers import cuda_available
from phost import ApplicabilityError, Registry, Selection, Session
from phost.stages import newton_linear as nl
from phost.testing.contract import check_impl
from warp_host import WarpFakeHost

pytestmark = pytest.mark.skipif(not cuda_available(), reason="needs CUDA")


@pytest.fixture
def registry():
    reg = Registry()
    nl.register(reg)
    impl.register(reg)
    return reg


def host(nv=100, worlds=4, coupled=12):
    return WarpFakeHost(nv=nv, world_count=worlds, nv_pad=((nv + 15) // 16) * 16, coupled=coupled)


@pytest.mark.parametrize("impl_id", [impl.SCAN, impl.STRUCTURAL])
@pytest.mark.parametrize("params", [{}, {"tile_size": 16}, {"max_coupled": 16}])
def test_implementations_satisfy_the_stage_contract(registry, impl_id, params):
    verdict = check_impl(host(), Selection(nl.NAME, impl_id, params), registry)
    assert verdict.ok, verdict.failures
    assert verdict.metrics["max_relative_residual"] < 1e-4


def test_overflow_goes_through_the_hosts_own_solver_and_stays_correct(registry):
    h = host(nv=100, coupled=40)
    sel = Selection(nl.NAME, impl.SCAN, {"max_coupled": 16})
    verdict = check_impl(h, sel, registry)
    assert verdict.ok, verdict.failures
    with Session(h, [sel], registry) as session:
        h.drive(nl.NAME)
        assert session.metrics()[nl.NAME]["fallbacks"] == h.data.nworld


def test_metrics_and_advisory_come_through_the_session(registry):
    h = host()
    with Session(h, [Selection(nl.NAME, impl.SCAN)], registry) as session:
        h.drive(nl.NAME)
        m = session.metrics()[nl.NAME]
    assert m["status"] == 0 and m["coupled_last_max"] > 0 and m["max_relative_residual"] < 1e-4
    assert session.advisories and "nv=100" in session.advisories[0]


def test_applicability_is_enforced(registry):
    big = host(nv=100)
    big.model.nv = 600                                  # outside the supported range
    with pytest.raises(ApplicabilityError, match="outside"):
        with Session(big, [Selection(nl.NAME, impl.SCAN)], registry):
            pass


def test_a_correct_structure_hint_verifies_clean(registry):
    h = host(nv=100, coupled=12)
    sel = Selection(nl.NAME, impl.STRUCTURAL, {"verify_structure": True, "rescan_above": 1000})
    with Session(h, [sel], registry) as session:
        h.drive(nl.NAME)
        assert session.metrics()[nl.NAME]["status"] == 0


def test_a_faulty_structure_hint_is_detected_by_verification(registry):
    h = host(nv=100, coupled=12)
    h.hint_drop = (3,)                                  # the host's hint misses coupled DOF 3
    sel = Selection(nl.NAME, impl.STRUCTURAL, {"verify_structure": True, "rescan_above": 1000})
    with Session(h, [sel], registry) as session:
        h.drive(nl.NAME)
        assert session.metrics()[nl.NAME]["status"] & 4
