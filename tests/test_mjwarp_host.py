"""The coupled Cholesky on the real MuJoCo Warp hook. Needs mujoco, mujoco_warp and CUDA."""

import numpy as np
import pytest

try:
    import mujoco
    import mujoco_warp as mjw
    import warp as wp
    HAVE = wp.is_cuda_available()
except Exception:                                    # pragma: no cover
    HAVE = False

pytestmark = pytest.mark.skipif(not HAVE, reason="needs mujoco, mujoco_warp and CUDA")

if HAVE:
    import coupledchol.phost_impl as impl
    from phost import GuardError, Registry, Selection, Session
    from phost.hosts import mjwarp as adapter
    from phost.stages import newton_linear as nl
    from phost.testing.contract import check_impl
    from phost.testing.mjwarp import DrivableMJWarpHost

SCENE = """<mujoco><option timestep="0.002" integrator="Euler" solver="Newton"/>
<worldbody><geom type="plane" size="3 3 .1"/>{bodies}</worldbody></mujoco>"""


def make_host(nworld=2, nbodies=60):
    """Many free spheres (no off-diagonal coupling) plus boxes in contact: a sparse-Hessian scene."""
    boxes = "".join(f'<body pos="{0.3*(i%4)-0.45} {0.3*(i//4)-0.3} {0.1+0.02*i}"><freejoint/>'
                    f'<geom type="box" size=".05 .05 .05" mass=".5"/></body>' for i in range(8))
    spheres = "".join(f'<body pos="{0.12*(i%10)-0.5} {0.12*(i//10)-0.3} 2" gravcomp="1"><freejoint/>'
                      f'<geom type="sphere" size=".02" mass=".1" contype="0" conaffinity="0"/></body>'
                      for i in range(nbodies - 8))
    cm = mujoco.MjModel.from_xml_string(SCENE.format(bodies=boxes + spheres))
    cd = mujoco.MjData(cm)
    for _ in range(40):
        mujoco.mj_step(cm, cd)
    with wp.ScopedDevice("cuda:0"):
        model = mjw.put_model(cm)
        data = mjw.put_data(cm, cd, nworld=nworld, nconmax=64 * nworld, njmax=512)
    return DrivableMJWarpHost(model, data, cpu_model=cm, cpu_data=cd)


def reviewed():
    try:
        adapter.Guard("t", adapter._read_hook, adapter.REVIEWED_HOOKS).check()
        return True
    except GuardError:
        return False


pytestmark = [pytestmark, pytest.mark.skipif(HAVE and not reviewed(), reason="this MJWarp build is not a reviewed build")]


@pytest.fixture
def registry():
    reg = Registry()
    nl.register(reg)
    impl.register(reg)
    return reg


@pytest.fixture(scope="module")
def host():
    return make_host()


PARAMS = [{}, {"max_coupled": 32}, {"tile_size": 16}]
STEP_CALLS = [dict(mode="step", index=0), dict(mode="step", index=1)]


@pytest.mark.parametrize("params", PARAMS)
def test_scan_meets_the_contract_on_the_real_hook_with_chosen_hessians(host, registry, params):
    verdict = check_impl(host, Selection(nl.NAME, impl.SCAN, params), registry)
    assert verdict.ok, verdict.failures


@pytest.mark.parametrize("impl_id", [impl.SCAN, impl.STRUCTURAL])
@pytest.mark.parametrize("params", PARAMS)
def test_meets_the_contract_on_hessians_the_host_assembled(host, registry, impl_id, params):
    sel = Selection(nl.NAME, impl_id, dict(params, verify_structure=True) if impl_id == impl.STRUCTURAL else params)
    verdict = check_impl(host, sel, registry, calls=STEP_CALLS)
    assert verdict.ok, verdict.failures


def test_the_structure_hint_is_only_exact_for_hessians_the_host_assembled(host, registry):
    """A synthetic H has no relation to the host's Jacobian, so the hint misses its coupling: the contract
    check must fail, and verify_structure must say why (status bit 4)."""
    sel = Selection(nl.NAME, impl.STRUCTURAL, {"verify_structure": True})
    verdict = check_impl(host, sel, registry)
    assert not verdict.ok
    with Session(host, [sel], registry) as session:
        host.drive(nl.NAME)
        assert session.metrics()[nl.NAME]["status"] & 4


def test_installed_dispatch_matches_native_after_a_real_rollout(registry):
    """Whole-simulation check against a native-vs-native control (MJWarp is not bitwise repeatable)."""
    h = make_host(nworld=2)
    m, cm = h.model, h.cpu_model

    def run(selections):
        with wp.ScopedDevice("cuda:0"):
            d = mjw.put_data(cm, mujoco.MjData(cm), nworld=2, nconmax=128, njmax=512)
            with Session(h, selections, registry) as session:
                mjw.step(m, d)                                        # warm up under the install
                wp.synchronize()
                with wp.ScopedCapture() as cap:
                    mjw.step(m, d)
            for _ in range(60):
                wp.capture_launch(cap.graph)
            wp.synchronize()
            return d.qpos.numpy().astype(np.float64), session

    native, _ = run([])
    control, _ = run([])
    for impl_id in (impl.SCAN, impl.STRUCTURAL):
        got, session = run([Selection(nl.NAME, impl_id, {"verify_structure": True} if impl_id == impl.STRUCTURAL else {})])
        metrics = session.metrics()[nl.NAME]
        assert metrics["status"] == 0 and metrics["fallbacks"] == 0
        assert np.isfinite(got).all()
        noise = max(np.abs(control - native).max(), 1e-6)
        assert np.abs(got - native).max() <= 10 * noise
