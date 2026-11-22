"""The coupled Cholesky as the ``effective.solve`` implementation on the real MuJoCo Warp hook.

Needs mujoco, mujoco_warp and CUDA, and a build whose functions are in phost's reviewed tables. The scene is a flex
sheet under the discrete integrator, the one case where the host solves an effective metric before its constraint solver.
"""

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
    import phost_coupledchol.phost_impl as impl
    from phost import GuardError, ApplicabilityError, Registry, Selection, Session
    from phost.hosts import mjwarp as adapter
    from phost.stages import effective_solve as es
    from phost.stages import newton_linear as nl
    from phost.testing.contract import check_impl
    from phost.testing.mjwarp import DrivableMJWarpHost

XML = """<mujoco model="sheet"><option timestep="0.002" integrator="{integrator}"/>
<worldbody>
 <geom type="plane" size="3 3 .1"/>
 <flexcomp name="sheet" type="grid" dim="2" count="{n} {n} 1" spacing="0.05 0.05 0.1" pos="0 0 .3" mass=".2" radius="5e-3">
   <pin gridrange="0 0 {m} 0"/><pin gridrange="0 {m} {m} {m}"/><pin gridrange="0 0 0 {m}"/><pin gridrange="{m} 0 {m} {m}"/>
   <elasticity elastic2d="both" poisson="0" thickness="2e-3" young="1e5" damping="0.01"/>
 </flexcomp>
 <body pos="0.12 0.12 0.6"><freejoint/><geom type="sphere" size="0.03" mass=".1"/></body>
</worldbody></mujoco>"""


def make_host(integrator="discrete", nworld=3, n=6):
    cm = mujoco.MjModel.from_xml_string(XML.format(integrator=integrator, n=n, m=n - 1))
    cd = mujoco.MjData(cm)
    for _ in range(10):
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
    es.register(reg)
    impl.register(reg)
    return reg


@pytest.fixture(scope="module")
def host():
    return make_host()


@pytest.mark.parametrize("params", [{}, {"tile_size": 16}, {"tile_size": 32}])
def test_meets_the_contract_on_the_matrix_the_host_assembled(host, registry, params):
    verdict = check_impl(host, Selection(es.NAME, impl.EFFECTIVE, params), registry)
    assert verdict.ok, verdict.failures
    assert verdict.metrics["max_native_difference"] < 1e-3
    assert verdict.metrics["max_relative_residual"] < 1e-4


def test_is_offered_only_where_the_host_solves_an_effective_metric(registry):
    flex = impl.EFFECTIVE
    euler = make_host("Euler", nworld=1)
    assert euler.facts().extra["effective_blocker"]
    with pytest.raises(ApplicabilityError, match="discrete integrator"):
        Session(euler, [Selection(es.NAME, flex)], registry).__enter__()


def test_a_system_larger_than_the_cholesky_limit_is_refused(registry):
    big = make_host(nworld=1, n=24)                  # 576 vertices x 3 > 512 dofs
    assert big.facts().nv > 512
    with pytest.raises(ApplicabilityError, match="outside 1..512"):
        Session(big, [Selection(es.NAME, impl.EFFECTIVE)], registry).__enter__()


def test_installed_step_matches_native_after_a_real_rollout(registry):
    """Whole-simulation check against a native-vs-native control (MJWarp is not bitwise repeatable). The step is
    captured as a CUDA graph with the replacement installed, so this also covers capture."""
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
    got, session = run([Selection(es.NAME, impl.EFFECTIVE)])
    metrics = session.metrics()[es.NAME]
    assert metrics["status"] == 0 and metrics["calls"] >= 1
    assert metrics["max_relative_residual"] < 1e-4
    assert np.isfinite(got).all()
    noise = max(np.abs(control - native).max(), 1e-6)
    assert np.abs(got - native).max() <= 10 * noise


def test_the_session_tells_the_user_to_time_it(host, registry):
    with Session(host, [Selection(es.NAME, impl.EFFECTIVE)], registry) as session:
        assert any("time this against the host" in a for a in session.advisories)
