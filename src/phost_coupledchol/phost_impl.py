"""Register the coupled-block Cholesky as implementations of phost's ``newton.linear_solve`` and ``effective.solve`` stages.

    import phost, phost_coupledchol.phost_impl
    phost_coupledchol.phost_impl.register(phost.default)

The algorithm is one thing and its array backends are another. ``effective.solve/coupled`` has two: Warp (a tile Cholesky on a
CUDA device) and NumPy (the same steps in float64, for a host whose ports are NumPy arrays); it builds the one that matches the
host's ``backend`` and says why not when a host's arrays are of another kind. ``newton.linear_solve`` is offered today by one
host, MuJoCo Warp's Newton step, which hands over Warp arrays, so its two implementations have a Warp backend only.

Two implementations of ``newton.linear_solve``: ``coupled`` finds the coupled DOFs by scanning H (always exact,
no host assumptions) and ``coupled-structural`` takes them from the host's structure hint (cheaper at high world
counts, relies on how the host assembles H, so it needs that guard to pass).

One of ``effective.solve``: ``coupled`` solves the host's effective metric M_hat x = rhs directly, with the same
factorization applied to the assembled M_hat. M_hat is the mass matrix plus a flex stiffness block, so most or all DOFs
end up coupled and the factorization is a blocked dense Cholesky of that block.

Importing this module needs phost and NumPy only; Warp is imported when a Warp runtime is built.
"""

from __future__ import annotations

from phost import ContractError, Impl, ImplKind, MetricSpec, ParamSpec, Registry
from phost.stages import effective_solve as es
from phost.stages import newton_linear as nl

from .limits import MAX_NV

SCAN = f"{nl.NAME}/coupled"
STRUCTURAL = f"{nl.NAME}/coupled-structural"
EFFECTIVE = f"{es.NAME}/coupled"


def _newton_runtime(structural: bool):
    def make(host, params):
        from .warp_runtime import CoupledRuntime
        return CoupledRuntime(host, params, structural=structural)
    return make


def _effective_runtime(host, params):
    """The backend that matches the arrays the host hands to ``effective.solve``."""
    if host.backend == "warp":
        from .warp_runtime import EffectiveRuntime
    elif host.backend == "numpy":
        from .numpy_solve import EffectiveRuntime
    else:
        raise ContractError(f"effective.solve/coupled has no backend for {host.backend!r} arrays (Warp and NumPy have one)")
    return EffectiveRuntime(host, params)


def _applicable_effective(facts):
    if facts.backend == "numpy":
        return None if facts.nv >= 1 else f"nv={facts.nv}: nothing to solve"
    if facts.backend != "warp":
        return f"no coupled-block solver for {facts.backend!r} arrays (Warp and NumPy have one)"
    if facts.device_kind != "cuda":
        return "the Warp backend needs a CUDA device"
    if "device" not in facts.extra:
        return "the host did not report its device in facts.extra['device']"
    if not 1 <= facts.nv <= MAX_NV:
        return f"nv={facts.nv} is outside 1..{MAX_NV} for the Warp backend"
    return None


def _applicable(facts):
    if facts.backend != "warp":
        return f"this stage's host hands over Warp arrays; this host's are {facts.backend!r}"
    if facts.device_kind != "cuda":
        return "needs a CUDA device"
    if facts.solver != "newton":
        return f"needs the Newton solver, host uses {facts.solver!r}"
    if facts.sleep:
        return "sleeping is not supported"
    if not 1 <= facts.nv <= MAX_NV:
        return f"nv={facts.nv} is outside 1..{MAX_NV}"
    if "device" not in facts.extra:
        return "the host did not report its device in facts.extra['device']"
    return None


def _advisory(facts):
    if facts.nv < 200:
        return (f"nv={facts.nv}: the coupled path was slower than the host's own solver at nv=138 "
                f"(0.74-0.98x) in the baseline; gains start around nv=300")
    return None


_PARAMS = (
    ParamSpec("tile_size", "int", 32, "tile edge of the dense block factorization", choices=(16, 32)),
    ParamSpec("max_coupled", "int", 128, "largest coupled set solved here; larger worlds go to the host", min=1, max=MAX_NV),
    ParamSpec("verify_structure", "bool", False, "also scan H and set status bit 4 if the hint missed a coupled DOF"),
    ParamSpec("rescan_above", "int", 0, "rescan a hinted world whose set exceeds this many DOFs (0: one tile)", min=0),
)
_EXTRA_METRICS = (
    MetricSpec("coupled_last_max", "dof", "none", "gauge", "largest coupled set in the most recent call"),
    MetricSpec("rescans", "count", "lower", "counter", "hinted sets replaced by an exact scan"),
)
_FIT = ("Exact. Helps when most DOFs have no off-diagonal entry in H (flex vertices, free bodies) and nv is large: "
        "baseline speedup was 1.2-2.4x at nv=330 and 1.8-5.1x at nv=462, shrinking as worlds grow, and below 1x at nv=138.")


_EFFECTIVE_PARAMS = (
    ParamSpec("tile_size", "int", 32, "tile edge of the dense block factorization (Warp backend; the NumPy one has no tiles)",
              choices=(16, 32)),
)
_EFFECTIVE_FIT = ("Exact, with a backend for each array type a host hands over. Warp (CUDA, nv up to 512): only worth installing "
                  "where the host's own iteration is slow; measured here (mujoco_warp 3.17.0, RTX PRO 2000 laptop GPU) on the "
                  "lab's 12x12 membrane, nv=330: slower than the host at every time step and stiffness tried, about 0.57x the "
                  "step rate at 1, 8 and 32 worlds, because the host's iteration converges in a few steps (0.11-0.79 ms) while one "
                  "factorization of the 300-DOF coupled block takes about 1.3 ms. NumPy (CPU, float64): the same algorithm for a "
                  "host whose ports are NumPy arrays; not written for speed, so time it against the host's own solve.")


def _effective_advisory(facts):
    if facts.backend == "warp":
        return (f"nv={facts.nv}: time this against the host's own solve before keeping it. On the lab's 12x12 membrane (nv=330) "
                f"the host's iteration took 0.11-0.79 ms and this took about 1.4 ms, so the step got slower. It can only win "
                f"where the host's iteration is slow.")
    return (f"nv={facts.nv}: this is the NumPy backend, a Python loop over worlds with dense NumPy calls. It is exact; time it "
            f"against the host's own solve before expecting it to be faster.")


def impls() -> list[Impl]:
    return [
        Impl(id=SCAN, stage=nl.NAME, stage_version=nl.CONTRACT.version, version="0.1", title="Coupled-block Cholesky (scan)",
             kind=ImplKind.BUILTIN, doc="Finds the coupled DOFs by scanning the upper triangle of H.", params=_PARAMS,
             fit=_FIT, metrics=_EXTRA_METRICS, extensions_used=("fallback",), applicable=_applicable, advisory=_advisory,
             factory=_newton_runtime(False),
             provenance=dict(package="phost-coupledchol")),
        Impl(id=STRUCTURAL, stage=nl.NAME, stage_version=nl.CONTRACT.version, version="0.1",
             title="Coupled-block Cholesky (structure hint)", kind=ImplKind.BUILTIN,
             doc="Takes the coupled DOFs from the host's structure hint instead of scanning H.", params=_PARAMS,
             fit=_FIT, metrics=_EXTRA_METRICS, extensions_used=("structure_hint", "fallback"), applicable=_applicable,
             advisory=_advisory, factory=_newton_runtime(True),
             provenance=dict(package="phost-coupledchol")),
        Impl(id=EFFECTIVE, stage=es.NAME, stage_version=es.CONTRACT.version, version="0.1",
             title="Direct solve of the effective metric (coupled-block Cholesky)", kind=ImplKind.BUILTIN,
             doc="Factors the assembled effective metric directly instead of iterating on it.", params=_EFFECTIVE_PARAMS,
             fit=_EFFECTIVE_FIT, metrics=(MetricSpec("coupled_last_max", "dof", "none", "gauge",
                                                     "largest coupled set in the most recent call"),),
             applicable=_applicable_effective, advisory=_effective_advisory,
             factory=_effective_runtime,
             provenance=dict(package="phost-coupledchol")),
    ]


def register(registry: Registry) -> None:
    for spec in (nl.SPEC, es.SPEC):      # a stage with its judge; adding the same one again is a no-op
        registry.add_stage(spec)
    for impl in impls():
        registry.add_impl(impl)
