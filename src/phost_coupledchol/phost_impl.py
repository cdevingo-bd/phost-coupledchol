"""Register the coupled-block Cholesky as implementations of phost's ``newton.linear_solve`` stage.

    import phost, phost_coupledchol.phost_impl
    phost_coupledchol.phost_impl.register(phost.default)

Two implementations of ``newton.linear_solve``: ``coupled`` finds the coupled DOFs by scanning H (always exact,
no host assumptions) and ``coupled-structural`` takes them from the host's structure hint (cheaper at high world
counts, relies on how the host assembles H, so it needs that guard to pass).

One of ``effective.solve``: ``coupled`` solves the host's effective metric M_hat x = rhs directly, with the same
factorization applied to the assembled M_hat. M_hat is the mass matrix plus a flex stiffness block, so most or all DOFs
end up coupled and the factorization is a blocked dense Cholesky of that block.
"""

from __future__ import annotations

import warp as wp
from phost import Impl, ImplKind, MetricSpec, ParamSpec, Registry
from phost.stages import effective_solve as es
from phost.stages import newton_linear as nl

from . import _kernels as k
from .workspace import MAX_NV, CoupledCholesky

SCAN = f"{nl.NAME}/coupled"
STRUCTURAL = f"{nl.NAME}/coupled-structural"
EFFECTIVE = f"{es.NAME}/coupled"


class CoupledRuntime:
    def __init__(self, host, params, *, structural):
        facts = host.facts()
        self.structural = structural
        self.chol = CoupledCholesky(
            facts.nv, facts.world_count, facts.extra["device"], tile_size=params["tile_size"],
            max_coupled=params["max_coupled"], verify_structure=params["verify_structure"],
            rescan_above=None if params["rescan_above"] == 0 else params["rescan_above"])

    def solve(self, system: nl.NewtonSystem) -> None:
        self.chol.enqueue(system.h, system.grad, system.active, system.refactor, system.search,
                          system.search_dot, system.newton_decrement,
                          hint=system.structure_hint if self.structural else None, fallback=system.fallback)

    def metrics(self):
        return self.chol.metrics()

    def reset_metrics(self):
        self.chol.reset_metrics()


class EffectiveRuntime:
    """``effective.solve`` through the coupled-block Cholesky: every world, every call, refactored each time."""

    def __init__(self, host, params):
        facts = host.facts()
        device = facts.extra["device"]
        n, nv = facts.world_count, facts.nv
        self.n, self.nv = n, nv
        self.chol = CoupledCholesky(nv, n, device, tile_size=params["tile_size"], max_coupled=MAX_NV)
        self.active = wp.ones(n, dtype=wp.int32, device=device)
        self.refactor = wp.ones(n, dtype=wp.int32, device=device)
        self.search = wp.zeros((n, nv), dtype=wp.float32, device=device)
        self.search_dot = wp.zeros(n, dtype=wp.float32, device=device)
        self.decrement = wp.zeros(n, dtype=wp.float32, device=device)

    def solve(self, system: es.EffectiveSystem) -> None:
        self.chol.enqueue(system.h, system.rhs, self.active, self.refactor, self.search, self.search_dot, self.decrement)
        wp.launch(k._negate, dim=(self.n, self.nv), inputs=[self.search], outputs=[system.qacc], device=self.chol.device)

    def metrics(self):
        m = self.chol.metrics()
        return dict(status=m["status"], max_relative_residual=m["max_relative_residual"], coupled_last_max=m["coupled_last_max"])

    def reset_metrics(self):
        self.chol.reset_metrics()


def _applicable_effective(facts):
    if facts.device_kind != "cuda":
        return "needs a CUDA device"
    if "device" not in facts.extra:
        return "the host did not report its device in facts.extra['device']"
    blocker = facts.extra.get("effective_blocker")
    if blocker is None:
        return "the host did not say whether the effective-metric solve can be replaced (facts.extra['effective_blocker'])"
    if blocker:
        return blocker
    if not 1 <= facts.nv <= MAX_NV:
        return f"nv={facts.nv} is outside 1..{MAX_NV}"
    return None


def _applicable(facts):
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
    ParamSpec("tile_size", "int", 32, "tile edge of the dense block factorization", choices=(16, 32)),
)
_EFFECTIVE_FIT = ("Exact, and only worth installing where the host's own iteration is slow. Measured here (mujoco_warp 3.17.0, "
                  "RTX PRO 2000 laptop GPU) on the lab's 12x12 membrane, nv=330: slower than the host at every time step and "
                  "stiffness tried, about 0.57x the step rate at 1, 8 and 32 worlds, because the host's iteration converges in "
                  "a few steps (0.11-0.79 ms) while one factorization of the 300-DOF coupled block takes about 1.3 ms. "
                  "Dense storage is worlds x nv x nv, so nv is limited to 512.")


def _effective_advisory(facts):
    return (f"nv={facts.nv}: time this against the host's own solve before keeping it. On the lab's 12x12 membrane (nv=330) "
            f"the host's iteration took 0.11-0.79 ms and this took about 1.4 ms, so the step got slower. It can only win "
            f"where the host's iteration is slow.")


def impls() -> list[Impl]:
    return [
        Impl(id=SCAN, stage=nl.NAME, stage_version=nl.CONTRACT.version, version="0.1", title="Coupled-block Cholesky (scan)",
             kind=ImplKind.BUILTIN, doc="Finds the coupled DOFs by scanning the upper triangle of H.", params=_PARAMS,
             fit=_FIT, metrics=_EXTRA_METRICS, extensions_used=("fallback",), applicable=_applicable, advisory=_advisory,
             factory=lambda host, params: CoupledRuntime(host, params, structural=False),
             provenance=dict(package="phost-coupledchol")),
        Impl(id=STRUCTURAL, stage=nl.NAME, stage_version=nl.CONTRACT.version, version="0.1",
             title="Coupled-block Cholesky (structure hint)", kind=ImplKind.BUILTIN,
             doc="Takes the coupled DOFs from the host's structure hint instead of scanning H.", params=_PARAMS,
             fit=_FIT, metrics=_EXTRA_METRICS, extensions_used=("structure_hint", "fallback"), applicable=_applicable,
             advisory=_advisory, factory=lambda host, params: CoupledRuntime(host, params, structural=True),
             provenance=dict(package="phost-coupledchol")),
        Impl(id=EFFECTIVE, stage=es.NAME, stage_version=es.CONTRACT.version, version="0.1",
             title="Direct solve of the effective metric (coupled-block Cholesky)", kind=ImplKind.BUILTIN,
             doc="Factors the assembled effective metric directly instead of iterating on it.", params=_EFFECTIVE_PARAMS,
             fit=_EFFECTIVE_FIT, metrics=(MetricSpec("coupled_last_max", "dof", "none", "gauge",
                                                     "largest coupled set in the most recent call"),),
             applicable=_applicable_effective, advisory=_effective_advisory,
             factory=lambda host, params: EffectiveRuntime(host, params),
             provenance=dict(package="phost-coupledchol")),
    ]


def register(registry: Registry) -> None:
    for contract in (nl.CONTRACT, es.CONTRACT):      # adding the same contract again is a no-op
        registry.add_contract(contract)
    for impl in impls():
        registry.add_impl(impl)
