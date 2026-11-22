"""Register the coupled-block Cholesky as implementations of phost's ``newton.linear_solve`` stage.

    import phost, phost_coupledchol.phost_impl
    phost_coupledchol.phost_impl.register(phost.default)

Two implementations: ``coupled`` finds the coupled DOFs by scanning H (always exact, no host
assumptions) and ``coupled-structural`` takes them from the host's structure hint (cheaper at high
world counts, relies on how the host assembles H, so it needs that guard to pass).
"""

from __future__ import annotations

from phost import Impl, ImplKind, MetricSpec, ParamSpec, Registry
from phost.stages import newton_linear as nl

from .workspace import MAX_NV, CoupledCholesky

SCAN = f"{nl.NAME}/coupled"
STRUCTURAL = f"{nl.NAME}/coupled-structural"


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
    ]


def register(registry: Registry) -> None:
    for impl in impls():
        registry.add_impl(impl)
