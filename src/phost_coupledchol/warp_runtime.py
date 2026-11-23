"""The Warp backend's runtimes for the coupled-block Cholesky: what ``phost_impl`` builds when a host's ports are Warp arrays.

Importing this module imports Warp, which is why ``phost_impl`` imports it only when it is asked for one.
"""

from __future__ import annotations

import warp as wp
from phost.stages import effective_solve as es
from phost.stages import newton_linear as nl

from . import _kernels as k
from .limits import MAX_NV
from .workspace import CoupledCholesky


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
