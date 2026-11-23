"""The coupled-block Cholesky on NumPy arrays: the same algorithm as the Warp backend, for a host whose ports are NumPy.

Solve ``H x = g`` per world where ``H`` is a diagonal plus a small dense block. A DOF with an off-diagonal entry in the upper
triangle is *coupled*; the coupled DOFs are compacted into a ``c x c`` block and factored, and every other DOF is ``g_i / H_ii``.
It is exact, and it is float64 inside: it exists so the algorithm runs where there is no CUDA device (a CPU simulator, a
reference to read, a test) and so the same stage check judges both backends. It is not written for speed: a world is solved
by a Python loop with dense NumPy calls.

Array contract: ``h`` ``[nworld, >=nv, >=nv]`` upper triangle valid, lower undefined (it is never read, so NaN is fine);
``rhs`` ``[nworld, >=nv]``; ``out`` ``[nworld, nv]``, written for every world.

Status bits (sticky until ``reset_metrics``, the Warp backend's): 1 non-finite or non-positive pivot, 2 relative residual above 1e-4,
measured in float64 on the answer as written (so on its float32 rounding too).
"""

from __future__ import annotations

import numpy as np

GATE = 1e-4


class NumpyCoupledCholesky:
    def __init__(self, nv: int, nworld: int):
        if isinstance(nv, bool) or int(nv) < 1:
            raise ValueError("nv must be a positive integer")
        if isinstance(nworld, bool) or int(nworld) < 1:
            raise ValueError("nworld must be a positive integer")
        self.nv, self.nworld = int(nv), int(nworld)
        self.reset_metrics()

    def reset_metrics(self) -> None:
        self.status, self.max_relative_residual, self.coupled_last_max = 0, 0.0, 0

    def metrics(self) -> dict:
        return dict(status=self.status, max_relative_residual=self.max_relative_residual, coupled_last_max=self.coupled_last_max)

    def solve(self, h, rhs, out) -> None:
        nv = self.nv
        widest = 0
        for world in range(self.nworld):
            matrix = np.asarray(h[world, :nv, :nv], dtype=np.float64)
            g = np.asarray(rhs[world, :nv], dtype=np.float64)
            upper = np.triu(matrix, 1)
            off = upper != 0
            coupled = np.flatnonzero(off.any(axis=0) | off.any(axis=1))        # a NaN above the diagonal counts: it fails below
            widest = max(widest, coupled.size)
            diagonal = np.diagonal(matrix)
            if not (np.isfinite(diagonal).all() and (diagonal > 0).all()):
                self.status |= 1
            with np.errstate(divide="ignore", invalid="ignore"):
                x = g / diagonal                                              # every DOF with no coupling
            if coupled.size:
                block = matrix[np.ix_(coupled, coupled)]
                block = np.triu(block) + np.triu(block, 1).T
                try:
                    factor = np.linalg.cholesky(block)
                    x[coupled] = np.linalg.solve(factor.T, np.linalg.solve(factor, g[coupled]))
                except np.linalg.LinAlgError:
                    self.status |= 1
                    x[coupled] = np.nan
            out[world, :nv] = x
            written = np.asarray(out[world, :nv], dtype=np.float64)
            if not np.isfinite(written).all():
                self.status |= 1
                continue
            symmetric = np.triu(matrix) + upper.T
            scale = np.linalg.norm(g)
            residual = np.linalg.norm(symmetric @ written - g) / (scale if scale > 0 else 1.0)
            self.max_relative_residual = max(self.max_relative_residual, float(residual))
            if residual > GATE:
                self.status |= 2
        self.coupled_last_max = int(widest)


class EffectiveRuntime:
    """``effective.solve`` through the coupled-block Cholesky on NumPy arrays: every world, every call."""

    def __init__(self, host, params):
        facts = host.facts()
        self.chol = NumpyCoupledCholesky(facts.nv, facts.world_count)

    def solve(self, system) -> None:
        self.chol.solve(system.h, system.rhs, system.qacc)

    def metrics(self) -> dict:
        return self.chol.metrics()

    def reset_metrics(self) -> None:
        self.chol.reset_metrics()
