"""Coupled-block Cholesky for batched Newton systems.

Solve ``H x = g`` per world where ``H`` is a diagonal plus a small dense block: most DOFs have no
off-diagonal entry, and only the "coupled" DOFs (those with one) need a real factorization.

1. Find the coupled DOFs of each world: scan the upper triangle of H for off-diagonal nonzeros
   (always exact), or take them from a ``hint`` the caller supplies (cheaper; may be a superset).
2. Compact the coupled DOFs into a dense ``c x c`` matrix and factor it with a tile Cholesky,
   launching work only for the tiles each world needs.
3. Every other DOF is solved by ``x_i = g_i / H_ii``.

This is exact, not an approximation: the cost per world is ``O(c^3)`` instead of ``O(nv^3)``. A float64
residual gate checks every solve. A world whose coupled set exceeds ``max_coupled`` is handed back to
the caller through ``fallback``.

The class knows nothing about any simulator: it takes arrays and two optional callables.

Array contract (all contiguous, on one CUDA device, ``nworld`` leading):
    h        float32 [nworld, >=nv, >=nv]  upper triangle valid, lower undefined
    grad     float32 [nworld, >=nv]
    active   int32   [nworld]  1: solve this world, 0: leave its outputs untouched
    refactor int32   [nworld]  1: H changed since this world's last solve; 0: a cached factor may be reused
    search, search_dot, newton_decrement: outputs, see the ``newton.linear_solve`` stage contract.

Status bits (sticky until ``reset_metrics``): 1 non-finite or non-positive pivot, 2 float64 relative
residual above 1e-4, 4 the hint missed a coupled DOF (only with ``verify_structure``).
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import warp as wp

from . import _kernels as k

MAX_NV = 512


class CoupledCholesky:
    """Per-world coupled-block Cholesky. Keep alive for as long as captured graphs are replayed."""

    def __init__(self, nv, nworld, device, *, tile_size=32, max_coupled=128, mark_chunk=8,
                 verify_structure=False, rescan_above=None):
        if tile_size not in (16, 32):
            raise ValueError("tile_size must be 16 or 32")
        if isinstance(max_coupled, bool) or int(max_coupled) < 1:
            raise ValueError("max_coupled must be a positive integer")
        if isinstance(nv, bool) or not 1 <= int(nv) <= MAX_NV:
            raise ValueError(f"nv must be in 1..{MAX_NV}")
        if isinstance(nworld, bool) or int(nworld) < 1:
            raise ValueError("nworld must be a positive integer")
        self.nv, self.nworld, self.device = int(nv), int(nworld), wp.get_device(device)
        if not self.device.is_cuda:
            raise ValueError("requires a CUDA device")
        self.tile_size = tile_size
        self.mark_chunk = int(mark_chunk)
        size = (self.nv + tile_size - 1) // tile_size * tile_size
        self.max_coupled = min(size, (int(max_coupled) + tile_size - 1) // tile_size * tile_size)
        self.can_overflow = self.max_coupled < self.nv
        self.verify_structure = bool(verify_structure)
        self.rescan_above = tile_size if rescan_above is None else int(rescan_above)
        n, c, i32, dev = self.nworld, self.max_coupled, wp.int32, self.device
        self.flag = wp.zeros((n, self.nv), dtype=i32, device=dev)
        self.idx = wp.zeros((n, c), dtype=i32, device=dev)
        self.pos = wp.zeros((n, self.nv), dtype=i32, device=dev)
        self.counts = wp.zeros((n, k.LANES), dtype=i32, device=dev)
        self.rescan = wp.zeros(n, dtype=i32, device=dev)
        self.rescans = wp.zeros(n, dtype=i32, device=dev)
        self.scanned = wp.zeros((n, self.nv), dtype=i32, device=dev) if self.verify_structure else None
        self.coupled = wp.zeros(n, dtype=i32, device=dev)
        self.cpad = wp.zeros(n, dtype=i32, device=dev)
        self.overflow = wp.zeros(n, dtype=i32, device=dev)
        self.fallbacks = wp.zeros(n, dtype=i32, device=dev)
        self.refactor = wp.zeros(n, dtype=i32, device=dev)
        self.initialized = wp.zeros(n, dtype=i32, device=dev)
        self.fac = wp.zeros(n, dtype=i32, device=dev)
        self.ok = wp.zeros(n, dtype=i32, device=dev)
        self.fallback_mask = wp.zeros(n, dtype=i32, device=dev)
        self.work = wp.zeros((n, c, c), dtype=wp.float32, device=dev)
        self.x = wp.zeros((n, c), dtype=wp.float32, device=dev)
        self.x_tiles = self.x.reshape((n, c, 1))
        self.status = wp.zeros(n, dtype=i32, device=dev)
        self.max_relative_residual = wp.zeros(n, dtype=wp.float32, device=dev)

    # -- validation ------------------------------------------------------------------------------
    def _check(self, **arrays):
        spec = dict(h=(wp.float32, 3), grad=(wp.float32, 2), active=(wp.int32, 1), refactor=(wp.int32, 1),
                    search=(wp.float32, 2), search_dot=(wp.float32, 1), newton_decrement=(wp.float32, 1))
        for name, (dtype, ndim) in spec.items():
            a = arrays[name]
            if (not isinstance(a, wp.array) or a.dtype != dtype or a.ndim != ndim or a.device != self.device
                    or not a.is_contiguous or a.shape[0] != self.nworld):
                raise ValueError(f"{name} must be a contiguous {dtype.__name__} array of rank {ndim} with "
                                 f"{self.nworld} worlds on {self.device}")
        h, grad, search = arrays["h"], arrays["grad"], arrays["search"]
        if min(h.shape[1:]) < self.nv or grad.shape[1] < self.nv or search.shape[1] != self.nv:
            raise ValueError("h/grad/search dimensions do not match nv")

    # -- the solve -------------------------------------------------------------------------------
    def enqueue(self, h, grad, active, refactor, search, search_dot, newton_decrement, *,
                hint: Callable | None = None, fallback: Callable | None = None):
        """Launch one solve. ``hint(flags, refactor)`` and ``fallback(solve_mask)`` are described in the
        ``newton.linear_solve`` stage contract. Launches only; nothing is read back to the host."""
        self._check(h=h, grad=grad, active=active, refactor=refactor, search=search, search_dot=search_dot,
                    newton_decrement=newton_decrement)
        if self.verify_structure and hint is None:
            raise ValueError("verify_structure needs a structure hint")
        if self.can_overflow and fallback is None:
            raise ValueError("max_coupled < nv: worlds can overflow, so a fallback is required")
        dev = self.device
        n, nv, c, t = self.nworld, self.nv, self.max_coupled, self.tile_size
        chunks = (nv + self.mark_chunk - 1) // self.mark_chunk
        wp.launch(k._prepare, dim=n, inputs=[active, refactor, self.initialized], outputs=[self.refactor], device=dev)
        if hint is not None:
            hint(self.flag, self.refactor)
            if self.verify_structure:
                wp.launch(k._clear_flags, dim=(n, nv), inputs=[self.refactor, self.scanned], device=dev)
                wp.launch(k._mark, dim=(n, nv, chunks), inputs=[nv, self.mark_chunk, self.refactor, h, self.scanned],
                          device=dev)
                wp.launch(k._compare_structure, dim=(n, nv), inputs=[self.refactor, self.flag, self.scanned],
                          outputs=[self.status], device=dev)
            if self.rescan_above < nv:
                wp.launch(k._count_chunks, dim=(n, k.LANES), inputs=[nv, k.LANES, self.refactor, self.flag],
                          outputs=[self.counts], device=dev)
                wp.launch(k._decide_rescan, dim=(n, k.LANES),
                          inputs=[nv, k.LANES, self.rescan_above, self.refactor, self.counts, self.flag],
                          outputs=[self.rescan, self.rescans], device=dev)
                wp.launch(k._mark, dim=(n, nv, chunks), inputs=[nv, self.mark_chunk, self.rescan, h, self.flag],
                          device=dev)
        else:
            wp.launch(k._clear_flags, dim=(n, nv), inputs=[self.refactor, self.flag], device=dev)
            wp.launch(k._mark, dim=(n, nv, chunks), inputs=[nv, self.mark_chunk, self.refactor, h, self.flag],
                      device=dev)
        wp.launch(k._count_chunks, dim=(n, k.LANES), inputs=[nv, k.LANES, self.refactor, self.flag],
                  outputs=[self.counts], device=dev)
        wp.launch(k._compact, dim=(n, k.LANES),
                  inputs=[nv, k.LANES, c, t, active, self.refactor, self.flag, self.counts],
                  outputs=[self.idx, self.pos, self.coupled, self.cpad, self.overflow, self.initialized,
                           self.fallbacks, self.fac, self.ok, self.fallback_mask], device=dev)
        wp.launch(k._pack, dim=(n, c), inputs=[self.fac, self.cpad, self.coupled, self.idx, h],
                  outputs=[self.work], device=dev)
        solve, finish = k._kernels(nv, c, t)
        wp.launch(k._gather_rhs, dim=(n, c), inputs=[self.ok, self.coupled, self.idx, grad],
                  outputs=[self.x], device=dev)
        wp.launch_tiled(solve, dim=n, inputs=[self.fac, self.ok, self.cpad, self.work, self.x_tiles],
                        block_dim=64, device=dev)
        wp.launch_tiled(
            finish, dim=n,
            inputs=[self.ok, h, grad, self.flag, self.pos, self.idx, self.coupled, self.x, self.work, search,
                    search_dot, newton_decrement, self.status, self.max_relative_residual],
            block_dim=max(32, 1 << (nv - 1).bit_length()), device=dev)
        if self.can_overflow:
            fallback(self.fallback_mask)

    # -- metrics ---------------------------------------------------------------------------------
    def metrics(self) -> dict:
        """Read back the per-world counters. Synchronizes the device."""
        wp.synchronize_device(self.device)
        return dict(
            status=int(np.bitwise_or.reduce(self.status.numpy())),
            max_relative_residual=float(self.max_relative_residual.numpy().max()),
            fallbacks=int(self.fallbacks.numpy().sum()),
            coupled_last_max=int(self.coupled.numpy().max()),
            rescans=int(self.rescans.numpy().sum()))

    def reset_metrics(self) -> None:
        for a in (self.status, self.max_relative_residual, self.fallbacks, self.rescans):
            a.zero_()
