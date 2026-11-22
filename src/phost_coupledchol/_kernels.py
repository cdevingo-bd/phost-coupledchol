"""Device kernels of the coupled-block Cholesky. See ``workspace.CoupledCholesky`` for the algorithm.

Numerics are carried over unchanged from the coupled workspace of the author's earlier deform3d project (retired). Two interface changes: the
fallback request is an int32 world mask, and ``_prepare`` derives the refactor mask from the
caller's ``active``/``refactor`` ports instead of from a host's skip counters.
"""

from functools import lru_cache

import warp as wp


@wp.kernel(enable_backward=False)
def _prepare(active: wp.array(dtype=wp.int32), refactor_in: wp.array(dtype=wp.int32),
             initialized: wp.array(dtype=wp.int32), refactor: wp.array(dtype=wp.int32)):
    # A world is refactored when the host says H changed OR this workspace holds no valid factor
    # for it (first call, or its last solve was handed to the host).
    w = wp.tid()
    on = active[w] != 0
    refactor[w] = int(on and (refactor_in[w] != 0 or initialized[w] == 0))
    if on:
        initialized[w] = 1


@wp.kernel(enable_backward=False)
def _clear_flags(refactor: wp.array(dtype=wp.int32), flag: wp.array2d(dtype=wp.int32)):
    w, i = wp.tid()
    if refactor[w] != 0:
        flag[w, i] = 0


@wp.kernel(enable_backward=False)
def _mark(nv: int, chunk: int, refactor: wp.array(dtype=wp.int32), h: wp.array3d(dtype=wp.float32),
          flag: wp.array2d(dtype=wp.int32)):
    # H is upper-triangular in the host's buffer. Both endpoints of an off-diagonal entry are
    # coupled; concurrent writers all store 1, so the race is benign. A NaN compares != 0 and is
    # flagged, which the finiteness gate then reports. Consecutive threads take consecutive
    # column chunks of one row, so a warp reads contiguous memory.
    w, i, block = wp.tid()
    if refactor[w] == 0:
        return
    first = block * chunk
    if first + chunk <= i + 1:
        return
    for q in range(chunk):
        j = first + q
        if j > i and j < nv:
            if h[w, i, j] != 0.0:
                flag[w, i] = 1
                flag[w, j] = 1


@wp.kernel(enable_backward=False)
def _compare_structure(refactor: wp.array(dtype=wp.int32), flag: wp.array2d(dtype=wp.int32),
                       scanned: wp.array2d(dtype=wp.int32), status: wp.array(dtype=wp.int32)):
    # A DOF the Hessian scan found coupled but the structure did not: the structural set is wrong.
    w, i = wp.tid()
    if refactor[w] != 0 and scanned[w, i] != 0 and flag[w, i] == 0:
        status[w] = status[w] | 4


LANES = 32


@wp.kernel(enable_backward=False)
def _count_chunks(nv: int, lanes: int, refactor: wp.array(dtype=wp.int32), flag: wp.array2d(dtype=wp.int32),
                  counts: wp.array2d(dtype=wp.int32)):
    # Lane g owns the contiguous DOFs [g * chunk, (g + 1) * chunk).
    w, g = wp.tid()
    if refactor[w] == 0:
        return
    chunk = (nv + lanes - 1) // lanes
    n = int(0)
    for q in range(chunk):
        i = g * chunk + q
        if i < nv:
            if flag[w, i] != 0:
                n += 1
    counts[w, g] = n


@wp.kernel(enable_backward=False)
def _decide_rescan(nv: int, lanes: int, limit: int, refactor: wp.array(dtype=wp.int32),
                   counts: wp.array2d(dtype=wp.int32), flag: wp.array2d(dtype=wp.int32),
                   rescan: wp.array(dtype=wp.int32), rescans: wp.array(dtype=wp.int32)):
    # A structural set above ``limit`` is probably loose (friction rows that cancel exactly are
    # flagged), so such a world discards it and takes the exact scan instead. Each lane clears its
    # own chunk of the flags so the scan starts from nothing.
    w, g = wp.tid()
    total = int(0)
    if refactor[w] != 0:
        for k in range(lanes):
            total += counts[w, k]
    over = total > limit
    if over:
        chunk = (nv + lanes - 1) // lanes
        for q in range(chunk):
            i = g * chunk + q
            if i < nv:
                flag[w, i] = 0
    if g == 0:
        rescan[w] = int(over)
        if over:
            rescans[w] = rescans[w] + 1


@wp.kernel(enable_backward=False)
def _compact(nv: int, lanes: int, cmax: int, tile: int, active: wp.array(dtype=wp.int32),
             refactor: wp.array(dtype=wp.int32), flag: wp.array2d(dtype=wp.int32),
             counts: wp.array2d(dtype=wp.int32),
             idx: wp.array2d(dtype=wp.int32), pos: wp.array2d(dtype=wp.int32),
             count: wp.array(dtype=wp.int32), cpad: wp.array(dtype=wp.int32),
             overflow: wp.array(dtype=wp.int32), initialized: wp.array(dtype=wp.int32),
             fallbacks: wp.array(dtype=wp.int32), fac: wp.array(dtype=wp.int32),
             ok: wp.array(dtype=wp.int32), fallback_mask: wp.array(dtype=wp.int32)):
    # Lane g scatters its chunk to the positions after the lower lanes' counts; lane 0 also does
    # the per-world bookkeeping. Every lane derives the same total from the chunk counts.
    w, g = wp.tid()
    total = int(0)
    first = int(0)
    if refactor[w] != 0:
        for k in range(lanes):
            n = counts[w, k]
            if k < g:
                first += n
            total += n
        chunk = (nv + lanes - 1) // lanes
        c = first
        for q in range(chunk):
            i = g * chunk + q
            if i < nv:
                if flag[w, i] != 0:
                    if c < cmax:
                        idx[w, c] = i
                        pos[w, i] = c
                    c += 1
    if g != 0:
        return
    fac[w] = 0
    if refactor[w] != 0:
        count[w] = total
        cpad[w] = (total + tile - 1) // tile * tile
        if total > cmax:
            overflow[w] = 1
            initialized[w] = 0  # the host path solves this world, so no factor of ours is valid
            fallbacks[w] = fallbacks[w] + 1
        else:
            overflow[w] = 0
            fac[w] = 1
    solved_here = active[w] != 0 and overflow[w] == 0
    ok[w] = int(solved_here)
    fallback_mask[w] = int(active[w] != 0 and overflow[w] != 0)


@wp.kernel(enable_backward=False)
def _pack(fac: wp.array(dtype=wp.int32), cpad: wp.array(dtype=wp.int32),
          count: wp.array(dtype=wp.int32), idx: wp.array2d(dtype=wp.int32),
          h: wp.array3d(dtype=wp.float32), work: wp.array3d(dtype=wp.float32)):
    # One thread per row of the compacted lower triangle; the factorization never reads the rest.
    w, a = wp.tid()
    if fac[w] == 0 or a >= cpad[w]:
        return
    c = count[w]
    for b in range(a + 1):
        value = float(0.0)
        if a < c and b < c:
            ia = idx[w, a]
            ib = idx[w, b]
            value = h[w, wp.min(ia, ib), wp.max(ia, ib)]
        elif a == b:
            value = 1.0  # identity padding keeps the last tile positive definite
        work[w, a, b] = value


@wp.kernel(enable_backward=False)
def _gather_rhs(ok: wp.array(dtype=wp.int32), count: wp.array(dtype=wp.int32),
                idx: wp.array2d(dtype=wp.int32), g: wp.array2d(dtype=wp.float32),
                x: wp.array2d(dtype=wp.float32)):
    w, a = wp.tid()
    if ok[w] == 0:
        return
    value = float(0.0)
    if a < count[w]:
        value = g[w, idx[w, a]]
    x[w, a] = value


@lru_cache(None)
def _kernels(nv, cmax, tile):
    options = {'enable_mathdx_solver': False, 'enable_mathdx_gemm': False}

    @wp.kernel(module='unique', enable_backward=False, module_options=options)
    def solve(fac: wp.array(dtype=wp.int32), ok: wp.array(dtype=wp.int32), cpad: wp.array(dtype=wp.int32),
              a: wp.array3d(dtype=wp.float32), solution: wp.array3d(dtype=wp.float32)):
        # One block per world factors (when ``fac``) and then substitutes (when ``ok``), looping
        # over only the ``cpad / tile`` tiles that world needs. A world with a cached factor has
        # ``ok`` without ``fac``. Tile loads and stores synchronize the block, so each step reads
        # what the previous one wrote.
        w = wp.tid()
        blocks = cpad[w] // tile
        if blocks == 0 or (fac[w] == 0 and ok[w] == 0):
            return
        if fac[w] != 0:
            for kb in range(blocks):
                k = kb * tile
                lower = wp.tile_load(a[w], shape=(tile, tile), offset=(k, k), storage='shared', bounds_check=False)
                wp.tile_cholesky_inplace(lower, fill_mode='lower')
                wp.tile_store(a[w], lower, offset=(k, k), bounds_check=False)
                for bi in range(kb + 1, blocks):
                    i = bi * tile
                    aik = wp.tile_load(a[w], shape=(tile, tile), offset=(i, k), storage='shared',
                                       bounds_check=False)
                    wp.tile_lower_solve_inplace(lower, wp.tile_transpose(aik))
                    wp.tile_store(a[w], aik, offset=(i, k), bounds_check=False)
                for bi in range(kb + 1, blocks):
                    i = bi * tile
                    li = wp.tile_load(a[w], shape=(tile, tile), offset=(i, k), storage='shared',
                                      bounds_check=False)
                    for bj in range(kb + 1, bi + 1):
                        j = bj * tile
                        lj = wp.tile_load(a[w], shape=(tile, tile), offset=(j, k), storage='shared',
                                          bounds_check=False)
                        target = wp.tile_load(a[w], shape=(tile, tile), offset=(i, j), storage='shared',
                                              bounds_check=False)
                        wp.tile_matmul(li, wp.tile_transpose(lj), target, alpha=wp.float32(-1.0))
                        wp.tile_store(a[w], target, offset=(i, j), bounds_check=False)
        if ok[w] != 0:
            x = wp.tile_load(solution[w], shape=(cmax, 1), offset=(0, 0), storage='shared', bounds_check=False)
            for block in range(blocks):
                k = block * tile
                part = wp.tile_view(x, shape=(tile, 1), offset=(k, 0))
                for previous in range(block):
                    j = previous * tile
                    lower = wp.tile_load(a[w], shape=(tile, tile), offset=(k, j), storage='shared',
                                         bounds_check=False)
                    y = wp.tile_view(x, shape=(tile, 1), offset=(j, 0))
                    wp.tile_matmul(lower, y, part, alpha=wp.float32(-1.0))
                lower = wp.tile_load(a[w], shape=(tile, tile), offset=(k, k), storage='shared',
                                     bounds_check=False)
                wp.tile_lower_solve_inplace(lower, part)
            for reverse in range(blocks):
                block = blocks - 1 - reverse
                k = block * tile
                part = wp.tile_view(x, shape=(tile, 1), offset=(k, 0))
                for following in range(block + 1, blocks):
                    j = following * tile
                    lower = wp.tile_load(a[w], shape=(tile, tile), offset=(j, k), storage='shared',
                                         bounds_check=False)
                    y = wp.tile_view(x, shape=(tile, 1), offset=(j, 0))
                    wp.tile_matmul(wp.tile_transpose(lower), y, part, alpha=wp.float32(-1.0))
                lower = wp.tile_load(a[w], shape=(tile, tile), offset=(k, k), storage='shared',
                                     bounds_check=False)
                wp.tile_upper_solve_inplace(wp.tile_transpose(lower), part)
            wp.tile_store(solution[w], x, bounds_check=False)

    @wp.kernel(module='unique', enable_backward=False)
    def finish(ok: wp.array(dtype=wp.int32), h: wp.array3d(dtype=wp.float32),
               g: wp.array2d(dtype=wp.float32), flag: wp.array2d(dtype=wp.int32),
               pos: wp.array2d(dtype=wp.int32), idx: wp.array2d(dtype=wp.int32),
               count: wp.array(dtype=wp.int32), x: wp.array2d(dtype=wp.float32),
               factor: wp.array3d(dtype=wp.float32), out: wp.array2d(dtype=wp.float32),
               norm: wp.array(dtype=wp.float32), decrement: wp.array(dtype=wp.float32),
               status: wp.array(dtype=wp.int32), max_residual: wp.array(dtype=wp.float32)):
        w, row = wp.tid()
        if ok[w] == 0:
            return
        value, b, ax = wp.float64(0.0), wp.float64(0.0), wp.float64(0.0)
        bad = int(0)
        if row < nv:
            b = wp.float64(g[w, row])
            diagonal_entry = h[w, row, row]
            if flag[w, row] == 0:
                # No off-diagonal entry anywhere in this row/column: exact scalar solve.
                solved = g[w, row] / diagonal_entry
                value = wp.float64(wp.float32(solved))
                ax = wp.float64(diagonal_entry) * value
                if not wp.isfinite(value) or not diagonal_entry > 0.0:
                    bad = 1
            else:
                p = pos[w, row]
                value = wp.float64(wp.float32(x[w, p]))
                for column in range(count[w]):
                    other = idx[w, column]
                    ax += wp.float64(h[w, wp.min(row, other), wp.max(row, other)]) * wp.float64(
                        wp.float32(x[w, column]))
                pivot = factor[w, p, p]
                if not wp.isfinite(value) or not wp.isfinite(pivot) or pivot <= 0.0:
                    bad = 1
            out[w, row] = -wp.float32(value)
        error = b - ax
        rr = wp.tile_reduce(wp.add, wp.tile(error * error))[0]
        bb = wp.tile_reduce(wp.add, wp.tile(b * b))[0]
        xx = wp.tile_reduce(wp.add, wp.tile(value * value))[0]
        bx = wp.tile_reduce(wp.add, wp.tile(b * value))[0]
        invalid = wp.tile_reduce(wp.add, wp.tile(bad))[0]
        relative = wp.float64(0.0)
        if bb > 0.0:
            relative = wp.sqrt(rr / bb)
        elif rr > 0.0:
            relative = wp.float64(wp.inf)
        if row == 0:
            code = int(0)
            if invalid > 0 or not wp.isfinite(relative) or not wp.isfinite(wp.float32(xx)) or not wp.isfinite(
                    wp.float32(bx)):
                code = 1
                relative = wp.float64(wp.inf)
            elif relative > wp.float64(1.0e-4):
                code = 2
            norm[w] = wp.float32(xx)
            decrement[w] = wp.float32(bx)
            status[w] = status[w] | code
            max_residual[w] = wp.max(max_residual[w], wp.float32(relative))

    return solve, finish
