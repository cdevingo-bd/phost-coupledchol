import numpy as np
import warp as wp

DEVICE = "cuda:0"


def cuda_available():
    return wp.is_cuda_available()


def make_batch(nworld, nv, nvpad, coupled, seed=0, scale=1.0):
    """SPD systems: a dense coupled block over the first `coupled` DOFs plus a positive diagonal.

    Only the upper triangle is meaningful; the lower triangle is filled with garbage on purpose.
    Returns (h float32 [w, nvpad, nvpad], grad float32 [w, nvpad], dense float64 [w, nv, nv]).
    """
    rng = np.random.default_rng(seed)
    h = np.full((nworld, nvpad, nvpad), 7.0, np.float32)
    grad = np.zeros((nworld, nvpad), np.float32)
    dense = np.zeros((nworld, nv, nv))
    for w in range(nworld):
        a = np.diag(rng.uniform(1.0, 3.0, nv) * scale)
        if coupled:
            b = rng.standard_normal((coupled, coupled))
            a[:coupled, :coupled] += b @ b.T / coupled + np.eye(coupled)
        a32 = a.astype(np.float32)
        h[w, :nv, :nv] = np.triu(a32) + np.tril(np.full((nv, nv), 7.0, np.float32), -1)
        dense[w] = np.triu(a32) + np.triu(a32, 1).T
        grad[w, :nv] = rng.standard_normal(nv)
    return h, grad, dense


class Arrays:
    """Device arrays for one enqueue call."""

    def __init__(self, h, grad, nv, active=None, refactor=None):
        nworld = h.shape[0]
        self.nv, self.nworld = nv, nworld
        self.h = wp.array(h, dtype=wp.float32, device=DEVICE)
        self.grad = wp.array(grad, dtype=wp.float32, device=DEVICE)
        self.active = wp.array(np.ones(nworld, np.int32) if active is None else np.asarray(active, np.int32),
                               dtype=wp.int32, device=DEVICE)
        self.refactor = wp.array(np.ones(nworld, np.int32) if refactor is None else np.asarray(refactor, np.int32),
                                 dtype=wp.int32, device=DEVICE)
        self.search = wp.full((nworld, nv), 123.0, dtype=wp.float32, device=DEVICE)
        self.search_dot = wp.full(nworld, 123.0, dtype=wp.float32, device=DEVICE)
        self.decrement = wp.full(nworld, 123.0, dtype=wp.float32, device=DEVICE)

    def args(self):
        return (self.h, self.grad, self.active, self.refactor, self.search, self.search_dot, self.decrement)

    def out(self):
        wp.synchronize()
        return self.search.numpy(), self.search_dot.numpy(), self.decrement.numpy()


def relative_residual(dense, grad, search, nv):
    """|H x - g| / |g| in float64 per world, with x = -search."""
    out = []
    for w in range(dense.shape[0]):
        x, g = -search[w].astype(np.float64), grad[w, :nv].astype(np.float64)
        out.append(np.linalg.norm(dense[w] @ x - g) / np.linalg.norm(g))
    return np.array(out)


def dense_from_upper(h, nv):
    out = np.zeros((h.shape[0], nv, nv))
    for w in range(h.shape[0]):
        u = np.triu(h[w, :nv, :nv].astype(np.float64))
        out[w] = u + np.triu(u, 1).T
    return out
