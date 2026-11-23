"""Coupled-block Cholesky for batched symmetric systems: a Warp backend and a NumPy one. Knows nothing about any simulator.

Importing the package imports neither backend; ``CoupledCholesky`` (Warp) and ``NumpyCoupledCholesky`` load on first use.
"""

from .limits import MAX_NV

__version__ = "0.1.0.dev0"
__all__ = ["CoupledCholesky", "NumpyCoupledCholesky", "MAX_NV"]


def __getattr__(name):
    if name == "CoupledCholesky":
        from .workspace import CoupledCholesky
        return CoupledCholesky
    if name == "NumpyCoupledCholesky":
        from .numpy_solve import NumpyCoupledCholesky
        return NumpyCoupledCholesky
    raise AttributeError(f"module 'phost_coupledchol' has no attribute {name!r}")
