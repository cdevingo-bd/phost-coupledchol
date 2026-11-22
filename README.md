# phost-coupledchol

Coupled-block Cholesky for the Newton linear solve in MuJoCo Warp, packaged as a [phost](https://github.com/cdevingo-bd/phost)
implementation. It is an exact solver for batched systems whose Hessian is a diagonal plus a small dense block (Warp, on the
GPU): it factors only the DOFs that have an off-diagonal entry, so the cost is about c^3 instead of nv^3.

No simulator is imported. `phost_coupledchol.phost_impl` registers it as an implementation of the `newton.linear_solve`
stage and, with the same factorization, as `effective.solve/coupled` for the `effective.solve` stage. The second is
measured slower than the host's own solve on the lab's flex scenes; the numbers are below.

## The effective-metric solve, and a measured slowdown

Under MuJoCo Warp's discrete integrator the host solves `M_hat x = rhs` by conjugate gradients before its constraint
solver runs, where `M_hat` is the mass matrix plus a diagonal plus a sparse flex stiffness block. The phost adapter
assembles `M_hat` densely per world, and `effective.solve/coupled` factors it directly (nv up to 512; refused otherwise,
and refused when the model holds any term other than mass, diagonal and flex stiffness). It is exact: on the lab's
`trampoline12:discrete` the residual is 3e-7 and the result matches the host's own solve.

**It is slower than the host here, at every setting tried.** RTX PRO 2000 laptop GPU, mujoco_warp 3.17.0, a 12x12
membrane (nv=330, 300 coupled DOFs), median of graph replays, one world unless stated:

| | native effective solve | `effective.solve/coupled` | whole step, native | whole step, replaced |
| --- | --- | --- | --- | --- |
| dt 0.5 ms, E 1e5 (the lab scene) | 0.11-0.15 ms | 1.45 ms | 1.46-1.96 ms (three runs) | 2.79-2.94 ms |
| dt 2 ms, E 1e5 | 0.28 ms | 1.45 ms | 2.39 ms | 3.89 ms |
| dt 5 ms, E 1e6 | 0.62 ms | 1.45 ms | 2.07 ms | 2.85 ms |
| dt 10 ms, E 1e7 | 0.79 ms | 1.45 ms | 2.96 ms | 2.92 ms |
| lab scene, 8 worlds | 0.16 ms | 1.50 ms | 2.02 ms | 3.09 ms |
| lab scene, 32 worlds (whole step only) | | | 2.22 ms | 3.92 ms |

Why: the host's iteration converges in a few steps on these membranes, and the factorization itself is the cost. One
tiled Cholesky of the 300-DOF block is 1.26 ms of the 1.44 ms (tile 32; 1.86 ms at tile 16), and the mark, compact and
pack kernels around it add about 0.1 ms. So a dedicated dense workspace would not change the result; only a
cheaper factorization (a banded one after a reordering, for instance) could. At the stiffest setting tried the whole
step reaches parity, and that is already near the float32 limit: above a condition number of about 1e4 no float32 solver
meets the 1e-4 residual gate. The deform3d project reported a native iteration of 2.3 ms against 0.8 ms for its dense
solve on a different scene (a generated bag, nv=357). That is the regime this is for, and it was not available here to
re-measure. `phost-lab profile` shows whether a model is in it.
The session prints an advisory saying so whenever this implementation is selected.

MIT licensed. Status: pre-release; being extracted from a working implementation.
