# phost-coupledchol

Exact coupled-block Cholesky for batched Newton systems whose Hessian is a diagonal plus a small
dense block, on the GPU with Warp. It factors only the DOFs that have an off-diagonal entry, so the
cost is about c^3 instead of nv^3.

No simulator is imported. `phost_coupledchol.phost_impl` registers it as an implementation of the
`newton.linear_solve` stage of [phost](../phost).

MIT licensed. Status: pre-release; being extracted from a working implementation.
