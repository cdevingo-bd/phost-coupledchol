"""phost's simulator-free host, with its numpy buffers bridged to Warp arrays around the runtime."""

import copy

import numpy as np
import warp as wp

from phost import HostInfo, ModelFacts, StageBinding
from phost.stages import newton_linear as nl
from phost.testing.fake import FakeHost, _make_dispatch as _numpy_dispatch

DEVICE = "cuda:0"


def _dev(a, dtype):
    return wp.array(np.ascontiguousarray(a), dtype=dtype, device=DEVICE)


class WarpFakeHost(FakeHost):
    info = HostInfo(name="fake-warp", version="1", build_id="fake-build-1", patched=False)

    def facts(self):
        return ModelFacts(nv=self.model.nv, world_count=self.data.nworld, device_kind="cuda", extra={"device": DEVICE})

    def stages(self):
        if self._stages is None:
            base = super().stages()[nl.NAME]
            self._stages = {nl.NAME: StageBinding(base.contract, base.point, _make_dispatch, base.extensions)}
        return self._stages


def _make_dispatch(original, runtime, host, allowed):
    initialized = np.zeros(host.data.nworld, bool)

    def dispatch(m, d, ctx, skip_unchanged=False, skip_noflip=False):
        if m is not host.model:
            return original(m, d, ctx, skip_unchanged, skip_noflip)
        active = ~ctx.done & (not skip_noflip or ctx.state_changed_count > 0)
        stale = np.full(active.shape, not skip_unchanged) | (ctx.quad_changed_count > 0) | ~initialized
        refactor = active & stale
        initialized[active] = True
        system = nl.NewtonSystem(
            h=_dev(ctx.h, wp.float32), grad=_dev(ctx.grad, wp.float32), active=_dev(active.astype(np.int32), wp.int32),
            refactor=_dev(refactor.astype(np.int32), wp.int32), search=_dev(ctx.search, wp.float32),
            search_dot=_dev(ctx.search_dot, wp.float32), newton_decrement=_dev(ctx.newton_decrement, wp.float32))

        def pull():
            wp.synchronize()
            ctx.search[:] = system.search.numpy()
            ctx.search_dot[:] = system.search_dot.numpy()
            ctx.newton_decrement[:] = system.newton_decrement.numpy()

        if "fallback" in allowed:
            def fallback(mask):
                pull()
                shadow = copy.copy(ctx)
                shadow.done = (mask.numpy() == 0) | ~active
                original(m, d, shadow, False, False)
                system.search.assign(ctx.search)
                system.search_dot.assign(ctx.search_dot)
                system.newton_decrement.assign(ctx.newton_decrement)
            system.fallback = fallback
        if "structure_hint" in allowed:
            def hint(flags, refactor_mask):
                wp.synchronize()
                flags_np, rf = flags.numpy().copy(), refactor_mask.numpy()
                for w in range(flags_np.shape[0]):
                    if rf[w]:
                        sym = nl._symmetric(ctx.h[w], m.nv)
                        flags_np[w, :m.nv] = ((np.abs(sym - np.diag(np.diag(sym))).sum(axis=1)) > 0)
                for i in getattr(host, 'hint_drop', ()):
                    flags_np[:, i] = 0                      # a faulty host hint: misses a coupled DOF
                flags.assign(flags_np)
            system.structure_hint = hint
        runtime.solve(system)
        pull()

    return dispatch
