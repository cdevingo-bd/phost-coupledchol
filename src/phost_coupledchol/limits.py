"""Limits that belong to the algorithm's storage, not to a backend, so they can be read without importing one."""

MAX_NV = 512            # the Warp backend's dense storage is worlds x nv x nv and its tiles stop here; the NumPy backend has no such limit
