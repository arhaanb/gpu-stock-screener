"""gpu backend selection and array conversion helpers."""

import numpy as np

_GPU_AVAILABLE = False
xp = np

try:
    import cupy as _cp  # type: ignore
    _cp.cuda.runtime.getDeviceCount()
    xp = _cp
    _GPU_AVAILABLE = True
except Exception:
    _cp = None  # type: ignore


def gpu_available() -> bool:
    return _GPU_AVAILABLE


def backend_name() -> str:
    return "cupy (GPU)" if _GPU_AVAILABLE else "numpy (CPU)"


def to_numpy(arr):
    if _GPU_AVAILABLE and isinstance(arr, _cp.ndarray):
        return _cp.asnumpy(arr)
    return np.asarray(arr)


def to_xp(arr):
    return xp.asarray(arr)
