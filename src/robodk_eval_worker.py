"""Compatibility wrapper for the moved RoboDK evaluation worker module."""

from src.robodk_runtime import eval_worker as _impl

globals().update(
    {
        name: getattr(_impl, name)
        for name in dir(_impl)
        if not name.startswith("__")
    }
)

__all__ = [name for name in globals() if not name.startswith("__")]
