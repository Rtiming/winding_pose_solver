"""Compatibility wrapper for the search package."""

from src.search import global_search as _impl

globals().update(
    {
        name: getattr(_impl, name)
        for name in dir(_impl)
        if not name.startswith("__")
    }
)

__all__ = [name for name in globals() if not name.startswith("__")]
