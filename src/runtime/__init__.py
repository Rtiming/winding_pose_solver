"""Runtime orchestration package.

Keep this package initializer intentionally light.

Server-safe entrypoints such as `python -m src.runtime.online.requester` import
submodules from this package directly. If we eagerly import `src.runtime.app`
here, that pulls in the visualization module and its `matplotlib` dependency,
which is not installed in the server environment.
"""

__all__ = []
