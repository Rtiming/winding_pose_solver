"""Compatibility CLI wrapper for the modular online roundtrip runtime.

The canonical implementation now lives in `src/runtime/online/roundtrip.py`.
Keep this root file as a stable entrypoint for existing scripts and docs.
"""

from src.runtime.online.roundtrip import *  # noqa: F401,F403
from src.runtime.online.roundtrip import main


if __name__ == "__main__":
    raise SystemExit(main())
