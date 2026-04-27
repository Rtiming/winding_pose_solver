"""Compatibility CLI wrapper for worker role."""

from src.runtime.online.worker import main


if __name__ == "__main__":
    raise SystemExit(main())
