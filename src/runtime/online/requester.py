"""Online requester compatibility entrypoint in the runtime package."""

from src.runtime.remote_search import main


if __name__ == "__main__":
    raise SystemExit(main())
