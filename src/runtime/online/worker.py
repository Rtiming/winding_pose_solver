"""Online worker compatibility entrypoint in the runtime package."""

from src.robodk_runtime.eval_worker import main


if __name__ == "__main__":
    raise SystemExit(main())
