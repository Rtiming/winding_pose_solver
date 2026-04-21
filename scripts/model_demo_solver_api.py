from __future__ import annotations

import argparse
import sys
from pathlib import Path


_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use src.six_axis_ik as backend solver for model_demo.",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8898)
    parser.add_argument("--log-level", type=str, default="info")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError(
            "uvicorn is required. Install dependencies from requirements.shared.txt."
        ) from exc

    print(f"[model_demo_solver_api] listening on http://{args.host}:{args.port}")
    uvicorn.run(
        "src.runtime.http_service:create_app",
        host=str(args.host),
        port=int(args.port),
        log_level=str(args.log_level),
        factory=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

