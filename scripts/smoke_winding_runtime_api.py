from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.runtime.http_service import create_app


def _assert_envelope(payload: dict, *, expect_ok: bool) -> None:
    required = {"ok", "code", "message", "request_id", "ts"}
    missing = sorted(required - set(payload.keys()))
    if missing:
        raise AssertionError(f"Envelope missing keys: {missing}")
    if bool(payload.get("ok")) is not bool(expect_ok):
        raise AssertionError(
            f"Envelope ok mismatch: expected {expect_ok}, got {payload.get('ok')}"
        )


def _latest_run_id_with_summary(repo_root: Path) -> str | None:
    online_root = repo_root / "artifacts" / "online_runs"
    if not online_root.is_dir():
        return None
    candidates: list[tuple[float, str]] = []
    for run_dir in online_root.iterdir():
        if not run_dir.is_dir():
            continue
        summary_path = run_dir / "summary.json"
        if summary_path.is_file():
            candidates.append((summary_path.stat().st_mtime, run_dir.name))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def run_smoke(*, with_live_run: bool, timeout_sec: int) -> None:
    client = TestClient(create_app())

    # Legacy endpoint compatibility check.
    legacy = client.get("/api/capabilities")
    if legacy.status_code != 200:
        raise AssertionError(f"/api/capabilities failed: {legacy.status_code}")
    if "ok" not in legacy.json():
        raise AssertionError("Legacy /api/capabilities missing 'ok'.")

    # New capabilities envelope.
    caps = client.get("/api/winding/capabilities")
    if caps.status_code != 200:
        raise AssertionError(f"/api/winding/capabilities failed: {caps.status_code}")
    caps_json = caps.json()
    _assert_envelope(caps_json, expect_ok=True)
    if "run_modes" not in (caps_json.get("data") or {}):
        raise AssertionError("winding capabilities payload missing run_modes.")

    # Dry-run orchestration.
    create_payload = {
        "run_mode": "online",
        "online_role": "server",
        "options": {"dry_run": True},
    }
    created = client.post("/api/winding/runs", json=create_payload)
    if created.status_code != 200:
        raise AssertionError(f"dry-run create failed: {created.status_code}")
    created_json = created.json()
    _assert_envelope(created_json, expect_ok=True)
    run_id = (created_json.get("data") or {}).get("run_id")
    if not run_id:
        raise AssertionError("dry-run response missing run_id")

    status = client.get(f"/api/winding/runs/{run_id}")
    if status.status_code != 200:
        raise AssertionError(f"status query failed: {status.status_code}")
    status_json = status.json()
    _assert_envelope(status_json, expect_ok=True)
    if ((status_json.get("data") or {}).get("status")) != "succeeded":
        raise AssertionError("dry-run status did not settle to succeeded")

    # Latest-summary should succeed if historical artifacts exist.
    latest = client.get("/api/winding/latest")
    if latest.status_code == 200:
        _assert_envelope(latest.json(), expect_ok=True)

    latest_run_id = _latest_run_id_with_summary(Path.cwd())
    if latest_run_id:
        summary = client.get(f"/api/winding/runs/{latest_run_id}/summary")
        if summary.status_code != 200:
            raise AssertionError(
                f"historical summary read failed for {latest_run_id}: {summary.status_code}"
            )
        _assert_envelope(summary.json(), expect_ok=True)

    if with_live_run:
        request_path = None
        if latest_run_id:
            candidate = Path("artifacts") / "online_runs" / latest_run_id / "request.json"
            if candidate.is_file():
                request_path = candidate
        if request_path is not None:
            live = client.post(
                "/api/winding/runs",
                json={
                    "run_mode": "online",
                    "online_role": "server",
                    "request_path": str(request_path),
                    "options": {"timeout_sec": timeout_sec},
                },
            )
            if live.status_code != 200:
                raise AssertionError(f"live run create failed: {live.status_code}")
            live_run_id = (live.json().get("data") or {}).get("run_id")
            if not live_run_id:
                raise AssertionError("live run response missing run_id")
            terminal = None
            for _ in range(max(1, timeout_sec * 5 + 20)):
                time.sleep(0.2)
                polled = client.get(f"/api/winding/runs/{live_run_id}")
                if polled.status_code != 200:
                    raise AssertionError(
                        f"live run polling failed: {polled.status_code}"
                    )
                payload = polled.json()
                data = payload.get("data") or {}
                if data.get("status") in {"succeeded", "failed", "canceled"}:
                    terminal = data
                    break
            if terminal is None:
                raise AssertionError("live run polling timed out without terminal state")

    print("Smoke passed.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Smoke test for winding runtime orchestration API.",
    )
    parser.add_argument(
        "--skip-live-run",
        action="store_true",
        help="Skip spawning a real subprocess run and test dry-run only.",
    )
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=3,
        help="Timeout used for the optional live run smoke.",
    )
    args = parser.parse_args(argv)
    run_smoke(with_live_run=not args.skip_live_run, timeout_sec=max(1, int(args.timeout_sec)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
