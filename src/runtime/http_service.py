from __future__ import annotations

import argparse
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.runtime.external_api import (
    ExternalAPIError,
    configure_robot,
    control_simulation,
    export_program,
    get_default_session,
    get_runtime_capabilities,
    plan_program,
    solve_fk,
    solve_ik,
    solve_path_batch,
    solve_path_request,
)


class ErrorPayload(BaseModel):
    code: str
    message: str
    details: Any | None = None


class ErrorResponse(BaseModel):
    ok: bool
    error: ErrorPayload
    error_legacy: str


class HealthResponse(BaseModel):
    ok: bool
    configured: bool
    kinematics_source: str
    kinematics_hash: str | None = None
    model_id: str | None = None


def _build_error_response(
    *,
    code: str,
    message: str,
    details: Any | None = None,
    status_code: int = 400,
) -> JSONResponse:
    payload = {
        "ok": False,
        "error": {
            "code": str(code),
            "message": str(message),
            "details": details,
        },
        "error_legacy": str(message),
    }
    return JSONResponse(payload, status_code=status_code)


def create_app() -> FastAPI:
    app = FastAPI(
        title="winding_pose_solver Runtime API",
        version="1.0.0",
        description=(
            "Stable HTTP facade for IK/FK and path solving. "
            "All solver logic is delegated to winding_pose_solver core/search/runtime modules."
        ),
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(ExternalAPIError)
    async def _external_api_error_handler(
        _request: Request,
        exc: ExternalAPIError,
    ) -> JSONResponse:
        return _build_error_response(
            code=exc.code,
            message=exc.message,
            details=exc.details,
            status_code=exc.status_code,
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error_handler(
        _request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        return _build_error_response(
            code="request_validation_failed",
            message="Request body validation failed.",
            details={"errors": exc.errors()},
            status_code=422,
        )

    @app.exception_handler(Exception)
    async def _unexpected_error_handler(
        _request: Request,
        exc: Exception,
    ) -> JSONResponse:
        return _build_error_response(
            code="internal_error",
            message=f"{type(exc).__name__}: {exc}",
            status_code=500,
        )

    @app.get("/health", response_model=HealthResponse)
    @app.get("/api/health", response_model=HealthResponse)
    def health() -> dict[str, Any]:
        session = get_default_session()
        return {
            "ok": True,
            "configured": bool(session.configured()),
            "kinematics_source": str(session.kinematics_source),
            "kinematics_hash": session.kinematics_hash,
            "model_id": session.model_id,
        }

    @app.get("/api/capabilities")
    def capabilities() -> dict[str, Any]:
        return get_runtime_capabilities()

    @app.post("/api/configure")
    def configure(payload: dict[str, Any]) -> dict[str, Any]:
        return configure_robot(payload)

    @app.post("/api/fk")
    def fk(payload: dict[str, Any]) -> dict[str, Any]:
        return solve_fk(payload)

    @app.post("/api/ik")
    def ik(payload: dict[str, Any]) -> dict[str, Any]:
        return solve_ik(payload)

    @app.post("/api/path/solve")
    def path_solve(payload: dict[str, Any]) -> dict[str, Any]:
        return solve_path_request(payload)

    @app.post("/api/path/solve-batch")
    def path_solve_batch(payload: dict[str, Any]) -> dict[str, Any]:
        return solve_path_batch(payload)

    @app.post("/api/simulation/control")
    def simulation_control(payload: dict[str, Any]) -> dict[str, Any]:
        return control_simulation(payload)

    @app.post("/api/program/plan")
    def program_plan(payload: dict[str, Any]) -> dict[str, Any]:
        return plan_program(payload)

    @app.post("/api/program/export")
    def program_export(payload: dict[str, Any]) -> dict[str, Any]:
        return export_program(payload)

    return app


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run winding_pose_solver FastAPI runtime service.",
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
            "uvicorn is required to run the FastAPI runtime service. "
            "Install dependencies from requirements.shared.txt."
        ) from exc

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
