"""Relay E2E fixture whose trusted provider fails only during close."""

from __future__ import annotations

from typing import Any

from pyodide_host_adapter import (
    build_db_service as _build_db_service,
)
from pyodide_host_adapter import (
    run_for_host_pyodide,
)


class _CloseFailingService:
    def __init__(self, service: Any) -> None:
        self._service = service

    def handle(self, method: str, params_json: str) -> str:
        return self._service.handle(method, params_json)

    def handle_duplex(self, method: str, params_json: str, control: Any) -> str:
        return self._service.handle_duplex(method, params_json, control)

    def close(self) -> None:
        self._service.close()
        raise RuntimeError("intentional provider close failure")


def build_db_service(
    db_path: str,
    contacts_db_path: str | None = None,
) -> tuple[_CloseFailingService, dict[str, Any]]:
    service, metadata = _build_db_service(db_path, contacts_db_path)
    return _CloseFailingService(service), metadata


__all__ = ["build_db_service", "run_for_host_pyodide"]
