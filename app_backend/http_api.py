from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .database import DEFAULT_DB
from .recommendation_service import RecommendationService


class RecommendationAPIHandler(BaseHTTPRequestHandler):
    service = RecommendationService(DEFAULT_DB)

    def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        parsed = urlparse(self.path)
        query = {key: values[-1] for key, values in parse_qs(parsed.query).items()}
        try:
            if parsed.path == "/health":
                self._json(HTTPStatus.OK, self.service.health())
            elif parsed.path == "/states":
                self._json(HTTPStatus.OK, {"states": self.service.list_states()})
            elif parsed.path == "/recommendation":
                payload = self.service.recommend(
                    state_code=self._required(query, "state"),
                    planting_class=self._required(query, "planting_class"),
                    cell_id=query.get("cell_id"),
                    latitude=self._float(query.get("latitude")),
                    longitude=self._float(query.get("longitude")),
                    mixture_size=self._int(query.get("mixture_size")),
                )
                self._json(HTTPStatus.OK, payload)
            else:
                self._json(HTTPStatus.NOT_FOUND, {"error": "unknown_endpoint"})
        except (ValueError, LookupError) as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"error": type(exc).__name__, "message": str(exc)})
        except Exception:
            self._json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": "internal_error", "message": "The recommendation service failed safely."},
            )

    def log_message(self, format: str, *args) -> None:
        return

    @staticmethod
    def _required(query: dict[str, str], name: str) -> str:
        if not query.get(name):
            raise ValueError(f"Missing required query parameter: {name}")
        return query[name]

    @staticmethod
    def _float(value: str | None) -> float | None:
        return float(value) if value is not None else None

    @staticmethod
    def _int(value: str | None) -> int | None:
        return int(value) if value is not None else None

    def _json(self, status: HTTPStatus, payload: object) -> None:
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def serve(database_path: Path | str = DEFAULT_DB, host: str = "127.0.0.1", port: int = 8765) -> None:
    RecommendationAPIHandler.service = RecommendationService(database_path)
    server = ThreadingHTTPServer((host, port), RecommendationAPIHandler)
    print(f"Pollinator recommendation API: http://{host}:{port}")
    server.serve_forever()
