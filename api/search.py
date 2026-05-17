from __future__ import annotations

import json
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from time import perf_counter
from urllib.parse import parse_qs, urlparse

ROOT_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT_DIR / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import eva_query_processing as search_backend


def parse_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def parse_int(value: str | None, default: int, minimum: int, maximum: int) -> int:
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return max(minimum, min(maximum, parsed))


def parse_float(value: str | None, default: float, minimum: float, maximum: float) -> float:
    if value is None:
        return default
    try:
        parsed = float(value)
    except ValueError:
        return default
    return max(minimum, min(maximum, parsed))


class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self) -> None:
        params = parse_qs(urlparse(self.path).query)
        query = (params.get("q") or [""])[0].strip()

        if not query:
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": "Missing required query parameter: q"})
            return

        top_k = parse_int((params.get("top_k") or [None])[0], default=25, minimum=1, maximum=50)
        feedback_docs = parse_int((params.get("feedback_docs") or [None])[0], default=5, minimum=1, maximum=20)
        feedback_terms = parse_int((params.get("feedback_terms") or [None])[0], default=5, minimum=1, maximum=20)
        embedding_top_n = parse_int((params.get("embedding_top_n") or [None])[0], default=5, minimum=1, maximum=20)
        use_synonyms = parse_bool((params.get("use_synonyms") or [None])[0], default=True)
        use_feedback = parse_bool((params.get("use_feedback") or [None])[0], default=True)
        embedding_backend = (params.get("embedding_backend") or [search_backend.EMBEDDING_BACKEND])[0]
        embedding_model_path = (params.get("embedding_model_path") or [None])[0]
        embedding_min_sim = parse_float(
            (params.get("embedding_min_sim") or [None])[0],
            default=0.35,
            minimum=0.0,
            maximum=1.0,
        )

        started_at = perf_counter()
        search_payload = search_backend.run_expanded_search_with_analysis(
            query,
            top_k=top_k,
            feedback_docs=feedback_docs,
            feedback_terms=feedback_terms,
            use_synonyms=use_synonyms,
            use_feedback=use_feedback,
            use_embeddings=True,
            embedding_backend=embedding_backend,
            embedding_model_path=embedding_model_path,
            embedding_top_n=embedding_top_n,
            embedding_min_sim=embedding_min_sim,
        )
        results = search_payload["results"]
        latency_ms = round((perf_counter() - started_at) * 1000, 2)

        self._write_json(
            HTTPStatus.OK,
            {
                "query": query,
                "count": len(results),
                "latency_ms": latency_ms,
                "options": {
                    "top_k": top_k,
                    "feedback_docs": feedback_docs,
                    "feedback_terms": feedback_terms,
                    "use_synonyms": use_synonyms,
                    "use_feedback": use_feedback,
                    "use_embeddings": True,
                    "embedding_backend": embedding_backend,
                    "embedding_model_path": embedding_model_path,
                    "embedding_top_n": embedding_top_n,
                    "embedding_min_sim": embedding_min_sim,
                },
                "expansion": search_payload["expansion"],
                "results": results,
            },
        )

    def _send_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _write_json(self, status: HTTPStatus, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._send_cors_headers()
        self.end_headers()
        self.wfile.write(body)
