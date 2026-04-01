import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict

from merchant_growth_mvp import (
    GAP_CONFIG,
    HEALTH_CONFIG,
    create_mock_inputs,
    run_llm_orchestrated_pipeline,
)


HOST = "127.0.0.1"
PORT = 8765


class CopilotHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(Path(__file__).parent), **kwargs)

    def _write_json(self, status: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/analyze":
            self._write_json(404, {"error": "not found"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
        except Exception:
            self._write_json(400, {"error": "invalid json payload"})
            return

        brand_id = str(payload.get("brand_id", "")).strip().upper()
        question = str(payload.get("question", "")).strip()
        mock_mode = bool(payload.get("mock_mode", False))

        if not brand_id:
            self._write_json(400, {"error": "brand_id is required"})
            return
        if not question:
            self._write_json(400, {"error": "question is required"})
            return

        try:
            merchant_df, peer_benchmark = create_mock_inputs()
            result = run_llm_orchestrated_pipeline(
                brand_id=brand_id,
                question=question,
                merchant_df=merchant_df,
                peer_benchmark=peer_benchmark,
                health_config=HEALTH_CONFIG,
                gap_config=GAP_CONFIG,
                mock_mode=mock_mode,
            )
            self._write_json(200, result)
        except Exception as exc:
            self._write_json(500, {"error": str(exc)})


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), CopilotHandler)
    print(f"Copilot server running at http://{HOST}:{PORT}/copilot_frontend.html")
    server.serve_forever()


if __name__ == "__main__":
    main()
