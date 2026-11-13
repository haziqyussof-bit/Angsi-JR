"""Exercise actual OpenAI Python SDK against a local, synthetic Gemma API."""
import json
import pytest
import sys
import threading
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ai_chat

class Handler(BaseHTTPRequestHandler):
    requests = []

    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        body = json.dumps({"object": "list", "data": [{"id": ai_chat.MODEL_ID, "object": "model"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Handler.requests.append(body)
        assert self.path.endswith("/v1/chat/completions")
        roles = [m["role"] for m in body["messages"]]
        assert roles[0] == "user"
        assert all(a != b for a,b in zip(roles,roles[1:]))
        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for text in ("Mock ", "Gemma reply"):
                event = {"id": "chatcmpl-test", "object":"chat.completion.chunk",
                         "created": 0, "model": ai_chat.MODEL_ID,
                         "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]}
                self.wfile.write(("data: " + json.dumps(event) + "\n\n").encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        else:
            content = {"id": "chatcmpl-test", "object": "chat.completion", "created": 0,
                       "model": ai_chat.MODEL_ID,
                       "choices": [{"index": 0, "message": {"role":"assistant", "content":"CONNECTION_OK"},
                                    "finish_reason":"stop"}]}
            raw = json.dumps(content).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)


def test_real_openai_sdk_http_path(monkeypatch):
    pytest.importorskip("openai", reason="OpenAI SDK is installed by Jupyter requirements on HPC")
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    monkeypatch.setenv("LLM_BASE_URL", f"http://127.0.0.1:{server.server_port}/v1")
    monkeypatch.setenv("LLM_API_KEY", "token-not-needed")
    try:
        report = ai_chat.check_connection()
        assert report["connected"] and report["model_list"] == "listed", report
        conversation = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "what is 4D?"},
        ]
        output = "".join(ai_chat.stream_chat(conversation, {"survey": "1995/2006"}))
        assert output == "Mock Gemma reply"
        assert Handler.requests[-1]["stream"] is True
        assert [m["role"] for m in Handler.requests[-1]["messages"]] == ["user", "assistant", "user"]
    finally:
        server.shutdown()
        server.server_close()
