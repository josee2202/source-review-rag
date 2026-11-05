"""Minimal Ollama client for the comparison: stdlib only, so the request sent is exactly what is recorded."""
from __future__ import annotations

import http.client
import json
import time


class OllamaClient:
    def __init__(self, host: str = "localhost", port: int = 11434):
        self.host, self.port = host, port

    def _json(self, method: str, path: str, body=None, timeout: float = 30) -> dict:
        conn = http.client.HTTPConnection(self.host, self.port, timeout=timeout)
        try:
            payload = json.dumps(body) if body is not None else None
            conn.request(method, path, body=payload, headers={"Content-Type": "application/json"})
            response = conn.getresponse()
            data = response.read()
            if response.status != 200:
                raise RuntimeError(f"{method} {path}: HTTP {response.status}: {data[:200]!r}")
            return json.loads(data)
        finally:
            conn.close()

    def version(self) -> str:
        return self._json("GET", "/api/version")["version"]

    def tags(self) -> dict:
        return {m["name"]: m for m in self._json("GET", "/api/tags")["models"]}

    def loaded(self) -> list:
        return self._json("GET", "/api/ps")["models"]

    def show(self, model: str) -> dict:
        return self._json("POST", "/api/show", {"model": model})

    def unload(self, model: str, wait_s: float = 30) -> None:
        self._json("POST", "/api/generate", {"model": model, "keep_alive": 0}, timeout=60)
        deadline = time.monotonic() + wait_s
        while time.monotonic() < deadline:
            if model not in [m["name"] for m in self.loaded()]:
                return
            time.sleep(0.5)
        raise RuntimeError(f"{model} is still loaded {wait_s} s after asking to unload it")

    def unload_all(self) -> list:
        names = [m["name"] for m in self.loaded()]
        for name in names:
            self.unload(name)
        return names

    def load(self, model: str, options: dict, keep_alive: str, timeout_s: float) -> dict:
        """Explicit load with an empty prompt, so that its cost is measured apart from inference."""
        started = time.perf_counter()
        result = self._json("POST", "/api/generate",
                            {"model": model, "options": options, "keep_alive": keep_alive, "stream": False},
                            timeout=timeout_s)
        return {"wall_s": time.perf_counter() - started, "load_duration_ns": result.get("load_duration")}

    def generate(self, payload: dict, timeout_s: float) -> dict:
        """Streams one generation under a wall-clock deadline and never retries.

        Closing the connection is what cancels the work on the server; run_benchmark verifies that it stopped.
        """
        started = time.perf_counter()
        deadline = started + timeout_s
        result = {"status": "request_error", "response": "", "thinking": "", "final": {}, "error": None,
                  "wall_s": None, "ttft_s": None}
        conn = http.client.HTTPConnection(self.host, self.port, timeout=timeout_s)
        try:
            conn.request("POST", "/api/generate", body=json.dumps(payload),
                         headers={"Content-Type": "application/json"})
            response = conn.getresponse()
            if response.status != 200:
                result["error"] = f"HTTP {response.status}: {response.read()[:300]!r}"
                return result
            sock = conn.sock or getattr(getattr(response.fp, "raw", None), "_sock", None)
            response_parts, thinking_parts = [], []
            while True:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    raise TimeoutError
                if sock is not None:
                    sock.settimeout(remaining)
                line = response.readline()
                if not line:
                    result["error"] = "the stream ended without a final message"
                    return result
                chunk = json.loads(line)
                if chunk.get("error"):
                    result["error"] = str(chunk["error"])[:300]
                    return result
                if chunk.get("thinking"):
                    thinking_parts.append(chunk["thinking"])
                if chunk.get("response"):
                    if result["ttft_s"] is None:
                        result["ttft_s"] = time.perf_counter() - started
                    response_parts.append(chunk["response"])
                result["response"], result["thinking"] = "".join(response_parts), "".join(thinking_parts)
                if chunk.get("done"):
                    result["final"] = {k: v for k, v in chunk.items() if k not in ("response", "thinking", "context")}
                    result["status"] = "completed"
                    return result
        except TimeoutError:
            result["status"] = "timeout"
            result["error"] = f"no complete answer within {timeout_s} s"
            return result
        except (OSError, http.client.HTTPException, json.JSONDecodeError) as e:
            result["error"] = f"{type(e).__name__}: {e}"[:300]
            return result
        finally:
            result["wall_s"] = time.perf_counter() - started
            conn.close()
