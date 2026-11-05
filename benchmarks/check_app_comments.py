#!/usr/bin/env python
"""Follow-up outside the frozen protocol: the app's own comment path, once the adapter turns thinking off.

The frozen run found that every Qwen3.5 comment came back empty with the adapter as it was. This runs the same end-to-end
examples through the unchanged prompts and API after that one-line fix. Comments are free text, so nothing here is scored
with labels, and the results do not change those of the frozen run.

    python benchmarks/check_app_comments.py --run-id <id>
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import yaml  # noqa: E402

import bench_resources  # noqa: E402
from bench_core import (BENCH_DIR, ROOT, append_jsonl, capitalized_not_in_inputs, load_protocol,  # noqa: E402
                        numbers_not_in_inputs, read_jsonl, value_mentioned)
from bench_ollama import OllamaClient  # noqa: E402


def without_list_markers(text: str) -> str:
    """The numbers of a numbered list are not figures."""
    return re.sub(r"(?m)^\s*\d+[.)]\s+", "", text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    run_dir = BENCH_DIR / "results" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    scratch = tempfile.mkdtemp(prefix="srr_comments_qdrant_")
    os.environ["QDRANT_LOCATION"] = scratch
    if shutil.which("caffeinate"):
        subprocess.Popen(["caffeinate", "-ims", "-w", str(os.getpid())])

    import api_rag
    import X_AGENT
    from fastapi.testclient import TestClient

    protocol = load_protocol()
    client = OllamaClient()
    examples = yaml.safe_load((BENCH_DIR / "dataset" / "e2e_examples.yaml").read_text("utf-8"))["examples"]
    api = TestClient(api_rag.app)
    output = run_dir / "comments.jsonl"
    done = {(r["alias"], r["modality"]) for r in read_jsonl(output)}
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    (run_dir / "manifest.json").write_text(json.dumps(
        {"run_id": args.run_id, "commit": commit, "ollama_version": client.version(), "hardware": bench_resources.hardware_info(),
         "note": "Outside the frozen protocol. App prompts and options, with think=False added in the adapter.",
         "models": {m["alias"]: {"tag": m["tag"], "digest": m["digest"]} for m in protocol["models"]}},
        indent=2, ensure_ascii=False), "utf-8")

    original = X_AGENT.Agente.get_answer
    seconds = []

    def timed(self, *a, **k):
        started = time.perf_counter()
        try:
            return original(self, *a, **k)
        finally:
            seconds.append(time.perf_counter() - started)
    X_AGENT.Agente.get_answer = timed
    try:
        for model in protocol["models"]:
            alias, tag = model["alias"], model["tag"]
            for mode, example in examples.items():
                if (alias, mode) in done:
                    continue
                client.unload_all()
                api_rag.agente_dinamico = X_AGENT.Agente(X_AGENT.ollama_endpoint, tag)
                files = [("file_ref", (Path(r["path"]).name, (ROOT / r["path"]).read_bytes())) if "path" in r
                         else ("file_ref", (r["filename"], r["text"].encode("utf-8"))) for r in example["references"]]
                metadata = [r.get("metadata", {}) for r in example["references"]]
                data = {"modalidad": mode, "texto_query": example["reviewed"], "usar_agente": "true"}
                if any(metadata):
                    data["ref_metadata"] = json.dumps(metadata)
                seconds.clear()
                with contextlib.redirect_stdout(io.StringIO()):
                    started = time.perf_counter()
                    response = api.post("/validar", files=files, data=data)
                    total = time.perf_counter() - started
                response.raise_for_status()
                comments = []
                for i, (result, wall, expected) in enumerate(zip(response.json()["resultados"], seconds, example["fragments"])):
                    text = without_list_markers(result.get("comentario") or "")
                    inputs = result["chunk"] + "\n" + result["context"]
                    values = expected.get("required_values", [])
                    comments.append({"fragment": result["fragmento"], "claim": result["chunk"], "expected": expected["label"],
                                     "wall_s": wall, "first_after_load": i == 0, "comment": result.get("comentario"),
                                     "error": result.get("comentario_error"),
                                     "values_present": all(value_mentioned(str(v), text) for v in values) if values else None,
                                     "numbers_not_in_inputs": numbers_not_in_inputs(text, inputs),
                                     "capitalized_not_in_inputs": capitalized_not_in_inputs(text, inputs)})
                append_jsonl(output, {"alias": alias, "tag": tag, "modality": mode, "total_s": total, "comments": comments})
                print(f"{alias} {mode}: total {total:.1f}s, usable {sum(1 for c in comments if c['comment'])}/{len(comments)}", flush=True)
        client.unload_all()
    finally:
        X_AGENT.Agente.get_answer = original
        shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    main()
