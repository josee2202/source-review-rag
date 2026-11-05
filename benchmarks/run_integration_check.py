#!/usr/bin/env python
"""Integration check of the production judge, after the owner chose qwen3.5:4b-q4_K_M. Not part of the frozen comparison.

Uses the production code in IA-modulable/juez.py, with no benchmark prompt or client:
  regression  the 30 evaluation cases with fixed context, one pass, as a regression on known cases
  controls    benchmarks/dataset/integration_cases.yaml, fixed before running it
  e2e         one full document per mode through the API: reading, embeddings, Qdrant, judge and Excel export

One request at a time. Expected labels are only read at analysis time (build_integration_report.py). Every answer is kept,
failed ones included. It can be resumed with the same --run-id, and refuses to start with uncommitted code.

    python benchmarks/run_integration_check.py --run-id <id> [--stages regression,controls,e2e] [--num-thread 8]
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import yaml  # noqa: E402

import bench_resources  # noqa: E402
from bench_core import BENCH_DIR, ROOT, append_jsonl, load_cases, load_protocol, ordered_cases, read_jsonl, sha256_bytes  # noqa: E402
from bench_ollama import OllamaClient  # noqa: E402

sys.path.insert(0, str(ROOT / "front-end"))

EVAL_CASES = BENCH_DIR / "dataset" / "eval_cases.yaml"
CONTROL_CASES = BENCH_DIR / "dataset" / "integration_cases.yaml"
E2E_EXAMPLES = BENCH_DIR / "dataset" / "e2e_examples.yaml"
STAGES = ("regression", "controls", "e2e")
MAX_CONSECUTIVE_TIMEOUTS = 3
BUDGET_S = 2 * 3600
EXPECTED_DIGEST = "d8b0f5e9760c"  # the digest measured in the frozen comparison


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def dirty_paths() -> list:
    """Uncommitted changes in anything that affects the results."""
    out = subprocess.run(["git", "status", "--porcelain", "--", "IA-modulable", "front-end", "benchmarks", "requirements.txt"],
                         cwd=ROOT, capture_output=True, text=True, check=True).stdout
    return [line for line in out.splitlines() if "benchmarks/results/" not in line and "__pycache__" not in line]


def write_manifest(run_dir: Path, args, juez, client) -> dict:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    path = run_dir / "manifest.json"
    if path.exists():
        manifest = json.loads(path.read_text("utf-8"))
        if manifest["commit"] != commit:
            raise SystemExit(f"{args.run_id} was started at {manifest['commit']}; use a new --run-id for {commit}.")
        return manifest
    cfg = juez.configuracion()
    tags = client.tags()
    if cfg["modelo"] not in tags:
        raise SystemExit(f"{cfg['modelo']} is not installed. Install it with: ollama pull {cfg['modelo']}")
    digest = tags[cfg["modelo"]]["digest"]
    if not digest.startswith(EXPECTED_DIGEST):
        raise SystemExit(f"Digest {digest[:12]} is not the {EXPECTED_DIGEST} of the frozen comparison.")
    request = juez.construir_peticion(cfg["modelo"], "", cfg["max_tokens"], cfg["num_ctx"], cfg["keep_alive"], cfg["num_thread"])
    manifest = {
        "run_id": args.run_id, "kind": "integration check of the production judge; outside the frozen comparison",
        "started_at": now(), "commit": commit, "review": "Cases and labels by the assistant. No human review.",
        "ollama_version": client.version(), "hardware": bench_resources.hardware_info(),
        "model": {"tag": cfg["modelo"], "digest": digest, "size_bytes": tags[cfg["modelo"]]["size"]},
        "request": {k: request[k] for k in ("stream", "think", "keep_alive", "options")}, "timeout_s": cfg["timeout_s"],
        "context_limit_tokens": juez.limite_tokens_prompt(cfg["num_ctx"]), "response_schema": request["format"],
        "prompt_template_sha256": hashlib.sha256(juez.prompt_de_plantilla().encode("utf-8")).hexdigest(),
        "stages": list(args.stages), "dataset_sha256": {p.name: sha256_bytes(p.read_bytes()) for p in (EVAL_CASES, CONTROL_CASES, E2E_EXAMPLES)},
    }
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), "utf-8")
    return manifest


class Abort(Exception):
    pass


class Budget:
    def __init__(self, spent_s: float):
        self.spent_s, self.timeouts = spent_s, 0

    def account(self, analysis: dict) -> None:
        self.spent_s += analysis["metricas"].get("tiempo_s") or 0
        self.timeouts = self.timeouts + 1 if "se ha cancelado" in (analysis.get("motivo") or "") else 0
        if self.timeouts >= MAX_CONSECUTIVE_TIMEOUTS:
            raise Abort(f"{self.timeouts} consecutive timeouts")
        if self.spent_s >= BUDGET_S:
            raise Abort(f"inference budget of {BUDGET_S} s used")


def case_row(stage: str, case: dict, result: dict, first_after_load: bool) -> dict:
    raw = result["bruto"]
    return {"finished_at": now(), "stage": stage, "case_id": case["id"], "modality": case["modality"],
            "first_after_load": first_after_load, "prompt_sha256": hashlib.sha256(raw["prompt"].encode("utf-8")).hexdigest(),
            "raw_response": raw["respuesta"], "done_reason": raw["fin"], "request_status": raw.get("estado_peticion"),
            "analysis": result["analisis"]}


def run_cases(stage: str, cases: list, juez, client, output: Path, budget: Budget, sampler) -> None:
    done = {r["case_id"] for r in read_jsonl(output) if r["stage"] == stage}
    pending = [c for c in cases if c["id"] not in done]
    sampler.set_block(stage)
    client.unload_all()
    for i, case in enumerate(pending):
        result = juez.ejecutar(case["modality"], case["claim"], case["passages"])
        append_jsonl(output, case_row(stage, case, result, first_after_load=i == 0))
        analysis = result["analisis"]
        print(f"{stage} {case['id']} {analysis['estado']} {analysis['etiqueta']} {analysis['metricas']['tiempo_s']}", flush=True)
        budget.account(analysis)


# * End-to-end through the API
# * ===========================================================================
def post_document(api, mode: str, example: dict, with_analysis: bool):
    files, metadata = [], []
    for ref in example["references"]:
        if "path" in ref:
            files.append(("file_ref", (Path(ref["path"]).name, (ROOT / ref["path"]).read_bytes())))
        else:
            files.append(("file_ref", (ref["filename"], ref["text"].encode("utf-8"))))
        metadata.append(ref.get("metadata", {}))
    data = {"modalidad": mode, "texto_query": example["reviewed"], "usar_agente": str(with_analysis).lower()}
    if any(metadata):
        data["ref_metadata"] = json.dumps(metadata)
    with contextlib.redirect_stdout(io.StringIO()):
        started = time.perf_counter()
        response = api.post("/validar", files=files, data=data)
        total = time.perf_counter() - started
    if response.status_code != 200:
        raise RuntimeError(f"/validar {mode}: {response.status_code} {response.text[:300]}")
    return response.json(), total


def excel_cells(body: dict) -> list:
    """What the exported workbook really contains, read back from its bytes."""
    from openpyxl import load_workbook
    from presentacion import exportar_excel

    rows = list(load_workbook(io.BytesIO(exportar_excel(body)))["Resultados"].values)
    return [dict(zip(rows[0], row)) for row in rows[1:]]


def check_shown_is_evaluated(body: dict, calls: list, juez) -> dict:
    """Recomputes each result from the model's raw answer, and compares it with the API response and with the Excel file."""
    results = body["resultados"]
    fragments = []
    excel = excel_cells(body)
    for result, call, cells in zip(results, calls, excel):
        passages = juez.pasajes_de_resultado(result)
        expected_prompt = juez.construir_prompt(body["modalidad"], result["chunk"], passages)
        data, reason = juez.validar_respuesta(call["response"], call["done_reason"], passages)
        analysis = result["analisis"]
        shown = {"label": analysis["etiqueta"], "explanation": analysis["explicacion"],
                 "evidence": [{k: e[k] for k in ("id", "pasaje", "cita")} for e in analysis["evidencia"]]}
        recomputed = {"label": data["etiqueta"], "explanation": data["explicacion"], "evidence": data["evidencia"]} if data else None
        quotes = [e["cita"] for e in analysis["evidencia"] if e["cita"]]
        fragments.append({
            "fragment": result["fragmento"],
            "prompt_is_the_evaluated_one": call["prompt"] == expected_prompt,
            "api_equals_validation_of_raw_answer": shown == recomputed if data else analysis["etiqueta"] is None,
            "excel_status": cells["Análisis del modelo"] == analysis["etiqueta_texto"],
            "excel_explanation": cells["Explicación del modelo"] == analysis["explicacion"],
            "excel_quotes": all(q in (cells["Evidencia citada"] or "") for q in quotes) and (bool(cells["Evidencia citada"]) == bool(quotes)),
            "excel_passages_kept": all(cells.get(f"Pasaje {n}") is not None for n in range(1, len(result["relevant_chunks"]) + 1)),
        })
    return {"one_generation_per_fragment": len(calls) == len(results), "excel_rows": len(excel) == len(results), "fragments": fragments,
            "all_ok": len(calls) == len(results) and len(excel) == len(results)
            and all(all(v for k, v in f.items() if k != "fragment") for f in fragments)}


def run_e2e(juez, client, output: Path, budget: Budget, sampler) -> bool:
    import api_rag
    import X_qderant
    from fastapi.testclient import TestClient

    if X_qderant.cfg_param["qdrant"]["location"] != os.environ.get("QDRANT_LOCATION"):
        raise SystemExit("The end-to-end stage must use a scratch index, never the user's own qdrant_storage.")
    examples = yaml.safe_load(E2E_EXAMPLES.read_text("utf-8"))["examples"]
    api = TestClient(api_rag.app)
    done = {r["modality"]: r for r in read_jsonl(output)}
    original = juez.generar
    calls = []

    def spy(peticion, timeout_s, host, port):
        salida = original(peticion, timeout_s, host, port)
        calls.append({"prompt": peticion["prompt"], "response": salida["response"], "wall_s": salida["wall_s"],
                      "done_reason": (salida["final"] or {}).get("done_reason")})
        return salida

    all_ok = all(r["checks"]["all_ok"] for r in done.values())
    juez.generar = spy
    try:
        for mode, example in examples.items():
            if mode in done:
                continue
            sampler.set_block(f"e2e:{mode}")
            client.unload_all()
            calls.clear()
            without, without_s = post_document(api, mode, example, with_analysis=False)
            if calls:
                raise RuntimeError("the analysis made a request with the analysis switched off")
            body, total_s = post_document(api, mode, example, with_analysis=True)
            judge_s = sum(c["wall_s"] for c in calls)
            checks = check_shown_is_evaluated(body, calls, juez)
            same_passages = [r["relevant_chunks"] for r in without["resultados"]] == [r["relevant_chunks"] for r in body["resultados"]]
            append_jsonl(output, {
                "finished_at": now(), "modality": mode, "without_analysis_s": without_s, "index_built_in_that_request": not without["indice"]["reutilizado"],
                "with_analysis_s": total_s, "judge_s": judge_s, "rest_of_document_s": total_s - judge_s, "fragments": len(body["resultados"]),
                "same_passages_with_and_without_analysis": same_passages,
                "no_analysis_fields_when_off": all("analisis" not in r and "comentario" not in r for r in without["resultados"]),
                "checks": checks, "excel": excel_cells(body), "response": body,
                "calls": [{"prompt_sha256": hashlib.sha256(c["prompt"].encode("utf-8")).hexdigest(), "response": c["response"],
                           "done_reason": c["done_reason"], "wall_s": c["wall_s"]} for c in calls]})
            all_ok = all_ok and checks["all_ok"] and same_passages
            print(f"e2e {mode}: {total_s:.1f}s, judge {judge_s:.1f}s, {len(calls)} generations, checks {checks['all_ok']}", flush=True)
            for result in body["resultados"]:
                budget.account(result["analisis"])
    finally:
        juez.generar = original
    return all_ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stages", type=lambda s: s.split(","), default=list(STAGES))
    parser.add_argument("--num-thread", type=int, default=8, help="threads sent to Ollama; recorded in the manifest")
    args = parser.parse_args()
    if dirty := dirty_paths():
        raise SystemExit("Commit first, so the manifest describes the code that ran:\n" + "\n".join(dirty))

    run_dir = BENCH_DIR / "results" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    scratch = tempfile.mkdtemp(prefix="srr_integration_qdrant_")
    os.environ["QDRANT_LOCATION"] = scratch
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    if shutil.which("caffeinate"):
        subprocess.Popen(["caffeinate", "-ims", "-w", str(os.getpid())])

    import juez

    juez.config.cfg_param["llm"]["text"]["num_thread"] = args.num_thread
    client = OllamaClient()
    write_manifest(run_dir, args, juez, client)
    responses, e2e_path = run_dir / "responses.jsonl", run_dir / "e2e.jsonl"
    budget = Budget(sum(r["analysis"]["metricas"].get("tiempo_s") or 0 for r in read_jsonl(responses)))
    sampler = bench_resources.ResourceSampler(run_dir / "resources.jsonl", 10).start()
    status, checks_ok = "completed", True
    try:
        if "regression" in args.stages:
            cases = ordered_cases(load_cases(EVAL_CASES), load_protocol()["request"]["options"]["seed"], 1)
            run_cases("regression", cases, juez, client, responses, budget, sampler)
        if "controls" in args.stages:
            run_cases("controls", load_cases(CONTROL_CASES), juez, client, responses, budget, sampler)
        if "e2e" in args.stages:
            checks_ok = run_e2e(juez, client, e2e_path, budget, sampler)
    except Abort as e:
        status = f"aborted: {e}"
    except KeyboardInterrupt:
        status = "cancelled"
    finally:
        sampler.stop()
        with contextlib.suppress(Exception):
            client.unload_all()
        shutil.rmtree(scratch, ignore_errors=True)
    (run_dir / "run_status.json").write_text(json.dumps(
        {"status": status, "finished_at": now(), "inference_s": budget.spent_s, "shown_equals_evaluated": checks_ok}, indent=2), "utf-8")
    print(f"{status}; shown equals evaluated: {checks_ok}", flush=True)
    return 0 if status == "completed" and checks_ok else 1


if __name__ == "__main__":
    sys.exit(main())
