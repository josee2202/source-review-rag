#!/usr/bin/env python
"""Runs the local model comparison and saves every result as soon as it exists.

    python benchmarks/run_benchmark.py freeze
    python benchmarks/run_benchmark.py run --run-id <id> [--stages pilot,eval,legacy,e2e]

Running the same command again resumes: finished requests are not repeated, and a run directory refuses to continue
if the protocol, the datasets or the harness changed. One inference at a time and one generative model in memory.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import yaml  # noqa: E402

import bench_resources  # noqa: E402
from bench_core import (BENCH_DIR, ROOT, append_jsonl, attribute_e2e, build_payload, build_prompt,  # noqa: E402
                        capitalized_not_in_inputs, format_passages, harness_hash, load_cases, load_protocol,
                        locked_hashes, normalize_for_match, numbers_not_in_inputs, ordered_cases, parse_response,
                        read_jsonl, sha256_text, timing_fields, value_mentioned)
from bench_ollama import OllamaClient  # noqa: E402

LOCK_FILE = BENCH_DIR / "protocol.lock.json"
EVAL_CASES = BENCH_DIR / "dataset" / "eval_cases.yaml"
DEV_CASES = BENCH_DIR / "dataset" / "dev_cases.yaml"
E2E_EXAMPLES = BENCH_DIR / "dataset" / "e2e_examples.yaml"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def request_key(stage: str, alias: str, pass_no: int, case_id: str) -> str:
    return f"{stage}|{alias}|{pass_no}|{case_id}"


class NullSampler:
    """Used when no telemetry is wanted, as in the harness tests."""
    def set_block(self, block): pass
    def critical_seconds(self): return 0.0


class Run:
    """State of one run directory. Everything needed to resume is read back from the saved files."""

    def __init__(self, run_dir: Path, protocol: dict, client, sampler=None,
                 verify_stopped=bench_resources.wait_inference_stopped):
        self.dir = Path(run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.protocol, self.client = protocol, client
        self.sampler = sampler or NullSampler()
        self.verify_stopped = verify_stopped
        self.models = {m["alias"]: m for m in protocol["models"]}
        self.responses_path = self.dir / "responses.jsonl"
        self.events_path = self.dir / "events.jsonl"
        self.e2e_path = self.dir / "e2e.jsonl"
        self.inputs_path = self.dir / "inputs.jsonl"
        responses, events, e2e = (read_jsonl(p) for p in (self.responses_path, self.events_path, self.e2e_path))
        self.done = {r["key"] for r in responses}
        self.saved_inputs = {r["id"] for r in read_jsonl(self.inputs_path)}
        self.viability = {e["alias"]: e for e in events if e["event"] == "pilot_result"}
        self.spent_s = sum(r.get("wall_s") or 0 for r in responses) + sum(e.get("model_seconds", 0) for e in events) \
            + sum(r.get("model_seconds", 0) for r in e2e)
        self.budget_s = protocol["limits"]["budget_model_hours"] * 3600

    def event(self, event: str, **fields) -> dict:
        row = {"event": event, "at": now(), **fields}
        append_jsonl(self.events_path, row)
        return row

    def budget_left(self) -> float:
        return self.budget_s - self.spent_s

    def is_viable(self, alias: str) -> bool:
        return alias in self.viability and self.viability[alias]["viable"]

    def save_input(self, case: dict, prompt: str) -> None:
        if case["id"] not in self.saved_inputs:
            append_jsonl(self.inputs_path, {"id": case["id"], "split": case["split"], "modality": case["modality"],
                                            "claim": case["claim"], "passages": case["passages"], "prompt": prompt,
                                            "prompt_sha256": sha256_text(prompt)})
            self.saved_inputs.add(case["id"])

    def load(self, alias: str, stage: str) -> bool:
        limits, request = self.protocol["limits"], self.protocol["request"]
        tag = self.models[alias]["tag"]
        started = time.perf_counter()
        try:
            load = self.client.load(tag, request["options"], request["keep_alive"], limits["load_timeout_s"])
        except Exception as e:
            elapsed = time.perf_counter() - started
            self.spent_s += elapsed
            self.event("load_failed", alias=alias, stage=stage, model_seconds=elapsed, error=str(e)[:300])
            return False
        self.spent_s += load["wall_s"]
        loaded = {m["name"]: m for m in self.client.loaded()}.get(tag, {})
        self.event("cold_load", alias=alias, stage=stage, model_seconds=load["wall_s"], wall_s=load["wall_s"],
                   load_duration_ns=load["load_duration_ns"], ps_size_bytes=loaded.get("size"),
                   ps_size_vram_bytes=loaded.get("size_vram"), ps_context_length=loaded.get("context_length"),
                   loaded_models=list({m["name"] for m in self.client.loaded()}))
        return True

    def request(self, stage: str, alias: str, pass_no: int, case_id: str, payload: dict, timeout_s: float,
                order_index: int, first_after_load: bool, parse=parse_response) -> dict:
        """One request, saved before anything else happens."""
        model = self.models[alias]
        result = self.client.generate(payload, timeout_s)
        final = result["final"]
        status, parsed, detail = result["status"], None, result["error"]
        if status == "completed":
            status, parsed, detail = parse(result["response"], final.get("done_reason"))
        row = {"key": request_key(stage, alias, pass_no, case_id), "stage": stage, "alias": alias, "tag": model["tag"],
               "digest": model["digest"], "pass": pass_no, "case_id": case_id, "order_index": order_index,
               "first_after_load": first_after_load, "finished_at": now(),
               "prompt_sha256": sha256_text(payload["prompt"]), "options": payload["options"],
               "think": payload.get("think"), "status": status, "detail": detail, "parsed": parsed,
               "raw_response": result["response"], "thinking_chars": len(result["thinking"]),
               "done_reason": final.get("done_reason"), "wall_s": result["wall_s"], "ttft_s": result["ttft_s"],
               "durations_ns": {k: final.get(k) for k in ("total_duration", "load_duration", "prompt_eval_duration",
                                                          "eval_duration")},
               **timing_fields(final)}
        append_jsonl(self.responses_path, row)
        self.done.add(row["key"])
        self.spent_s += result["wall_s"] or 0
        return row


def run_block(run: Run, stage: str, alias: str, pass_no: int, cases: list, payload_for=None, timeout_s=None,
              parse=parse_response, unload_at_end: bool = True) -> str:
    """Cold-loads one model, runs its pending cases one by one and unloads it. Returns how the block ended."""
    limits = run.protocol["limits"]
    timeout_s = timeout_s or limits["request_timeout_s"]
    tag = run.models[alias]["tag"]
    indexed = list(enumerate(cases))
    pending = [(i, c) for i, c in indexed if request_key(stage, alias, pass_no, c["id"]) not in run.done]
    if not pending:
        return "done"
    run.sampler.set_block(f"{stage}:{alias}:{pass_no}")
    run.event("block_start", alias=alias, stage=stage, pass_no=pass_no, pending=len(pending),
              system=bench_resources.system_snapshot())
    unloaded = run.client.unload_all()
    if unloaded:
        run.event("unloaded_before_block", alias=alias, stage=stage, models=unloaded)
    if run.budget_left() <= 0:
        run.event("budget_exhausted", alias=alias, stage=stage, pass_no=pass_no)
        return "budget"
    if not run.load(alias, stage):
        return "load_failed"
    outcome, first, consecutive = "done", True, 0
    for index, case in pending:
        if run.budget_left() <= 0:
            run.event("budget_exhausted", alias=alias, stage=stage, pass_no=pass_no)
            outcome = "budget"
            break
        if run.sampler.critical_seconds() >= run.protocol["pilot"]["nonviable_if"]["severe_memory_pressure_s"]:
            run.event("memory_abort", alias=alias, stage=stage, pass_no=pass_no,
                      critical_s=round(run.sampler.critical_seconds(), 1))
            outcome = "memory"
            break
        prompt = build_prompt(run.protocol, case["modality"], case["claim"], case["passages"])
        run.save_input(case, prompt)
        payload = payload_for(case, prompt) if payload_for else build_payload(run.protocol, tag, prompt)
        row = run.request(stage, alias, pass_no, case["id"], payload, timeout_s, index, first, parse=parse)
        first = False
        print(f"[{now()}] {stage} {alias} p{pass_no} {case['id']}: {row['status']} {row['wall_s']:.1f}s "
              f"(budget used {run.spent_s / 3600:.2f} h)", flush=True)
        if row["status"] == "timeout":
            stopped = run.verify_stopped()
            run.event("timeout_cancel_check", alias=alias, stage=stage, case_id=case["id"], **stopped)
            if not stopped["stopped"]:
                outcome = "cancel_failed"
                break
            consecutive += 1
            if consecutive >= limits["consecutive_timeouts_abort"]:
                run.event("timeouts_abort", alias=alias, stage=stage, pass_no=pass_no, consecutive=consecutive)
                outcome = "timeouts"
                break
        else:
            consecutive = 0
    if unload_at_end:
        run.client.unload(tag)
    return outcome


# * Stages
# * ===========================================================================
def cancel_check(run: Run, alias: str) -> dict:
    """Aborts a long generation and verifies that the runner stops using CPU before anything else starts."""
    tag = run.models[alias]["tag"]
    options = {**run.protocol["request"]["options"], "num_predict": 2000}
    payload = {"model": tag, "prompt": "Escribe una lista numerada muy larga de ciudades inventadas.", "stream": True,
               "think": False, "keep_alive": run.protocol["request"]["keep_alive"], "options": options}
    before = {r["pid"]: r["cpu_s"] for r in bench_resources.runners()}
    result = run.client.generate(payload, timeout_s=8)
    after = {r["pid"]: r["cpu_s"] for r in bench_resources.runners()}
    used = max((after[p] - before.get(p, after[p]) for p in after), default=0.0)
    run.spent_s += result["wall_s"]
    check = run.verify_stopped()
    return run.event("cancel_check", alias=alias, model_seconds=result["wall_s"], request_status=result["status"],
                     tokens_received=len(result["response"]) > 0, runner_cpu_s_during_request=round(used, 1), **check)


def token_fit_check(run: Run, alias: str, cases: list) -> dict:
    """Counts the tokens of the three longest prompts. Asks for one token and does not keep it."""
    request = run.protocol["request"]
    prompts = sorted((build_prompt(run.protocol, c["modality"], c["claim"], c["passages"]) for c in cases),
                     key=len, reverse=True)[:3]
    counts, seconds = [], 0.0
    for prompt in prompts:
        payload = {"model": run.models[alias]["tag"], "prompt": prompt, "stream": True, "think": False,
                   "keep_alive": request["keep_alive"], "options": {**request["options"], "num_predict": 1}}
        result = run.client.generate(payload, run.protocol["limits"]["request_timeout_s"])
        seconds += result["wall_s"]
        counts.append(result["final"].get("prompt_eval_count"))
    run.spent_s += seconds
    limit = request["options"]["num_ctx"] - request["options"]["num_predict"]
    fits = all(isinstance(c, int) and c <= limit for c in counts)
    return run.event("token_fit", alias=alias, model_seconds=seconds, prompt_tokens=counts, limit=limit, fits=fits)


def stage_pilot(run: Run) -> None:
    dev_data = yaml.safe_load(DEV_CASES.read_text("utf-8"))
    dev = {c["id"]: c for c in load_cases(DEV_CASES)}
    pilot = [dev[i] for i in dev_data["pilot_cases"]]
    eval_cases = load_cases(EVAL_CASES)
    rules = run.protocol["pilot"]["nonviable_if"]
    for alias in run.models:
        if alias in run.viability:
            continue
        outcome = run_block(run, "pilot", alias, 0, pilot, unload_at_end=False)
        reasons = []
        if outcome == "load_failed":
            reasons.append("the model failed to load")
        elif outcome == "memory":
            reasons.append("severe sustained memory pressure")
        rows = [r for r in read_jsonl(run.responses_path) if r["stage"] == "pilot" and r["alias"] == alias]
        if sum(r["status"] == "timeout" for r in rows) >= rules["timeouts_in_three_warm_cases"]:
            reasons.append("two or more timeouts in the three pilot cases")
        fit = None
        if outcome != "load_failed":
            cancel = cancel_check(run, alias)
            if not cancel["stopped"]:
                reasons.append("cancelling a request did not stop inference")
            fit = token_fit_check(run, alias, eval_cases)
            run.client.unload(run.models[alias]["tag"])
        result = run.event("pilot_result", alias=alias, viable=not reasons, reasons=reasons, block_outcome=outcome,
                           token_fit=None if fit is None else fit["fits"])
        run.viability[alias] = result
        print(f"[{now()}] pilot {alias}: viable={not reasons} {reasons}", flush=True)
        if fit is not None and not fit["fits"]:
            raise SystemExit(f"Prompts do not fit the context of {alias}: fix the protocol before evaluating.")


def stage_eval(run: Run) -> str:
    cases = load_cases(EVAL_CASES)
    order = run.protocol["order"]
    for pass_no in range(1, order["passes"] + 1):
        ordered = ordered_cases(cases, order["case_shuffle_seed"], pass_no)
        run.event("case_order", pass_no=pass_no, order=[c["id"] for c in ordered])
        for alias in order["model_order_by_pass"][pass_no]:
            if not run.is_viable(alias):
                continue
            outcome = run_block(run, "eval", alias, pass_no, ordered)
            run.event("block_end", stage="eval", alias=alias, pass_no=pass_no, outcome=outcome)
            if outcome == "budget":
                return "budget"
    return "done"


def stage_legacy(run: Run) -> str:
    """The 1B with the app's own prompt and settings, in free text. Not comparable with the common protocol."""
    from prompt import PROMPTS_MODALIDAD
    from RAG3 import formatear_contexto
    from config import config

    settings = config.cfg_param["llm"]["text"]
    legacy = run.protocol["legacy_1b"]
    alias = legacy["alias"]
    if not run.is_viable(alias):
        return "skipped"
    by_id = {c["id"]: c for c in load_cases(EVAL_CASES)}
    cases = [by_id[i] for i in legacy["cases"]]

    def payload_for(case, _prompt):
        sources = {}
        hits = []
        for i, passage in enumerate(case["passages"], start=1):
            ref = sources.setdefault(passage["source"], f"R{len(sources) + 1}")
            hits.append({"ref_id": ref, "location": f"párr. {i}", "filename": passage["source"], "text": passage["text"]})
        prompt = PROMPTS_MODALIDAD[case["modality"]].format(contexto=formatear_contexto(hits), pregunta=case["claim"])
        return {"model": run.models[alias]["tag"], "prompt": prompt, "stream": True,
                "keep_alive": run.protocol["request"]["keep_alive"],
                "options": {"temperature": 0, "num_predict": settings.get("max_tokens", 512)}}

    def parse_free_text(raw, done_reason):
        if done_reason == "length":
            return "truncated", None, "num_predict reached"
        return ("ok", None, None) if raw.strip() else ("format_error", None, "empty answer")

    return run_block(run, "legacy", alias, 0, cases, payload_for=payload_for, parse=parse_free_text,
                     timeout_s=settings.get("timeout_s", 300))


def _timed_pipeline():
    """Times the existing pipeline steps from outside, without changing the application."""
    import RAG3
    import X_AGENT
    records = {"qdrant_record": [], "process_query_chunks_v2": [], "get_answer": []}

    def wrap(owner, name):
        original = getattr(owner, name)

        def timed(*args, **kwargs):
            started = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                records[name].append(time.perf_counter() - started)
        setattr(owner, name, timed)
        return owner, name, original

    patches = [wrap(RAG3.RAG, "qdrant_record"), wrap(RAG3.RAG, "process_query_chunks_v2"),
               wrap(X_AGENT.Agente, "get_answer")]

    @contextlib.contextmanager
    def capture():
        for values in records.values():
            values.clear()
        yield records

    def restore():
        for owner, name, original in patches:
            setattr(owner, name, original)
    return capture, restore


def _source_label(source: dict) -> str:
    fields = [source.get(k) for k in ("filename", "title", "outlet", "date", "location")]
    return " · ".join(str(f) for f in fields if f)


def stage_e2e(run: Run) -> str:
    """One example per mode through the real API code, with the search kept as it is."""
    import api_rag
    import X_AGENT
    import X_qderant
    from fastapi.testclient import TestClient

    location = X_qderant.cfg_param["qdrant"]["location"]
    if location != os.environ.get("QDRANT_LOCATION"):
        raise SystemExit("The end-to-end stage must use the scratch index, never the user's own qdrant_storage.")
    examples = yaml.safe_load(E2E_EXAMPLES.read_text("utf-8"))["examples"]
    api = TestClient(api_rag.app)
    capture, restore = _timed_pipeline()
    previous = read_jsonl(run.e2e_path)
    finished = {(r["alias"], r["modality"]) for r in previous if r["step"] == "example_done"}
    keep_alive, options = run.protocol["request"]["keep_alive"], run.protocol["request"]["options"]

    def save(**row):
        append_jsonl(run.e2e_path, {"finished_at": now(), **row})
        run.spent_s += row.get("model_seconds", 0)

    def post(mode, example, with_comment):
        files, metadata = [], []
        for ref in example["references"]:
            if "path" in ref:
                files.append(("file_ref", (Path(ref["path"]).name, (ROOT / ref["path"]).read_bytes())))
            else:
                files.append(("file_ref", (ref["filename"], ref["text"].encode("utf-8"))))
            metadata.append(ref.get("metadata", {}))
        data = {"modalidad": mode, "texto_query": example["reviewed"], "usar_agente": str(with_comment).lower()}
        if any(metadata):
            data["ref_metadata"] = json.dumps(metadata)
        with capture() as records, contextlib.redirect_stdout(io.StringIO()):
            started = time.perf_counter()
            response = api.post("/validar", files=files, data=data)
            total = time.perf_counter() - started
        if response.status_code != 200:
            raise RuntimeError(f"/validar {mode}: {response.status_code} {response.text[:300]}")
        return response.json(), total, {k: list(v) for k, v in records.items()}

    try:
        for alias in run.protocol["e2e"]["order"]:
            if not run.is_viable(alias):
                continue
            tag = run.models[alias]["tag"]
            for mode, example in examples.items():
                if (alias, mode) in finished:
                    continue
                if run.budget_left() <= 0:
                    run.event("budget_exhausted", stage="e2e", alias=alias, modality=mode)
                    return "budget"
                attempt = 1 + sum(1 for r in previous if r["alias"] == alias and r["modality"] == mode
                                  and r["step"] == "index_new")
                common = {"alias": alias, "tag": tag, "modality": mode, "attempt": attempt}
                run.sampler.set_block(f"e2e:{alias}:{mode}")
                run.client.unload_all()
                qdrant = X_qderant.get_qdrant_client()
                for collection in qdrant.get_collections().collections:
                    qdrant.delete_collection(collection.name)

                # 1. New index, then the same request reusing it
                body, total, records = post(mode, example, False)
                if body["indice"]["reutilizado"] or len(body["resultados"]) != len(example["fragments"]):
                    raise RuntimeError(f"{mode}: unexpected index state or fragments: {body['indice']}, "
                                       f"{len(body['resultados'])} vs {len(example['fragments'])}")
                save(step="index_new", total_s=total, index_s=sum(records["qdrant_record"]),
                     search_s=sum(records["process_query_chunks_v2"]), index_reused=False,
                     chunks=body["indice"]["fragmentos"], fragments=len(body["resultados"]), model_seconds=total, **common)
                body, total, records = post(mode, example, False)
                if not body["indice"]["reutilizado"]:
                    raise RuntimeError(f"{mode}: the second request should reuse the index")
                save(step="index_reused", total_s=total, index_s=sum(records["qdrant_record"]),
                     search_s=sum(records["process_query_chunks_v2"]), index_reused=True,
                     fragments=len(body["resultados"]), model_seconds=total, **common)

                # 2. Structured judge on what was really retrieved, to separate search failures from judge failures
                if not run.load(alias, f"e2e_judge:{mode}"):
                    save(step="load_failed", model_seconds=0, **common)
                    continue
                judge_started = time.perf_counter()
                for index, (result, expected) in enumerate(zip(body["resultados"], example["fragments"]), start=1):
                    passages = [{"id": f"P{i}", "source": _source_label(source), "text": text}
                                for i, (text, source) in enumerate(zip(result["relevant_chunks"], result["sources"]), 1)]
                    prompt = build_prompt(run.protocol, mode, result["chunk"], passages)
                    evidence = expected.get("evidence")
                    needed = None if not evidence else any(
                        normalize_for_match(evidence) in normalize_for_match(p["text"]) for p in passages)
                    answer = run.client.generate(build_payload(run.protocol, tag, prompt),
                                                 run.protocol["limits"]["request_timeout_s"])
                    if answer["status"] == "timeout":
                        run.event("timeout_cancel_check", alias=alias, stage="e2e_judge", **run.verify_stopped())
                    status, parsed, detail = answer["status"], None, answer["error"]
                    if status == "completed":
                        status, parsed, detail = parse_response(answer["response"], answer["final"].get("done_reason"))
                    predicted = parsed["label"] if parsed else "no_valid_answer"
                    values = expected.get("required_values", [])
                    save(step="judge", fragment=index, claim=result["chunk"], passages=passages, scores=result["scores"],
                         status=status, detail=detail, parsed=parsed, raw_response=answer["response"],
                         expected=expected["label"], predicted=predicted, needed_passage_retrieved=needed,
                         outcome=attribute_e2e(expected["label"], predicted, needed), tags=expected.get("tags", []),
                         values_named=None if not (values and parsed) else all(
                             value_mentioned(str(v), parsed["explanation"]) for v in values),
                         wall_s=answer["wall_s"], ttft_s=answer["ttft_s"], first_after_load=index == 1,
                         model_seconds=answer["wall_s"], **timing_fields(answer["final"]), **common)
                    print(f"[{now()}] e2e judge {alias} {mode} #{index}: {status} {predicted} "
                          f"(expected {expected['label']}) retrieved={needed} {answer['wall_s']:.1f}s", flush=True)
                save(step="judge_document", total_s=time.perf_counter() - judge_started, fragments=len(body["resultados"]),
                     **common)
                run.client.unload(tag)

                # 3. The app as it is today: its own prompt, free text, no think option, cold model on the first fragment
                api_rag.agente_dinamico = X_AGENT.Agente(X_AGENT.ollama_endpoint, tag)
                body, total, records = post(mode, example, True)
                comments = [{"fragment": r["fragmento"], "wall_s": seconds, "comment": r.get("comentario"),
                             "error": r.get("comentario_error"), "first_after_load": i == 0,
                             "numbers_not_in_inputs": numbers_not_in_inputs(r.get("comentario") or "", r["chunk"] + r["context"]),
                             "capitalized_not_in_inputs": capitalized_not_in_inputs(r.get("comentario") or "", r["chunk"] + r["context"])}
                            for i, (r, seconds) in enumerate(zip(body["resultados"], records["get_answer"]))]
                save(step="app_comment", total_s=total, index_s=sum(records["qdrant_record"]),
                     search_s=sum(records["process_query_chunks_v2"]), comment_s=sum(records["get_answer"]),
                     index_reused=body["indice"]["reutilizado"], comments=comments, fragments=len(body["resultados"]),
                     model_seconds=total, **common)
                print(f"[{now()}] e2e app {alias} {mode}: total {total:.1f}s, "
                      f"empty or failed {sum(1 for c in comments if not c['comment'])}/{len(comments)}", flush=True)
                run.client.unload_all()
                save(step="example_done", **common)
                write_state(run)
    finally:
        restore()
    return "done"


# * Manifest and state
# * ===========================================================================
def _model_details(client, tag: str, digest: str) -> dict:
    listed = client.tags().get(tag)
    if listed is None:
        raise SystemExit(f"{tag} is not installed.")
    if listed["digest"] != digest:
        raise SystemExit(f"{tag} has digest {listed['digest']}, not the {digest} in the protocol.")
    shown = client.show(tag)
    info = shown.get("model_info", {})
    template = shown.get("template") or ""
    details = {k: v for k, v in shown.get("details", {}).items() if k != "parent_model"}  # may hold a local path
    repo, _, version = tag.partition(":")
    manifest_path = Path.home() / ".ollama/models/manifests/registry.ollama.ai/library" / repo / (version or "latest")
    layers = None
    if manifest_path.exists():
        layers = [{"media_type": layer["mediaType"], "size_bytes": layer["size"], "digest": layer["digest"]}
                  for layer in json.loads(manifest_path.read_text())["layers"]]
    return {"tag": tag, "digest": listed["digest"], "size_bytes": listed["size"], "details": details,
            "capabilities": shown.get("capabilities"), "thinking": shown.get("thinking"),
            "inherited_parameters": shown.get("parameters"), "template_sha256": sha256_text(template),
            "template": template, "renderer": shown.get("renderer"), "parser": shown.get("parser"),
            "context_length": next((v for k, v in info.items() if k.endswith("context_length")), None),
            "parameter_count": info.get("general.parameter_count"), "layers": layers}


def _git(*args) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def _server_log() -> list:
    path = Path.home() / ".ollama/logs/server.log"
    return path.read_text("utf-8", errors="ignore").splitlines() if path.exists() else []


def count_truncations(lines: list) -> int:
    """Server log lines that report a truncated prompt. "truncated = 0" is the normal case and does not count."""
    return sum(bool(re.search(r"truncated = [1-9]|truncating input", line, re.I)) for line in lines)


def backend_from_log(lines: list) -> dict:
    """What the server itself says it runs on. Only these fields are kept, never the log."""
    for line in reversed(lines):
        if "inference compute" in line:
            return {k: re.search(rf'\b{k}=("[^"]*"|\S+)', line).group(1).strip('"') if re.search(rf'\b{k}=', line) else None
                    for k in ("id", "library", "total")}
    return {}


def ensure_manifest(run: Run, frozen: bool) -> None:
    hashes, harness = locked_hashes(), harness_hash()
    path = run.dir / "manifest.json"
    if path.exists():
        saved = json.loads(path.read_text("utf-8"))
        if saved["locked_files"] != hashes or saved["harness_sha256"] != harness:
            raise SystemExit("Protocol, datasets or harness differ from this run: use a new --run-id instead of mixing "
                             "configurations.")
        run.event("resumed", commit=_git("rev-parse", "HEAD"))
        return
    if frozen:
        if not LOCK_FILE.exists():
            raise SystemExit("No protocol.lock.json: run the freeze command first (or use --unfrozen for development).")
        lock = json.loads(LOCK_FILE.read_text("utf-8"))
        if lock["files"] != hashes:
            raise SystemExit("The protocol or the datasets changed after the freeze: new version required.")
    protocol = run.protocol
    manifest = {
        "run_id": run.dir.name, "created_at": now(), "frozen": frozen, "protocol_version": protocol["version"],
        "commit": _git("rev-parse", "HEAD"), "tracked_changes_at_start": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "locked_files": hashes, "harness_sha256": harness, "ollama_version": run.client.version(),
        "hardware": bench_resources.hardware_info(),
        "backend": backend_from_log(_server_log()),
        "python": sys.version.split()[0],
        "models": {m["alias"]: _model_details(run.client, m["tag"], m["digest"]) for m in protocol["models"]},
        "embedding_model": {k: v for k, v in run.client.tags().get(protocol["embedding_model"], {}).items()
                            if k in ("name", "digest", "size")},
        "effective_request": protocol["request"],
        "data_note": yaml.safe_load(EVAL_CASES.read_text("utf-8"))["review_status"],
        "server_log_truncation_lines_at_start": count_truncations(_server_log()),
        "system_at_start": bench_resources.system_snapshot(),
    }
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), "utf-8")


def write_state(run: Run) -> None:
    """Which cases finished, failed or never ran, recomputed from the saved results."""
    rows = read_jsonl(run.responses_path)
    protocol, order = run.protocol, run.protocol["order"]
    eval_ids = [c["id"] for c in load_cases(EVAL_CASES)]
    blocks = {}
    for pass_no in range(1, order["passes"] + 1):
        for alias in order["model_order_by_pass"][pass_no]:
            saved = {r["case_id"]: r["status"] for r in rows if r["stage"] == "eval" and r["alias"] == alias
                     and r["pass"] == pass_no}
            statuses = {}
            for status in saved.values():
                statuses[status] = statuses.get(status, 0) + 1
            blocks[f"eval|{alias}|{pass_no}"] = {"planned": len(eval_ids), "statuses": statuses,
                                                 "not_executed": [i for i in eval_ids if i not in saved]}
    for stage, planned in (("pilot", 3), ("legacy", len(protocol["legacy_1b"]["cases"]))):
        for alias in run.models:
            saved = [r for r in rows if r["stage"] == stage and r["alias"] == alias]
            if saved or stage == "pilot":
                blocks[f"{stage}|{alias}"] = {"planned": planned, "attempted": len(saved)}
    e2e = {}
    for row in read_jsonl(run.e2e_path):
        if row["step"] == "example_done":
            e2e.setdefault(row["alias"], []).append(row["modality"])
    state = {"updated_at": now(), "budget_used_hours": round(run.spent_s / 3600, 3),
             "budget_hours": protocol["limits"]["budget_model_hours"],
             "viability": {a: {"viable": v["viable"], "reasons": v["reasons"]} for a, v in run.viability.items()},
             "blocks": blocks, "e2e_examples_finished": e2e,
             "server_log_truncation_lines": count_truncations(_server_log())}
    (run.dir / "state.json").write_text(json.dumps(state, indent=2, ensure_ascii=False), "utf-8")


# * Commands
# * ===========================================================================
def freeze(note: str) -> None:
    protocol = load_protocol()
    LOCK_FILE.write_text(json.dumps({"protocol_version": protocol["version"], "frozen_at": now(), "note": note,
                                     "files": locked_hashes()}, indent=2) + "\n", "utf-8")
    print(f"Frozen protocol v{protocol['version']} in {LOCK_FILE.name}")


def run_command(args) -> None:
    run_dir = Path(args.results_dir) / args.run_id
    # The index of the end-to-end stage lives outside the repository and is deleted at the end
    scratch = tempfile.mkdtemp(prefix="srr_benchmark_qdrant_")
    os.environ["QDRANT_LOCATION"] = scratch
    os.environ.pop("LLM_MODEL", None)

    lock = open(Path(tempfile.gettempdir()) / "srr_benchmark.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        raise SystemExit("Another benchmark process is running: one inference at a time.")
    if shutil.which("caffeinate"):
        subprocess.Popen(["caffeinate", "-ims", "-w", str(os.getpid())])  # ends with this process; no permanent change

    protocol = load_protocol()
    client = OllamaClient()
    run_dir.mkdir(parents=True, exist_ok=True)
    sampler = bench_resources.ResourceSampler(run_dir / "resources.jsonl", protocol["resources"]["interval_s"])
    run = Run(run_dir, protocol, client, sampler=sampler)
    ensure_manifest(run, frozen=not args.unfrozen)
    sampler.start()
    stages = {"pilot": stage_pilot, "eval": stage_eval, "legacy": stage_legacy, "e2e": stage_e2e}
    try:
        for name in args.stages.split(","):
            run.event("stage_start", stage=name, system=bench_resources.system_snapshot())
            outcome = stages[name](run)
            run.event("stage_end", stage=name, outcome=outcome, budget_used_hours=round(run.spent_s / 3600, 3),
                      system=bench_resources.system_snapshot())
            write_state(run)
            if outcome == "budget":
                print("Model budget exhausted: the remaining cases stay as not executed.", flush=True)
                break
    finally:
        with contextlib.suppress(Exception):
            client.unload_all()
        sampler.stop()
        write_state(run)
        shutil.rmtree(scratch, ignore_errors=True)
    print(f"Finished. Model time used: {run.spent_s / 3600:.2f} h. Results in {run_dir.name}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    frozen = commands.add_parser("freeze", help="store the hashes of the protocol and the datasets")
    frozen.add_argument("--note", default="")
    runner = commands.add_parser("run", help="run or resume stages")
    runner.add_argument("--run-id", required=True)
    runner.add_argument("--stages", default="pilot,eval,legacy,e2e")
    runner.add_argument("--results-dir", default=str(BENCH_DIR / "results"))
    runner.add_argument("--unfrozen", action="store_true", help="development only: do not require the freeze")
    args = parser.parse_args()
    if args.command == "freeze":
        freeze(args.note)
    else:
        run_command(args)


if __name__ == "__main__":
    main()
