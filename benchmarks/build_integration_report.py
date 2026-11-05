#!/usr/bin/env python
"""Summarises an integration check: writes summary.json and tables.md next to its results, and changes nothing else.

An answer has two readings. The raw label is whatever the model wrote. The accepted label is what the application shows:
only an answer that passed validation has one, so a rejected answer is never a hit, even if its raw label was right.

    python benchmarks/build_integration_report.py --run-id <id>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import yaml  # noqa: E402

import bench_core as core  # noqa: E402

EVAL_CASES = core.BENCH_DIR / "dataset" / "eval_cases.yaml"
CONTROL_CASES = core.BENCH_DIR / "dataset" / "integration_cases.yaml"
E2E_EXAMPLES = core.BENCH_DIR / "dataset" / "e2e_examples.yaml"


def raw_view(raw_response, done_reason) -> dict:
    """What the model wrote, before the application validates its evidence."""
    status, parsed, _ = core.parse_response(raw_response, done_reason)
    return {"alias": "4B", "pass": 1, "status": status, "parsed": parsed}


def accepted_view(analysis: dict, raw_response, done_reason) -> dict:
    """What the application accepts. no_validado and no_disponible carry no label, whatever the model wrote."""
    if analysis["estado"] == "validado":
        return raw_view(raw_response, done_reason)
    if analysis["estado"] == "no_validado":
        status = "truncated" if done_reason == "length" else "rejected"
    else:
        status = "timeout" if "se ha cancelado" in (analysis["motivo"] or "") else "request_error"
    return {"alias": "4B", "pass": 1, "status": status, "parsed": None}


def score_answer(analysis: dict, raw_response, done_reason, case: dict) -> dict:
    raw = core.score_row(raw_view(raw_response, done_reason), case)
    accepted = core.score_row(accepted_view(analysis, raw_response, done_reason), case)
    return {"case_id": case["id"], "expected": case["label"], "critical": case["critical"], "tags": case["tags"],
            "state": analysis["estado"], "reason": analysis["motivo"], "raw": raw, "accepted": accepted,
            "raw_label": raw["predicted"], "shown_label": accepted["predicted"], "explanation": analysis["explicacion"],
            "evidence": analysis["evidencia"]}


def stage_summary(scored: list, planned: int) -> dict:
    rejected = [s for s in scored if s["state"] == "no_validado"]
    unavailable = [s for s in scored if s["state"] == "no_disponible"]
    return {
        "planned": planned, "attempted": len(scored), "accepted": sum(s["state"] == "validado" for s in scored),
        "rejected": len(rejected), "unavailable": len(unavailable), "without_result": len(rejected) + len(unavailable),
        "rejected_cases": [{"case_id": s["case_id"], "raw_label": s["raw_label"], "expected": s["expected"], "reason": s["reason"]} for s in rejected],
        "unavailable_cases": [{"case_id": s["case_id"], "reason": s["reason"]} for s in unavailable],
        "raw": core.quality_metrics([s["raw"] for s in scored], planned),
        "accepted_view": core.quality_metrics([s["accepted"] for s in scored], planned),
    }


def latency_row(row: dict) -> dict:
    m = row["analysis"]["metricas"]
    return {"status": "ok" if row["request_status"] == "completed" else "timeout" if row["request_status"] == "timeout" else "request_error",
            "first_after_load": row["first_after_load"], "wall_s": m["tiempo_s"], "ttft_s": m["primer_token_s"], "prompt_eval_s": m["evaluacion_prompt_s"],
            "eval_s": m["generacion_s"], "prompt_tokens": m["tokens_prompt"], "prompt_cached_tokens": m["tokens_prompt_en_cache"], "output_tokens": m["tokens_salida"],
            "generation_tokens_per_s": m["tokens_salida"] / m["generacion_s"] if m["tokens_salida"] and m["generacion_s"] else None, "load_s": m["carga_s"]}


def e2e_summary(row: dict, example: dict) -> dict:
    fragments = []
    for result, expected, call in zip(row["response"]["resultados"], example["fragments"], row["calls"]):
        analysis = result["analisis"]
        evidence = core.normalize_for_match(expected.get("evidence") or "")
        retrieved = any(evidence in core.normalize_for_match(p) for p in result["relevant_chunks"]) if evidence else None
        shown = analysis["etiqueta"] or core.NO_ANSWER
        raw = raw_view(call["response"], call["done_reason"])
        fragments.append({
            "fragment": result["fragmento"], "claim": result["chunk"], "expected": expected["label"], "shown": shown, "state": analysis["estado"],
            "raw_label": raw["parsed"]["label"] if raw["parsed"] else core.NO_ANSWER, "reason": analysis["motivo"], "retrieved_needed_passage": retrieved,
            "attribution": core.attribute_e2e(expected["label"], shown, retrieved) if analysis["estado"] == "validado" else "no_result",
            "required_values_ok": all(core.value_mentioned(str(v), analysis["explicacion"] or "") for v in expected["required_values"]) if expected.get("required_values") else None,
            "explanation": analysis["explicacion"], "quotes": [e["cita"] for e in analysis["evidencia"] if e["cita"]], "wall_s": call["wall_s"], "load_s": analysis["metricas"]["carga_s"],
            "prompt_tokens": analysis["metricas"]["tokens_prompt"], "output_tokens": analysis["metricas"]["tokens_salida"]})
    walls = [f["wall_s"] for f in fragments]
    return {"modality": row["modality"], "fragments": len(fragments), "fragments_expected": len(example["fragments"]), "without_analysis_s": row["without_analysis_s"],
            "with_analysis_s": row["with_analysis_s"], "judge_s": row["judge_s"], "rest_of_document_s": row["rest_of_document_s"], "first_fragment_s": walls[0] if walls else None,
            "median_other_fragments_s": core.median(walls[1:]), "correct": sum(f["expected"] == f["shown"] for f in fragments),
            "same_passages_with_and_without_analysis": row["same_passages_with_and_without_analysis"], "checks_ok": row["checks"]["all_ok"], "detail": fragments}


def memory_summary(path: Path) -> dict:
    rows = core.read_jsonl(path) if path.exists() else []
    rss = [r["rss_mib"] for s in rows for r in s["runners"] if r.get("rss_mib")]
    free = [s["memory_free_pct"] for s in rows if s.get("memory_free_pct") is not None]
    swap = [s["swap_used_mib"] for s in rows if s.get("swap_used_mib") is not None]
    return {"samples": len(rows), "max_runner_rss_mib_sampled": max(rss, default=None), "min_memory_free_pct": min(free, default=None),
            "swap_used_mib_first": swap[0] if swap else None, "swap_used_mib_max": max(swap, default=None)}


def fmt_s(value) -> str:
    return "n/a" if value is None else f"{value:.1f} s"


def build_tables(summary: dict) -> str:
    reg, ctl = summary["regression"], summary["controls"]
    lat = summary["latency"]
    lines = ["<!-- tables:start -->", "### Regression on the 30 known cases", "",
             "| Reading | Correct | Macro-F1 | Wrongly supported |", "|---|---|---|---|"]
    for name, view in (("Raw label, before validation", reg["raw"]), ("Shown by the application", reg["accepted_view"])):
        lines.append(f"| {name} | {view['correct']}/{reg['planned']} | {view['macro_f1']:.3f} | {view['false_support']}/{view['negatives_attempted']} |")
    lines += ["", f"Accepted {reg['accepted']}/{reg['planned']}; rejected by validation {reg['rejected']}; technical failures {reg['unavailable']}; "
              f"without a result {reg['without_result']}/{reg['planned']}.", "", "### Control cases, fixed context", "",
              "| Case | Tags | Expected | Raw label | Shown |", "|---|---|---|---|---|"]
    for s in summary["control_cases"]:
        lines.append(f"| {s['case_id']} | {', '.join(s['tags'])} | {s['expected']} | {s['raw_label']} | {s['shown_label']} |")
    lines += ["", f"Shown correctly {ctl['accepted_view']['correct']}/{ctl['planned']}; rejected {ctl['rejected']}; technical failures {ctl['unavailable']}.", "",
              "### One document per mode, through the API", "",
              "| Mode | Fragments | Shown correctly | Without analysis | With analysis | Judge only | First fragment | Median of the others |", "|---|---|---|---|---|---|---|---|"]
    for e in summary["e2e"]:
        lines.append(f"| `{e['modality']}` | {e['fragments']} | {e['correct']}/{e['fragments']} | {fmt_s(e['without_analysis_s'])} | {fmt_s(e['with_analysis_s'])} | {fmt_s(e['judge_s'])} | {fmt_s(e['first_fragment_s'])} | {fmt_s(e['median_other_fragments_s'])} |")
    warm = lat["warm_wall_s"]
    lines += ["", "### Latency and memory of the fixed-context stages", "",
              f"Median per fragment {fmt_s(warm['median'])}, p95 {fmt_s(warm['p95'])} (n = {warm['n']}, model already loaded). "
              f"First request after loading: median {fmt_s(lat['first_after_load_wall_s']['median'])} (n = {lat['first_after_load_wall_s']['n']}). "
              f"Prompt median {lat['prompt_tokens']['median']} tokens, output median {lat['output_tokens']['median']} tokens. "
              f"Maximum sampled RSS of the runner {summary['memory']['max_runner_rss_mib_sampled']} MiB; lowest free memory {summary['memory']['min_memory_free_pct']} %.",
              "<!-- tables:end -->"]
    return "\n".join(lines) + "\n"


def build_summary(run_dir: Path) -> dict:
    rows = core.read_jsonl(run_dir / "responses.jsonl")
    summary = {"run_id": run_dir.name, "manifest": json.loads((run_dir / "manifest.json").read_text("utf-8")),
               "run_status": json.loads((run_dir / "run_status.json").read_text("utf-8"))}
    for stage, path in (("regression", EVAL_CASES), ("controls", CONTROL_CASES)):
        cases = {c["id"]: c for c in core.load_cases(path)}
        selected = [r for r in rows if r["stage"] == stage]
        scored = [score_answer(r["analysis"], r["raw_response"], r["done_reason"], cases[r["case_id"]]) for r in selected]
        summary[stage] = stage_summary(scored, len(cases))
        summary[f"{'regression' if stage == 'regression' else 'control'}_cases"] = [{k: v for k, v in s.items() if k not in ("raw", "accepted")} for s in scored]
    summary["latency"] = core.latency_metrics([latency_row(r) for r in rows])
    load = [r["analysis"]["metricas"]["carga_s"] for r in rows if r["first_after_load"] and r["analysis"]["metricas"]["carga_s"] is not None]
    summary["latency"]["first_after_load_load_s"] = core.median(load)
    examples = yaml.safe_load(E2E_EXAMPLES.read_text("utf-8"))["examples"]
    summary["e2e"] = [e2e_summary(r, examples[r["modality"]]) for r in core.read_jsonl(run_dir / "e2e.jsonl")]
    summary["memory"] = memory_summary(run_dir / "resources.jsonl")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    run_dir = core.BENCH_DIR / "results" / args.run_id
    summary = build_summary(run_dir)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), "utf-8")
    (run_dir / "tables.md").write_text(build_tables(summary), "utf-8")
    print(build_tables(summary))


if __name__ == "__main__":
    main()
