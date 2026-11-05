#!/usr/bin/env python
"""Turns the saved results of a run into metrics, a CSV, a chart and the tables of docs/model-comparison.md.

    python benchmarks/build_report.py --run-id <id>

No model is called. Every table and figure comes from the files in benchmarks/results/<id>/, and only the text between
<!-- BEGIN GENERATED: name --> and <!-- END GENERATED: name --> is replaced in the document.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_core import (BENCH_DIR, LABELS, NO_ANSWER, ROOT, capitalized_not_in_inputs, latency_metrics,  # noqa: E402
                        load_cases, load_protocol, median, numbers_not_in_inputs, percentile_nearest_rank,
                        quality_metrics, read_jsonl, score_row, value_mentioned)

DOC = ROOT / "docs" / "model-comparison.md"
FIGURE = ROOT / "docs" / "figures" / "model-comparison.svg"
SERIES_COLORS = {"1B": ("#2a78d6", "#3987e5"), "4B": ("#eb6834", "#d95926"), "9B": ("#1baf7a", "#199e70")}  # all-pairs validated


def fmt(value, digits=1, empty="N/D"):
    return empty if value is None else f"{value:.{digits}f}"


def fraction(k, n):
    return f"{k}/{n}"


def pct_interval(interval):
    return "N/D" if not interval else f"{interval[0] * 100:.0f}–{interval[1] * 100:.0f} %"


# * Computation
# * ===========================================================================
def compute(run_dir: Path, protocol: dict) -> dict:
    cases = {c["id"]: c for c in load_cases(BENCH_DIR / "dataset" / "eval_cases.yaml")}
    rows = read_jsonl(run_dir / "responses.jsonl")
    events = read_jsonl(run_dir / "events.jsonl")
    resources = read_jsonl(run_dir / "resources.jsonl")
    e2e = read_jsonl(run_dir / "e2e.jsonl")
    manifest = json.loads((run_dir / "manifest.json").read_text("utf-8"))
    aliases = [m["alias"] for m in protocol["models"]]
    passes = list(range(1, protocol["order"]["passes"] + 1))
    planned = len(cases)

    results = {"aliases": aliases, "passes": passes, "planned_per_pass": planned, "models": {}, "manifest": manifest}
    all_scores = []
    for alias in aliases:
        per_pass = {}
        for pass_no in passes:
            subset = [r for r in rows if r["stage"] == "eval" and r["alias"] == alias and r["pass"] == pass_no]
            scores = [score_row(r, cases[r["case_id"]]) for r in subset]
            all_scores.extend(scores)
            per_pass[pass_no] = {"quality": quality_metrics(scores, planned), "latency": latency_metrics(subset),
                                 "thinking_chars": sum(r.get("thinking_chars") or 0 for r in subset),
                                 "first_token_reported": sum(r.get("ttft_s") is not None for r in subset),
                                 "max_prompt_tokens": max((r.get("prompt_tokens") or 0 for r in subset), default=None)}
        results["models"][alias] = {"passes": per_pass, "repeatability": _repeatability(rows, alias),
                                    "by_tag": _by_tag(all_scores, alias, protocol["selection"]["primary_pass"]),
                                    "loads": [e for e in events if e["event"] == "cold_load" and e.get("alias") == alias
                                              and e.get("stage") == "eval"],
                                    "pilot": _pilot(rows, events, alias), "resources": _resources(resources, manifest, alias),
                                    "cpu_limit_by_pass": {p: _cpu_limit(resources, alias, p) for p in passes}}
    results["scores"] = all_scores
    results["discordant"] = _discordant(all_scores, aliases, protocol["selection"]["primary_pass"])
    results["selection"] = apply_selection_rule(results, protocol)
    results["e2e"] = _e2e(e2e)
    results["legacy"] = _legacy(rows, cases)
    results["timeline"] = _timeline(events, rows, e2e)
    results["state"] = json.loads((run_dir / "state.json").read_text("utf-8")) if (run_dir / "state.json").exists() else {}
    return results


def _repeatability(rows, alias):
    first = {r["case_id"]: r for r in rows if r["stage"] == "eval" and r["alias"] == alias and r["pass"] == 1}
    second = {r["case_id"]: r for r in rows if r["stage"] == "eval" and r["alias"] == alias and r["pass"] == 2}
    both = sorted(set(first) & set(second))

    def label(r):
        return (r.get("parsed") or {}).get("label", NO_ANSWER)
    return {"cases_in_both": len(both), "same_label": sum(label(first[c]) == label(second[c]) for c in both),
            "same_text": sum(first[c]["raw_response"] == second[c]["raw_response"] for c in both),
            "label_changes": [{"case_id": c, "pass_1": label(first[c]), "pass_2": label(second[c])}
                              for c in both if label(first[c]) != label(second[c])]}


def _by_tag(scores, alias, primary):
    table = {}
    for score in scores:
        if score["model"] == alias and score["pass"] == primary:
            for tag in score["tags"]:
                entry = table.setdefault(tag, [0, 0])
                entry[0] += score["correct"]
                entry[1] += 1
    return table


def _pilot(rows, events, alias):
    subset = [r for r in rows if r["stage"] == "pilot" and r["alias"] == alias]
    result = next((e for e in reversed(events) if e["event"] == "pilot_result" and e["alias"] == alias), None)
    cancel = next((e for e in reversed(events) if e["event"] == "cancel_check" and e["alias"] == alias), None)
    fit = next((e for e in reversed(events) if e["event"] == "token_fit" and e["alias"] == alias), None)
    return {"statuses": [r["status"] for r in subset], "wall_s": [r["wall_s"] for r in subset],
            "viable": None if result is None else result["viable"], "reasons": [] if result is None else result["reasons"],
            "cancel": cancel, "token_fit": fit}


def _resources(samples, manifest, alias):
    details = manifest["models"][alias]
    blob = next((l["digest"].split(":")[1][:12] for l in details.get("layers") or []
                 if l["media_type"].endswith("image.model")), None)
    block = [s for s in samples if s["block"].startswith(("eval:" + alias + ":", "pilot:" + alias + ":"))]
    rss = [r["rss_mib"] for s in block for r in s["runners"] if r["blob"] == blob]

    def values(key):
        return [s[key] for s in block if s.get(key) is not None]
    swap = values("swap_used_mib")
    return {"samples": len(block), "rss_median_mib": median(rss), "rss_max_sampled_mib": max(rss) if rss else None,
            "memory_free_pct_min": min(values("memory_free_pct"), default=None),
            "memory_free_pct_median": median(values("memory_free_pct")),
            "pressure_level_max": max(values("pressure_level"), default=None),
            "pressure_critical_samples": sum(1 for v in values("pressure_level") if v == 4),
            "pressure_warn_samples": sum(1 for v in values("pressure_level") if v == 2),
            "swap_first_mib": swap[0] if swap else None, "swap_last_mib": swap[-1] if swap else None,
            "swap_max_mib": max(swap) if swap else None,
            "cpu_speed_limit_min_pct": min(values("cpu_speed_limit_pct"), default=None)}


def _cpu_limit(samples, alias, pass_no):
    values = [s["cpu_speed_limit_pct"] for s in samples if s["block"] == f"eval:{alias}:{pass_no}"
              and s.get("cpu_speed_limit_pct") is not None]
    return {"median": median(values), "min": min(values, default=None)}


def _discordant(scores, aliases, primary):
    correct = {(s["model"], s["case_id"]): s["correct"] for s in scores if s["pass"] == primary}
    table = {}
    for i, a in enumerate(aliases):
        for b in aliases[i + 1:]:
            shared = [c for m, c in correct if m == a and (b, c) in correct]
            table[f"{a}|{b}"] = {"shared": len(shared), "only_first_right": sum(correct[(a, c)] and not correct[(b, c)] for c in shared),
                                 "only_second_right": sum(correct[(b, c)] and not correct[(a, c)] for c in shared)}
    return table


def apply_selection_rule(results, protocol) -> dict:
    """The rule frozen in the protocol, applied mechanically to the primary pass."""
    selection = protocol["selection"]
    primary = selection["primary_pass"]
    planned = results["planned_per_pass"]
    complete = [a for a in results["aliases"]
                if all(results["models"][a]["passes"][p]["quality"]["attempted"] == planned for p in results["passes"])]
    def key(alias, pass_no):
        quality = results["models"][alias]["passes"][pass_no]["quality"]
        return quality["false_support"], -(quality["macro_f1"] or 0)
    ranking = sorted(complete, key=lambda a: key(a, primary))
    other = [p for p in results["passes"] if p != primary]
    ranking_other = {p: sorted(complete, key=lambda a: key(a, p)) for p in other}
    outcome = {"complete": complete, "ranking_primary": ranking, "ranking_other_pass": ranking_other,
               "flips_between_passes": any(r != ranking for r in ranking_other.values())}
    critical = {a: results["models"][a]["passes"][primary]["quality"]["critical_false_support"] for a in complete}
    outcome["critical_false_supports"] = critical
    target = selection["latency_target_s"]
    outcome["meets_latency_target"] = {a: (results["models"][a]["passes"][primary]["latency"]["warm_wall_s"]["median"] or 1e9) <= target
                                       for a in complete}
    if len(ranking) >= 2:
        best, second = (results["models"][a]["passes"][primary]["quality"] for a in ranking[:2])
        gap_cases = abs(best["correct"] - second["correct"])
        outcome["top_gap"] = {"false_support": second["false_support"] - best["false_support"],
                              "correct": best["correct"] - second["correct"], "macro_f1": (best["macro_f1"] or 0) - (second["macro_f1"] or 0),
                              "same_critical": best["critical_false_support"] == second["critical_false_support"]}
        small = gap_cases <= 1 and abs(best["false_support"] - second["false_support"]) <= 1 and outcome["top_gap"]["same_critical"]
        outcome["robust_winner"] = None if small or outcome["flips_between_passes"] else ranking[0]
    else:
        outcome["top_gap"] = None
        outcome["robust_winner"] = ranking[0] if ranking else None
    outcome["keep_comments_off_by_default"] = any(critical.values()) or not complete
    return outcome


def _latest_complete(rows):
    latest = {}
    for row in rows:
        if row["step"] == "example_done":
            latest[(row["alias"], row["modality"])] = row["attempt"]
    return latest


def _e2e(rows):
    attempts = _latest_complete(rows)
    table = []
    fragments = []
    for (alias, mode), attempt in attempts.items():
        subset = [r for r in rows if r["alias"] == alias and r["modality"] == mode and r["attempt"] == attempt]
        by_step = {r["step"]: r for r in subset}
        judges = [r for r in subset if r["step"] == "judge"]
        app = by_step.get("app_comment", {})
        comments = app.get("comments", [])
        warm = [c["wall_s"] for c in comments if not c["first_after_load"]]
        judge_warm = [r["wall_s"] for r in judges if not r["first_after_load"]]
        table.append({
            "alias": alias, "modality": mode, "fragments": by_step["index_new"]["fragments"],
            "chunks": by_step["index_new"]["chunks"], "index_new_total_s": by_step["index_new"]["total_s"],
            "index_new_index_s": by_step["index_new"]["index_s"], "index_new_search_s": by_step["index_new"]["search_s"],
            "index_reused_total_s": by_step["index_reused"]["total_s"], "index_reused_index_s": by_step["index_reused"]["index_s"],
            "index_reused_search_s": by_step["index_reused"]["search_s"],
            "judge_document_s": by_step.get("judge_document", {}).get("total_s"),
            "judge_warm_median_s": median(judge_warm), "judge_first_s": judges[0]["wall_s"] if judges else None,
            "judge_correct": sum(r["outcome"] == "correct" for r in judges),
            "judge_errors": sum(r["outcome"] == "judge_error" for r in judges),
            "retrieval_misses": sum(r["outcome"] == "retrieval_miss" for r in judges),
            "needed_retrieved": sum(r["needed_passage_retrieved"] is True for r in judges),
            "needed_total": sum(r["needed_passage_retrieved"] is not None for r in judges),
            "judge_failed_requests": sum(r["status"] != "ok" for r in judges),
            "app_total_s": app.get("total_s"), "app_comment_s": app.get("comment_s"),
            "app_warm_median_s": median(warm), "app_warm_max_s": max(warm) if warm else None,
            "app_first_s": comments[0]["wall_s"] if comments else None,
            "app_comments_usable": sum(1 for c in comments if c["comment"]), "app_comments_total": len(comments),
            "app_errors": sorted({c["error"] for c in comments if c["error"]}),
        })
        fragments.extend({"alias": alias, "modality": mode, "fragment": r["fragment"], "claim": r["claim"],
                          "expected": r["expected"], "predicted": r["predicted"], "status": r["status"],
                          "needed_passage_retrieved": r["needed_passage_retrieved"], "outcome": r["outcome"],
                          "values_named": r["values_named"], "tags": r["tags"], "wall_s": r["wall_s"],
                          "explanation": (r["parsed"] or {}).get("explanation"), "top_score": (r["scores"] or [None])[0]}
                         for r in judges)
    return {"documents": table, "fragments": fragments}


def _legacy(rows, cases):
    table = []
    for row in rows:
        if row["stage"] != "legacy":
            continue
        case = cases[row["case_id"]]
        inputs = case["claim"] + "\n" + "\n".join(p["text"] + " " + p["source"] for p in case["passages"])
        text = row["raw_response"]
        table.append({"case_id": row["case_id"], "expected": case["label"], "claim": case["claim"], "status": row["status"],
                      "wall_s": row["wall_s"], "output_tokens": row.get("output_tokens"), "comment": text,
                      "values_named": all(value_mentioned(v, text) for v in case["required_values"]) if case["required_values"] else None,
                      "required_values": case["required_values"],
                      "numbers_not_in_inputs": numbers_not_in_inputs(text, inputs),
                      "capitalized_not_in_inputs": capitalized_not_in_inputs(text, inputs)})
    return table


def _timeline(events, rows, e2e):
    stamps = [e["at"] for e in events] + [r["finished_at"] for r in rows] + [r["finished_at"] for r in e2e]
    stage_wall = {}
    for row in rows:
        stage_wall[row["stage"]] = stage_wall.get(row["stage"], 0) + (row["wall_s"] or 0)
    return {"first": min(stamps) if stamps else None, "last": max(stamps) if stamps else None,
            "wall_s_by_stage": stage_wall, "resumed": sum(e["event"] == "resumed" for e in events),
            "cold_load_s": sum(e.get("model_seconds", 0) for e in events if e["event"] == "cold_load"),
            "e2e_model_s": sum(r.get("model_seconds", 0) for r in e2e)}


# * Outputs
# * ===========================================================================
def write_summary_csv(results: dict, path: Path) -> None:
    columns = ["model", "tag", "pass", "planned", "attempted", "not_executed", "ok", "format_error", "truncated", "timeout",
               "request_error", "completion_rate", "correct", "accuracy_over_attempted", "macro_f1", "false_support",
               "negatives_attempted", "false_support_rate", "critical_false_support", "traceable_correct",
               "unknown_id_answers", "quote_found_in_cited", "quote_denominator", "expected_evidence_cited",
               "expected_evidence_denominator", "both_values_named", "both_values_denominator",
               "answers_with_number_not_in_inputs", "answers_with_capitalized_not_in_inputs", "warm_n", "warm_median_s",
               "warm_p95_s", "first_after_load_median_s", "warm_ttft_median_s", "generation_tokens_per_s_median",
               "prompt_tokens_median", "output_tokens_median", "total_wall_s"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for alias in results["aliases"]:
            for pass_no in results["passes"]:
                entry = results["models"][alias]["passes"][pass_no]
                quality, latency = entry["quality"], entry["latency"]
                writer.writerow({"model": alias, "tag": results["manifest"]["models"][alias]["tag"], "pass": pass_no,
                                 **{k: quality[k] for k in columns if k in quality},
                                 "warm_n": latency["warm_wall_s"]["n"], "warm_median_s": latency["warm_wall_s"]["median"],
                                 "warm_p95_s": latency["warm_wall_s"]["p95"],
                                 "first_after_load_median_s": latency["first_after_load_wall_s"]["median"],
                                 "warm_ttft_median_s": latency["warm_ttft_s"]["median"],
                                 "generation_tokens_per_s_median": latency["generation_tokens_per_s"]["median"],
                                 "prompt_tokens_median": latency["prompt_tokens"]["median"],
                                 "output_tokens_median": latency["output_tokens"]["median"],
                                 "total_wall_s": latency["total_wall_s"]})


def table(headers: list, rows: list) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def block_overview(results, protocol):
    primary = protocol["selection"]["primary_pass"]
    critical_cases = sum(c["critical"] for c in load_cases(BENCH_DIR / "dataset" / "eval_cases.yaml"))
    rows = []
    for alias in results["aliases"]:
        entry = results["models"][alias]["passes"][primary]
        q, l = entry["quality"], entry["latency"]
        rows.append([f"**{alias}** `{results['manifest']['models'][alias]['tag']}`",
                     f"{q['ok']}/{q['planned']}", fraction(q["correct"], q["attempted"]), pct_interval(q["accuracy_ci95"]),
                     fmt(q["macro_f1"], 2), fraction(q["false_support"], q["negatives_attempted"]),
                     fraction(q["critical_false_support"], critical_cases), f"{fmt(l['warm_wall_s']['median'])} s (n={l['warm_wall_s']['n']})",
                     f"{fmt(l['warm_wall_s']['p95'])} s"])
    return table(["Model", "Valid answers", "Correct / attempted", "95 % CI (Wilson)", "Macro-F1", "False supports / negatives",
                  "Critical false supports / critical cases", "Warm median", "Warm p95"], rows)


def block_confusion(results, protocol):
    primary = protocol["selection"]["primary_pass"]
    parts = []
    for alias in results["aliases"]:
        q = results["models"][alias]["passes"][primary]["quality"]
        rows = [[f"**{e}**"] + [q["confusion"][e][p] for p in LABELS + (NO_ANSWER,)] for e in LABELS]
        parts.append(f"**{alias}** (pass {primary}; rows are expected, columns predicted)\n\n"
                     + table(["expected \\ predicted", *LABELS, NO_ANSWER], rows))
    return "\n\n".join(parts)


def block_per_class(results, protocol):
    primary = protocol["selection"]["primary_pass"]
    rows = []
    for alias in results["aliases"]:
        q = results["models"][alias]["passes"][primary]["quality"]
        for label in LABELS:
            c = q["per_class"][label]
            rows.append([alias, label, f"{fmt(c['precision'], 2)} ({c['tp']}/{c['predicted']})",
                         f"{fmt(c['recall'], 2)} ({c['tp']}/{c['actual']})", fmt(c["f1"], 2)])
    return table(["Model", "Class", "Precision", "Recall", "F1"], rows)


def block_failures_by_type(results, protocol):
    primary = protocol["selection"]["primary_pass"]
    rows = []
    for alias in results["aliases"]:
        q = results["models"][alias]["passes"][primary]["quality"]
        rows.append([alias, q["attempted"], q["ok"], q["format_error"], q["truncated"], q["timeout"], q["request_error"],
                     q["not_executed"], f"{(q['completion_rate'] or 0) * 100:.0f} %"])
    return table(["Model", "Attempted", "Valid", "Format errors", "Truncated", "Timeouts", "Request errors", "Not executed",
                  "Completion"], rows)


def block_traceability(results, protocol):
    primary = protocol["selection"]["primary_pass"]
    rows = []
    for alias in results["aliases"]:
        q = results["models"][alias]["passes"][primary]["quality"]
        rows.append([alias, fraction(q["unknown_id_answers"], q["answers"]),
                     fraction(q["quote_found_in_cited"], q["quote_denominator"]),
                     fraction(q["expected_evidence_cited"], q["expected_evidence_denominator"]),
                     fraction(q["both_values_named"], q["both_values_denominator"]),
                     fraction(q["traceable_correct"], q["attempted"]),
                     fraction(q["answers_with_number_not_in_inputs"], q["answers"]),
                     fraction(q["answers_with_capitalized_not_in_inputs"], q["answers"])])
    return table(["Model", "Answers citing a nonexistent passage ID", "Non-empty quote found literally in a cited passage",
                  "Expected passage cited", "Both values named in the explanation", "Correct label + expected passage + literal quote",
                  "Explanations with a figure absent from the inputs", "Explanations with a capitalised word absent from the inputs"], rows)


def block_performance(results, protocol):
    rows = []
    for alias in results["aliases"]:
        loads = [e["wall_s"] for e in results["models"][alias]["loads"]]
        for pass_no in results["passes"]:
            entry = results["models"][alias]["passes"][pass_no]
            l = entry["latency"]
            limit = results["models"][alias]["cpu_limit_by_pass"][pass_no]
            rows.append([alias, pass_no, l["completed"], l["timeouts"], f"{fmt(l['total_wall_s'] / 60)} min",
                         fmt(loads[pass_no - 1]) if len(loads) >= pass_no else "N/D",
                         fmt(l["first_after_load_wall_s"]["median"]), f"{fmt(l['warm_wall_s']['median'])} / {fmt(l['warm_wall_s']['p95'])}",
                         fmt(l["warm_ttft_s"]["median"]), fmt(l["warm_prompt_eval_s"]["median"]), fmt(l["warm_eval_s"]["median"]),
                         f"{fmt(l['prompt_tokens']['median'], 0)} / {fmt(l['prompt_cached_tokens']['median'], 0)}",
                         fmt(l["output_tokens"]["median"], 0), fmt(l["generation_tokens_per_s"]["median"]),
                         "{} / {} %".format(fmt(limit["median"], 0), fmt(limit["min"], 0))])
    return table(["Model", "Pass", "Completed", "Timeouts", "Total request time", "Cold load (s)", "First request after load (s)",
                  "Warm wall median / p95 (s)", "Warm time to first token (s)", "Warm prompt eval (s)", "Warm generation (s)",
                  "Prompt tokens / cached (median)", "Output tokens (median)", "Generation tok/s (median)", "CPU speed limit, median / min"], rows)


def block_memory(results, protocol):
    rows = []
    for alias in results["aliases"]:
        details = results["manifest"]["models"][alias]
        layers = {l["media_type"].rsplit(".", 1)[-1]: l["size_bytes"] for l in details.get("layers") or []}
        res = results["models"][alias]["resources"]
        loads = results["models"][alias]["loads"]
        size = loads[0].get("ps_size_bytes") if loads else None
        rows.append([alias, f"{details['size_bytes'] / 1e9:.2f} GB", f"{layers['model'] / 1e9:.2f} GB" if "model" in layers else "N/D",
                     f"{size / 1e9:.2f} GB" if size else "N/D",
                     f"{fmt(res['rss_median_mib'], 0)} / {fmt(res['rss_max_sampled_mib'], 0)} MiB",
                     f"{fmt(res['memory_free_pct_median'], 0)} / {fmt(res['memory_free_pct_min'], 0)} %",
                     f"{fmt(res['swap_first_mib'], 0)} → {fmt(res['swap_last_mib'], 0)} MiB",
                     f"{res['pressure_warn_samples']} / {res['pressure_critical_samples']} of {res['samples']}",
                     f"{fmt(res['cpu_speed_limit_min_pct'], 0)} %",
                     "0" if loads and all(e.get("ps_size_vram_bytes") == 0 for e in loads) else "N/D"])
    return table(["Model", "Files on disk (all layers)", "Weights file", "Size in `/api/ps`", "Runner RSS median / max sampled",
                  "System memory free, median / min", "Swap used, first → last sample", "Samples at warn / critical pressure",
                  "Lowest CPU speed limit", "`size_vram`"], rows)


def block_repeatability(results, protocol):
    rows = []
    for alias in results["aliases"]:
        r = results["models"][alias]["repeatability"]
        passes = results["models"][alias]["passes"]
        changes = ", ".join(f"{c['case_id']} ({c['pass_1']} → {c['pass_2']})" for c in r["label_changes"]) or "none"
        rows.append([alias, fraction(r["same_label"], r["cases_in_both"]), fraction(r["same_text"], r["cases_in_both"]),
                     " / ".join(fraction(passes[p]["quality"]["correct"], passes[p]["quality"]["attempted"]) for p in results["passes"]),
                     " / ".join(fraction(passes[p]["quality"]["false_support"], passes[p]["quality"]["negatives_attempted"]) for p in results["passes"]),
                     " / ".join(fmt(passes[p]["quality"]["macro_f1"], 2) for p in results["passes"]),
                     " / ".join(fmt(passes[p]["latency"]["warm_wall_s"]["median"]) for p in results["passes"]), changes])
    return table(["Model", "Same label", "Identical text", "Correct (pass 1 / 2)", "False supports (1 / 2)", "Macro-F1 (1 / 2)",
                  "Warm median s (1 / 2)", "Label changes"], rows)


def block_by_tag(results, protocol):
    tags = sorted({t for a in results["aliases"] for t in results["models"][a]["by_tag"]})
    rows = [[f"`{t}`"] + [fraction(*results["models"][a]["by_tag"].get(t, [0, 0])) for a in results["aliases"]] for t in tags]
    return table(["Case property", *results["aliases"]], rows)


def block_discordant(results, protocol):
    rows = [[pair.replace("|", " vs "), d["shared"], d["only_first_right"], d["only_second_right"]]
            for pair, d in results["discordant"].items()]
    return table(["Pair", "Shared cases", "Only the first right", "Only the second right"], rows)


def block_selection(results, protocol):
    s = results["selection"]
    target = protocol["selection"]["latency_target_s"]
    lines = [f"- Complete candidates in both passes: {', '.join(s['complete']) or 'none'}.",
             f"- Order by false supports, then macro-F1, in pass {protocol['selection']['primary_pass']}: "
             f"{' > '.join(s['ranking_primary']) or 'N/D'}.",
             f"- Order changes in the other pass: {'yes' if s['flips_between_passes'] else 'no'}.",
             f"- Robust winner under the rule: {s['robust_winner'] or 'none'}.",
             "- Median warm time within the " + f"{target} s preference: "
             + (", ".join(f"{a} {'yes' if v else 'no'}" for a, v in s["meets_latency_target"].items()) or "N/D") + ".",
             "- Critical false supports: " + (", ".join(f"{a} {v}" for a, v in s["critical_false_supports"].items()) or "N/D") + ".",
             f"- Comments stay off by default under the rule: {'yes' if s['keep_comments_off_by_default'] else 'no'}."]
    return "\n".join(lines)


def block_failure_examples(results, protocol):
    """Literal model output for false supports, the errors the protocol weighs the most."""
    cases = {c["id"]: c for c in load_cases(BENCH_DIR / "dataset" / "eval_cases.yaml")}
    primary = protocol["selection"]["primary_pass"]
    rows_file = read_jsonl(RUN_DIR / "responses.jsonl")
    parts = []
    for alias in results["aliases"]:
        scores = [s for s in results["scores"] if s["model"] == alias and s["pass"] == primary]
        failed = [s for s in scores if s["critical_false_support"]] or [s for s in scores if not s["correct"]]
        shown = sorted(failed, key=lambda s: s["case_id"])[:3]
        if not shown:
            parts.append(f"**{alias}:** no error to show in pass {primary}.")
            continue
        entries = []
        for s in shown:
            raw = next(r for r in rows_file if r["stage"] == "eval" and r["alias"] == alias and r["pass"] == primary
                       and r["case_id"] == s["case_id"])
            parsed = raw.get("parsed") or {}
            entries.append(f"- `{s['case_id']}` · claim: «{cases[s['case_id']]['claim']}» · expected `{s['expected']}`, "
                           f"got `{s['predicted']}`. Quote: «{parsed.get('quote', '')}». Explanation: «{parsed.get('explanation', raw.get('detail'))}»")
        parts.append(f"**{alias}** ({len(failed)} in this category, first {len(shown)} shown)\n\n" + "\n".join(entries))
    return "\n\n".join(parts)


NAMES = {"text_vs_text": "text vs text", "report_vs_news": "report vs news", "news_vs_newsletter": "news vs newsletter"}


def block_e2e_documents(results, protocol):
    rows = []
    for d in results["e2e"]["documents"]:
        rows.append([d["alias"], NAMES[d["modality"]], d["fragments"], f"{fmt(d['index_new_total_s'])} ({fmt(d['index_new_index_s'])} + {fmt(d['index_new_search_s'])})",
                     f"{fmt(d['index_reused_total_s'])} ({fmt(d['index_reused_index_s'], 2)} + {fmt(d['index_reused_search_s'])})",
                     f"{fmt(d['judge_document_s'])} ({fmt(d['judge_first_s'])} / {fmt(d['judge_warm_median_s'])})",
                     f"{fmt(d['app_total_s'])} ({fmt(d['app_first_s'])} / {fmt(d['app_warm_median_s'])})",
                     f"{d['app_comments_usable']}/{d['app_comments_total']}"])
    return table(["Model", "Mode", "Fragments", "New index: total (index + search), s", "Reused index: total (index + search), s",
                  "Structured judge, whole document (first / warm median), s", "App comment path, whole request (first / warm median), s",
                  "Non-empty app comments"], rows)


def block_e2e_outcomes(results, protocol):
    rows = []
    for d in results["e2e"]["documents"]:
        rows.append([d["alias"], NAMES[d["modality"]], fraction(d["needed_retrieved"], d["needed_total"]),
                     fraction(d["judge_correct"], d["fragments"]), d["judge_errors"], d["retrieval_misses"], d["judge_failed_requests"]])
    return table(["Model", "Mode", "Needed passage in top 3", "Correct label", "Judge errors", "Retrieval misses", "Failed requests"], rows)


def block_e2e_fragments(results, protocol):
    rows = []
    for f in results["e2e"]["fragments"]:
        retrieved = {True: "yes", False: "**no**", None: "n/a"}[f["needed_passage_retrieved"]]
        values = {True: "yes", False: "no", None: "n/a"}[f["values_named"]]
        rows.append([f["alias"], NAMES[f["modality"]], f["fragment"], f"`{f['expected']}`", f"`{f['predicted']}`", retrieved, values,
                     f["outcome"], fmt(f["wall_s"])])
    return table(["Model", "Mode", "#", "Expected", "Predicted", "Needed passage retrieved", "Values named", "Outcome", "Wall s"], rows)


def block_legacy(results, protocol):
    rows = []
    for r in results["legacy"]:
        named = {True: "yes", False: "**no**", None: "n/a"}[r["values_named"]]
        rows.append([f"`{r['case_id']}`", f"`{r['expected']}`", r["status"], fmt(r["wall_s"]), r["output_tokens"], named,
                     ", ".join(r["numbers_not_in_inputs"]) or "none", ", ".join(r["capitalized_not_in_inputs"]) or "none"])
    return table(["Case", "Expected", "Status", "Wall s", "Output tokens", "Both values named", "Figures absent from the inputs",
                  "Capitalised words absent from the inputs"], rows)


def block_environment(results, protocol):
    m = results["manifest"]
    hardware = m["hardware"]
    model_rows = []
    for alias in results["aliases"]:
        d = m["models"][alias]
        model_rows.append([alias, f"`{d['tag']}`", f"`{d['digest'][:12]}`", d["details"].get("family"), d["details"].get("parameter_size"),
                           d["details"].get("quantization_level"), d.get("context_length"), ", ".join(d.get("capabilities") or []),
                           (d.get("inherited_parameters") or "none").replace("\n", "; ").replace("  ", "").strip() or "none",
                           f"renderer {d['renderer']}" if d.get("renderer") else f"template `{d['template_sha256'][:12]}`"])
    t = results["timeline"]
    state = results["state"]
    lines = [f"- **Machine:** {hardware['cpu']}, {hardware['physical_cores']} cores / {hardware['logical_cores']} threads, "
             f"{hardware['memory_gib']:.0f} GiB RAM, {hardware['os']}. The system also reports "
             f"{' and '.join(hardware['gpus_reported_by_system'])}; Ollama does not use them on Intel Macs.",
             f"- **Ollama:** {m['ollama_version']}. Backend according to its log: `library={m['backend'].get('library')}`; `/api/ps` reported "
             f"`size_vram` = 0 on every load, which is what `ollama ps` shows as 100 % CPU.",
             f"- **Run:** `{m['run_id']}`, protocol v{m['protocol_version']}, commit `{m['commit'][:12]}`, frozen: {'yes' if m['frozen'] else 'no'}. "
             f"From {t['first']} to {t['last']}, resumed {t['resumed']} time(s).",
             f"- **Model execution time:** {state.get('budget_used_hours', 'N/D')} h of the {state.get('budget_hours', 'N/D')} h budget. "
             f"Server log lines mentioning truncation during the run: {state.get('server_log_truncation_lines', 0) - m.get('server_log_truncation_lines_at_start', 0)}.",
             "", table(["Alias", "Tag", "Digest", "Family", "Parameters", "Quantization", "Model context limit", "Capabilities",
                        "Inherited parameters (overridden)", "Prompt format"], model_rows)]
    return "\n".join(lines)


def block_pilot(results, protocol):
    rows = []
    for alias in results["aliases"]:
        p = results["models"][alias]["pilot"]
        cancel, fit = p["cancel"] or {}, p["token_fit"] or {}
        rows.append([alias, ", ".join(p["statuses"]) or "N/D", " / ".join(fmt(w) for w in p["wall_s"]) or "N/D",
                     f"{'stopped' if cancel.get('stopped') else 'NOT stopped'} in {cancel.get('waited_s', 'N/D')} s "
                     f"({cancel.get('runner_cpu_s_during_request', 'N/D')} CPU s used before cancelling)",
                     f"{max(fit.get('prompt_tokens') or [0])} ≤ {fit.get('limit', 'N/D')}" if fit else "N/D",
                     {True: "viable", False: "not viable", None: "N/D"}[p["viable"]] + (f": {'; '.join(p['reasons'])}" if p["reasons"] else "")])
    return table(["Model", "Pilot statuses", "Pilot wall (s)", "Cancel check", "Longest eval prompt, tokens", "Decision"], rows)


def render_figure(results, protocol, path: Path) -> bool:
    """Quality against waiting time. Only drawn if at least two models completed the primary pass."""
    primary = protocol["selection"]["primary_pass"]
    points = []
    for alias in results["aliases"]:
        for pass_no in results["passes"]:
            entry = results["models"][alias]["passes"][pass_no]
            x, y = entry["latency"]["warm_wall_s"]["median"], entry["quality"]["macro_f1"]
            if x is not None and y is not None and entry["quality"]["attempted"] == results["planned_per_pass"]:
                points.append({"alias": alias, "pass": pass_no, "x": x, "y": y, "fs": entry["quality"]["false_support"],
                               "neg": entry["quality"]["negatives_attempted"]})
    if len({p["alias"] for p in points if p["pass"] == primary}) < 2:
        return False
    target = protocol["selection"]["latency_target_s"]
    width, height, left, right, top, bottom = 760, 420, 64, 24, 64, 56
    x_max = max(max(p["x"] for p in points) * 1.25, target * 1.2)
    sx = lambda v: left + v / x_max * (width - left - right)
    sy = lambda v: height - bottom - v * (height - top - bottom)
    css = ":root{--bg:#fcfcfb;--ink:#0b0b0b;--muted:#52514e;--grid:#e4e3df;"
    css += "".join(f"--s{a}:{c[0]};" for a, c in SERIES_COLORS.items()) + "}"
    css += "@media (prefers-color-scheme:dark){:root{--bg:#1a1a19;--ink:#f0efec;--muted:#c3c2b7;--grid:#3a3a38;"
    css += "".join(f"--s{a}:{c[1]};" for a, c in SERIES_COLORS.items()) + "}}"
    css += "text{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;fill:var(--ink);font-size:13px}.m{fill:var(--muted);font-size:12px}.t{font-size:15px;font-weight:600}"
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
             '<title id="title">Macro-F1 against median warm time per request</title>',
             '<desc id="desc">' + "; ".join(f"{p['alias']} pass {p['pass']}: macro-F1 {p['y']:.2f}, {p['x']:.1f} s, {p['fs']} of {p['neg']} false supports" for p in points) + "</desc>",
             f"<style>{css}</style>", f'<rect width="{width}" height="{height}" fill="var(--bg)"/>',
             f'<text x="{left}" y="24" class="t">Label quality and waiting time, 30 synthetic cases</text>']
    legend_x = left
    for alias in results["aliases"]:
        parts.append(f'<circle cx="{legend_x + 5}" cy="44" r="5" fill="var(--s{alias})"/><text x="{legend_x + 15}" y="48" class="m">{alias}</text>')
        legend_x += 56
    parts.append(f'<circle cx="{legend_x + 5}" cy="44" r="4" fill="none" stroke="var(--muted)" stroke-width="2"/><text x="{legend_x + 15}" y="48" class="m">hollow: pass 2</text>')
    for tick in (0, 0.25, 0.5, 0.75, 1):
        parts.append(f'<line x1="{left}" x2="{width - right}" y1="{sy(tick):.1f}" y2="{sy(tick):.1f}" stroke="var(--grid)" stroke-width="1"/>')
        parts.append(f'<text x="{left - 8}" y="{sy(tick) + 4:.1f}" text-anchor="end" class="m">{tick:g}</text>')
    step = 10 if x_max <= 80 else 20 if x_max <= 160 else 50
    tick = 0
    while tick <= x_max:
        parts.append(f'<text x="{sx(tick):.1f}" y="{height - bottom + 20}" text-anchor="middle" class="m">{tick}</text>')
        tick += step
    parts.append(f'<line x1="{sx(target):.1f}" x2="{sx(target):.1f}" y1="{top}" y2="{height - bottom}" stroke="var(--muted)" stroke-width="1" stroke-dasharray="4 4"/>')
    parts.append(f'<text x="{sx(target) + 6:.1f}" y="{top + 12}" class="m">{target} s preference</text>')
    parts.append(f'<text x="{(left + width - right) / 2}" y="{height - 14}" text-anchor="middle" class="m">Median wall time per warm request, seconds (lower is better)</text>')
    parts.append(f'<text transform="translate(16 {(top + height - bottom) / 2}) rotate(-90)" text-anchor="middle" class="m">Macro-F1 over attempted cases (higher is better)</text>')
    for p in sorted(points, key=lambda p: -p["pass"]):
        if p["pass"] == primary:
            parts.append(f'<circle cx="{sx(p["x"]):.1f}" cy="{sy(p["y"]):.1f}" r="6" fill="var(--s{p["alias"]})" stroke="var(--bg)" stroke-width="2"/>')
        else:
            parts.append(f'<circle cx="{sx(p["x"]):.1f}" cy="{sy(p["y"]):.1f}" r="5" fill="var(--bg)" stroke="var(--s{p["alias"]})" stroke-width="2"/>')
    for p in points:
        if p["pass"] == primary:
            anchor, dx = ("end", -12) if sx(p["x"]) > width - 170 else ("start", 12)
            parts.append(f'<text x="{sx(p["x"]) + dx:.1f}" y="{sy(p["y"]) - 8:.1f}" text-anchor="{anchor}">{p["alias"]}: {p["fs"]}/{p["neg"]} false supports</text>')
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts) + "\n", "utf-8")
    return True


BLOCKS = {"environment": block_environment, "pilot": block_pilot, "overview": block_overview, "confusion": block_confusion,
          "per_class": block_per_class, "failures_by_type": block_failures_by_type, "traceability": block_traceability,
          "performance": block_performance, "memory": block_memory, "repeatability": block_repeatability, "by_tag": block_by_tag,
          "discordant": block_discordant, "selection": block_selection, "failure_examples": block_failure_examples,
          "e2e_documents": block_e2e_documents, "e2e_outcomes": block_e2e_outcomes, "e2e_fragments": block_e2e_fragments,
          "legacy": block_legacy}


def update_document(results: dict, protocol: dict, doc: Path) -> list:
    text = doc.read_text("utf-8")
    updated = []
    for name, build in BLOCKS.items():
        pattern = re.compile(rf"(<!-- BEGIN GENERATED: {name} -->\n).*?(\n<!-- END GENERATED: {name} -->)", re.S)
        if pattern.search(text):
            content = build(results, protocol)
            text = pattern.sub(lambda m: m.group(1) + content + m.group(2), text)
            updated.append(name)
    doc.write_text(text, "utf-8")
    return updated


def main() -> None:
    global RUN_DIR
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--results-dir", default=str(BENCH_DIR / "results"))
    parser.add_argument("--doc", default=str(DOC))
    parser.add_argument("--figure", default=str(FIGURE))
    args = parser.parse_args()
    RUN_DIR = Path(args.results_dir) / args.run_id
    protocol = load_protocol()
    results = compute(RUN_DIR, protocol)
    published = {k: v for k, v in results.items() if k != "manifest"}
    (RUN_DIR / "metrics.json").write_text(json.dumps(published, indent=1, ensure_ascii=False, default=str), "utf-8")
    write_summary_csv(results, RUN_DIR / "summary.csv")
    drawn = render_figure(results, protocol, Path(args.figure))
    updated = update_document(results, protocol, Path(args.doc)) if Path(args.doc).exists() else []
    print(f"metrics.json and summary.csv in {RUN_DIR.name}; figure: {'written' if drawn else 'not enough data'}; "
          f"document blocks updated: {len(updated)}")


if __name__ == "__main__":
    main()
