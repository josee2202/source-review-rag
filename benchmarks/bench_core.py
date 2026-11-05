"""Pure logic of the model comparison: dataset, prompt, response checks and metrics.

Nothing here talks to Ollama, so all of it can be tested in CI without models.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import yaml

BENCH_DIR = Path(__file__).resolve().parent
ROOT = BENCH_DIR.parent
sys.path.insert(0, str(ROOT / "IA-modulable"))

from prompt import neutralizar_delimitadores  # noqa: E402  (same delimiter protection as the app)

LABELS = ("supported", "contradicted", "insufficient_evidence")
NO_ANSWER = "no_valid_answer"
COMPLETED_STATUSES = ("ok", "format_error", "truncated")  # the model answered, usefully or not
FAILURE_STATUSES = ("format_error", "truncated", "timeout", "request_error")
LOCKED_FILES = ("protocol.yaml", "dataset/eval_cases.yaml", "dataset/dev_cases.yaml", "dataset/e2e_examples.yaml")
HARNESS_FILES = ("run_benchmark.py", "bench_core.py", "bench_ollama.py", "bench_resources.py")  # what affects a run
NS_PER_S = 1e9


# * Files and hashes
# * ===========================================================================
def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def locked_hashes(bench_dir: Path = BENCH_DIR) -> dict:
    return {name: sha256_bytes((bench_dir / name).read_bytes()) for name in LOCKED_FILES}


def harness_hash(bench_dir: Path = BENCH_DIR) -> str:
    return sha256_text("".join(sha256_bytes((bench_dir / name).read_bytes()) for name in HARNESS_FILES))


def append_jsonl(path: Path, row: dict) -> None:
    """Writes one row and forces it to disk: a crash loses at most the request in flight."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path: Path) -> list:
    """Reads saved rows. A partial last line (crash while writing) is ignored, so that request runs again."""
    if not Path(path).exists():
        return []
    rows = []
    for line in Path(path).read_text("utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


# * Dataset and prompt
# * ===========================================================================
def load_protocol(path: Path = BENCH_DIR / "protocol.yaml") -> dict:
    return yaml.safe_load(Path(path).read_text("utf-8"))


def load_cases(path: Path) -> list:
    """Expands the scenarios into self-contained cases. Passages are renamed P1, P2... in prompt order."""
    data = yaml.safe_load(Path(path).read_text("utf-8"))
    cases = []
    for scenario_id, scenario in data["scenarios"].items():
        for raw in scenario["cases"]:
            ids = {key: f"P{i}" for i, key in enumerate(raw["passage_order"], start=1)}
            cases.append({
                "id": raw["id"],
                "split": data["split"],
                "scenario": scenario_id,
                "modality": scenario["modality"],
                "claim": raw["claim"],
                "passages": [{"id": ids[key], "source": scenario["passages"][key]["source"],
                              "text": " ".join(scenario["passages"][key]["text"].split())}
                             for key in raw["passage_order"]],
                "label": raw["label"],
                "evidence_ids": [ids[key] for key in raw.get("evidence", [])],
                "quote": raw.get("quote", ""),
                "required_values": [str(v) for v in raw.get("required_values", [])],
                "critical": bool(raw.get("critical", False)),
                "tags": raw.get("tags", []),
                "rationale": raw["rationale"],
            })
    return sorted(cases, key=lambda c: c["id"])


def format_passages(passages: list) -> str:
    return neutralizar_delimitadores("\n\n".join(f"[{p['id']}] {p['source']}\n{p['text']}" for p in passages))


def build_prompt(protocol: dict, modality: str, claim: str, passages: list) -> str:
    """Receives only what the model may see: never the case, so labels and rationales cannot leak."""
    values = {"role": protocol["judge"]["roles"][modality],
              "passages": format_passages(passages),
              "claim": neutralizar_delimitadores(claim)}
    return re.sub(r"\{(role|passages|claim)\}", lambda m: values[m.group(1)], protocol["judge"]["prompt"])


def build_payload(protocol: dict, model_tag: str, prompt: str) -> dict:
    request = protocol["request"]
    return {"model": model_tag, "prompt": prompt, "stream": request["stream"], "think": request["think"],
            "format": protocol["judge"]["response_schema"], "keep_alive": request["keep_alive"],
            "options": dict(request["options"])}


def ordered_cases(cases: list, seed: int, pass_no: int) -> list:
    """One reproducible order per pass, independent of labels and shared by all the models."""
    result = sorted(cases, key=lambda c: c["id"])
    random.Random(f"{seed}:{pass_no}").shuffle(result)
    return result


# * Response checks
# * ===========================================================================
def parse_response(raw: str, done_reason) -> tuple:
    """Returns (status, parsed, detail). Nothing is repaired: an unusable answer is reported as such."""
    if done_reason == "length":
        return "truncated", None, "num_predict reached before the end of the answer"
    try:
        obj = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return "format_error", None, "not valid JSON"
    if not isinstance(obj, dict):
        return "format_error", None, "not a JSON object"
    if obj.get("label") not in LABELS:
        return "format_error", None, f"invalid label: {obj.get('label')!r}"
    ids = obj.get("passage_ids")
    if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
        return "format_error", None, "passage_ids is not a list of strings"
    if not isinstance(obj.get("quote"), str) or not isinstance(obj.get("explanation"), str):
        return "format_error", None, "quote or explanation is not a string"
    if not obj["explanation"].strip():
        return "format_error", None, "empty explanation"
    return "ok", {k: obj[k] for k in ("passage_ids", "quote", "explanation", "label")}, None


def normalize_for_match(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    for source, target in {"“": '"', "”": '"', "‘": "'", "’": "'", "—": "-", "–": "-", "…": "..."}.items():
        text = text.replace(source, target)
    return " ".join(text.split())


def check_citations(parsed: dict, passages: list) -> dict:
    """Mechanical traceability: cited IDs exist and the quote is literally in a cited passage.

    It says nothing about whether the explanation is right.
    """
    by_id = {p["id"]: normalize_for_match(p["text"]) for p in passages}
    cited = [i.strip().strip("[]").strip() for i in parsed["passage_ids"]]
    unknown = [i for i in cited if i not in by_id]
    quote = normalize_for_match(parsed["quote"]).strip(" \"'«»").rstrip(".").strip()
    in_cited = in_any = None
    if quote:
        in_cited = any(quote in by_id[i] for i in cited if i in by_id)
        in_any = any(quote in text for text in by_id.values())
    return {"cited_ids": cited, "unknown_ids": unknown, "quote_empty": not quote,
            "quote_in_cited": in_cited, "quote_in_any": in_any}


def _normalize_numbers(text: str) -> str:
    return re.sub(r"(?<=\d)\.(?=\d{3}(?:\D|$))", "", text).lower()


def value_mentioned(value: str, text: str) -> bool:
    """A whole-token match: "3" is not found in "30", "2023" or "P3", and "12.500" is found in "12500"."""
    pattern = rf"(?<![\w.,]){re.escape(_normalize_numbers(value))}(?![\w]|[.,]\d)"
    return re.search(pattern, _normalize_numbers(text)) is not None


def numbers_in(text: str) -> set:
    numbers = set()
    for match in re.findall(r"(?<!\w)\d+(?:[.,]\d+)*", text):
        numbers.add(re.sub(r"\.(?=\d{3}(?:\D|$))", "", match).replace(",", "."))
    return numbers


def numbers_not_in_inputs(explanation: str, inputs: str) -> list:
    """Figures in the explanation that appear neither in the claim nor in the passages."""
    return sorted(numbers_in(explanation) - numbers_in(inputs))


def capitalized_not_in_inputs(explanation: str, inputs: str) -> list:
    """Crude proxy for invented names: capitalised words, not opening a sentence, absent from the inputs."""
    vocabulary = set(re.findall(r"\w+", inputs.lower()))
    found = set()
    for sentence in re.split(r"(?<=[.!?:])\s+", explanation):
        for match in re.finditer(r"\b[A-ZÁÉÍÓÚÑ][\wáéíóúñü]+", sentence):
            if match.start() > 0 and match.group().lower() not in vocabulary:
                found.add(match.group())
    return sorted(found)


def score_row(row: dict, case: dict) -> dict:
    """Compares one saved answer with the expectation. Only runs at analysis time, never before a request."""
    parsed = row.get("parsed") if row["status"] == "ok" else None
    predicted = parsed["label"] if parsed else NO_ANSWER
    expected = case["label"]
    score = {"model": row["alias"], "pass": row["pass"], "case_id": case["id"], "modality": case["modality"],
             "expected": expected, "predicted": predicted, "status": row["status"], "correct": predicted == expected,
             "false_support": predicted == "supported" and expected != "supported",
             "critical_false_support": predicted == "supported" and expected != "supported" and case["critical"],
             "tags": case["tags"], "evidence_hit": None, "required_values_ok": None, "traceable_correct": False,
             "quote_in_cited": None, "quote_empty": None, "unknown_ids": [], "numbers_not_in_inputs": [],
             "capitalized_not_in_inputs": []}
    if not parsed:
        return score
    citations = check_citations(parsed, case["passages"])
    inputs = f"{case['claim']}\n{format_passages(case['passages'])}"
    score.update(quote_in_cited=citations["quote_in_cited"], quote_empty=citations["quote_empty"],
                 unknown_ids=citations["unknown_ids"],
                 numbers_not_in_inputs=numbers_not_in_inputs(parsed["explanation"], inputs),
                 capitalized_not_in_inputs=capitalized_not_in_inputs(parsed["explanation"], inputs))
    if case["evidence_ids"]:
        score["evidence_hit"] = bool(set(citations["cited_ids"]) & set(case["evidence_ids"]))
    if case["required_values"]:
        score["required_values_ok"] = all(value_mentioned(v, parsed["explanation"]) for v in case["required_values"])
    if score["correct"]:
        score["traceable_correct"] = expected == "insufficient_evidence" or bool(
            score["evidence_hit"] and citations["quote_in_cited"])
    return score


def attribute_e2e(expected: str, predicted: str, retrieved_needed_passage) -> str:
    """Separates a search failure from a judge failure in the end-to-end examples."""
    if predicted == expected:
        return "correct"
    if retrieved_needed_passage is False:
        return "retrieval_miss"
    return "judge_error"


# * Metrics
# * ===========================================================================
def timing_fields(final: dict) -> dict:
    """Converts Ollama's nanosecond durations to seconds and computes tokens/s only from a valid duration."""
    def seconds(key):
        value = final.get(key)
        return value / NS_PER_S if isinstance(value, (int, float)) else None

    eval_ns, eval_count = final.get("eval_duration"), final.get("eval_count")
    prompt_ns, prompt_count = final.get("prompt_eval_duration"), final.get("prompt_eval_count")
    cached = final.get("prompt_eval_cached_count")
    evaluated = prompt_count - cached if isinstance(prompt_count, int) and isinstance(cached, int) else prompt_count
    return {"total_s": seconds("total_duration"), "load_s": seconds("load_duration"),
            "prompt_eval_s": seconds("prompt_eval_duration"), "eval_s": seconds("eval_duration"),
            "prompt_tokens": prompt_count, "prompt_cached_tokens": cached, "output_tokens": eval_count,
            "generation_tokens_per_s": eval_count * NS_PER_S / eval_ns if eval_ns and eval_count else None,
            "prompt_tokens_per_s": evaluated * NS_PER_S / prompt_ns if prompt_ns and evaluated else None}


def median(values: list):
    if not values:
        return None
    ordered, middle = sorted(values), len(values) // 2
    return ordered[middle] if len(values) % 2 else (ordered[middle - 1] + ordered[middle]) / 2


def percentile_nearest_rank(values: list, q: float):
    if not values:
        return None
    return sorted(values)[math.ceil(q * len(values)) - 1]


def wilson_interval(k: int, n: int, z: float = 1.96):
    if not n:
        return None
    p = k / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def _ratio(k: int, n: int):
    return k / n if n else None


def quality_metrics(scores: list, planned: int) -> dict:
    """Failures count as not correct. Rates use the attempted cases; nothing is computed over valid answers alone."""
    attempted = len(scores)
    statuses = Counter(s["status"] for s in scores)
    columns = LABELS + (NO_ANSWER,)
    confusion = {e: {p: sum(1 for s in scores if s["expected"] == e and s["predicted"] == p) for p in columns}
                 for e in LABELS}
    per_class = {}
    f1_values = []
    for label in LABELS:
        tp = confusion[label][label]
        predicted_as = sum(confusion[e][label] for e in LABELS)
        actual = sum(confusion[label].values())
        precision, recall = _ratio(tp, predicted_as), _ratio(tp, actual)
        f1 = 2 * precision * recall / (precision + recall) if precision and recall else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "tp": tp,
                            "predicted": predicted_as, "actual": actual}
        f1_values.append(f1)
    negatives = sum(1 for s in scores if s["expected"] != "supported")
    false_support = sum(s["false_support"] for s in scores)
    correct = sum(s["correct"] for s in scores)
    with_evidence = [s for s in scores if s["evidence_hit"] is not None]
    with_quote = [s for s in scores if s["quote_in_cited"] is not None]
    answered = [s for s in scores if s["status"] == "ok"]
    with_values = [s for s in scores if s["required_values_ok"] is not None]
    return {
        "planned": planned, "attempted": attempted, "not_executed": planned - attempted,
        "ok": statuses["ok"], "format_error": statuses["format_error"], "truncated": statuses["truncated"],
        "timeout": statuses["timeout"], "request_error": statuses["request_error"],
        "completion_rate": _ratio(statuses["ok"], planned),
        "correct": correct, "accuracy_over_attempted": _ratio(correct, attempted),
        "accuracy_ci95": wilson_interval(correct, attempted),
        "macro_f1": sum(f1_values) / len(LABELS) if attempted else None,
        "per_class": per_class, "confusion": confusion,
        "false_support": false_support, "negatives_attempted": negatives,
        "false_support_rate": _ratio(false_support, negatives),
        "false_support_ci95": wilson_interval(false_support, negatives),
        "critical_false_support": sum(s["critical_false_support"] for s in scores),
        "critical_false_support_cases": sorted(s["case_id"] for s in scores if s["critical_false_support"]),
        "false_support_cases": sorted(s["case_id"] for s in scores if s["false_support"]),
        "traceable_correct": sum(s["traceable_correct"] for s in scores),
        "unknown_id_answers": sum(1 for s in answered if s["unknown_ids"]),
        "answers": len(answered),
        "expected_evidence_cited": sum(1 for s in with_evidence if s["evidence_hit"]),
        "expected_evidence_denominator": len(with_evidence),
        "quote_found_in_cited": sum(1 for s in with_quote if s["quote_in_cited"]),
        "quote_denominator": len(with_quote),
        "both_values_named": sum(1 for s in with_values if s["required_values_ok"]),
        "both_values_denominator": len(with_values),
        "answers_with_number_not_in_inputs": sum(1 for s in answered if s["numbers_not_in_inputs"]),
        "answers_with_capitalized_not_in_inputs": sum(1 for s in answered if s["capitalized_not_in_inputs"]),
    }


def latency_metrics(rows: list) -> dict:
    """Wall time is what the person waits for. Warm excludes the first request after each load."""
    completed = [r for r in rows if r["status"] in COMPLETED_STATUSES]
    warm = [r for r in completed if not r.get("first_after_load")]
    first = [r for r in completed if r.get("first_after_load")]

    def stats(key, subset):
        values = [r[key] for r in subset if r.get(key) is not None]
        return {"n": len(values), "median": median(values), "p95": percentile_nearest_rank(values, 0.95)}

    return {"completed": len(completed), "timeouts": sum(1 for r in rows if r["status"] == "timeout"),
            "total_wall_s": sum(r["wall_s"] for r in rows if r.get("wall_s") is not None),
            "warm_wall_s": stats("wall_s", warm), "first_after_load_wall_s": stats("wall_s", first),
            "warm_ttft_s": stats("ttft_s", warm), "warm_prompt_eval_s": stats("prompt_eval_s", warm),
            "warm_eval_s": stats("eval_s", warm), "prompt_tokens": stats("prompt_tokens", completed),
            "prompt_cached_tokens": stats("prompt_cached_tokens", warm), "output_tokens": stats("output_tokens", completed),
            "generation_tokens_per_s": stats("generation_tokens_per_s", completed),
            "prompt_tokens_per_s": stats("prompt_tokens_per_s", warm)}
