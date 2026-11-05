# Local model comparison

Measures how `llama3.2:1b`, `qwen3.5:4b-q4_K_M` and `qwen3.5:9b-q4_K_M` label a claim against given passages, and what that
costs on one machine. Results and conclusions are in [`docs/model-comparison.md`](../docs/model-comparison.md).

This is not part of the app and not part of normal CI: it needs Ollama and about 10 GB of models, and it takes hours on a CPU.
The `tests/test_benchmark.py` tests of the harness do run in CI, with no models.

| Path | What it is |
|---|---|
| `protocol.yaml` | Models, request parameters, judge prompt, limits, order and selection rule |
| `protocol.lock.json` | SHA-256 of the protocol and the datasets, stored before any model answered an evaluation case |
| `dataset/eval_cases.yaml` | 30 evaluation cases, written and labelled by the assistant |
| `dataset/dev_cases.yaml` | 6 development cases, 3 of them used for the pilot |
| `dataset/e2e_examples.yaml` | One document per mode for the end-to-end stage |
| `run_benchmark.py` | Resumable runner |
| `build_report.py` | Metrics, CSV, chart and the tables of the document, from saved results only |
| `check_app_comments.py` | Follow-up outside the frozen protocol: the app's own comment path after turning thinking off |
| `dataset/integration_cases.yaml` | Control cases of the integration check, written and labelled by the assistant before running them |
| `run_integration_check.py` | Integration check of the production judge, using `IA-modulable/juez.py` and the real API |
| `build_integration_report.py` | Summary and tables of an integration check, from saved results only |
| `results/<run_id>/` | One run: see below |

## Reproduce or resume

```bash
ollama pull llama3.2:1b && ollama pull qwen3.5:4b-q4_K_M && ollama pull qwen3.5:9b-q4_K_M && ollama pull nomic-embed-text
.venv/bin/python benchmarks/run_benchmark.py run --run-id my-run
.venv/bin/python benchmarks/build_report.py --run-id my-run --doc /tmp/my-copy.md --figure /tmp/my-figure.svg
```

Repeating the `run` command with the same `--run-id` resumes it: finished requests are not repeated, and failed ones are not
retried. It refuses to continue if the protocol, the datasets or the harness have changed, so configurations cannot be mixed.
Any change to the protocol or the datasets needs a new `version` in the protocol, a new `freeze` and a new run ID:

```bash
.venv/bin/python benchmarks/run_benchmark.py freeze --note "what changed and why"
```

The runner keeps the machine awake with `caffeinate` for as long as it lives, runs one inference at a time, keeps one
generative model loaded, and stops at the model-time budget in the protocol. The end-to-end index goes to a temporary
directory, never to `qdrant_storage/`.

## The published document

The document in `docs/` was generated with:

```bash
.venv/bin/python benchmarks/check_app_comments.py --run-id 2026-11-05_app-comments-followup
.venv/bin/python benchmarks/build_report.py --run-id 2026-11-05_protocol-v1 --comments-run-id 2026-11-05_app-comments-followup
```

The first command is resumable by model and mode. It runs nothing from the frozen protocol and changes none of its results. A test
checks that the tables in the document are exactly what `build_report.py` generates from the saved results.

## Reproducing the frozen run after the integration

The app later replaced its free-text comment with the judge evaluated here ([integration check](../docs/integration-check.md)), so
`IA-modulable/X_AGENT.py` no longer exists. The frozen harness is unchanged, and its fixed-evidence stages still run from this commit,
but its end-to-end stage and `check_app_comments.py` exercised that earlier comment path. Run those from the commits they were
written for, in a separate worktree:

```bash
git worktree add ../srr-frozen 1af7098     # frozen run; b52eb98 for the comments follow-up
```

The saved results and the commits in their manifests are the record of what ran.

## The integration check

Separate from the comparison: one model, the production code, and no change to the frozen files or results.

```bash
ollama pull qwen3.5:4b-q4_K_M && ollama pull nomic-embed-text
.venv/bin/python benchmarks/run_integration_check.py --run-id my-check     # --num-thread 8 by default, recorded in the manifest
.venv/bin/python benchmarks/build_integration_report.py --run-id my-check
```

It refuses to start with uncommitted code, resumes with the same `--run-id`, keeps failed answers, stops after three consecutive
timeouts or two hours of inference, and uses a temporary index. It writes `manifest.json`, `responses.jsonl`, `e2e.jsonl`,
`resources.jsonl`, `run_status.json`, `summary.json` and `tables.md`. Results are in [`docs/integration-check.md`](../docs/integration-check.md).

## Files in a run

| File | Content |
|---|---|
| `manifest.json` | Commit, hashes, Ollama version, anonymised hardware, model digests, templates and inherited parameters |
| `inputs.jsonl` | Passages, claim and full prompt of every case sent |
| `responses.jsonl` | One row per request: raw answer, status, wall time, time to first token, durations and token counts |
| `e2e.jsonl` | End-to-end steps: index, search, structured judge per fragment, and the app's own comment path |
| `events.jsonl` | Loads, order of cases, cancel checks, token counts, pilot decisions, resumes |
| `resources.jsonl` | Telemetry every 5 s: runner RSS, memory pressure, free memory, swap, CPU speed limit |
| `state.json` | Cases completed, failed and not executed; model time used |
| `metrics.json`, `summary.csv` | Computed by `build_report.py` |
| `comments.jsonl` | Follow-up only: the comments, their times and the text checks |

Expected labels are not stored with the answers: they are joined from the dataset only at analysis time.
