# Local model comparison

Measures how `llama3.2:1b`, `qwen3.5:4b-q4_K_M` and `qwen3.5:9b-q4_K_M` label a claim against given passages, and what that
costs on one machine. Results and conclusions are in [`docs/model-comparison.md`](../docs/model-comparison.md).

This is not part of the app and not part of normal CI: it needs Ollama and about 10 GB of models, and it takes hours on a CPU.
The `tests/test_benchmark.py` tests of the harness do run in CI, with no models.

| Path | What it is |
|---|---|
| `protocol.yaml` | Models, request parameters, judge prompt, limits, order and selection rule |
| `protocol.lock.json` | SHA-256 of the protocol and the datasets, stored before any evaluation case reached a model |
| `dataset/eval_cases.yaml` | 30 evaluation cases, written and labelled by the assistant |
| `dataset/dev_cases.yaml` | 6 development cases, 3 of them used for the pilot |
| `dataset/e2e_examples.yaml` | One document per mode for the end-to-end stage |
| `run_benchmark.py` | Resumable runner |
| `build_report.py` | Metrics, CSV, chart and the tables of the document, from saved results only |
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

Expected labels are not stored with the answers: they are joined from the dataset only at analysis time.
