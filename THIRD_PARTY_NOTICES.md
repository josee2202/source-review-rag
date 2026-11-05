# Third-party notices

## Downloaded at runtime (not redistributed)

| Resource | Source | Notes |
|---|---|---|
| `nomic-embed-text` embedding model | `ollama pull nomic-embed-text` | Apache License 2.0 |
| `qwen3.5:4b-q4_K_M` (optional analysis, configured model) | `ollama pull qwen3.5:4b-q4_K_M` | Apache License 2.0, as reported by `ollama show --license` |
| `llama3.2:1b` and `qwen3.5:9b-q4_K_M` (optional, only for the comparison or if configured) | `ollama pull <model>` | Llama 3.2 Community License; Apache License 2.0 |
| Python packages | PyPI, pinned in `requirements.txt` | Each under its own license |

## Synthetic data

Everything in `examples/synthetic/` was written for the original thesis project and is fictitious.
"Banco Ficticio S.A." does not exist.
