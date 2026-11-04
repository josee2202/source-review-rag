# Third-party notices

## Redistributed in this repository

| File | Origin | License |
|---|---|---|
| `IA-modulable/descargas/source.spm` | SentencePiece tokenizer of [Helsinki-NLP/opus-mt-es-en](https://huggingface.co/Helsinki-NLP/opus-mt-es-en) (unmodified; SHA-256 `e236ee6d…ad0e3`) | Apache License 2.0 — full text in `IA-modulable/descargas/LICENSE-Apache-2.0.txt` |

The tokenizer is only used to estimate prompt lengths when grouping SAS code blocks.

## Downloaded at runtime (not redistributed)

| Resource | Source | Notes |
|---|---|---|
| `nomic-embed-text` embedding model | `ollama pull nomic-embed-text` | Apache License 2.0 |
| `llama3.2:1b` / `llama3.2:3b` | `ollama pull llama3.2:1b` | Llama 3.2 Community License |
| Annex 9 of Banco de España Circular 4/2017 | Official Spanish Gazette (BOE-A-2017-14334), fetched by `examples/public/download_anejo9.py` | Official regulatory text published at [boe.es](https://www.boe.es/eli/es/cir/2017/11/27/4) |
| Python packages | PyPI, pinned in `requirements.txt` | Each under its own license |

## Synthetic data

Everything in `examples/synthetic/` (regulatory guide, methodology and technical documents, SAS code)
was written for this repository and is fictitious. "Banco Ficticio S.A." does not exist.
