# API

The UI talks to a small FastAPI service on port 8010. You only need this page if you want to call it directly.

## `POST /validar`

Exactly one of `file_ref` / `texto_ref` and one of `file_query` / `texto_query` is required.

| Field | Type | Description |
|---|---|---|
| `modalidad` | text, required | `text_vs_text`, `report_vs_news` or `news_vs_newsletter`. Anything else returns 400, with no fallback |
| `file_ref` | file, repeatable | Reference file(s): PDF, DOCX, TXT or Markdown |
| `texto_ref` | text | Pasted reference (one reference) |
| `ref_metadata` | JSON text, optional | One object per reference, in order, with any of `title`, `outlet`, `date`, `url` |
| `file_query` | file | Text under review as a file |
| `texto_query` | text | Text under review, pasted |
| `usar_agente` | bool, default `false` | Add the optional analysis by the configured model |
| `fraccion_inicial` | float in (0, 1], default `1` | Review only the first fraction of the fragments |

```bash
curl -s -X POST localhost:8010/validar \
  -F modalidad=report_vs_news \
  -F file_ref=@examples/synthetic/normativa_sintetica_pd.pdf \
  -F 'texto_query=Una exposición entra en default con más de 60 días de impago.' \
  -F usar_agente=true
```

Response shape, abridged:

```json
{
  "modalidad": "report_vs_news",
  "modalidad_nombre": "Informe vs. noticia",
  "comentario_solicitado": true,
  "modelo_analisis": "qwen3.5:4b-q4_K_M",
  "referencias": [{"ref_id": "R1", "filename": "normativa_sintetica_pd.pdf", "title": null, "outlet": null, "date": null, "url": null}],
  "indice": {"coleccion": "srr_…", "fragmentos": 6, "reutilizado": false},
  "texto_revisado": {"filename": "Texto pegado (revisado)", "fragmentos_totales": 1, "fragmentos_revisados": 1, "fragmentos_sin_revisar": 0, "sin_revisar_desde": null},
  "resultados": [{
    "fragmento": 1,
    "chunk": "Una exposición entra en default con más de 60 días de impago.",
    "chunk_location": {"filename": "Texto pegado (revisado)", "page": null, "paragraph": 1, "paragraph_end": 1, "char_start": 0, "char_end": 61, "location": "párr. 1"},
    "relevant_chunks": ["…", "…", "…"],
    "sources": [{"ref_id": "R1", "filename": "normativa_sintetica_pd.pdf", "page": 1, "location": "p. 1", "title": null, "outlet": null, "date": null, "url": null}],
    "scores": [0.7133, 0.6388, 0.6335],
    "context": "[R1 · p. 1] normativa_sintetica_pd.pdf\n…",
    "analisis": {
      "estado": "validado",
      "etiqueta": "contradicted",
      "etiqueta_texto": "Contradicción con los pasajes",
      "explicacion": "…",
      "evidencia": [{"id": "P1", "pasaje": 1, "cita": "más de 90 días de impago", "fuente": {"ref_id": "R1", "location": "p. 1"}}],
      "motivo": null,
      "modelo": "qwen3.5:4b-q4_K_M",
      "metricas": {"tiempo_s": 21.4, "carga_s": 0.1, "tokens_prompt": 900, "tokens_salida": 80, "fin": "stop"}
    },
    "comentario": "Contradicción con los pasajes. … Pasaje 1: «más de 90 días de impago»"
  }]
}
```

`analisis` is only present when `usar_agente=true`.

| `estado` | Meaning | `etiqueta` |
|---|---|---|
| `validado` | The answer has the right format, its passage IDs exist and its quote is literally in a cited passage | `supported`, `contradicted` or `insufficient_evidence` |
| `no_validado` | The model answered, but the output was cut off, broke the format or its evidence could not be checked. `motivo` says why | `null` |
| `no_disponible` | No answer: Ollama unreachable, model not installed, time limit reached, or the prompt does not fit the context. `motivo` says why | `null` |

`validado` means the evidence checks out mechanically, not that the interpretation is right. The raw answer of a rejected output
is not returned.

`comentario` is kept for clients of the earlier API. It is built from `analisis` by code, with no further model call, and is
`null` without a validated result. `comentario_error` then holds the state and the reason.

| Situation | Response |
|---|---|
| Invalid input: unknown mode, empty text, unsupported or unreadable file, non-UTF-8 text, mixed inputs, too many references | 400 with the reason |
| Embedding, indexing or search failure | 500 with the cause, never an empty success |
| Analysis failure | 200 with the passages kept, and `estado` other than `validado` on that fragment |

Requests are processed one at a time. A request with the analysis switched on makes one model call per fragment and no others.

## `GET /colecciones`

Lists this project's collections in Qdrant, for diagnostics.
