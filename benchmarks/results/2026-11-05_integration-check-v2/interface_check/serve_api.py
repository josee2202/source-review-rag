"""Real API over HTTP on port 8010, with a scratch index, recording every generation and the analysis the API returned."""
import json
import os
import sys
import tempfile
from pathlib import Path

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
os.environ["QDRANT_LOCATION"] = tempfile.mkdtemp(prefix="srr_ui_qdrant_")
os.environ["LLM_NUM_THREAD"] = "8"
sys.path.insert(0, "IA-modulable")

import uvicorn  # noqa: E402

import api_rag  # noqa: E402
import juez  # noqa: E402

generar, analizar = juez.generar, juez.analizar_resultados


def log(name, row):
    with open(out / name, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def spy(peticion, timeout_s, host, port):
    salida = generar(peticion, timeout_s, host, port)
    log("generations.jsonl", {"model": peticion["model"], "options": peticion["options"], "think": peticion["think"],
                              "response": salida["response"], "wall_s": salida["wall_s"], "final": salida["final"]})
    return salida


def analizar_spy(resultados, modalidad):
    analizar(resultados, modalidad)
    log("api_analysis.jsonl", {"modalidad": modalidad, "resultados": [
        {"fragmento": r["fragmento"], "chunk": r["chunk"], "analisis": r["analisis"], "relevant_chunks": r["relevant_chunks"]}
        for r in resultados]})


juez.generar, juez.analizar_resultados = spy, analizar_spy
uvicorn.run(api_rag.app, host="127.0.0.1", port=8010, log_level="warning")
