"""Configuración común de los tests.

Los tests no necesitan Ollama ni un servidor Qdrant: usan Qdrant en memoria y una función de
embedding sustituta (bolsa de palabras con hash). Validan la lógica del pipeline y de la API,
no la calidad de los modelos. tests/smoke_e2e.py comprueba el recorrido con modelos reales.
"""
import hashlib
import json
import os
import re
import sys
import uuid
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
os.environ["QDRANT_LOCATION"] = ":memory:"  # antes de importar config
sys.path.insert(0, str(ROOT / "IA-modulable"))
sys.path.insert(0, str(ROOT / "front-end"))

SYNTHETIC = ROOT / "examples" / "synthetic"


class FakeEmbeddings:
    """Embedding determinista (doble de prueba): textos con más palabras en común quedan más cerca."""

    dim = 256

    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def _vec(self, text):
        v = np.zeros(self.dim)
        for w in re.findall(r"\w+", text.lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1.0
        n = np.linalg.norm(v)
        return (v / n if n else v + 1e-6).tolist()

    def embed_documents(self, texts):
        if self.fail:
            raise ConnectionError("Ollama no disponible (simulado)")
        self.calls.append(len(texts))
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self.embed_documents([text])[0]


def salida_ollama(texto="", status="completed", done_reason="stop", http_status=200, error=None, prompt_tokens=900):
    """Lo que devuelve juez.generar, para los dobles de prueba."""
    final = {"done_reason": done_reason, "prompt_eval_count": prompt_tokens, "eval_count": 50, "load_duration": 10**9,
             "prompt_eval_duration": 2 * 10**9, "eval_duration": 3 * 10**9} if status == "completed" else {}
    return {"status": status, "response": texto, "final": final, "error": error, "http_status": http_status,
            "wall_s": 5.0, "ttft_s": 2.0}


class FakeOllama:
    """Sustituye a juez.generar: registra cada petición y por defecto respalda citando el inicio del primer pasaje."""

    def __init__(self):
        self.peticiones = []
        self.respuestas = []  # salidas de salida_ollama(), en orden de llamada

    def __call__(self, peticion, timeout_s, host, port):
        self.peticiones.append(peticion)
        if self.respuestas:
            return self.respuestas.pop(0)
        primer_pasaje = re.search(r"\[P1\][^\n]*\n(.+)", peticion["prompt"]).group(1)
        return salida_ollama(json.dumps({"passage_ids": ["P1"], "quote": " ".join(primer_pasaje.split()[:3]),
                                         "explanation": "Explicación simulada del modelo.", "label": "supported"}))

    @property
    def prompts(self):
        return [p["prompt"] for p in self.peticiones]


@pytest.fixture
def fake_embeddings():
    return FakeEmbeddings()


@pytest.fixture
def unico():
    """Marca única: el cliente en memoria se comparte entre tests y un corpus igual reutilizaría su índice."""
    return f"marca{uuid.uuid4().hex[:10]}"
