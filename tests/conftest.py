"""Configuración común de los tests.

Los tests no necesitan Ollama ni un servidor Qdrant: usan Qdrant en memoria y una función de
embedding sustituta (bolsa de palabras con hash). Validan la lógica del pipeline y de la API,
no la calidad de los modelos; esa se mide con eval/run_eval.py contra Ollama real.
"""
import hashlib
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
sys.path.insert(0, str(ROOT / "examples"))

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


@pytest.fixture
def fake_embeddings():
    return FakeEmbeddings()


@pytest.fixture
def coleccion():
    """Nombre de colección único: el cliente en memoria se comparte entre tests."""
    return f"test_{uuid.uuid4().hex[:8]}"
