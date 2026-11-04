"""Tests de la API FastAPI con Qdrant en memoria y dobles de prueba para Ollama y el LLM."""
import pytest
from fastapi.testclient import TestClient

from conftest import SYNTHETIC, FakeEmbeddings

NORMATIVA = SYNTHETIC / "normativa_sintetica_pd.pdf"
METODOLOGIA = SYNTHETIC / "metodologia_banco_ficticio.docx"
TECNICO = SYNTHETIC / "tecnico_banco_ficticio.docx"
SAS = SYNTHETIC / "codigo_pd_ficticio.sas"


class FakeAgente:
    """Sustituye al agente de Ollama: registra el prompt usado y devuelve un texto fijo."""

    def __init__(self):
        self.prompt = None

    def set_prompt(self, prompt):
        self.prompt = prompt

    def get_answer(self, contexto, pregunta):
        return f"veredicto simulado ({len(contexto)} caracteres de contexto)"


class FakeCodigo:
    def get_answer(self, codigo):
        return "Descripción simulada: selecciona la muestra, marca el default y estima la PD."


@pytest.fixture
def client(monkeypatch):
    import api_rag
    import RAG3
    monkeypatch.setattr(RAG3, "nomic_embedding_model", FakeEmbeddings())
    agente = FakeAgente()
    monkeypatch.setattr(api_rag, "agente_dinamico", agente)
    c = TestClient(api_rag.app)
    c.agente = agente
    return c


def validar(client, refs, consulta, coleccion, tipo="Normativo vs Funcional", agente=False, ratio=1.0):
    files = [("file_ref", (p.name, p.read_bytes())) for p in refs]
    files.append(("file_query", (consulta.name, consulta.read_bytes())))
    return client.post("/validar", files=files, data={
        "collection_name": coleccion, "doc_sample_ratio": ratio,
        "usar_agente": str(agente).lower(), "tipo_validacion": tipo})


def test_validar_devuelve_evidencias_trazables(client, coleccion):
    r = validar(client, [NORMATIVA], METODOLOGIA, coleccion)
    assert r.status_code == 200
    resultados = r.json()["resultados"]
    assert len(resultados) > 0
    for res in resultados:
        assert len(res["relevant_chunks"]) == 3
        assert len(res["scores"]) == 3
        # Se conserva el nombre original del fichero (antes aparecía como "ref_<nombre>")
        assert {s["filename"] for s in res["sources"]} == {"normativa_sintetica_pd.pdf"}
        assert "respuesta" not in res
    assert coleccion in client.get("/colecciones").json()["colecciones"]


def test_muestreo_reduce_los_fragmentos_analizados(client, coleccion):
    completo = validar(client, [NORMATIVA], METODOLOGIA, coleccion).json()["resultados"]
    muestra = validar(client, [NORMATIVA], METODOLOGIA, coleccion, ratio=0.3).json()["resultados"]
    assert 1 <= len(muestra) < len(completo)


def test_indexa_todos_los_documentos_de_referencia(client, coleccion):
    """Regresión: la UI permitía subir varios documentos de referencia pero solo se indexaba el primero."""
    from X_qderant import get_qdrant_client

    assert validar(client, [NORMATIVA, TECNICO], METODOLOGIA, coleccion).status_code == 200
    puntos, _ = get_qdrant_client().scroll(coleccion, limit=1000, with_payload=True)
    assert {p.payload["filename"] for p in puntos} == {NORMATIVA.name, TECNICO.name}


@pytest.mark.parametrize("tipo,prompt_esperado", [
    ("Normativo vs Funcional", "evaluar_alineacion_normativa"),
    ("Funcional vs Técnico", "evaluar_tecnico_funcional"),  # antes: evaluar_tecnico_codigo
])
def test_agente_usa_el_prompt_del_tipo_de_validacion(client, coleccion, tipo, prompt_esperado):
    from prompt import prompt_definitivos

    r = validar(client, [NORMATIVA], METODOLOGIA, coleccion, tipo=tipo, agente=True)
    assert r.status_code == 200
    assert all(res["respuesta"].startswith("veredicto simulado") for res in r.json()["resultados"])
    assert client.agente.prompt is prompt_definitivos[prompt_esperado]


def test_prompt_normativo_situa_la_norma_como_texto_del_supervisor():
    """Regresión: la plantilla presentaba la norma recuperada como 'información a analizar' y viceversa."""
    from prompt import prompt_definitivos

    texto = prompt_definitivos["evaluar_alineacion_normativa"].format(contexto="NORMA", pregunta="BANCO")
    assert texto.index("Información a analizar") < texto.index("BANCO") < texto.index("Texto del supervisor")
    assert texto.index("Texto del supervisor") < texto.index("NORMA")


@pytest.mark.parametrize("tipo", ["Código vs Técnico", "Código vs Estadístico"])
def test_consulta_sas_se_traduce_a_texto(client, coleccion, monkeypatch, tipo):
    """Regresión: en 'Código vs Estadístico' un .sas devolvía error de formato no soportado."""
    import X_AGENT

    monkeypatch.setattr(X_AGENT, "codigo", FakeCodigo())
    monkeypatch.setattr(X_AGENT, "resumen", FakeCodigo())
    r = validar(client, [TECNICO], SAS, coleccion, tipo=tipo)
    assert r.status_code == 200
    assert any("Descripción simulada" in res["chunk"] for res in r.json()["resultados"])


def test_formato_no_soportado_devuelve_400(client, coleccion, tmp_path):
    txt = tmp_path / "notas.txt"
    txt.write_text("texto plano")
    r = validar(client, [txt], METODOLOGIA, coleccion)
    assert r.status_code == 400
    assert "no soportada" in r.json()["detail"]


def test_fallo_de_ollama_devuelve_500_con_detalle(client, coleccion, monkeypatch):
    """Regresión: antes el error se silenciaba y la API respondía 200 con resultados vacíos."""
    import RAG3

    monkeypatch.setattr(RAG3, "nomic_embedding_model", FakeEmbeddings(fail=True))
    r = validar(client, [NORMATIVA], METODOLOGIA, coleccion)
    assert r.status_code == 500
    assert "Ollama no disponible" in r.json()["detail"]
