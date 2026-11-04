"""Tests de la interfaz Streamlit con AppTest (sin backend: las llamadas HTTP se simulan)."""
from types import SimpleNamespace
from unittest import mock

from streamlit.testing.v1 import AppTest

from conftest import ROOT

APP = str(ROOT / "front-end" / "app.py")


def fichero(nombre):
    return SimpleNamespace(name=nombre, type="application/pdf", getvalue=lambda: b"%PDF-1.4")


def test_la_app_arranca_sin_backend():
    with mock.patch("requests.get", side_effect=ConnectionError("backend caído")):
        at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    assert at.title[0].value == "🔍 Validación de Documentos"


def test_resultados_con_menos_de_tres_fragmentos_no_rompen_la_tabla():
    """Regresión: con menos de 3 fragmentos recuperados la app lanzaba IndexError."""
    respuesta = mock.Mock(status_code=200)
    respuesta.json.return_value = {"resultados": [{
        "chunk": "fragmento validado", "relevant_chunks": ["único fragmento"],
        "sources": [{"filename": "norma.pdf", "page": 3}], "scores": [0.81234], "respuesta": "alineado"}]}
    with mock.patch("requests.get", side_effect=ConnectionError), \
         mock.patch("requests.post", return_value=respuesta) as post:
        at = AppTest.from_file(APP, default_timeout=120)
        at.session_state["documento_referencia"] = [fichero("norma.pdf"), fichero("anexo.pdf")]
        at.session_state["documentos_consulta"] = [fichero("metodologia.pdf")]
        at.session_state["colecciones"] = ["demo"]
        at.session_state["coleccion_select"] = "demo"
        at.session_state["procesando"] = True
        at.run()

    assert not at.exception
    # Se envían los dos documentos de referencia
    enviados = [campo for campo, _ in post.call_args.kwargs["files"]]
    assert enviados.count("file_ref") == 2
    df = at.session_state["resultado_validacion"]
    assert df.loc[0, "Texto 1"] == "único fragmento" and df.loc[0, "Texto 2"] is None
    assert df.loc[0, "Fuentes (similitud)"] == "norma.pdf p.3 · 0.812"
    assert df.loc[0, "Respuesta"] == "alineado"
