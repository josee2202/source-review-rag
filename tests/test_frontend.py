"""Tests de la interfaz Streamlit con AppTest (sin backend: las llamadas HTTP se simulan)."""
import io
import json
from types import SimpleNamespace
from unittest import mock

import pytest
import requests
from openpyxl import load_workbook
from streamlit.testing.v1 import AppTest

from conftest import ROOT
from presentacion import (ORIGEN_ARCHIVO, ORIGEN_TEXTO, aviso_sin_revisar, construir_peticion, entradas_completas,
                          escapar_markdown, exportar_excel, resultados_a_dataframe)

APP = str(ROOT / "front-end" / "app.py")

FUENTE = {"ref_id": "R1", "filename": "norma.pdf", "page": 3, "location": "p. 3", "title": "Informe", "outlet": None,
          "date": "2025-05-02", "url": "https://ejemplo.org/a"}


def respuesta_api(**extra):
    base = {
        "modalidad": "report_vs_news", "modalidad_nombre": "Informe vs. noticia", "comentario_solicitado": False,
        "referencias": [{"ref_id": "R1", "filename": "norma.pdf", "title": "Informe", "outlet": None,
                         "date": "2025-05-02", "url": "https://ejemplo.org/a"}],
        "indice": {"coleccion": "srr_x", "fragmentos": 1, "reutilizado": False},
        "texto_revisado": {"filename": "noticia.txt", "fragmentos_totales": 1, "fragmentos_revisados": 1,
                           "fragmentos_sin_revisar": 0, "sin_revisar_desde": None},
        "resultados": [{"fragmento": 1, "chunk": "fragmento revisado", "chunk_location": {"location": "párr. 1"},
                        "relevant_chunks": ["único pasaje"], "sources": [FUENTE], "scores": [0.81234]}],
    }
    return {**base, **extra}


def fichero(nombre, contenido=b"contenido"):
    return SimpleNamespace(name=nombre, type="text/plain", getvalue=lambda: contenido)


def app():
    with mock.patch("requests.get", side_effect=ConnectionError("backend caído")):
        return AppTest.from_file(APP, default_timeout=120).run()


def rellenar_y_revisar(at, post):
    at.text_area(key="ref_texto").set_value("texto de referencia")
    at.text_area(key="rev_texto").set_value("texto a revisar")
    at.run()
    with mock.patch("requests.post", post):
        at.button(key="revisar").click().run()
    return at


def test_la_app_arranca_sin_backend():
    at = app()
    assert not at.exception
    assert at.title[0].value == "🔍 Source Review RAG"
    assert at.radio(key="modalidad").options == ["Texto vs. texto", "Informe vs. noticia", "Noticias vs. newsletter"]
    assert at.radio(key="modalidad").value == "text_vs_text"
    assert at.checkbox(key="usar_comentario").value is False
    assert at.checkbox(key="solo_inicio").value is False


@pytest.mark.parametrize("modalidad,referencia,revisado", [
    ("text_vs_text", "Texto de referencia", "Texto a revisar"),
    ("report_vs_news", "Informe o documento original", "Noticia a revisar"),
    ("news_vs_newsletter", "Noticias de referencia", "Newsletter a revisar"),
])
def test_elegir_modalidad_actualiza_etiquetas_y_ayuda(modalidad, referencia, revisado):
    at = app()
    at.radio(key="modalidad").set_value(modalidad).run()
    assert not at.exception
    assert referencia in at.subheader[0].value and revisado in at.subheader[1].value
    aviso_pegado = any("una sola referencia" in c.value for c in at.caption)
    assert aviso_pegado == (modalidad == "news_vs_newsletter")


def test_no_se_puede_revisar_con_entradas_vacias():
    at = app()
    assert at.button(key="revisar").disabled
    at.text_area(key="ref_texto").set_value("   ").run()
    at.text_area(key="rev_texto").set_value("texto").run()
    assert at.button(key="revisar").disabled


def test_texto_pegado_se_envia_con_la_modalidad_elegida_y_se_muestra():
    post = mock.Mock(return_value=mock.Mock(status_code=200, json=lambda: respuesta_api()))
    at = app()
    at.radio(key="modalidad").set_value("report_vs_news").run()
    rellenar_y_revisar(at, post)

    assert not at.exception
    enviado = post.call_args.kwargs
    assert enviado["data"]["modalidad"] == "report_vs_news"
    assert enviado["data"]["texto_ref"] == "texto de referencia"
    assert enviado["data"]["texto_query"] == "texto a revisar"
    assert enviado["data"]["usar_agente"] == "false" and enviado["data"]["fraccion_inicial"] == "1.0"
    assert enviado["files"] == []

    textos = " ".join(m.value for m in at.markdown)
    assert "fragmento revisado" in textos and "único pasaje" in textos and "0.812" in textos
    assert any("R1 · norma.pdf · p. 3 · Informe" in c.value.replace("\\", "") for c in at.caption)
    assert not any("Comentario del modelo" in m.value for m in at.markdown)
    assert any("no es una probabilidad" in c.value for c in at.caption)


def test_cambiar_de_modalidad_limpia_los_resultados_anteriores():
    post = mock.Mock(return_value=mock.Mock(status_code=200, json=lambda: respuesta_api()))
    at = rellenar_y_revisar(app(), post)
    assert at.session_state["resultado_validacion"] is not None
    at.radio(key="modalidad").set_value("news_vs_newsletter").run()
    assert at.session_state["resultado_validacion"] is None
    assert not any("Resultados" in s.value for s in at.subheader)


def test_error_del_servidor_se_muestra_y_no_deja_resultados():
    fallo = mock.Mock(status_code=400, json=lambda: {"detail": "Modalidad desconocida"})
    at = rellenar_y_revisar(app(), mock.Mock(return_value=fallo))
    assert not at.exception
    assert any("Modalidad desconocida" in e.value for e in at.error)
    assert at.session_state["resultado_validacion"] is None


def test_backend_caido_se_muestra_como_error():
    post = mock.Mock(side_effect=requests.exceptions.ConnectionError("rechazada"))
    at = rellenar_y_revisar(app(), post)
    assert not at.exception
    assert any("No se ha podido conectar con el backend" in e.value for e in at.error)


def test_fallo_del_comentario_se_muestra_junto_a_los_pasajes():
    res = {**respuesta_api()["resultados"][0], "comentario": None, "comentario_error": "ConnectionError: sin Ollama"}
    cuerpo = respuesta_api(comentario_solicitado=True, resultados=[res])
    at = rellenar_y_revisar(app(), mock.Mock(return_value=mock.Mock(status_code=200, json=lambda: cuerpo)))
    assert not at.exception
    assert any("1 de 1 fragmentos" in e.value for e in at.error)
    assert any("sin Ollama" in e.value for e in at.error)
    assert "único pasaje" in " ".join(m.value for m in at.markdown)


def test_comentario_del_modelo_se_distingue_de_los_pasajes():
    res = {**respuesta_api()["resultados"][0], "comentario": "primera [R1 · p. 3]\nsegunda"}
    cuerpo = respuesta_api(comentario_solicitado=True, resultados=[res])
    at = rellenar_y_revisar(app(), mock.Mock(return_value=mock.Mock(status_code=200, json=lambda: cuerpo)))
    assert any("Comentario del modelo" in m.value and "no es un veredicto" in m.value for m in at.markdown)
    assert any("primera" in m.value and "  \nsegunda" in m.value for m in at.markdown)  # conserva los saltos de línea


def test_solo_el_inicio_se_nombra_y_avisa_de_lo_no_revisado():
    rev = {"filename": "n.txt", "fragmentos_totales": 5, "fragmentos_revisados": 2, "fragmentos_sin_revisar": 3,
           "sin_revisar_desde": "párr. 3"}
    cuerpo = respuesta_api(texto_revisado=rev)
    post = mock.Mock(return_value=mock.Mock(status_code=200, json=lambda: cuerpo))
    at = app()
    at.checkbox(key="solo_inicio").check().run()
    at.slider(key="porcentaje_inicio").set_value(40).run()
    rellenar_y_revisar(at, post)
    assert post.call_args.kwargs["data"]["fraccion_inicial"] == "0.4"
    assert any("quedan sin revisar 3 de 5 fragmentos, a partir de párr. 3" in w.value for w in at.warning)


def test_varias_noticias_se_envian_todas_con_sus_metadatos_en_orden():
    metas = [{"title": "Uno", "outlet": "", "date": "", "url": ""}, {"title": "", "outlet": "", "date": "", "url": "https://b.org"}]
    files, data = construir_peticion("news_vs_newsletter", ORIGEN_ARCHIVO, [fichero("a.txt"), fichero("b.md")], "", metas,
                                     ORIGEN_ARCHIVO, fichero("news.md"), "", False)
    assert [(campo, f[0]) for campo, f in files] == [("file_ref", "a.txt"), ("file_ref", "b.md"), ("file_query", "news.md")]
    assert json.loads(data["ref_metadata"]) == metas
    assert "texto_ref" not in data and "texto_query" not in data


@pytest.mark.parametrize("ref_origen,rev_origen,campos_form,n_files", [
    (ORIGEN_TEXTO, ORIGEN_TEXTO, {"texto_ref", "texto_query"}, 0),
    (ORIGEN_ARCHIVO, ORIGEN_TEXTO, {"texto_query"}, 1),
    (ORIGEN_TEXTO, ORIGEN_ARCHIVO, {"texto_ref"}, 1),
    (ORIGEN_ARCHIVO, ORIGEN_ARCHIVO, set(), 2),
])
def test_solo_se_envia_el_origen_seleccionado(ref_origen, rev_origen, campos_form, n_files):
    """Aunque quede texto escrito o un archivo cargado en el otro origen, no se combinan."""
    files, data = construir_peticion("text_vs_text", ref_origen, [fichero("r.txt")], "pegado ref", [{}],
                                     rev_origen, fichero("q.txt"), "pegado rev", False)
    assert set(data) & {"texto_ref", "texto_query"} == campos_form and len(files) == n_files


def test_metadatos_vacios_no_se_envian():
    _, data = construir_peticion("text_vs_text", ORIGEN_TEXTO, [], "r", [{"title": " ", "url": ""}], ORIGEN_TEXTO, None, "q", False)
    assert "ref_metadata" not in data


def test_entradas_completas():
    assert entradas_completas(ORIGEN_TEXTO, [], "r", ORIGEN_ARCHIVO, fichero("q.txt"), "")
    assert not entradas_completas(ORIGEN_ARCHIVO, [], "texto ignorado", ORIGEN_TEXTO, None, "q")
    assert not entradas_completas(ORIGEN_TEXTO, [], "r", ORIGEN_ARCHIVO, None, "texto ignorado")


def test_menos_de_tres_pasajes_dejan_columnas_vacias():
    df = resultados_a_dataframe(respuesta_api())
    fila = df.iloc[0]
    assert fila["Pasaje 1"] == "único pasaje" and fila["Pasaje 2"] is None and fila["Fuente 3"] is None
    assert fila["Fuentes (similitud)"] == "R1 · norma.pdf · p. 3 · Informe · 2025-05-02 · https://ejemplo.org/a · 0.812"
    assert "Comentario del modelo" not in df.columns


def test_excel_incluye_referencias_comentarios_y_lo_no_revisado():
    res = {**respuesta_api()["resultados"][0], "comentario": "observación del modelo"}
    rev = {"filename": "n.txt", "fragmentos_totales": 5, "fragmentos_revisados": 1, "fragmentos_sin_revisar": 4,
           "sin_revisar_desde": "párr. 2"}
    libro = load_workbook(io.BytesIO(exportar_excel(respuesta_api(comentario_solicitado=True, resultados=[res], texto_revisado=rev))))
    assert libro.sheetnames == ["Resultados", "Referencias", "Resumen"]
    filas = list(libro["Resultados"].values)
    fila = dict(zip(filas[0], filas[1]))
    assert fila["Ubicación"] == "párr. 1" and fila["Comentario del modelo"] == "observación del modelo"
    assert fila["Similitud 1"] == 0.81234 and "p. 3" in fila["Fuente 1"]
    resumen = dict(list(libro["Resumen"].values)[1:])
    assert resumen["Fragmentos sin revisar"] == 4 and resumen["Sin revisar a partir de"] == "párr. 2"
    assert "no es una probabilidad" in resumen["Nota sobre la similitud"]


def test_excel_trata_los_textos_aportados_como_datos():
    res = {**respuesta_api()["resultados"][0], "chunk": "=HYPERLINK(\"http://x\")\x0b y más"}
    libro = load_workbook(io.BytesIO(exportar_excel(respuesta_api(resultados=[res]))))
    celda = next(c for c in libro["Resultados"][2] if c.value and "HYPERLINK" in str(c.value))
    assert celda.data_type == "s" and "\x0b" not in celda.value


def test_aviso_sin_revisar_es_none_si_se_reviso_todo():
    assert aviso_sin_revisar(respuesta_api()) is None


def test_escapar_markdown_muestra_el_texto_literalmente():
    assert escapar_markdown("**a** [b](c) $d$ :red[e]") == r"\*\*a\*\* \[b\]\(c\) \$d\$ \:red\[e\]"
