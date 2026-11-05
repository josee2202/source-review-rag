"""Integration check: fixed cases, a rejected answer is never a hit, and what is shown has to match what was evaluated."""
import json
import sys
from pathlib import Path

import pytest
import yaml

import juez
from conftest import ROOT

sys.path.insert(0, str(ROOT / "benchmarks"))
sys.path.insert(0, str(ROOT / "front-end"))

import bench_core as core  # noqa: E402
import build_integration_report as report  # noqa: E402
import run_integration_check as check  # noqa: E402

CONTROLES = core.load_cases(ROOT / "benchmarks/dataset/integration_cases.yaml")
EVALUACION = {c["id"]: c for c in core.load_cases(ROOT / "benchmarks/dataset/eval_cases.yaml")}
EJEMPLOS = yaml.safe_load((ROOT / "benchmarks/dataset/e2e_examples.yaml").read_text("utf-8"))["examples"]


def test_los_casos_de_control_cubren_lo_que_se_pidio():
    etiquetas = {t for c in CONTROLES for t in c["tags"]}
    assert {"days_60_90", "exceptions", "correct_contradicted_unsupported", "correct_unsupported", "all_supported", "irrelevant_omission"} <= etiquetas
    assert sum("simple_control" in c["tags"] for c in CONTROLES) >= 3
    assert {c["modality"] for c in CONTROLES} == {"text_vs_text", "report_vs_news", "news_vs_newsletter"}
    assert {c["label"] for c in CONTROLES} == set(core.LABELS)
    assert EVALUACION["EV-24"]["label"] == "insufficient_evidence"  # sigue en la regresión de los 30 casos
    assert "60 días" in EJEMPLOS["report_vs_news"]["reviewed"] and "movilidad reducida" in EJEMPLOS["text_vs_text"]["reviewed"]


def test_los_controles_son_coherentes_y_nuevos():
    previos = {c["claim"] for c in EVALUACION.values()} | {c["claim"] for c in core.load_cases(ROOT / "benchmarks/dataset/dev_cases.yaml")}
    for caso in CONTROLES:
        assert caso["claim"] not in previos, caso["id"]
        assert caso["rationale"]
        if caso["quote"]:
            textos = [p["text"] for p in caso["passages"] if p["id"] in caso["evidence_ids"]]
            assert any(core.normalize_for_match(caso["quote"]) in core.normalize_for_match(t) for t in textos), caso["id"]
        assert (caso["label"] == "insufficient_evidence") == (not caso["evidence_ids"]), caso["id"]
    assert len({c["id"] for c in CONTROLES}) == len(CONTROLES)


CASO = {**CONTROLES[0]}
PASAJE = next(p for p in CASO["passages"] if p["id"] in CASO["evidence_ids"])


def analisis(estado, etiqueta=None, motivo=None):
    return {"estado": estado, "etiqueta": etiqueta, "etiqueta_texto": "", "explicacion": "Explicación." if etiqueta else None, "evidencia": [], "motivo": motivo,
            "metricas": {}}


def crudo(label, quote="más de 90 días de impago"):
    return json.dumps({"passage_ids": [PASAJE["id"]], "quote": quote, "explanation": "El fragmento dice 60 y el pasaje 90.", "label": label})


def test_un_rechazo_no_cuenta_como_acierto_aunque_la_etiqueta_bruta_fuera_correcta():
    puntuado = report.score_answer(analisis("no_validado", motivo="La cita no aparece"), crudo("contradicted", "cita inventada"), "stop", CASO)
    assert puntuado["raw_label"] == "contradicted" and puntuado["raw"]["correct"]
    assert puntuado["shown_label"] == core.NO_ANSWER and not puntuado["accepted"]["correct"]
    resumen = report.stage_summary([puntuado], 1)
    assert resumen["accepted_view"]["correct"] == 0 and resumen["raw"]["correct"] == 1
    assert resumen["rejected"] == resumen["without_result"] == 1


def test_un_respaldo_falso_rechazado_solo_cuenta_en_la_lectura_bruta():
    puntuado = report.score_answer(analisis("no_validado", motivo="La cita no aparece"), crudo("supported", "cita inventada"), "stop", CASO)
    resumen = report.stage_summary([puntuado], 1)
    assert resumen["raw"]["false_support"] == 1 and resumen["accepted_view"]["false_support"] == 0


def test_un_respaldo_falso_aceptado_cuenta_en_ambas_lecturas():
    puntuado = report.score_answer(analisis("validado", "supported"), crudo("supported"), "stop", CASO)
    resumen = report.stage_summary([puntuado], 1)
    assert resumen["raw"]["false_support"] == 1 and resumen["accepted_view"]["false_support"] == 1


def test_un_fallo_tecnico_se_cuenta_como_tal_y_no_tiene_etiqueta():
    tiempo = report.score_answer(analisis("no_disponible", motivo="El modelo no ha respondido y se ha cancelado la petición."), "", None, CASO)
    conexion = report.score_answer(analisis("no_disponible", motivo="No se ha podido conectar"), "", None, CASO)
    resumen = report.stage_summary([tiempo, conexion], 3)
    assert resumen["unavailable"] == 2 and resumen["accepted_view"]["timeout"] == 1 and resumen["accepted_view"]["request_error"] == 1
    assert resumen["accepted_view"]["not_executed"] == 1 and resumen["accepted_view"]["correct"] == 0


# * === Lo mostrado es lo evaluado ==============================================================

def respuesta_de_api(explicacion="El fragmento dice 60 días y el pasaje dice 90.", etiqueta="contradicted"):
    fuente = {"ref_id": "R1", "filename": "a.pdf", "location": "p. 1"}
    resultado = {"fragmento": 1, "chunk": "El plazo es de 60 días.", "chunk_location": {"location": "párr. 1"}, "relevant_chunks": ["El plazo es de 90 días naturales."],
                 "sources": [fuente], "scores": [0.8], "context": "c",
                 "analisis": {"estado": "validado", "etiqueta": etiqueta, "etiqueta_texto": juez.TEXTO_ETIQUETA[etiqueta], "explicacion": explicacion, "motivo": None, "modelo": "m", "metricas": {},
                              "evidencia": [{"id": "P1", "pasaje": 1, "cita": "90 días naturales", "fuente": fuente}]},
                 "comentario": "resumen"}
    return {"modalidad": "report_vs_news", "modalidad_nombre": "Informe vs. noticia", "comentario_solicitado": True, "modelo_analisis": "m",
            "referencias": [{"ref_id": "R1", "filename": "a.pdf"}], "indice": {"coleccion": "c", "fragmentos": 1, "reutilizado": False},
            "texto_revisado": {"filename": None, "fragmentos_totales": 1, "fragmentos_revisados": 1, "fragmentos_sin_revisar": 0, "sin_revisar_desde": None}, "resultados": [resultado]}


def llamada(cuerpo):
    resultado = cuerpo["resultados"][0]
    prompt = juez.construir_prompt(cuerpo["modalidad"], resultado["chunk"], juez.pasajes_de_resultado(resultado))
    return {"prompt": prompt, "done_reason": "stop", "wall_s": 1.0,
            "response": json.dumps({"passage_ids": ["P1"], "quote": "90 días naturales", "explanation": "El fragmento dice 60 días y el pasaje dice 90.", "label": "contradicted"})}


def test_la_comprobacion_acepta_lo_que_coincide():
    cuerpo = respuesta_de_api()
    assert check.check_shown_is_evaluated(cuerpo, [llamada(cuerpo)], juez)["all_ok"] is True


@pytest.mark.parametrize("cambio", [
    {"explicacion": "Un texto generado después."},
    {"etiqueta": "supported"},
])
def test_la_comprobacion_detecta_un_resultado_distinto_del_evaluado(cambio):
    cuerpo = respuesta_de_api(**cambio)
    comprobado = check.check_shown_is_evaluated(cuerpo, [llamada(respuesta_de_api())], juez)
    assert comprobado["all_ok"] is False and comprobado["fragments"][0]["api_equals_validation_of_raw_answer"] is False


def test_la_comprobacion_detecta_una_generacion_de_mas():
    cuerpo = respuesta_de_api()
    assert check.check_shown_is_evaluated(cuerpo, [llamada(cuerpo), llamada(cuerpo)], juez)["all_ok"] is False


def test_la_ejecucion_se_detiene_tras_varios_tiempos_agotados():
    presupuesto = check.Budget(0)
    agotado = {"motivo": "El modelo no ha respondido en 120 s y se ha cancelado la petición.", "metricas": {"tiempo_s": 120}}
    presupuesto.account(agotado)
    presupuesto.account({"motivo": None, "metricas": {"tiempo_s": 20}})  # una respuesta corta la racha
    presupuesto.account(agotado)
    presupuesto.account(agotado)
    with pytest.raises(check.Abort):
        presupuesto.account(agotado)


# * === CI ======================================================================================

def test_la_ci_arranca_ollama_y_no_oculta_fallos():
    flujo = (ROOT / ".github/workflows/ci.yml").read_text("utf-8")
    assert "continue-on-error" not in flujo
    assert "ollama serve > ollama.log" in flujo and "did not answer on port 11434" in flujo
    assert 'if [ -f "$log" ]' in flujo and "cat api.log" not in flujo
    assert "qwen3.5" not in flujo and "llama3.2" not in flujo  # los modelos generativos no se descargan en CI


# * === Resultados publicados ===================================================================

RESULTADOS = ROOT / "benchmarks/results/2026-11-05_integration-check-v2"


def test_las_tablas_del_informe_son_las_generadas_con_los_resultados_guardados():
    tablas = report.build_tables(report.build_summary(RESULTADOS))
    assert tablas == (RESULTADOS / "tables.md").read_text("utf-8")
    assert tablas.strip() in (ROOT / "docs/integration-check.md").read_text("utf-8")


def test_la_ejecucion_publicada_se_completo_y_uso_el_modelo_configurado():
    manifiesto = json.loads((RESULTADOS / "manifest.json").read_text("utf-8"))
    assert manifiesto["model"]["tag"] == juez.configuracion()["modelo"] == "qwen3.5:4b-q4_K_M"
    assert manifiesto["request"]["think"] is False and manifiesto["request"]["options"]["num_thread"] == 8
    assert json.loads((RESULTADOS / "run_status.json").read_text("utf-8"))["status"] == "completed"
    assert json.loads((RESULTADOS / "interface_check/ui_check.json").read_text("utf-8"))["all_ok"] is True
    assert json.loads((RESULTADOS / "interface_check/missing_model_check.json").read_text("utf-8"))["all_ok"] is True


def test_las_cifras_del_readme_coinciden_con_los_resultados():
    readme = (ROOT / "README.md").read_text("utf-8")
    resumen = report.build_summary(RESULTADOS)
    regresion, controles = resumen["regression"], resumen["controls"]
    documentos = resumen["e2e"]
    aciertos, fragmentos = sum(d["correct"] for d in documentos), sum(d["fragments"] for d in documentos)
    assert (f"| The 30 known cases | {regresion['accepted_view']['correct']}/30 | {regresion['without_result']} | "
            f"{regresion['accepted_view']['false_support']}/{regresion['accepted_view']['negatives_attempted']} |") in readme
    assert (f"| 15 control cases, 5 of them unseen by the final prompt | {controles['accepted_view']['correct']}/{controles['planned']} | {controles['without_result']} | "
            f"{controles['accepted_view']['false_support']}/{controles['accepted_view']['negatives_attempted']} |") in readme
    assert f"14 fragments | {aciertos}/{fragmentos} | {fragmentos - sum(f['state'] == 'validado' for d in documentos for f in d['detail'])} | none |" in readme
    assert not any(f["shown"] == "supported" and f["expected"] != "supported" for d in documentos for f in d["detail"])
    assert regresion["unavailable"] == controles["unavailable"] == 0


def test_el_readme_no_atribuye_el_resultado_del_benchmark_a_la_aplicacion():
    readme = (ROOT / "README.md").read_text("utf-8")
    assert "96.7" not in readme and "96,7" not in readme
    for prohibido in ("real time", "state of the art", "better than Claude", "fully private", "GDPR", "encrypted at rest"):
        assert prohibido.lower() not in readme.lower(), prohibido
    assert "not real news and not the application as a whole" in readme


def test_los_enlaces_e_imagenes_locales_de_la_documentacion_existen():
    import re
    for ruta in ("README.md", "docs/how-it-works.md", "docs/api.md", "docs/integration-check.md", "docs/commit-history.md", "benchmarks/README.md"):
        fichero = ROOT / ruta
        for destino in re.findall(r"\]\(([^)#\s]+?)(?:#[^)]*)?\)", fichero.read_text("utf-8")):
            if not destino.startswith(("http://", "https://", "mailto:")):
                assert (fichero.parent / destino).exists(), f"{ruta}: {destino}"


def test_cada_commit_de_un_manifiesto_tiene_su_equivalente_publicado():
    nota = (ROOT / "docs/commit-history.md").read_text("utf-8")
    manifiestos = sorted((ROOT / "benchmarks/results").glob("*/manifest.json"))
    assert len(manifiestos) == 4
    for manifiesto in manifiestos:
        commit = json.loads(manifiesto.read_text("utf-8"))["commit"]
        assert f"`{commit[:12]}`" in nota, manifiesto.parent.name
