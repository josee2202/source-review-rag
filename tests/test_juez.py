"""Análisis opcional: contrato con el juez evaluado, validación de la salida y tratamiento de fallos, sin Ollama ni modelos."""
import ast
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import yaml

import juez
import prompt as prompt_produccion
from conftest import FakeOllama, salida_ollama

ROOT = Path(__file__).resolve().parents[1]
PROTOCOLO = yaml.safe_load((ROOT / "benchmarks/protocol.yaml").read_text("utf-8"))
PASAJES = [{"id": "P1", "source": "informe.pdf · p. 1", "text": "El plazo máximo es de 90 días naturales desde la solicitud."},
           {"id": "P2", "source": "informe.pdf · p. 2", "text": "La tarifa   general será de 28 euros al mes."}]


def respuesta(label="contradicted", ids=("P1",), quote="90 días naturales", explanation="El fragmento dice 60 días y el pasaje dice 90."):
    return json.dumps({"passage_ids": list(ids), "quote": quote, "explanation": explanation, "label": label})


@pytest.fixture
def ollama(monkeypatch):
    doble = FakeOllama()
    monkeypatch.setattr(juez, "generar", doble)
    return doble


def analizar(ollama, *salidas, **ajustes):
    ollama.respuestas = list(salidas)
    return juez.ejecutar("report_vs_news", "El plazo máximo es de 60 días.", PASAJES, **ajustes)["analisis"]


# * === Es el juez que se evaluó ================================================================

def test_el_esquema_y_los_roles_son_los_del_protocolo_congelado():
    assert prompt_produccion.ESQUEMA_RESPUESTA == PROTOCOLO["judge"]["response_schema"]
    assert prompt_produccion.ROLES == PROTOCOLO["judge"]["roles"]


def test_el_prompt_solo_anade_las_normas_declaradas():
    congelado = PROTOCOLO["judge"]["prompt"].splitlines()
    produccion = prompt_produccion.PLANTILLA.splitlines()
    assert [linea for linea in produccion if linea not in prompt_produccion.NORMAS_AÑADIDAS] == congelado


def test_los_parametros_son_los_evaluados_salvo_los_hilos(monkeypatch):
    monkeypatch.delitem(juez.config.cfg_param["llm"]["text"], "num_thread", raising=False)
    ollama = FakeOllama()
    monkeypatch.setattr(juez, "generar", ollama)
    juez.ejecutar("text_vs_text", "Dato.", PASAJES)
    peticion = ollama.peticiones[0]
    esperado = dict(PROTOCOLO["request"]["options"])
    esperado.pop("num_thread")
    assert peticion["options"] == esperado and "num_thread" not in peticion["options"]
    assert peticion["think"] is False and peticion["stream"] is True
    assert peticion["format"] == PROTOCOLO["judge"]["response_schema"]
    assert peticion["model"] == "qwen3.5:4b-q4_K_M"


def test_los_hilos_solo_se_envian_si_se_configuran(monkeypatch):
    monkeypatch.setitem(juez.config.cfg_param["llm"]["text"], "num_thread", 6)
    ollama = FakeOllama()
    monkeypatch.setattr(juez, "generar", ollama)
    juez.ejecutar("text_vs_text", "Dato.", PASAJES)
    assert ollama.peticiones[0]["options"]["num_thread"] == 6


def test_produccion_no_depende_de_los_scripts_de_benchmark():
    for ruta in (ROOT / "IA-modulable").glob("*.py"):
        importados = {n.name.split(".")[0] for nodo in ast.walk(ast.parse(ruta.read_text("utf-8"))) if isinstance(nodo, ast.Import) for n in nodo.names}
        importados |= {nodo.module.split(".")[0] for nodo in ast.walk(ast.parse(ruta.read_text("utf-8"))) if isinstance(nodo, ast.ImportFrom) and nodo.module}
        assert not importados & {"benchmarks", "bench_core", "bench_ollama", "bench_resources", "run_benchmark", "run_integration_check"}, ruta
        assert "OllamaLLM" not in ruta.read_text("utf-8"), ruta  # ninguna generación libre


def test_el_recorrido_produce_el_mismo_formato_de_pasajes_que_la_etapa_evaluada():
    resultado = {"relevant_chunks": ["Texto A", "Texto B"],
                 "sources": [{"ref_id": "R1", "filename": "a.pdf", "title": "T", "location": "p. 2"}, {"filename": "b.txt", "location": "párr. 1"}]}
    assert juez.pasajes_de_resultado(resultado) == [{"id": "P1", "source": "a.pdf · T · p. 2", "text": "Texto A"},
                                                    {"id": "P2", "source": "b.txt · párr. 1", "text": "Texto B"}]


# * === Validación ==============================================================================

@pytest.mark.parametrize("label,texto", [("supported", "Respaldado por los pasajes"), ("contradicted", "Contradicción con los pasajes")])
def test_una_respuesta_con_evidencia_literal_se_valida(ollama, label, texto):
    analisis = analizar(ollama, salida_ollama(respuesta(label)))
    assert analisis["estado"] == "validado" and analisis["etiqueta"] == label and analisis["etiqueta_texto"] == texto
    assert analisis["evidencia"] == [{"id": "P1", "pasaje": 1, "cita": "90 días naturales"}]
    assert analisis["explicacion"].startswith("El fragmento dice 60") and analisis["motivo"] is None


def test_insuficiente_admite_no_citar_nada(ollama):
    analisis = analizar(ollama, salida_ollama(respuesta("insufficient_evidence", (), "")))
    assert analisis["estado"] == "validado" and analisis["etiqueta_texto"] == "Información insuficiente en los pasajes"
    assert analisis["evidencia"] == []


@pytest.mark.parametrize("cuerpo,fragmento_motivo", [
    (respuesta("supported", (), ""), "exige un pasaje citado"),
    (respuesta("contradicted", ("P1",), ""), "exige un pasaje citado"),
    (respuesta("contradicted", (), "90 días naturales"), "exige un pasaje citado"),
    (respuesta("contradicted", ("P9",), "90 días naturales"), "no existen: P9"),
    (respuesta("supported", ("P1",), "un plazo de noventa días"), "literalmente"),  # coincidencia semántica: no es literal
    (respuesta("supported", ("P2",), "90 días naturales"), "literalmente"),  # está en otro pasaje, no en el citado
    (respuesta("supported", ("P1",), "90 Días Naturales"), "literalmente"),  # las mayúsculas no se normalizan
    (respuesta("supported", ("P1",), "90"), "demasiado corta"),
    (respuesta("insufficient_evidence", ("P1",), "cita inventada"), "literalmente"),
    (respuesta("insufficient_evidence", (), "90 días naturales"), "literalmente"),
    (respuesta("otra"), "etiqueta admitida"),
    ('["no es un objeto"]', "etiqueta admitida"),
    ("no es JSON", "JSON válido"),
    (json.dumps({"label": "supported", "passage_ids": "P1", "quote": "x", "explanation": "y"}), "esquema"),
    (respuesta(explanation="   "), "explicación"),
])
def test_una_respuesta_que_no_se_puede_comprobar_no_lleva_etiqueta(ollama, cuerpo, fragmento_motivo):
    analisis = analizar(ollama, salida_ollama(cuerpo))
    assert analisis["estado"] == "no_validado" and analisis["etiqueta"] is None and analisis["explicacion"] is None
    assert analisis["etiqueta_texto"] == "Resultado no validado" and analisis["evidencia"] == []
    assert fragmento_motivo in analisis["motivo"]


def test_la_cita_solo_normaliza_espacios_comillas_y_punto_final(ollama):
    assert analizar(ollama, salida_ollama(respuesta(ids=("P2",), quote="«La tarifa general\nserá de 28 euros al mes.»")))["estado"] == "validado"
    assert analizar(ollama, salida_ollama(respuesta(ids=("[P1]",))))["evidencia"][0]["id"] == "P1"


def test_una_salida_cortada_no_se_valida_aunque_sea_utilizable(ollama):
    analisis = analizar(ollama, salida_ollama(respuesta(), done_reason="length"))
    assert analisis["estado"] == "no_validado" and "límite de tokens" in analisis["motivo"]


def test_el_campo_comentario_se_construye_sin_otra_llamada(ollama, monkeypatch):
    ollama.respuestas = [salida_ollama(respuesta())]
    resultados = [{"chunk": "El plazo es de 60 días.", "relevant_chunks": [PASAJES[0]["text"]], "sources": [{"ref_id": "R1", "filename": "a.pdf"}]}]
    juez.analizar_resultados(resultados, "report_vs_news")
    assert len(ollama.peticiones) == 1
    assert resultados[0]["comentario"] == ("Contradicción con los pasajes. El fragmento dice 60 días y el pasaje dice 90. "
                                           "Pasaje 1: «90 días naturales»")
    assert resultados[0]["analisis"]["evidencia"][0]["fuente"] == {"ref_id": "R1", "filename": "a.pdf"}


# * === Fallos técnicos =========================================================================

@pytest.mark.parametrize("salida,texto_motivo", [
    (salida_ollama(status="timeout", error="sin respuesta"), "se ha cancelado la petición"),
    (salida_ollama(status="error", http_status=None, error="ConnectionRefusedError"), "conectar con Ollama"),
    (salida_ollama(status="error", http_status=404, error="not found"), "ollama pull qwen3.5:4b-q4_K_M"),
    (salida_ollama(status="error", http_status=500, error="sin memoria"), "sin memoria"),
])
def test_un_fallo_tecnico_no_se_convierte_en_informacion_insuficiente(ollama, salida, texto_motivo):
    analisis = analizar(ollama, salida)
    assert analisis["estado"] == "no_disponible" and analisis["etiqueta"] is None
    assert analisis["etiqueta_texto"] == "Análisis no disponible" and texto_motivo in analisis["motivo"]
    assert len(ollama.peticiones) == 1  # sin reintentos


# * === Contexto ================================================================================

def test_lo_que_no_cabe_no_se_envia_ni_se_recorta(ollama):
    largos = [{"id": "P1", "source": "a", "text": "palabra " * 2000}]
    resultado = juez.ejecutar("text_vs_text", "Dato.", largos)["analisis"]
    assert resultado["estado"] == "no_disponible" and "no caben en el contexto" in resultado["motivo"]
    assert ollama.peticiones == []


def test_un_contexto_que_ollama_ha_podido_recortar_no_se_da_por_bueno(ollama):
    """Ollama 0.40.14 deja 2050 tokens de un prompt de 11624 con num_ctx 4096 y no lo indica en la respuesta."""
    analisis = analizar(ollama, salida_ollama(respuesta(), prompt_tokens=2050))
    assert analisis["estado"] == "no_validado" and "recortar el contexto" in analisis["motivo"]
    assert analizar(ollama, salida_ollama(respuesta(), prompt_tokens=2047))["estado"] == "validado"


def test_los_fragmentos_y_pasajes_mas_largos_de_la_configuracion_caben():
    cfg = juez.config.cfg_param["chunking"]
    pasajes = [{"id": f"P{n}", "source": "documento_con_nombre_largo.pdf · Un título de informe bastante largo · Medio · 2025-01-01 · p. 123",
                "text": "x" * cfg["reference"]["size"]} for n in (1, 2, 3)]
    prompt = prompt_produccion.construir_prompt("news_vs_newsletter", "y" * cfg["review"]["size"], pasajes)
    assert len(prompt) / juez.CARACTERES_POR_TOKEN < juez.limite_tokens_prompt(juez.configuracion()["num_ctx"])


# * === Sin estado compartido ===================================================================

def test_peticiones_concurrentes_de_distintas_modalidades_no_se_mezclan(monkeypatch):
    vistos = {}

    def generar(peticion, timeout_s, host, port):
        time.sleep(0.01)
        marca = peticion["prompt"].split("<fragmento_revisado>\n")[1].split("\n")[0]
        vistos[marca] = peticion["prompt"]
        return salida_ollama(respuesta("insufficient_evidence", (), ""))

    monkeypatch.setattr(juez, "generar", generar)
    modalidades = list(prompt_produccion.ROLES) * 10
    hilos = [threading.Thread(target=juez.ejecutar, args=(m, f"marca-{i}-{m}", PASAJES)) for i, m in enumerate(modalidades)]
    [h.start() for h in hilos]
    [h.join() for h in hilos]
    assert len(vistos) == len(modalidades)
    for marca, prompt in vistos.items():
        modalidad = marca.split("-")[2]
        assert prompt_produccion.ROLES[modalidad] in prompt
        assert not any(rol in prompt for m, rol in prompt_produccion.ROLES.items() if m != modalidad)


# * === Cliente real contra un servidor falso ===================================================

class ServidorFalso(BaseHTTPRequestHandler):
    def do_POST(self):
        cuerpo = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.peticiones.append(cuerpo)
        modo = self.server.modo
        if modo == "falta":
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'{"error":"model not found"}')
            return
        self.send_response(200)
        self.end_headers()
        if modo == "lento":
            try:
                for _ in range(60):
                    self.wfile.write(b'{"response":"x","done":false}\n')
                    self.wfile.flush()
                    time.sleep(0.05)
            except OSError:
                self.server.cancelada = True
            return
        if modo == "error":
            self.wfile.write(b'{"error":"sin memoria"}\n')
            return
        for parte in (respuesta()[:10], respuesta()[10:]):
            self.wfile.write(json.dumps({"response": parte, "done": False}).encode() + b"\n")
        self.wfile.write(json.dumps({"response": "", "done": True, "done_reason": "stop", "prompt_eval_count": 900, "eval_count": 40,
                                     "load_duration": 2 * 10**9, "prompt_eval_duration": 3 * 10**9, "eval_duration": 4 * 10**9}).encode() + b"\n")

    def log_message(self, *args):
        pass


@pytest.fixture
def servidor():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), ServidorFalso)
    srv.modo, srv.peticiones, srv.cancelada = "ok", [], False
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv
    srv.shutdown()


def contra(servidor, **ajustes):
    return juez.ejecutar("report_vs_news", "El plazo es de 60 días.", PASAJES, host="127.0.0.1", port=servidor.server_address[1], **ajustes)["analisis"]


def test_el_cliente_lee_el_flujo_y_convierte_las_duraciones(servidor):
    analisis = contra(servidor)
    assert analisis["estado"] == "validado"
    metricas = analisis["metricas"]
    assert metricas["carga_s"] == 2.0 and metricas["evaluacion_prompt_s"] == 3.0 and metricas["generacion_s"] == 4.0
    assert metricas["tokens_prompt"] == 900 and metricas["tokens_salida"] == 40 and metricas["tiempo_s"] > 0
    assert servidor.peticiones[0]["think"] is False and servidor.peticiones[0]["format"]["required"][-1] == "label"


def test_el_plazo_se_aplica_aunque_siga_llegando_salida_y_cierra_la_conexion(servidor):
    servidor.modo = "lento"
    inicio = time.perf_counter()
    analisis = contra(servidor, timeout_s=0.3)
    assert time.perf_counter() - inicio < 1.5
    assert analisis["estado"] == "no_disponible" and "se ha cancelado" in analisis["motivo"]
    limite = time.monotonic() + 4
    while not servidor.cancelada and time.monotonic() < limite:
        time.sleep(0.05)
    assert servidor.cancelada


def test_el_modelo_que_falta_y_el_error_del_flujo_son_no_disponible(servidor):
    servidor.modo = "falta"
    assert "ollama pull" in contra(servidor)["motivo"]
    servidor.modo = "error"
    analisis = contra(servidor)
    assert analisis["estado"] == "no_disponible" and "sin memoria" in analisis["motivo"]


def test_sin_servidor_es_no_disponible():
    analisis = juez.ejecutar("text_vs_text", "Dato.", PASAJES, host="127.0.0.1", port=1)["analisis"]
    assert analisis["estado"] == "no_disponible" and "conectar con Ollama" in analisis["motivo"]
