"""Análisis opcional: contrasta un fragmento con sus pasajes y devuelve un resultado estructurado y validado.

Es la única implementación del contraste: la usan la API y el seguimiento de integración (benchmarks/). No genera ningún
comentario libre: lo que se muestra es exactamente lo que devolvió el modelo y superó estas validaciones.

Una respuesta acaba en uno de tres estados, y solo el primero lleva etiqueta:
  validado      el esquema, los IDs y la cita literal son correctos. Que una cita exista no prueba que la interpretación
                sea correcta.
  no_validado   el modelo respondió, pero la salida se cortó, no cumple el esquema o su evidencia no se puede comprobar.
  no_disponible no hubo respuesta: Ollama no responde, falta el modelo, se agotó el tiempo o el contexto no cabe.
Un fallo técnico nunca se convierte en información insuficiente, y no hay reintentos.
"""
import http.client
import json
import re
import time
import unicodedata

from config import config
from prompt import ESQUEMA_RESPUESTA, ETIQUETAS, PLANTILLA, construir_prompt

VALIDADO, NO_VALIDADO, NO_DISPONIBLE = "validado", "no_validado", "no_disponible"
TEXTO_ETIQUETA = {"supported": "Respaldado por los pasajes",
                  "contradicted": "Contradicción con los pasajes",
                  "insufficient_evidence": "Información insuficiente en los pasajes"}
TEXTO_ESTADO = {NO_VALIDADO: "Resultado no validado", NO_DISPONIBLE: "Análisis no disponible"}

# Parámetros del juez evaluado, explícitos para no heredar los del Modelfile (Qwen3.5 trae presence_penalty 1.5,
# temperature 1, top_k 20 y top_p 0.95). num_thread solo se envía si se configura: no se imponen ocho hilos a todos.
OPCIONES_EVALUADAS = {"temperature": 0, "seed": 42, "top_k": 40, "top_p": 0.9, "min_p": 0, "presence_penalty": 0,
                      "frequency_penalty": 0, "repeat_penalty": 1.0}

CARACTERES_POR_TOKEN = 3  # estimación prudente previa; en español se han medido unos 3,3 caracteres por token


def configuracion() -> dict:
    """Valores de config.yaml y del entorno. Se leen en cada petición: nada se guarda entre peticiones."""
    llm = config.cfg_param["llm"]["text"]
    return {"modelo": llm["name"], "max_tokens": llm["max_tokens"], "timeout_s": llm["timeout_s"], "num_ctx": llm["num_ctx"],
            "keep_alive": llm["keep_alive"], "num_thread": llm.get("num_thread"),
            "host": config.cfg_param["ollama"]["host"], "port": config.cfg_param["ollama"]["port"]}


def prompt_de_plantilla() -> str:
    """La plantilla que se envía, para registrar su huella al evaluar."""
    return PLANTILLA


def limite_tokens_prompt(num_ctx: int) -> int:
    """Cuando un prompt llega a num_ctx, Ollama lo recorta en silencio a cerca de la mitad (2050 de 4096, medido en 0.40.14)
    y la respuesta no lo indica. Un recuento de ese tamaño no distingue un prompt completo de uno recortado, así que desde
    la mitad del contexto no se da por completo. Con los tamaños de fragmento configurados no se llega a este límite."""
    return num_ctx // 2


def construir_peticion(modelo: str, prompt: str, max_tokens: int, num_ctx: int, keep_alive: str, num_thread=None) -> dict:
    opciones = {**OPCIONES_EVALUADAS, "num_ctx": num_ctx, "num_predict": max_tokens}
    if num_thread:
        opciones["num_thread"] = num_thread
    return {"model": modelo, "prompt": prompt, "stream": True, "think": False, "format": ESQUEMA_RESPUESTA,
            "keep_alive": keep_alive, "options": opciones}


def generar(peticion: dict, timeout_s: float, host: str, port: int) -> dict:
    """Una generación con plazo de pared y sin reintentos. Cerrar la conexión es lo que cancela el trabajo en Ollama."""
    inicio = time.perf_counter()
    limite = inicio + timeout_s
    salida = {"status": "error", "response": "", "final": {}, "error": None, "http_status": None, "wall_s": None, "ttft_s": None}
    conexion = http.client.HTTPConnection(host, port, timeout=timeout_s)
    try:
        conexion.request("POST", "/api/generate", body=json.dumps(peticion), headers={"Content-Type": "application/json"})
        respuesta = conexion.getresponse()
        salida["http_status"] = respuesta.status
        if respuesta.status != 200:
            salida["error"] = respuesta.read()[:300].decode("utf-8", "replace")
            return salida
        sock = conexion.sock or getattr(getattr(respuesta.fp, "raw", None), "_sock", None)
        partes = []
        while True:
            restante = limite - time.perf_counter()
            if restante <= 0:
                raise TimeoutError
            if sock is not None:
                sock.settimeout(restante)
            linea = respuesta.readline()
            if not linea:
                salida["error"] = "la conexión terminó sin un mensaje final"
                return salida
            dato = json.loads(linea)
            if dato.get("error"):
                salida["error"] = str(dato["error"])[:300]
                return salida
            if dato.get("response"):
                if salida["ttft_s"] is None:
                    salida["ttft_s"] = time.perf_counter() - inicio
                partes.append(dato["response"])
            salida["response"] = "".join(partes)
            if dato.get("done"):
                salida["final"] = {k: v for k, v in dato.items() if k not in ("response", "thinking", "context")}
                salida["status"] = "completed"
                return salida
    except TimeoutError:
        salida["status"] = "timeout"
        salida["error"] = f"sin respuesta completa en {timeout_s:g} s"
        return salida
    except (OSError, http.client.HTTPException, json.JSONDecodeError) as e:
        salida["error"] = f"{type(e).__name__}: {e}"[:300]
        return salida
    finally:
        salida["wall_s"] = time.perf_counter() - inicio
        conexion.close()


# * Validación
# * ===========================================================================
def normalizar_cita(texto: str) -> str:
    """Solo unifica representación: Unicode NFC, comillas y guiones tipográficos y espacios. No hace coincidencia semántica."""
    texto = unicodedata.normalize("NFC", texto)
    for origen, destino in {"“": '"', "”": '"', "‘": "'", "’": "'", "—": "-", "–": "-", "…": "..."}.items():
        texto = texto.replace(origen, destino)
    return " ".join(texto.split())


def partes_de_cita(cita: str) -> list:
    """Unos puntos suspensivos separan trozos de una cita, y cada trozo debe aparecer tal cual en un pasaje citado: un párrafo
    con varias afirmaciones puede necesitar más de una cita. Las comillas que la envuelven y su punto final no forman parte."""
    trozos = re.split(r"\[?\.{3,}\]?", normalizar_cita(cita))
    limpios = (t.strip(" \"'«»").rstrip(".").strip() for t in trozos)
    return [t for t in limpios if t]


def validar_respuesta(texto: str, done_reason, pasajes: list) -> tuple:
    """Devuelve (datos, None) si la respuesta es utilizable, o (None, motivo). Nada se repara."""
    if done_reason == "length":
        return None, "La respuesta se cortó al llegar al límite de tokens."
    try:
        obj = json.loads(texto)
    except (json.JSONDecodeError, TypeError):
        return None, "La respuesta no es JSON válido."
    if not isinstance(obj, dict) or obj.get("label") not in ETIQUETAS:
        return None, "La respuesta no trae una etiqueta admitida."
    ids, cita, explicacion = obj.get("passage_ids"), obj.get("quote"), obj.get("explanation")
    if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids) or not isinstance(cita, str) or not isinstance(explicacion, str):
        return None, "La respuesta no cumple el esquema."
    if not explicacion.strip():
        return None, "La respuesta no trae explicación."

    textos = {p["id"]: normalizar_cita(p["text"]) for p in pasajes}
    citados = list(dict.fromkeys(i.strip().strip("[]").strip() for i in ids))
    desconocidos = [i for i in citados if i not in textos]
    if desconocidos:
        return None, f"Cita pasajes que no existen: {', '.join(desconocidos)}."
    partes = partes_de_cita(cita)
    if obj["label"] != "insufficient_evidence" and not (citados and partes):
        return None, "Falta la evidencia: esta etiqueta exige un pasaje citado y una cita literal."
    citas = {i: [] for i in citados}
    for parte in partes:
        destino = next((i for i in citados if parte in textos[i]), None)
        if destino is None:
            return None, "La cita no aparece literalmente en los pasajes citados."
        if len(parte.split()) < 2:
            return None, "La cita es demasiado corta para servir de evidencia."
        citas[destino].append(parte)

    numero = {p["id"]: n for n, p in enumerate(pasajes, start=1)}
    evidencia = [{"id": i, "pasaje": numero[i], "cita": c} for i in citados for c in (citas[i] or [None])]
    return {"etiqueta": obj["label"], "explicacion": explicacion.strip(), "evidencia": evidencia}, None


# * Análisis de un fragmento
# * ===========================================================================
def _metricas(salida: dict) -> dict:
    final = salida.get("final") or {}
    segundos = lambda clave: final[clave] / 1e9 if isinstance(final.get(clave), (int, float)) else None
    return {"tiempo_s": salida.get("wall_s"), "primer_token_s": salida.get("ttft_s"), "carga_s": segundos("load_duration"),
            "evaluacion_prompt_s": segundos("prompt_eval_duration"), "generacion_s": segundos("eval_duration"),
            "tokens_prompt": final.get("prompt_eval_count"), "tokens_prompt_en_cache": final.get("prompt_eval_cached_count"),
            "tokens_salida": final.get("eval_count"), "fin": final.get("done_reason")}


def _sin_resultado(estado: str, motivo: str, modelo: str, salida=None) -> dict:
    return {"estado": estado, "etiqueta": None, "etiqueta_texto": TEXTO_ESTADO[estado], "explicacion": None, "evidencia": [],
            "motivo": motivo, "modelo": modelo, "metricas": _metricas(salida or {})}


def _motivo_sin_respuesta(salida: dict, cfg: dict, modelo: str) -> str:
    if salida["status"] == "timeout":
        return f"El modelo no ha respondido en {cfg['timeout_s']:g} s y se ha cancelado la petición."
    if salida["http_status"] == 404:
        return f"El modelo {modelo} no está instalado en Ollama. Instálalo con: ollama pull {modelo}"
    if salida["http_status"] is None:
        return f"No se ha podido conectar con Ollama en {cfg['host']}:{cfg['port']} ({salida['error']})."
    return f"Ollama ha devuelto un error: {salida['error']}"


def ejecutar(modalidad: str, fragmento: str, pasajes: list, **ajustes) -> dict:
    """Devuelve {"analisis", "bruto"}. El análisis es lo que se muestra; lo bruto es para diagnosticar y evaluar, y no sale de la API."""
    cfg = {**configuracion(), **ajustes}
    modelo = cfg["modelo"]
    prompt = construir_prompt(modalidad, fragmento, pasajes)
    limite = limite_tokens_prompt(cfg["num_ctx"])
    bruto = {"prompt": prompt, "respuesta": None, "fin": None}

    # No se recorta evidencia a escondidas: si no cabe, no se envía al modelo y se dice
    if len(prompt) / CARACTERES_POR_TOKEN >= limite:
        motivo = (f"El fragmento y sus pasajes no caben en el contexto del modelo ({len(prompt)} caracteres, unos "
                  f"{len(prompt) // CARACTERES_POR_TOKEN} tokens; límite {limite}). No se ha enviado ni se ha recortado nada.")
        return {"analisis": _sin_resultado(NO_DISPONIBLE, motivo, modelo), "bruto": bruto}

    peticion = construir_peticion(modelo, prompt, cfg["max_tokens"], cfg["num_ctx"], cfg["keep_alive"], cfg["num_thread"])
    salida = generar(peticion, cfg["timeout_s"], cfg["host"], cfg["port"])
    bruto.update(respuesta=salida["response"], fin=(salida["final"] or {}).get("done_reason"), estado_peticion=salida["status"])
    if salida["status"] != "completed":
        return {"analisis": _sin_resultado(NO_DISPONIBLE, _motivo_sin_respuesta(salida, cfg, modelo), modelo, salida), "bruto": bruto}

    tokens = (salida["final"] or {}).get("prompt_eval_count")
    if isinstance(tokens, int) and tokens >= limite:
        motivo = f"Ollama ha podido recortar el contexto ({tokens} tokens evaluados; límite {limite}): el resultado no es fiable."
        return {"analisis": _sin_resultado(NO_VALIDADO, motivo, modelo, salida), "bruto": bruto}

    datos, motivo = validar_respuesta(salida["response"], bruto["fin"], pasajes)
    if datos is None:
        return {"analisis": _sin_resultado(NO_VALIDADO, motivo, modelo, salida), "bruto": bruto}
    return {"analisis": {"estado": VALIDADO, "etiqueta": datos["etiqueta"], "etiqueta_texto": TEXTO_ETIQUETA[datos["etiqueta"]],
                         "explicacion": datos["explicacion"], "evidencia": datos["evidencia"], "motivo": None,
                         "modelo": modelo, "metricas": _metricas(salida)},
            "bruto": bruto}


# * Pasajes de la búsqueda real
# * ===========================================================================
def _procedencia(fuente: dict) -> str:
    return " · ".join(str(fuente[c]) for c in ("filename", "title", "outlet", "date", "location") if fuente.get(c))


def pasajes_de_resultado(resultado: dict) -> list:
    """Los pasajes recuperados, con los IDs P1, P2... que el modelo cita, en el orden de la búsqueda."""
    return [{"id": f"P{n}", "source": _procedencia(fuente), "text": texto}
            for n, (texto, fuente) in enumerate(zip(resultado["relevant_chunks"], resultado["sources"]), start=1)]


def resumen_textual(analisis: dict) -> str:
    """Resumen del resultado construido por código, sin otra llamada al modelo. Mantiene el campo comentario de la API."""
    citas = [f"Pasaje {e['pasaje']}: «{e['cita']}»" for e in analisis["evidencia"] if e["cita"]]
    return " ".join([f"{analisis['etiqueta_texto']}.", analisis["explicacion"], *citas])


def analizar_resultados(resultados: list, modalidad: str) -> None:
    """Añade a cada fragmento su análisis, una petición por fragmento. Un fallo no pierde los pasajes."""
    for resultado in resultados:
        analisis = ejecutar(modalidad, resultado["chunk"], pasajes_de_resultado(resultado))["analisis"]
        for evidencia in analisis["evidencia"]:
            evidencia["fuente"] = resultado["sources"][evidencia["pasaje"] - 1]
        resultado["analisis"] = analisis
        resultado["comentario"] = resumen_textual(analisis) if analisis["estado"] == VALIDADO else None
        if analisis["estado"] != VALIDADO:
            resultado["comentario_error"] = f"{analisis['etiqueta_texto']}: {analisis['motivo']}"
