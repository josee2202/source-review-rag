"""Lógica de la interfaz sin dependencia de Streamlit: petición, tablas y exportación a Excel."""
import io
import json
import re

import pandas as pd
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE

ORIGEN_TEXTO = "Pegar texto"
ORIGEN_ARCHIVO = "Subir archivo"
FORMATOS = ["pdf", "docx", "txt", "md", "markdown"]
NOTA_SIMILITUD = ("La similitud coseno indica cercanía entre textos: no es una probabilidad de que algo sea cierto "
                  "ni demuestra que la referencia respalde el fragmento. Se muestran siempre hasta tres pasajes, "
                  "aunque ninguno sea realmente pertinente.")
NOTA_ANALISIS = ("Análisis experimental de un modelo local a partir solo de los pasajes recuperados. Compara el fragmento "
                 "con esos pasajes: no comprueba si algo es cierto en general y puede equivocarse. Lee siempre los pasajes.")


def construir_peticion(modalidad, ref_origen, ref_archivos, ref_texto, ref_metadatos,
                       rev_origen, rev_archivo, rev_texto, usar_comentario, fraccion_inicial=1.0):
    """Devuelve (files, data) para /validar. Solo se envía el origen seleccionado de cada entrada."""
    files, data = [], {
        "modalidad": modalidad,
        "usar_agente": str(bool(usar_comentario)).lower(),
        "fraccion_inicial": str(fraccion_inicial),
    }
    if ref_origen == ORIGEN_TEXTO:
        data["texto_ref"] = ref_texto or ""
    else:
        files += [("file_ref", (a.name, a.getvalue(), a.type)) for a in ref_archivos or []]
    if rev_origen == ORIGEN_TEXTO:
        data["texto_query"] = rev_texto or ""
    elif rev_archivo is not None:
        files.append(("file_query", (rev_archivo.name, rev_archivo.getvalue(), rev_archivo.type)))

    # Los metadatos solo se envían si el usuario ha rellenado alguno; nunca se inventan
    if any(any((v or "").strip() for v in m.values()) for m in ref_metadatos or []):
        data["ref_metadata"] = json.dumps(ref_metadatos, ensure_ascii=False)
    return files, data


def entradas_completas(ref_origen, ref_archivos, ref_texto, rev_origen, rev_archivo, rev_texto) -> bool:
    ref_ok = bool((ref_texto or "").strip()) if ref_origen == ORIGEN_TEXTO else bool(ref_archivos)
    rev_ok = bool((rev_texto or "").strip()) if rev_origen == ORIGEN_TEXTO else rev_archivo is not None
    return ref_ok and rev_ok


def escapar_markdown(texto) -> str:
    """Los textos aportados se muestran literalmente, sin interpretarlos como Markdown."""
    return re.sub(r"([!-/:-@\[-`{-~])", r"\\\1", str(texto))


def etiqueta_fuente(fuente: dict) -> str:
    """'R1 · informe.pdf · p. 3 · Título · Medio · Fecha · URL'. Solo incluye lo que existe."""
    campos = ("ref_id", "filename", "location", "title", "outlet", "date", "url")
    return " · ".join(str(fuente[c]) for c in campos if fuente.get(c))


def _pasajes(resultado: dict):
    textos = resultado.get("relevant_chunks") or []
    fuentes = resultado.get("sources") or []
    similitudes = resultado.get("scores") or []
    return list(zip(textos, fuentes, similitudes))


def citas_de(analisis: dict) -> list:
    """Cada cita validada con el pasaje del que procede: [(cita, 'Pasaje 1 · R1 · informe.pdf · p. 3')]."""
    return [(e["cita"], f"Pasaje {e['pasaje']} · {etiqueta_fuente(e.get('fuente') or {})}".rstrip(" ·"))
            for e in analisis.get("evidencia") or [] if e.get("cita")]


def resumen_analisis(respuesta: dict) -> str:
    """Recuento por estado, con su denominador: un fragmento sin resultado no cuenta como ningún estado."""
    analisis = [r["analisis"] for r in respuesta.get("resultados", []) if r.get("analisis")]
    partes = [(texto, sum(1 for a in analisis if a["etiqueta_texto"] == texto)) for texto in dict.fromkeys(a["etiqueta_texto"] for a in analisis)]
    return f"Análisis de {len(analisis)} fragmentos: " + " · ".join(f"{texto}: {n}" for texto, n in partes) + "."


def resultados_a_dataframe(respuesta: dict) -> pd.DataFrame:
    """Una fila por fragmento revisado. Puede haber menos de tres pasajes: las columnas quedan vacías."""
    filas = []
    for res in respuesta.get("resultados", []):
        pasajes = _pasajes(res)
        fila = {
            "Nº": res.get("fragmento"),
            "Fragmento revisado": res.get("chunk", ""),
            "Ubicación": (res.get("chunk_location") or {}).get("location"),
            "Fuentes (similitud)": " | ".join(f"{etiqueta_fuente(f)} · {s:.3f}" for _, f, s in pasajes),
        }
        if respuesta.get("comentario_solicitado"):
            analisis = res.get("analisis") or {}
            fila["Análisis del modelo"] = analisis.get("etiqueta_texto")
            fila["Explicación del modelo"] = analisis.get("explicacion")
            fila["Evidencia citada"] = " | ".join(f"{origen}: «{cita}»" for cita, origen in citas_de(analisis)) or None
            fila["Motivo sin resultado"] = analisis.get("motivo")
        for i in range(3):
            texto, fuente, similitud = pasajes[i] if i < len(pasajes) else (None, None, None)
            fila[f"Pasaje {i + 1}"] = texto
            fila[f"Fuente {i + 1}"] = etiqueta_fuente(fuente) if fuente else None
            fila[f"Similitud {i + 1}"] = similitud
        filas.append(fila)
    return pd.DataFrame(filas)


def referencias_a_dataframe(respuesta: dict) -> pd.DataFrame:
    columnas = {"ref_id": "ID", "filename": "Archivo", "title": "Título", "outlet": "Medio", "date": "Fecha", "url": "URL"}
    filas = [{nombre: ref.get(campo) for campo, nombre in columnas.items()} for ref in respuesta.get("referencias", [])]
    return pd.DataFrame(filas, columns=list(columnas.values()))


def aviso_sin_revisar(respuesta: dict):
    """Mensaje cuando se ha analizado solo el inicio del texto; None si se ha revisado entero."""
    rev = respuesta.get("texto_revisado") or {}
    if not rev.get("fragmentos_sin_revisar"):
        return None
    return (f"Solo se ha analizado el inicio del texto: quedan sin revisar {rev['fragmentos_sin_revisar']} de "
            f"{rev['fragmentos_totales']} fragmentos, a partir de {rev.get('sin_revisar_desde') or 'el último revisado'}.")


def _resumen_a_dataframe(respuesta: dict) -> pd.DataFrame:
    rev = respuesta.get("texto_revisado") or {}
    filas = [
        ("Modalidad", respuesta.get("modalidad_nombre")),
        ("Texto revisado", rev.get("filename")),
        ("Fragmentos revisados", rev.get("fragmentos_revisados")),
        ("Fragmentos sin revisar", rev.get("fragmentos_sin_revisar")),
        ("Sin revisar a partir de", rev.get("sin_revisar_desde")),
        ("Análisis del modelo solicitado", "Sí" if respuesta.get("comentario_solicitado") else "No"),
        ("Nota sobre la similitud", NOTA_SIMILITUD),
    ]
    if respuesta.get("comentario_solicitado"):
        filas.append(("Modelo del análisis", respuesta.get("modelo_analisis")))
        filas.append(("Resumen del análisis", resumen_analisis(respuesta)))
        filas.append(("Nota sobre el análisis", NOTA_ANALISIS))
    return pd.DataFrame(filas, columns=["Campo", "Valor"])


def _celda_segura(valor):
    return ILLEGAL_CHARACTERS_RE.sub("", valor) if isinstance(valor, str) else valor


def exportar_excel(respuesta: dict) -> bytes:
    """Excel con los resultados, las referencias y un resumen de lo revisado."""
    hojas = {"Resultados": resultados_a_dataframe(respuesta),
             "Referencias": referencias_a_dataframe(respuesta),
             "Resumen": _resumen_a_dataframe(respuesta)}
    salida = io.BytesIO()
    with pd.ExcelWriter(salida, engine="openpyxl") as writer:
        for nombre, df in hojas.items():
            df.map(_celda_segura).to_excel(writer, sheet_name=nombre, index=False)
            # Un texto aportado que empiece por "=" es dato, no una fórmula que Excel deba ejecutar
            for fila in writer.sheets[nombre].iter_rows():
                for celda in fila:
                    if celda.data_type == "f":
                        celda.data_type = "s"
    return salida.getvalue()
