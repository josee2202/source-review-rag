import os
import sys
from pathlib import Path

import requests
import streamlit as st

# Módulos del proyecto: las modalidades se definen una sola vez, en el backend
base_dir = Path(__file__).resolve().parent
for ruta in (base_dir, base_dir.parent / "IA-modulable"):
    if str(ruta) not in sys.path:
        sys.path.insert(0, str(ruta))

from modalidades import MODALIDADES
from presentacion import (FORMATOS, NOTA_COMENTARIO, NOTA_SIMILITUD, ORIGEN_ARCHIVO, ORIGEN_TEXTO, aviso_sin_revisar,
                          construir_peticion, entradas_completas, escapar_markdown, etiqueta_fuente, exportar_excel,
                          referencias_a_dataframe, resultados_a_dataframe)

# URL del backend FastAPI
API_URL = os.getenv("API_URL", "http://localhost:8010")

CAMPOS_METADATOS = {"title": "Título", "outlet": "Medio", "date": "Fecha", "url": "URL (solo como enlace)"}

st.set_page_config(page_title="Source Review RAG", page_icon="🔍", layout="wide")
st.session_state.setdefault("resultado_validacion", None)

st.title("🔍 Source Review RAG")
st.markdown("*Revisa un texto, fragmento a fragmento, frente a las referencias que tú eliges.*")


def metadatos_opcionales(clave: str, titulo: str) -> dict:
    """Campos opcionales de una referencia: lo que se deje vacío no se inventa."""
    with st.expander(f"Datos opcionales de la fuente · {titulo}"):
        return {campo: st.text_input(etiqueta, key=f"meta_{clave}_{campo}")
                for campo, etiqueta in CAMPOS_METADATOS.items()}


# * === MODALIDAD ================================================================================

def limpiar_resultados():
    """Los resultados pertenecen a la modalidad con la que se generaron."""
    st.session_state["resultado_validacion"] = None


modalidad_id = st.radio("Modalidad", options=list(MODALIDADES), key="modalidad", horizontal=True,
                        format_func=lambda m: MODALIDADES[m].nombre, on_change=limpiar_resultados)
modalidad = MODALIDADES[modalidad_id]
st.info(modalidad.descripcion)

# * === ENTRADAS =================================================================================

col_ref, col_rev = st.columns(2)

with col_ref:
    st.subheader(f"1️⃣ {modalidad.etiqueta_referencia}", help=modalidad.ayuda_referencia)
    ref_origen = st.radio("Origen de la referencia", [ORIGEN_TEXTO, ORIGEN_ARCHIVO], key="ref_origen", horizontal=True)
    ref_texto, ref_archivos, ref_metadatos = "", [], []
    if ref_origen == ORIGEN_TEXTO:
        ref_texto = st.text_area(modalidad.etiqueta_referencia, key="ref_texto", height=220, label_visibility="collapsed")
        if modalidad.multiples_referencias:
            st.caption("Un texto pegado cuenta como una sola referencia: no se separa en artículos. "
                       "Para saber de qué noticia procede cada pasaje, sube un archivo por noticia.")
        ref_metadatos = [metadatos_opcionales("pegado", "texto pegado")]
    elif modalidad.multiples_referencias:
        ref_archivos = st.file_uploader(modalidad.etiqueta_referencia, type=FORMATOS, accept_multiple_files=True,
                                        key="ref_archivos_varios", label_visibility="collapsed") or []
        ref_metadatos = [metadatos_opcionales(f"{i}_{a.name}", a.name) for i, a in enumerate(ref_archivos)]
    else:
        archivo = st.file_uploader(modalidad.etiqueta_referencia, type=FORMATOS, key="ref_archivo_unico",
                                   label_visibility="collapsed")
        ref_archivos = [archivo] if archivo else []
        ref_metadatos = [metadatos_opcionales(f"0_{a.name}", a.name) for a in ref_archivos]

with col_rev:
    st.subheader(f"2️⃣ {modalidad.etiqueta_revisado}", help=modalidad.ayuda_revisado)
    rev_origen = st.radio("Origen del texto a revisar", [ORIGEN_TEXTO, ORIGEN_ARCHIVO], key="rev_origen", horizontal=True)
    rev_texto, rev_archivo = "", None
    if rev_origen == ORIGEN_TEXTO:
        rev_texto = st.text_area(modalidad.etiqueta_revisado, key="rev_texto", height=220, label_visibility="collapsed")
    else:
        rev_archivo = st.file_uploader(modalidad.etiqueta_revisado, type=FORMATOS, key="rev_archivo",
                                       label_visibility="collapsed")

    usar_comentario = st.checkbox("Añadir comentario del modelo (opcional y lento)", value=False, key="usar_comentario",
                                  help=NOTA_COMENTARIO)
    solo_inicio = st.checkbox("Analizar solo el inicio del texto", value=False, key="solo_inicio",
                              help="Por defecto se revisa todo el texto. Esta opción conserva los primeros fragmentos, "
                                   "no una muestra aleatoria, e indica lo que queda sin revisar.")
    porcentaje_inicio = 100
    if solo_inicio:
        porcentaje_inicio = st.slider("Porcentaje inicial de fragmentos", min_value=5, max_value=95, value=50, step=5,
                                      key="porcentaje_inicio")

completas = entradas_completas(ref_origen, ref_archivos, ref_texto, rev_origen, rev_archivo, rev_texto)

# * === PROCESAMIENTO ============================================================================

if st.button("🔎 Revisar", key="revisar", type="primary", disabled=not completas):
    files, data = construir_peticion(modalidad.id, ref_origen, ref_archivos, ref_texto, ref_metadatos,
                                     rev_origen, rev_archivo, rev_texto, usar_comentario, porcentaje_inicio / 100)
    st.session_state["resultado_validacion"] = None
    with st.spinner("Procesando. Puede tardar unos minutos, sobre todo con el comentario del modelo..."):
        try:
            response = requests.post(f"{API_URL}/validar", files=files, data=data)
        except requests.exceptions.RequestException as e:
            st.error(f"No se ha podido conectar con el backend en {API_URL}: {e}")
        else:
            if response.status_code == 200:
                st.session_state["resultado_validacion"] = response.json()
            else:
                try:
                    detalle = response.json().get("detail", response.text)
                except ValueError:
                    detalle = response.text
                st.error(f"Error del servidor ({response.status_code}): {detalle}")

# * === RESULTADOS ===============================================================================

respuesta = st.session_state["resultado_validacion"]
if respuesta is not None:
    st.divider()
    st.subheader(f"Resultados · {respuesta['modalidad_nombre']}")

    resultados = respuesta["resultados"]
    revisado = respuesta["texto_revisado"]
    con_fallo = [r for r in resultados if r.get("comentario_error")]

    col1, col2, col3 = st.columns(3)
    col1.metric("Fragmentos revisados", revisado["fragmentos_revisados"])
    col2.metric("Referencias", len(respuesta["referencias"]))
    col3.metric("Comentario del modelo", "Activado" if respuesta["comentario_solicitado"] else "Desactivado")

    aviso = aviso_sin_revisar(respuesta)
    if aviso:
        st.warning(aviso)
    if con_fallo:
        st.error(f"No se ha podido generar el comentario en {len(con_fallo)} de {len(resultados)} fragmentos. "
                 "Los pasajes recuperados se conservan.")
    st.caption(NOTA_SIMILITUD)

    tab_detalle, tab_tabla, tab_referencias = st.tabs(["Detalle por fragmento", "Tabla", "Referencias"])

    with tab_detalle:
        for res in resultados:
            ubicacion = (res.get("chunk_location") or {}).get("location") or "ubicación no disponible"
            with st.expander(f"Fragmento {res['fragmento']} · {ubicacion}", expanded=res["fragmento"] == 1):
                st.markdown("**Texto revisado**")
                st.markdown(escapar_markdown(res["chunk"]))
                for i, (texto, fuente, similitud) in enumerate(zip(res["relevant_chunks"], res["sources"], res["scores"]), 1):
                    st.markdown(f"**Pasaje {i}** · similitud {similitud:.3f}")
                    st.caption(escapar_markdown(etiqueta_fuente(fuente)))
                    if fuente.get("url"):
                        st.link_button("Abrir fuente ↗", fuente["url"])
                    st.markdown(escapar_markdown(texto))
                if res.get("comentario"):
                    st.markdown("**🤖 Comentario del modelo** (orientativo, no es un veredicto)")
                    st.markdown(escapar_markdown(res["comentario"]).replace("\n", "  \n"))
                if res.get("comentario_error"):
                    st.error(f"Fallo del comentario: {res['comentario_error']}")

    with tab_tabla:
        st.dataframe(resultados_a_dataframe(respuesta), use_container_width=True, height=400, hide_index=True)

    with tab_referencias:
        st.dataframe(referencias_a_dataframe(respuesta), use_container_width=True, hide_index=True)

    st.download_button(
        label="📥 Descargar resultados en Excel",
        data=exportar_excel(respuesta),
        file_name=f"source_review_{respuesta['modalidad']}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="descargar_excel",
    )
