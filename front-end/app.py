import streamlit as st
from pathlib import Path
import sys
import pandas as pd
import requests
import time
import os

# URL del backend FastAPI (configurable para despliegues distintos de localhost)
API_URL = os.getenv("API_URL", "http://localhost:8000")

# * === CONFIGURACIÓN DEL ENTORNO Y CARGA DEL MÓDULO EXTERNO ===================================

base_dir = Path(__file__).resolve().parent
ruta_modulo = (base_dir.parent / "IA-MODULABLE").resolve()

if ruta_modulo.is_dir() and str(ruta_modulo) not in sys.path:
    sys.path.insert(0, str(ruta_modulo))

# * === CONFIGURACIÓN DE PÁGINA ===============================================================

st.set_page_config(
    page_title="Validación de Documentos",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado para mejorar la apariencia
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 0.5rem 0;
        margin-bottom: 1rem;
    }
    .step-container {
        border-left: 4px solid #007bff;
        padding-left: 1rem;
        margin: 1rem 0;
    }
    .spinner-container {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100px;
    }
    .spinner {
        border: 4px solid #f3f3f3;
        border-top: 4px solid #007bff;
        border-radius: 50%;
        width: 50px;
        height: 50px;
        animation: spin 1s linear infinite;
    }
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        color: #155724;
    }
    .info-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin-top: -6rem;
    }
    .info-card h3 {
        color: white !important;
        margin-top: 0 !important;
        margin-bottom: 1rem !important;
    }
    .info-card h4 {
        color: white !important;
        margin-bottom: 0.5rem !important;
        margin-top: 1rem !important;
    }
    .info-card p {
        margin-bottom: 1rem !important;
        opacity: 0.9;
        line-height: 1.4;
    }
    .info-card p:last-child {
        margin-bottom: 0 !important;
    }
</style>
""", unsafe_allow_html=True)

# * === SESIÓN Y ESTRUCTURA DE DATOS ==========================================================

# Inicializar variables de sesión
st.session_state.setdefault("documento_referencia", None)
st.session_state.setdefault("documentos_consulta", [])
st.session_state.setdefault("resultado_validacion", None)
st.session_state.setdefault("colecciones", [])
st.session_state.setdefault("coleccion_select", None)
st.session_state.setdefault("usar_agente", False)
st.session_state.setdefault("porcentaje_referencia", 100)
st.session_state.setdefault("procesando", False)

# * === TÍTULO Y DESCRIPCIÓN PRINCIPAL ========================================================

st.title("🔍 Validación de Documentos")
st.markdown("*Sistema inteligente de comparación y validación documental*")

# * === LAYOUT PRINCIPAL EN COLUMNAS ==========================================================

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    ## 📋 Cómo usar el sistema  
    ### 🛠️ Siempre necesitas:
    - **📁 Documento de referencia**  
    - **🗂️ Colección** es el nombre con el que guardas el documento   
    - **📄 Documento de consulta** a validar

    ### 🎯 Pasos a seguir   

     1️⃣ - **📁 Sube documentos de referencia**  → Crea o elige tu **colección** (más rápido si ya existe)

     2️⃣ - **📄 Añade documento de consulta**  → Ajusta el **% a validar** si solo quieres una prueba
    
     3️⃣ - **🔍 Elige tipo de validación**  → Pulsa **🚀 Ejecutar** y revisa resultados
    """)




with col2:
    st.markdown(
        """
<div class="info-card">
  <h3>✨ Información clave</h3>

  <h4>📁 Formatos compatibles</h4>
  <ul>
    <li>📄 PDF</li>
    <li>📝 Word (.docx)</li>
    <li>💻 SAS (solo como consulta)</li>
  </ul>

  <h4>📊 Obtendrás</h4>
  <ul>
    <li>🧩 Fragmentos clave analizados</li>
    <li>🧠 Contexto de referencia</li>
    <li>🤖 Validación automatizada</li>
    <li>📥 Descarga en Excel</li>
  </ul>

</div>
        """,
        unsafe_allow_html=True   #  <-- imprescindible
    )



# * === BARRA LATERAL DE CONFIGURACIÓN ========================================================

st.sidebar.markdown("# ⚙️ Configuración")

# * Paso 1: Documento de referencia
with st.sidebar:
    st.markdown("---")
    st.markdown("## 1️⃣ Documentos de referencia")
    
    doc_ref = st.file_uploader(
        "📘 Sube uno o más documentos",
        type=['pdf', 'docx'],
        accept_multiple_files=True,
        key="doc_referencia",
        help="Documentos que servirán como base de comparación"
    )
    if doc_ref:
        st.session_state["documento_referencia"] = doc_ref
        st.success(f"✅ {len(doc_ref)} documento(s) cargado(s)")

# * Paso 2: Gestión de colecciones
with st.sidebar:
    st.markdown("### 📚 Gestión de colecciones")
    
    # Obtener colecciones existentes
    try:
        response = requests.get(f"{API_URL}/colecciones")
        colecciones_existentes = response.json().get("colecciones", [])
    except:
        colecciones_existentes = []

    if "colecciones" not in st.session_state or not st.session_state["colecciones"]:
        st.session_state["colecciones"] = colecciones_existentes

    # Nueva colección
    col_a, col_b = st.columns([3, 1])
    with col_a:
        nueva_col = st.text_input("🔖 Nueva colección", placeholder="Nombre de la colección")
    with col_b:
        crear = st.button("➕", help="Crear colección")

    if crear and nueva_col.strip():
        nueva_col = nueva_col.strip()
        if nueva_col not in st.session_state["colecciones"]:
            st.session_state["colecciones"].append(nueva_col)
        st.session_state["coleccion_select"] = nueva_col

    # Selector de colección
    if st.session_state["colecciones"]:
        coleccion_default = st.session_state.get("coleccion_select", st.session_state["colecciones"][0])
        nombre_final = st.selectbox(
            "Selecciona colección:",
            options=st.session_state["colecciones"],
            index=st.session_state["colecciones"].index(coleccion_default) if coleccion_default in st.session_state["colecciones"] else 0,
            key="coleccion_select"
        )
    else:
        st.warning("⚠️ Crea una colección para continuar")
        nombre_final = None

# * Paso 3: Documento de consulta
with st.sidebar:
    st.markdown("---")
    st.markdown("## 2️⃣ Documento de consulta")
    
    doc_consulta = st.file_uploader(
        "📄 Documento a validar",
        type=['pdf', 'docx', 'sas'],
        key="doc_consulta",
        help="Documento que será comparado con las referencias"
    )
    if doc_consulta:
        st.session_state["documentos_consulta"] = [doc_consulta]
        st.success(f"✅ {doc_consulta.name}")

    # Configuración de porcentaje
    porcentaje = st.slider(
        "📊 Porcentaje del documento a analizar",
        min_value=2,
        max_value=100,
        value=100,
        step=2,
        help="Reduce el porcentaje para análisis más rápidos"
    )

# * Paso 4: Tipo de validación
with st.sidebar:
    st.markdown("---")
    st.markdown("## 3️⃣ Tipo de validación")
    
    tipos_validacion = [
        "Normativo vs Funcional",
        "Funcional vs Técnico", 
        "Código vs Técnico",
        "Código vs Estadístico"
    ]

    tipo_seleccionado = st.selectbox(
        "🔍 Método de análisis:",
        options=tipos_validacion,
        index=tipos_validacion.index(st.session_state.get("tipo_validacion", "Normativo vs Funcional")),
    )
    st.session_state["tipo_validacion"] = tipo_seleccionado

    # Descripciones en formato más compacto
    descripciones = {
        "Normativo vs Funcional": "📋 Realiza una comparación entre legislación y el documento metodológico del banco",
        "Funcional vs Técnico": "⚙️ Compara el documento técnico (que explica el desarrollo) con el metodológico",
        "Código vs Técnico": "💻 Compara lo que se ha implementado con lo que explica el banco que ha hecho",
        "Código vs Estadístico": "📊 Verificar si cumple los test estadísticos relevantes"
    }
    st.info(descripciones[tipo_seleccionado])

    # Opción avanzada
    usar_agente = st.checkbox(
        "🤖 Validación avanzada con IA",
        value=False,
        help="Análisis más profundo con agente inteligente"
    )
    st.session_state["usar_agente"] = usar_agente

# * Botón de procesamiento
with st.sidebar:
    st.markdown("---")
    
    # Verificar requisitos
    puede_procesar = (
        st.session_state["documentos_consulta"] and 
        st.session_state.get("coleccion_select") and
        st.session_state["documento_referencia"]
    )
    
    if st.button("🚀 Validar documentos", disabled=not puede_procesar or st.session_state.get("procesando", False)):
        if not puede_procesar:
            if not st.session_state["documento_referencia"]:
                st.error("📘 Falta documento de referencia")
            if not st.session_state["documentos_consulta"]:
                st.error("📄 Falta documento de consulta")
            if not st.session_state.get("coleccion_select"):
                st.error("📚 Selecciona una colección")
        else:
            st.session_state["procesando"] = True
            st.rerun()

# * === PROCESAMIENTO Y VISUALIZACIÓN ========================================================

# Mostrar spinner durante procesamiento
if st.session_state.get("procesando", False):
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div class="spinner-container">
            <div class="spinner"></div>
        </div>
        <p style="text-align: center; margin-top: 1rem;">
            <strong>🔄 Procesando documentos...</strong><br>
            <small>Esto puede tardar unos minutos</small>
        </p>
        """, unsafe_allow_html=True)
    
    # Procesamiento real
    try:
        # Se envían todos los documentos de referencia (antes solo el primero)
        files = [
            ("file_ref", (doc.name, doc.getvalue(), doc.type))
            for doc in st.session_state["documento_referencia"]
        ]
        files.append((
            "file_query",
            (
                st.session_state["documentos_consulta"][0].name,
                st.session_state["documentos_consulta"][0].getvalue(),
                st.session_state["documentos_consulta"][0].type,
            ),
        ))

        form_data = {
            "collection_name": nombre_final,
            "doc_sample_ratio": porcentaje / 100,
            "usar_agente": str(st.session_state.get("usar_agente", False)).lower(),
            "tipo_validacion": st.session_state.get("tipo_validacion", "Normativo vs Funcional")
        }

        response = requests.post(f"{API_URL}/validar", files=files, data=form_data)

        if response.status_code == 200:
            resultados = response.json()["resultados"]
            
            # Reformatear resultados
            formateados = []
            for res in resultados:
                # Puede haber menos de 3 fragmentos recuperados (colección pequeña): se rellena con None
                relevantes = (res.get("relevant_chunks") or []) + [None, None, None]
                fuentes = [
                    f"{s.get('filename')}" + (f" p.{s['page']}" if s.get("page") else "") + f" · {score:.3f}"
                    for s, score in zip(res.get("sources", []), res.get("scores", []))
                ]
                # Fuentes y veredicto antes que los textos completos, para verlos sin desplazamiento horizontal
                fila = {
                    "Consulta": res.get("chunk", ""),
                    "Fuentes (similitud)": " | ".join(fuentes),
                }
                if "respuesta" in res:
                    fila["Respuesta"] = res["respuesta"]
                fila.update({"Texto 1": relevantes[0], "Texto 2": relevantes[1], "Texto 3": relevantes[2]})
                formateados.append(fila)

            df_result = pd.DataFrame(formateados)
            st.session_state["resultado_validacion"] = df_result
            st.session_state["procesando"] = False
            st.rerun()
        else:
            st.session_state["procesando"] = False
            try:
                detalle = response.json().get("detail", response.text)
            except ValueError:
                detalle = response.text
            st.error(f"❌ Error del servidor ({response.status_code}): {detalle}")

    except Exception as e:
        st.session_state["procesando"] = False
        st.error(f"❌ Error durante el procesamiento: {e}")

# * === MOSTRAR RESULTADOS ====================================================================

if st.session_state["resultado_validacion"] is not None and not st.session_state.get("procesando", False):
    st.markdown("---")
    # Un único bloque HTML: Streamlit renderiza cada st.markdown por separado y el <div> quedaba vacío
    st.markdown('<div class="success-box"><h3>✅ Validación completada exitosamente</h3></div>', unsafe_allow_html=True)
    
    # Métricas rápidas
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📄 Fragmentos analizados", len(st.session_state["resultado_validacion"]))
    with col2:
        st.metric("🔍 Tipo de validación", st.session_state.get("tipo_validacion", "N/A"))
    with col3:
        st.metric("🤖 IA Avanzada", "Activada" if st.session_state.get("usar_agente") else "Desactivada")
    with col4:
        st.metric("📊 Porcentaje analizado", f"{porcentaje}%")
    
    st.markdown("### 📊 Resultados detallados")
    st.dataframe(
        st.session_state["resultado_validacion"], 
        use_container_width=True,
        height=400
    )

    # Botón de descarga mejorado
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        import io
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            st.session_state["resultado_validacion"].to_excel(writer, index=False)
        output.seek(0)

        st.download_button(
            label="📥 Descargar resultados en Excel",
            data=output,
            file_name=f"validacion_{tipo_seleccionado.lower().replace(' ', '_')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

elif not st.session_state.get("procesando", False):
    # Mensaje inicial más atractivo
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 2rem; border-radius: 10px;">
            <h3> ¡Listo para comenzar!</h3>
            <p>Configura tus documentos en la barra lateral y ejecuta la validación</p>
        </div>
        """, unsafe_allow_html=True)