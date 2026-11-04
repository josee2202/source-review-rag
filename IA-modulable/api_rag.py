import json
import os
import shutil
import sys
import tempfile
from typing import List, Optional
from urllib.parse import urlparse

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from langchain_core.documents import Document

# El backend se puede arrancar desde cualquier directorio
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import config
from modalidades import MODALIDADES
from prompt import PROMPTS_MODALIDAD
from RAG3 import RAG
from X_AGENT import agente_dinamico
from X_FUNCTIONS import EntradaInvalida
from X_qderant import PREFIJO_COLECCION

CAMPOS_METADATOS = ("title", "outlet", "date", "url")
_chunking = config.cfg_param['chunking']

app = FastAPI(title="Source Review RAG")


def obtener_modalidad(modalidad: Optional[str]):
    """Valida el ID recibido: nunca se recurre a una modalidad o prompt por defecto."""
    if modalidad not in MODALIDADES:
        raise EntradaInvalida(f"Modalidad desconocida: {modalidad!r}. Valores admitidos: {', '.join(MODALIDADES)}.")
    return MODALIDADES[modalidad]


def seleccionar_entrada(archivos: List[UploadFile], texto: Optional[str], nombre_entrada: str):
    """Cada entrada es o bien texto pegado o bien archivos: nunca se combinan."""
    archivos = [f for f in archivos if f is not None and f.filename]
    hay_texto = bool(texto and texto.strip())
    if archivos and hay_texto:
        raise EntradaInvalida(f"{nombre_entrada}: se han recibido a la vez un archivo y texto pegado. "
                              "Envía solo uno de los dos; no se combinan.")
    if not archivos and not hay_texto:
        raise EntradaInvalida(f"{nombre_entrada}: no hay contenido. Pega texto o sube un archivo.")
    return archivos, (texto if hay_texto else None)


def parsear_metadatos(ref_metadata: Optional[str], n_referencias: int) -> List[dict]:
    """Metadatos opcionales de cada referencia, en el orden de las referencias. Nunca se inventan."""
    if not ref_metadata or not ref_metadata.strip():
        return [{} for _ in range(n_referencias)]
    try:
        lista = json.loads(ref_metadata)
    except json.JSONDecodeError as e:
        raise EntradaInvalida(f"ref_metadata no es un JSON válido: {e}")
    if not isinstance(lista, list) or len(lista) != n_referencias or not all(isinstance(m, dict) for m in lista):
        raise EntradaInvalida(f"ref_metadata debe ser una lista con un objeto por referencia ({n_referencias}).")

    limpios = []
    for meta in lista:
        desconocidos = set(meta) - set(CAMPOS_METADATOS)
        if desconocidos:
            raise EntradaInvalida(f"Campos no admitidos en ref_metadata: {', '.join(sorted(desconocidos))}. "
                                  f"Admitidos: {', '.join(CAMPOS_METADATOS)}.")
        limpio = {k: str(v).strip() for k, v in meta.items() if v is not None and str(v).strip()}
        if "url" in limpio:
            # La URL solo se conserva como enlace: nunca se descarga
            url = urlparse(limpio["url"])
            if url.scheme not in ("http", "https") or not url.netloc:
                raise EntradaInvalida(f"URL no válida en ref_metadata: {limpio['url']!r}. Debe empezar por http(s)://.")
        limpios.append(limpio)
    return limpios


def guardar_upload(upload: UploadFile, carpeta: str) -> str:
    """Guarda un fichero subido conservando su nombre original (aparece como fuente en los resultados)."""
    os.makedirs(carpeta, exist_ok=True)
    ruta = os.path.join(carpeta, os.path.basename(upload.filename))
    with open(ruta, "wb") as f:
        shutil.copyfileobj(upload.file, f)
    return ruta


def leer_archivo(rag: RAG, upload: UploadFile, carpeta: str) -> List[Document]:
    rag.path = guardar_upload(upload, carpeta)
    docs = [d for d in rag.clean(rag.read_document()) if d.page_content.strip()]
    if not docs:
        raise EntradaInvalida(f"'{upload.filename}' no contiene texto extraíble (¿está vacío o es un PDF escaneado?).")
    return docs


def leer_pegado(rag: RAG, texto: str, nombre: str) -> List[Document]:
    return rag.clean([Document(page_content=texto, metadata={"filename": nombre})])


def leer_referencias(rag: RAG, archivos, texto, metadatos, tmp) -> List[Document]:
    """Lee todas las referencias. Cada una tiene su carpeta: dos archivos con el mismo nombre no se pisan."""
    if texto is not None:
        lecturas = [leer_pegado(rag, texto, "Texto pegado (referencia)")]
    else:
        lecturas = [leer_archivo(rag, f, f"{tmp}/ref/{i}") for i, f in enumerate(archivos)]

    docs = []
    for i, (lectura, meta) in enumerate(zip(lecturas, metadatos), start=1):
        for doc in lectura:
            doc.metadata.update(ref_id=f"R{i}", **meta)
        docs.extend(lectura)
    return docs


@app.post("/validar")
async def validar_documentos(modalidad: Optional[str] = Form(None),
                             file_ref: List[UploadFile] = File(default=[]),
                             texto_ref: Optional[str] = Form(None),
                             ref_metadata: Optional[str] = Form(None),
                             file_query: Optional[UploadFile] = File(None),
                             texto_query: Optional[str] = Form(None),
                             fraccion_inicial: float = Form(1.0),
                             usar_agente: bool = Form(False)):
    print(f"🔍 Modalidad: {modalidad} · 🤖 Comentario: {'sí' if usar_agente else 'no'}")
    try:
        return _ejecutar_revision(modalidad, file_ref, texto_ref, ref_metadata, file_query, texto_query,
                                  fraccion_inicial, usar_agente)
    except EntradaInvalida as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Un fallo de embeddings o búsqueda nunca se devuelve como un éxito vacío
        raise HTTPException(status_code=500, detail=f"Error en el pipeline (¿está Ollama en marcha y el modelo "
                                                    f"de embeddings descargado?): {type(e).__name__}: {e}")


def _ejecutar_revision(modalidad, file_ref, texto_ref, ref_metadata, file_query, texto_query,
                       fraccion_inicial, usar_agente):
    config_modalidad = obtener_modalidad(modalidad)
    if not 0 < fraccion_inicial <= 1:
        raise EntradaInvalida("fraccion_inicial debe estar entre 0 (excluido) y 1.")

    archivos_ref, pegado_ref = seleccionar_entrada(file_ref, texto_ref, "Referencia")
    archivos_rev, pegado_rev = seleccionar_entrada([file_query], texto_query, "Texto a revisar")

    n_referencias = 1 if pegado_ref is not None else len(archivos_ref)
    if n_referencias > 1 and not config_modalidad.multiples_referencias:
        raise EntradaInvalida(f"La modalidad '{config_modalidad.nombre}' admite una sola referencia "
                              f"y se han recibido {n_referencias}.")
    metadatos = parsear_metadatos(ref_metadata, n_referencias)

    with tempfile.TemporaryDirectory() as tmp:
        # El mismo recorrido para las tres modalidades: indexar la referencia, buscar cada fragmento revisado
        referencia = RAG()
        docs = leer_referencias(referencia, archivos_ref, pegado_ref, metadatos, tmp)
        chunks = referencia.split(docs, _chunking['reference']['size'], _chunking['reference']['overlap'])
        indice = referencia.qdrant_record(chunks)

        consulta = RAG(embeddings_func=referencia.embeddings_func)
        if pegado_rev is not None:
            query_docs = leer_pegado(consulta, pegado_rev, "Texto pegado (revisado)")
        else:
            query_docs = leer_archivo(consulta, archivos_rev[0], f"{tmp}/query")
        query_chunks = consulta.split_query(query_docs, _chunking['review']['size'], _chunking['review']['overlap'],
                                            fraccion_inicial=fraccion_inicial)
        if not query_chunks:
            raise EntradaInvalida("El texto a revisar no contiene texto procesable.")

        results = consulta.process_query_chunks_v2(query_chunks, collection_name=indice["coleccion"])

        if usar_agente:
            print("🧠 Generando comentarios...")
            agente_dinamico.set_prompt(PROMPTS_MODALIDAD[config_modalidad.id])
            consulta.generar_respuesta(agente_dinamico)
            results = consulta.results

        referencias = []
        vistos = set()
        for doc in docs:
            if doc.metadata["ref_id"] not in vistos:
                vistos.add(doc.metadata["ref_id"])
                referencias.append({c: doc.metadata.get(c) for c in ("ref_id", "filename") + CAMPOS_METADATOS})

        return JSONResponse(content={
            "modalidad": config_modalidad.id,
            "modalidad_nombre": config_modalidad.nombre,
            "comentario_solicitado": usar_agente,
            "referencias": referencias,
            "indice": indice,
            "texto_revisado": {
                "filename": query_docs[0].metadata.get("filename"),
                "fragmentos_totales": consulta.fragmentos_totales,
                "fragmentos_revisados": len(query_chunks),
                "fragmentos_sin_revisar": consulta.fragmentos_totales - len(query_chunks),
                "sin_revisar_desde": consulta.sin_revisar_desde,
            },
            "resultados": results,
        })


@app.get("/colecciones")
def listar_colecciones():
    """Diagnóstico: colecciones de este proyecto en el almacenamiento configurado."""
    from X_qderant import get_qdrant_client
    collections = get_qdrant_client().get_collections().collections
    return {"colecciones": [col.name for col in collections if col.name.startswith(PREFIJO_COLECCION)]}
