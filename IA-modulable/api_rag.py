from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import List
import shutil
import tempfile

import os
import sys

# ? Añadimos dinámicamente la carpeta padre al path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from RAG3 import RAG
from prompt import prompt_definitivos
from X_AGENT import agente_dinamico  # Asegúrate de que esté inicializado

def split_by_consulta_pattern(query_docs):
    import re
    from langchain.schema import Document
    
    # Si es una lista de Documents, extraer el texto
    if isinstance(query_docs, list):
        # Concatenar todo el contenido de los documentos
        full_text = "\n".join([doc.page_content for doc in query_docs])
    elif isinstance(query_docs, str):
        full_text = query_docs
    else:
        # Si es un solo Document
        full_text = query_docs.page_content
    
    # Dividir por el patrón "Consulta N:"
    parts = re.split(r'Consulta \d+:', full_text)
    
    # Filtrar partes vacías y crear Documents
    documents = []
    for i, part in enumerate(parts):
        if part.strip():
            documents.append(Document(
                page_content=part.strip(),
                metadata={"chunk_id": i, "source": "consulta_split"}
            ))
    
    print(f"✂️ Split por consultas: {len(documents)} chunks generados")
    return documents    



def get_prompt_by_validation_type(tipo_validacion):
    """
    Retorna el prompt correspondiente según el tipo de validación.
    
    Args:
        tipo_validacion (str): Tipo de validación seleccionado
        
    Returns:
        ChatPromptTemplate: Prompt correspondiente
    """
    prompts = {
        "Normativo vs Funcional": prompt_definitivos['evaluar_alineacion_normativa'],
        "Funcional vs Técnico": prompt_definitivos['evaluar_tecnico_funcional'],
        "Código vs Técnico": prompt_definitivos['evaluar_descriptivo_tecnico'],
        "Código vs Estadístico": prompt_definitivos['evaluar_alineacion_normativa']
    }
    
    return prompts.get(tipo_validacion, prompt_definitivos['evaluar_alineacion_normativa'])

app = FastAPI()

def guardar_upload(upload: UploadFile, carpeta: str) -> str:
    """Guarda un fichero subido conservando su nombre original (aparece como fuente en los resultados)."""
    os.makedirs(carpeta, exist_ok=True)
    ruta = os.path.join(carpeta, os.path.basename(upload.filename))
    with open(ruta, "wb") as f:
        shutil.copyfileobj(upload.file, f)
    return ruta


@app.post("/validar")
async def validar_documentos(file_ref: List[UploadFile] = File(...),
                            file_query: UploadFile = File(...), 
                            collection_name: str = Form(...),
                            doc_sample_ratio: float = Form(1.0),
                            usar_agente: bool = Form(False),
                            tipo_validacion: str = Form("Normativo vs Funcional")):
    
    print(f"⚙️ Porcentaje solicitado: {doc_sample_ratio * 100}%")
    print(f"🤖 ¿Validación con agente?: {'Sí' if usar_agente else 'No'}")
    print(f"🔍 Tipo de validación: {tipo_validacion}")

    try:
        return _ejecutar_validacion(file_ref, file_query, collection_name, doc_sample_ratio, usar_agente, tipo_validacion)
    except (ValueError, TypeError, FileNotFoundError) as e:
        # Entrada no válida: formato no soportado, documento sin texto extraíble...
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Antes los fallos de Ollama/Qdrant se silenciaban y se devolvía 200 sin evidencias
        raise HTTPException(status_code=500, detail=f"Error en el pipeline (¿están Ollama y Qdrant en marcha?): {e}")


def _ejecutar_validacion(file_ref, file_query, collection_name, doc_sample_ratio, usar_agente, tipo_validacion):
    with tempfile.TemporaryDirectory() as tmp:
        # Guardar archivos (la referencia puede ser uno o varios documentos)
        ref_paths = [guardar_upload(f, f"{tmp}/ref") for f in file_ref]
        query_path = guardar_upload(file_query, f"{tmp}/query")

        # Ejecutar pipeline
        supervisor = RAG(ref_paths[0])
        docs = []
        for ref_path in ref_paths:
            supervisor.path = ref_path
            docs.extend(supervisor.read_document())
        docs = supervisor.clean(docs)
        chunks = supervisor.split(docs, chunk_ss=800, chunk_so=200)
        supervisor.qdrant_record(chunks, collection_name=collection_name)

        # El código SAS se traduce primero a texto, sea cual sea el tipo de validación
        # (antes solo en "Código vs Técnico": en "Código vs Estadístico" un .sas fallaba)
        if query_path.lower().endswith(".sas"):
            from Codigo_sas import ProcesarSAS
            procesador_sas = ProcesarSAS()
            
            # Crear directorio para el archivo SAS
            sas_dir = f"{tmp}/sas_files"
            os.makedirs(sas_dir, exist_ok=True)
            
            # Mover el archivo SAS al directorio
            sas_file_path = f"{sas_dir}/{os.path.basename(query_path)}"
            shutil.copy2(query_path, sas_file_path)
            
            # Cargar y procesar documentos SAS desde el directorio
            sas_docs = procesador_sas.read_directory(sas_dir)
            split_documents = procesador_sas.split_documents_2(sas_docs)
            
            # Agrupar bloques en chunks largos optimizados
            grupos = procesador_sas.agrupar_bloques_por_longitud(split_documents, max_length=2400)
            
            # Generar los textos explicando los bloques de código
            respuestas = procesador_sas.generar_respuestas(grupos)
            
            # Guardar el Word en el directorio temporal y asegurar que query_path tenga la ruta correcta
            word_path = f"{tmp}/resultados_bloques_definitivo.docx"
            procesador_sas.guardar_respuestas_en_word(respuestas, word_path)
            query_path = word_path  # Asegurar que query_path tenga la ruta correcta
            
            print(f"📄 Archivo Word guardado en: {query_path}")  # Debug

        consulta = RAG(query_path, embeddings_func=supervisor.embeddings_func)
        
        # Verificar que query_path no sea None
        if query_path is None:
            raise ValueError("Error: query_path es None después del procesamiento SAS")
            
        print(f"🔍 Leyendo documento desde: {query_path}")  # Debug
        query_docs = consulta.read_document()
        query_docs = consulta.clean(query_docs)
        if not any(d.page_content.strip() for d in query_docs):
            raise ValueError("El documento de consulta no contiene texto extraíble (¿PDF escaneado?).")
        query_chunks = consulta.split_query(query_docs, 600, 200, sample_ratio=doc_sample_ratio)
        # query_chunks = split_by_consulta_pattern(query_docs)
        results = consulta.process_query_chunks_v2(query_chunks, collection_name=collection_name)

        consulta.results = results

        if usar_agente:
            print("🧠 Generando respuestas con agente...")
            
            # Obtener el prompt después de procesar el código SAS
            prompt_seleccionado = get_prompt_by_validation_type(tipo_validacion)
            agente_dinamico.set_prompt(prompt_seleccionado)
            
            consulta.generar_respuesta(agente_dinamico)
            results = consulta.results

            print("Respuesta generada con éxito")
            
        return JSONResponse(content={"resultados": results})




# @app.post("/validar")
# async def validar_documentos(file_ref: UploadFile = File(...),
#                             file_query: UploadFile = File(...), 
#                             collection_name: str = Form(...),
#                             doc_sample_ratio: float = Form(1.0),
#                             usar_agente: bool = Form(False)):
    
#     print(f"⚙️ Porcentaje solicitado: {doc_sample_ratio * 100}%")
#     print(f"🤖 ¿Validación con agente?: {'Sí' if usar_agente else 'No'}")
    
#     with tempfile.TemporaryDirectory() as tmp:
#         # Guardar archivos
#         ref_path = f"{tmp}/ref_{file_ref.filename}"
#         with open(ref_path, "wb") as f:
#             shutil.copyfileobj(file_ref.file, f)

#         query_path = f"{tmp}/query_{file_query.filename}"
#         with open(query_path, "wb") as f:
#             shutil.copyfileobj(file_query.file, f)

#         # Ejecutar pipeline
#         supervisor = RAG(ref_path)
#         docs = supervisor.read_document()
#         docs = supervisor.clean(docs)
#         chunks = supervisor.split(docs, chunk_ss=800, chunk_so=200)
#         supervisor.qdrant_record(chunks, collection_name=collection_name)

#         consulta = RAG(query_path, embeddings_func=supervisor.embeddings_func)
#         query_docs = consulta.read_document()
#         query_docs = consulta.clean(query_docs)
#         query_chunks = consulta.split_query(query_docs, 600, 200, sample_ratio=doc_sample_ratio)
#         results = consulta.process_query_chunks_v2(query_chunks, collection_name=collection_name)

#         consulta.results = results  # Asignamos resultados para el método

#         if usar_agente:
#             print("🧠 Generando respuestas con agente...")
#             consulta.generar_respuesta()
#             results = consulta.results  # actualizar con respuestas

#         return JSONResponse(content={"resultados": results})
    

@app.get("/colecciones")
def listar_colecciones():
    from X_qderant import get_qdrant_client
    collections = get_qdrant_client().get_collections().collections
    nombres = [col.name for col in collections]
    return {"colecciones": nombres}


