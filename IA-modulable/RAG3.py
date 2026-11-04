
# * LIBRERIAS 
# * ==================================================================================================================
# pip install -qU langchain_community pypdf
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, Docx2txtLoader
# from langchain_community.document_loaders import UnstructuredWordDocumentLoader     # Mantiene la estructura del documento y puede identificar elementos como títulos, listas, párrafos, o tablas.

from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv


import numpy as np


# from langchain_community.vectorstores import Qdrant     # No uso langchain 

from langchain_community.vectorstores import FAISS
# import faiss

import os 

import logging

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("uvicorn.error").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("fastapi").setLevel(logging.WARNING)


from config import config
cfg_param = config.cfg_param

from langchain_ollama import OllamaEmbeddings
ollama_endpoint = f"http://{cfg_param['ollama']['host']}:{cfg_param['ollama']['port']}"
embed_model_name = cfg_param['llm']['embed']['name']


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Cargar las variables de entorno del archivo .env
load_dotenv()


nomic_embedding_model = OllamaEmbeddings(
            base_url=ollama_endpoint,
            model=embed_model_name
        )

from typing import List, Optional

from langchain_community.document_loaders import (DirectoryLoader, 
                                                  UnstructuredFileLoader) # deprecated in LangChain 0.2.8 and will be removed in 1.0.
# from langchain_unstructured import UnstructuredLoader
from langchain_core.documents import Document

from X_qderant import QdrantHandler

# * FUNCIÓN RAG  
# * ==================================================================================================================

class RAG:
    def __init__(self, docPath, embeddings_func=None):
        self.path = docPath
        # Función de embedding (por defecto, nomic-embed-text vía Ollama; se resuelve aquí para poder sustituirla)
        self.embeddings_func = embeddings_func or nomic_embedding_model
        self.texts = []  # Lista de textos para almacenar chunks
       
        self.results = None

        # Handler de Qdrant (usa la misma función de embedding que el resto del pipeline)
        self.qdrant = QdrantHandler(embeddings_func=self.embeddings_func, collection_name='prueba')

    # ! Preprocesamiento
    def read_document(
        self,
        recursive: bool = False,
        extension: Optional[str] = None
            ) -> List[Document]:
        """
        Carga documentos desde un archivo o carpeta. Solo se admiten .pdf y .docx.
        """
        path = self.path

        def get_loader(file_path):
            ext = os.path.splitext(file_path)[1].lower()
            if ext == ".pdf":
                return PyPDFLoader(file_path)
            elif ext == ".docx":
                return Docx2txtLoader(file_path)
            else:
                raise ValueError(f"Extensión no soportada: {ext}")

        if os.path.isdir(path):
            raise ValueError("Para carpetas, usa read_directory().")

        print("Leyendo archivo individual")
        loader = get_loader(path)
        docs = loader.load()

        for doc in docs:
            src = doc.metadata.get("source", path)
            doc.metadata["filename"] = os.path.basename(src)

        return docs

    

    def read_directory(self, ext: str = "pdf", recursive: bool = True) -> List[Document]:
        """
        Carga todos los documentos de una extensión específica dentro de un directorio.
        """
        print(f"Leyendo todos los .{ext} en el directorio")

        loader = DirectoryLoader(
            path=self.path,
            glob=f"**/*.{ext.lstrip('.')}",
            show_progress=True,
            recursive=recursive,
            loader_cls=PyPDFLoader if ext == "pdf" else Docx2txtLoader,
            loader_kwargs={"mode": "single"}
        )

        docs = loader.load()

        for doc in docs:
            source_path = doc.metadata.get("source", "")
            doc.metadata["filename"] = os.path.basename(source_path)

        print(f"Documentos cargados: {len(docs)}")
        return docs

    
    # * Limpieza  
    # * ===========================================================================
    def clean(self, docs):

        from X_FUNCTIONS import clean_text

        # Aseguramos que siempre trabajamos con una lista
        docs_list = docs if isinstance(docs, list) else [docs]

        for doc in docs_list:
            doc.page_content = clean_text(doc.page_content)

        # Opcional: devolver los docs (limpios) para poder encadenar llamadas
        return docs_list


    def split(self, texto, chunk_ss, chunk_so, sample_ratio: Optional[float] = None):
        """
        Divide el texto en chunks y aplica un muestreo si se especifica.
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_ss,
            chunk_overlap=chunk_so
        )
        splits = text_splitter.split_documents(texto)

        if sample_ratio is not None and 0 < sample_ratio < 1.0:
            original_count = len(splits)
            sample_size = max(1, int(original_count * sample_ratio))
            splits = splits[:sample_size]
            print(f"✂️ Sampling: {sample_size} de {original_count} chunks ({sample_ratio*100:.0f}%)")

        self.texts = splits
        print(f"GENERACIÓN SPLITS - Se han generado {len(splits)} chunks")
        return splits


    # * Qderant (embeddings)
    # * ===========================================================================
    def qdrant_record(self, splits: List[Document], collection_name: str = None):
        # Delegar en el handler
        return self.qdrant.qdrant_record(splits, collection_name)
    
    def qdrant_retriever(self, query: str, k: int = 3, collection_name: str = None):
        return self.qdrant.qdrant_retriever(query, k, collection_name)

    # * Query 
    # * =========================================================================== 
    def split_query(self, query, chunk_ss, chunk_so, separators=None, sample_ratio: Optional[float] = None):
        """
        Divide la consulta en chunks y aplica un muestreo si se especifica.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_ss,
            chunk_overlap=chunk_so,
            separators=separators
        )
        
        if isinstance(query, str):
            docs = [Document(page_content=query)]
        elif isinstance(query, list) and all(hasattr(d, 'page_content') for d in query):
            docs = query
        else:
            raise TypeError("split_query espera str o lista de Document")
        
        splits = splitter.split_documents(docs)

        if sample_ratio is not None and 0 < sample_ratio < 1.0:
            original_count = len(splits)
            sample_size = max(1, int(original_count * sample_ratio))
            splits = splits[:sample_size]
            print(f"✂️ Sampling QUERY: {sample_size} de {original_count} chunks ({sample_ratio*100:.0f}%)")
        
        self.texts = splits
        print(f"🔍 SPLIT QUERY - Se han generado {len(splits)} chunks")
        return splits


    def process_query_chunks_v2(self, splits_query, collection_name: str = "document_embeddings"):
        """
        Procesa cada chunk de consulta y recupera contexto.
        """
        results = []
        print('Procesando cada chunk como query...')

        for i, chunk in enumerate(splits_query):
            text = chunk.page_content if hasattr(chunk, 'page_content') else str(chunk)
            print(f"Chunk {i+1}/{len(splits_query)}: {text[:50]}...")
            hits = self.qdrant.qdrant_search(text, k=3, collection_name=collection_name)
            ctx = '\n\n'.join(f"Documento: {h['filename']}\n{h['text']}" for h in hits)
            results.append({'chunk': text, 
                            'relevant_chunks': [h['text'] for h in hits], 
                            'context': ctx,
                            # Trazabilidad: fuente, página (PDF) y similitud coseno de cada fragmento recuperado
                            'sources': [{'filename': h['filename'], 'page': h['page']} for h in hits],
                            'scores': [round(h['score'], 4) for h in hits]})
        self.results = results
        
        return results

    def generar_respuesta(self, agente):
            
        for result in self.results:
            # Llamada al Modelo para cada 'chunk' y su contexto
            answer = agente.get_answer(contexto=result['context'], pregunta=result['chunk'])
            
            # Asociar la respuesta al 'chunk' correspondiente
            result['respuesta'] = answer





