
# * LIBRERIAS
# * ==================================================================================================================
import logging
import os
from typing import List, Optional

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

logging.getLogger("httpx").setLevel(logging.WARNING)

from config import config
cfg_param = config.cfg_param

ollama_endpoint = f"http://{cfg_param['ollama']['host']}:{cfg_param['ollama']['port']}"
embed_model_name = cfg_param['llm']['embed']['name']

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

nomic_embedding_model = OllamaEmbeddings(
            base_url=ollama_endpoint,
            model=embed_model_name
        )

from prompt import neutralizar_delimitadores
from X_FUNCTIONS import EntradaInvalida, clean_text
from X_qderant import QdrantHandler

EXTENSIONES_TEXTO = (".txt", ".md", ".markdown")
EXTENSIONES_ADMITIDAS = (".pdf", ".docx") + EXTENSIONES_TEXTO
TOP_K = 3


def etiqueta_ubicacion(metadata: dict) -> str:
    """Ubicación legible: la página real en PDF; el párrafo del texto normalizado en el resto."""
    if isinstance(metadata.get("page"), int):
        return f"p. {metadata['page']}"
    inicio, fin = metadata.get("paragraph"), metadata.get("paragraph_end")
    if inicio is None:
        return "ubicación no disponible"
    return f"párr. {inicio}" if inicio == fin else f"párr. {inicio}-{fin}"


def anotar_ubicacion(fragmento: Document, texto_documento: str) -> None:
    """Añade al fragmento su posición dentro del documento (o de la página, en un PDF) del que procede."""
    metadata = fragmento.metadata
    inicio = metadata.pop("start_index", -1)
    metadata.update(char_start=None, char_end=None, paragraph=None, paragraph_end=None)
    if inicio >= 0:
        fin = inicio + len(fragmento.page_content)
        metadata.update(char_start=inicio, char_end=fin)
        if not isinstance(metadata.get("page"), int):
            metadata.update(paragraph=texto_documento.count("\n\n", 0, inicio) + 1,
                            paragraph_end=texto_documento.count("\n\n", 0, fin) + 1)
    metadata["location"] = etiqueta_ubicacion(metadata)


def tramos_por_parrafo(texto: str):
    """Pares (posición, párrafo) de un texto cuyos párrafos están separados por una línea en blanco."""
    posicion = 0
    for parrafo in texto.split("\n\n"):
        if parrafo.strip():
            yield posicion, parrafo
        posicion += len(parrafo) + 2


def formatear_contexto(hits: List[dict]) -> str:
    """Pasajes como un solo texto, cada uno con la etiqueta de su procedencia (campo context de la API)."""
    bloques = []
    for h in hits:
        datos = " · ".join(str(h[c]) for c in ("filename", "title", "outlet", "date", "url") if h.get(c))
        etiqueta = " · ".join(str(h[c]) for c in ("ref_id", "location") if h.get(c))
        bloques.append(f"[{etiqueta}] {datos}\n{h['text']}")
    return neutralizar_delimitadores("\n\n".join(bloques))


# * FUNCIÓN RAG
# * ==================================================================================================================

class RAG:
    def __init__(self, docPath=None, embeddings_func=None):
        self.path = docPath
        # Función de embedding (por defecto, nomic-embed-text vía Ollama; se resuelve aquí para poder sustituirla)
        self.embeddings_func = embeddings_func or nomic_embedding_model
        self.texts = []  # Lista de textos para almacenar chunks
        self.config_segmentacion = {}
        self.fragmentos_totales = 0
        self.sin_revisar_desde = None

        self.results = None

        # Handler de Qdrant (usa la misma función de embedding que el resto del pipeline)
        self.qdrant = QdrantHandler(embeddings_func=self.embeddings_func, collection_name='sin_indexar')

    # ! Preprocesamiento
    def read_document(self) -> List[Document]:
        """
        Carga un archivo PDF, DOCX, TXT o Markdown. Un error de lectura es un problema de la entrada.
        """
        path = self.path
        nombre = os.path.basename(path)
        ext = os.path.splitext(path)[1].lower()

        if ext not in EXTENSIONES_ADMITIDAS:
            raise EntradaInvalida(f"Formato no soportado en '{nombre}' ({ext or 'sin extensión'}). "
                                  "Formatos admitidos: PDF, DOCX, TXT y Markdown.")
        try:
            if ext == ".pdf":
                docs = PyPDFLoader(path).load()
            elif ext == ".docx":
                docs = Docx2txtLoader(path).load()
            else:
                with open(path, encoding="utf-8-sig") as f:
                    docs = [Document(page_content=f.read())]
        except UnicodeDecodeError:
            raise EntradaInvalida(f"'{nombre}' no está codificado en UTF-8.")
        except Exception as e:
            raise EntradaInvalida(f"No se ha podido leer '{nombre}' como {ext}: {e}")

        # PyPDFLoader numera las páginas desde 0; se guardan desde 1
        for doc in docs:
            pagina = doc.metadata.get("page")
            doc.metadata = {"filename": nombre}
            if isinstance(pagina, int):
                doc.metadata["page"] = pagina + 1

        return docs


    # * Limpieza
    # * ===========================================================================
    def clean(self, docs):
        # Aseguramos que siempre trabajamos con una lista
        docs_list = docs if isinstance(docs, list) else [docs]

        for doc in docs_list:
            doc.page_content = clean_text(doc.page_content)

        # Opcional: devolver los docs (limpios) para poder encadenar llamadas
        return docs_list


    def _trocear(self, docs: List[Document], chunk_ss: int, chunk_so: int, separators=None,
                 respetar_parrafos: bool = False) -> List[Document]:
        """
        Divide cada documento por separado (nunca mezcla documentos en un fragmento), prioriza los
        límites de párrafo del separador por defecto y conserva los metadatos y la ubicación.
        Con respetar_parrafos, además, dos párrafos no se fusionan en un mismo fragmento.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_ss,
            chunk_overlap=chunk_so,
            separators=separators,
            add_start_index=True
        )
        self.config_segmentacion = {"chunk_size": chunk_ss, "chunk_overlap": chunk_so, "separators": separators}
        fragmentos = []
        for doc in docs:
            tramos = tramos_por_parrafo(doc.page_content) if respetar_parrafos else [(0, doc.page_content)]
            for posicion, tramo in tramos:
                for fragmento in splitter.create_documents([tramo], [doc.metadata]):
                    if fragmento.metadata["start_index"] >= 0:
                        fragmento.metadata["start_index"] += posicion
                    anotar_ubicacion(fragmento, doc.page_content)
                    fragmentos.append(fragmento)
        return fragmentos


    def split(self, texto: List[Document], chunk_ss: int, chunk_so: int) -> List[Document]:
        """
        Divide la referencia en fragmentos.
        """
        splits = self._trocear(texto, chunk_ss, chunk_so)
        self.texts = splits
        print(f"GENERACIÓN SPLITS - Se han generado {len(splits)} chunks")
        return splits


    # * Qderant (embeddings)
    # * ===========================================================================
    def qdrant_record(self, splits: List[Document]) -> dict:
        # Delegar en el handler
        return self.qdrant.qdrant_record(splits, self.config_segmentacion)

    # * Query
    # * ===========================================================================
    def split_query(self, query, chunk_ss, chunk_so, separators=None, fraccion_inicial: float = 1.0):
        """
        Divide el texto revisado en fragmentos sin fusionar párrafos: cada afirmación breve de una
        newsletter debe tener sus propios pasajes. Por defecto se procesan todos; con fraccion_inicial < 1
        se conserva solo el inicio (no es un muestreo aleatorio); fragmentos_totales y sin_revisar_desde
        indican lo que queda sin revisar.
        """
        if isinstance(query, str):
            docs = [Document(page_content=query)]
        elif isinstance(query, list) and all(hasattr(d, 'page_content') for d in query):
            docs = query
        else:
            raise TypeError("split_query espera str o lista de Document")

        splits = self._trocear(docs, chunk_ss, chunk_so, separators, respetar_parrafos=True)
        self.fragmentos_totales = len(splits)
        self.sin_revisar_desde = None

        if 0 < fraccion_inicial < 1.0 and splits:
            n_inicio = max(1, int(len(splits) * fraccion_inicial))
            if n_inicio < len(splits):
                self.sin_revisar_desde = splits[n_inicio].metadata.get("location")
            splits = splits[:n_inicio]
            print(f"✂️ Solo el inicio: {len(splits)} de {self.fragmentos_totales} chunks")

        self.texts = splits
        print(f"🔍 SPLIT QUERY - Se han generado {len(splits)} chunks")
        return splits


    def process_query_chunks_v2(self, splits_query, collection_name: str = None):
        """
        Procesa cada chunk del texto revisado y recupera sus pasajes con su procedencia y similitud.
        """
        results = []
        print('Procesando cada chunk como query...')

        for i, chunk in enumerate(splits_query):
            text = chunk.page_content if hasattr(chunk, 'page_content') else str(chunk)
            print(f"Chunk {i+1}/{len(splits_query)}: {text[:50]}...")
            hits = self.qdrant.qdrant_search(text, k=TOP_K, collection_name=collection_name)
            metadata = getattr(chunk, 'metadata', {}) or {}
            results.append({'fragmento': i + 1,
                            'chunk': text,
                            'chunk_location': {c: metadata.get(c) for c in ("filename", "page", "paragraph",
                                                                            "paragraph_end", "char_start",
                                                                            "char_end", "location")},
                            'relevant_chunks': [h['text'] for h in hits],
                            'context': formatear_contexto(hits),
                            # Trazabilidad: procedencia de cada pasaje y similitud coseno (no es una probabilidad)
                            'sources': [{k: v for k, v in h.items() if k not in ('text', 'score')} for h in hits],
                            'scores': [round(h['score'], 4) for h in hits]})
        self.results = results

        return results
