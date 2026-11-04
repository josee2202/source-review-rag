import hashlib
import json
from typing import Optional, List

from qdrant_client import QdrantClient, models
from langchain_core.documents import Document
import numpy as np

from config import config
from X_FUNCTIONS import EntradaInvalida
cfg_param = config.cfg_param

from langchain_ollama import OllamaEmbeddings


host = cfg_param['ollama']['host']
port = cfg_param['ollama']['port']
base_url = f"http://{host}:{port}"

embed_model_name = cfg_param['llm']['embed']['name']
embed_batch_size = cfg_param['llm']['embed'].get('batch_size', 32)

nomic_embedding_model = OllamaEmbeddings(
    base_url=base_url,
    model=embed_model_name
)

# Procedencia que acompaña a cada fragmento desde la lectura hasta los resultados
CAMPOS_PROCEDENCIA = ("ref_id", "filename", "page", "paragraph", "paragraph_end", "char_start", "char_end",
                      "location", "title", "outlet", "date", "url")

PREFIJO_COLECCION = "srr_"  # evita colisiones si QDRANT_LOCATION apunta a un servidor compartido
VERSION_HUELLA = 1          # subirla si cambia el formato del payload


def payload_fragmento(chunk: Document) -> dict:
    """Lo que se guarda en Qdrant para un fragmento: su texto y su procedencia."""
    metadata = getattr(chunk, 'metadata', {}) or {}
    return {"text": chunk.page_content, **{campo: metadata.get(campo) for campo in CAMPOS_PROCEDENCIA}}


def huella_corpus(splits: List[Document], config_indice: dict) -> str:
    """Huella estable del contenido, los metadatos y la configuración de embeddings y segmentación."""
    contenido = {"version": VERSION_HUELLA, "config": config_indice,
                 "fragmentos": [payload_fragmento(chunk) for chunk in splits]}
    return hashlib.sha256(json.dumps(contenido, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


# * Cliente compartido: el modo ":memory:" y el modo ruta local solo ven sus datos
# * desde la misma instancia, así que se reutiliza un cliente por ubicación.
_clients = {}

def get_qdrant_client(location: Optional[str] = None) -> QdrantClient:
    location = location or cfg_param['qdrant']['location']
    if location not in _clients:
        if location.startswith(("http://", "https://")):
            _clients[location] = QdrantClient(url=location)
        elif location == ":memory:":
            _clients[location] = QdrantClient(location=":memory:")
        else:
            _clients[location] = QdrantClient(path=location)
    return _clients[location]


class QdrantHandler:
    def __init__(self, embeddings_func = nomic_embedding_model, collection_name: str = "document_embeddings", location: Optional[str] = None):
        """
        Handler para operaciones con Qdrant: inserción y recuperación de embeddings.
        """
        self.embeddings_func = embeddings_func
        self.client = get_qdrant_client(location)
        self.collection_name = collection_name

    def embed_in_batches(self, texts: List[str], batch_size: int = embed_batch_size) -> np.ndarray:
        """
        Calcula los embeddings en lotes: Ollama falla con peticiones de cientos de textos.
        """
        vectors = []
        for i in range(0, len(texts), batch_size):
            vectors.extend(self.embeddings_func.embed_documents(texts=texts[i:i + batch_size]))
            print(f"   · embeddings {min(i + batch_size, len(texts))}/{len(texts)}")
        return np.array(vectors)

    def qdrant_record(self, splits: List[Document], config_segmentacion: dict) -> dict:
        """
        Indexa los fragmentos en una colección cuyo nombre es la huella del corpus.

        Un corpus distinto produce siempre otra colección, así que nunca se consulta el anterior.
        Solo se reutiliza una colección con todos los puntos esperados; una indexación
        interrumpida se descarta y se repite.

        Returns:
            dict: coleccion, fragmentos y reutilizado.
        """
        if not splits:
            raise EntradaInvalida("No hay fragmentos que indexar: la referencia no contiene texto extraíble.")

        huella = huella_corpus(splits, {"embed_model": embed_model_name, **config_segmentacion})
        name = f"{PREFIJO_COLECCION}{huella[:32]}"
        self.collection_name = name
        esperados = len(splits)

        if self.client.collection_exists(name):
            actuales = self.client.count(name, exact=True).count
            if actuales == esperados:
                print(f"✅ Índice completo reutilizado: '{name}' ({actuales} vectores).")
                return {"coleccion": name, "fragmentos": esperados, "reutilizado": True}
            print(f"⚠️ Índice parcial en '{name}' ({actuales}/{esperados}): se descarta y se repite.")
            self.client.delete_collection(name)

        # Generar embeddings
        print(f"🧠 Generando embeddings para {esperados} fragmentos...")
        texts = [chunk.page_content for chunk in splits]
        embedding_matrix = self.embed_in_batches(texts)

        self.client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(size=embedding_matrix.shape[1], distance=models.Distance.COSINE)
        )

        points = [
            models.PointStruct(id=i, vector=vector.tolist(), payload=payload_fragmento(splits[i]))
            for i, vector in enumerate(embedding_matrix)
        ]

        # Upsert por batches
        batch_size = 100
        print(f"📤 Subiendo {len(points)} vectores en lotes de {batch_size}...")
        for i in range(0, len(points), batch_size):
            self.client.upsert(collection_name=name, points=points[i:i + batch_size])

        guardados = self.client.count(name, exact=True).count
        if guardados != esperados:
            raise RuntimeError(f"Indexación incompleta en '{name}': {guardados} de {esperados} vectores.")

        print(f"✅ {guardados} vectores almacenados en '{name}'.")
        return {"coleccion": name, "fragmentos": esperados, "reutilizado": False}


    def qdrant_search(self, query: str, k: int = 3, collection_name: str = None) -> List[dict]:
        """
        Devuelve hasta k fragmentos similares con su texto, procedencia y similitud coseno.
        Puede haber menos de k si la referencia es pequeña; ninguno es un error, no un éxito vacío.
        """
        name = collection_name or self.collection_name
        if not self.client.collection_exists(name):
            raise RuntimeError(f"La colección '{name}' no existe: la referencia no se ha indexado.")

        q_vec = self.embeddings_func.embed_query(query)
        results = self.client.query_points(
            collection_name=name,
            query=q_vec,
            limit=k,
            with_payload=True,
            with_vectors=False
        ).points
        if not results:
            raise RuntimeError(f"La búsqueda en '{name}' no ha devuelto ningún pasaje.")

        hits = []
        for pt in results:
            payload = pt.payload or {}
            hits.append({
                **{campo: payload.get(campo) for campo in CAMPOS_PROCEDENCIA},
                "text": payload.get('text', ''),
                "filename": payload.get('filename') or 'desconocido',
                "score": float(pt.score)
            })
        return hits
