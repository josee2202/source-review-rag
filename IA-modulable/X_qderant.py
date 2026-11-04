from typing import Optional, List
from qdrant_client import QdrantClient, models
from langchain_core.documents import Document
import numpy as np

from config import config
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

    def ensure_collection(self, vector_size: int):
        """
        Crea la colección en Qdrant si no existe.
        """
        existing = self.client.get_collections().collections
        names = [col.name for col in existing]
        if self.collection_name not in names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE
                )
            )

    def embed_in_batches(self, texts: List[str], batch_size: int = embed_batch_size) -> np.ndarray:
        """
        Calcula los embeddings en lotes: Ollama falla con peticiones de cientos de textos.
        """
        vectors = []
        for i in range(0, len(texts), batch_size):
            vectors.extend(self.embeddings_func.embed_documents(texts=texts[i:i + batch_size]))
            print(f"   · embeddings {min(i + batch_size, len(texts))}/{len(texts)}")
        return np.array(vectors)

    def qdrant_record(self, splits: List[Document], collection_name: str = None):
        """
        Almacena embeddings y chunks en Qdrant solo si la colección está vacía.

        Args:
            splits (list[Document]): Lista de chunks de Document.
            collection_name (str): Nombre de la colección Qdrant. Si no se especifica, usa el por defecto.

        Returns:
            int: número de vectores en la colección tras la operación.
        """
        try:
            name = collection_name or self.collection_name
            self.collection_name = name

            print(f"📦 Usando colección: {name}")
            print("🔍 Verificando existencia y contenido de la colección...")

            # Si la colección ya existe y tiene puntos, no hacemos nada
            collections = self.client.get_collections().collections
            existing = [col.name for col in collections]

            if name in existing:
                info = self.client.get_collection(collection_name=name)
                if info.points_count > 0:
                    print(f"✅ La colección '{name}' ya contiene {info.points_count} vectores. No se recalculan embeddings.")
                    return info.points_count
                else:
                    print(f"⚠️ La colección '{name}' existe pero está vacía. Procediendo a generar embeddings...")
            else:
                print(f"🆕 La colección '{name}' no existe. Será creada.")

            if not splits:
                raise ValueError("No hay chunks que indexar: el documento de referencia no contiene texto extraíble.")

            # Generar embeddings
            print(f"🧠 Generando embeddings para {len(splits)} chunks...")
            texts = [chunk.page_content if hasattr(chunk, 'page_content') else str(chunk) for chunk in splits]
            embedding_matrix = self.embed_in_batches(texts)
            vector_size = embedding_matrix.shape[1]

            # Asegurar colección
            self.ensure_collection(vector_size)
            print(f"🛠️ Colección '{name}' lista para recibir vectores.")

            # Preparar puntos (PyPDFLoader numera las páginas desde 0; se guardan desde 1)
            metadatas = [getattr(chunk, 'metadata', {}) for chunk in splits]
            points = [
                models.PointStruct(
                    id=i,
                    vector=vector.tolist(),
                    payload={
                        "text": texts[i],
                        "filename": metadatas[i].get('filename', None),
                        "page": metadatas[i]['page'] + 1 if isinstance(metadatas[i].get('page'), int) else None
                    }
                ) for i, vector in enumerate(embedding_matrix)
            ]

            # Upsert por batches
            batch_size = 100
            print(f"📤 Subiendo {len(points)} vectores en lotes de {batch_size}...")
            for i in range(0, len(points), batch_size):
                self.client.upsert(collection_name=name, points=points[i:i + batch_size])

            print(f"✅ {len(points)} vectores almacenados en '{name}'.")
            return len(points)

        except Exception as e:
            # Antes se imprimía y se continuaba: la API devolvía 200 con resultados vacíos
            print(f"❌ Error al almacenar en Qdrant: {e}")
            raise


    def qdrant_search(self, query: str, k: int = 3, collection_name: str = None) -> List[dict]:
        """
        Devuelve los k fragmentos más similares con su texto, fuente y similitud coseno.
        """
        name = collection_name or self.collection_name
        # Embed query
        q_vec = self.embeddings_func.embed_query(query)
        # Asegurar colección vacía si no existe
        self.collection_name = name
        self.ensure_collection(len(q_vec))
        # Search
        results = self.client.query_points(
            collection_name=name,
            query=q_vec,
            limit=k,
            with_payload=True,
            with_vectors=False
        ).points
        hits = []
        for pt in results:
            payload = pt.payload or {}
            hits.append({
                "text": payload.get('text', ''),
                "filename": payload.get('filename') or 'desconocido',
                "page": payload.get('page'),
                "score": float(pt.score)
            })
        return hits


    def qdrant_retriever(self, query: str, k: int = 3, collection_name: str = None):
        """
        Recuperador usando Qdrant.

        Args:
            query (str): Texto de consulta.
            k (int): Número de resultados a recuperar.
            collection_name (str): Nombre de la colección Qdrant.

        Returns:
            relevant_chunks (list[str]), combined_context (str)
        """
        try:
            hits = self.qdrant_search(query, k, collection_name)
            relevant_chunks = [h['text'] for h in hits]
            combined = [f"Documento: {h['filename']}\n{h['text']}" for h in hits]
            return relevant_chunks, '\n\n'.join(combined)
        except Exception as e:
            print(f"Error en retriever de Qdrant: {e}")
            raise
