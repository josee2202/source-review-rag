"""Tests del pipeline: limpieza, troceado de SAS, indexación y recuperación en Qdrant."""
import pytest

from conftest import SYNTHETIC, FakeEmbeddings
from X_FUNCTIONS import clean_text


def test_clean_text_normaliza_guiones_espacios_y_puntuacion():
    texto = "La informa-\nción   tiene “comillas”\r\n\r\n\r\n\r\ny más —texto…"
    assert clean_text(texto) == 'La información tiene "comillas"\n\ny más -texto...'


def test_split_sas_por_macros_conserva_bloques_completos():
    from Codigo_sas import ProcesarSAS
    from langchain_core.documents import Document
    import synthetic_corpus

    bloques = ProcesarSAS().split_documents_2([Document(page_content=synthetic_corpus.CODIGO_SAS,
                                                        metadata={"filename": "demo.sas"})])
    macros = [b for b in bloques if b.metadata["block_type"] == "macro"]
    assert len(macros) == 3
    assert all("%mend" in b.page_content.lower() for b in macros)
    # Todo el código queda cubierto por algún bloque y los metadatos se conservan
    assert all(b.metadata["filename"] == "demo.sas" for b in bloques)
    assert "roc 'PD estimada'" in bloques[-1].page_content


def test_agrupar_bloques_respeta_el_presupuesto_de_tokens():
    from Codigo_sas import ProcesarSAS
    from langchain_core.documents import Document
    import synthetic_corpus

    proc = ProcesarSAS()
    bloques = proc.split_documents_2([Document(page_content=synthetic_corpus.CODIGO_SAS * 6)])
    grupos = proc.agrupar_bloques_por_longitud(bloques, max_length=1200)
    assert len(grupos) > 1
    from prompt import prompt_codigo_texto
    presupuesto = 1200 - len(proc.sp.encode(prompt_codigo_texto.template)) - 200
    assert all(len(proc.sp.encode(g)) <= presupuesto + 5 for g in grupos)  # margen por saltos de línea


def test_agentes_llm_limitan_los_tokens_generados():
    """Sin límite, llama3.2:1b entró en bucle con un SAS real (38.000 tokens en 2 h, sin respuesta)."""
    import X_AGENT

    assert X_AGENT.llm_max_tokens > 0
    for agente in (X_AGENT.agente_dinamico, X_AGENT.codigo, X_AGENT.resumen):
        assert agente.llm.num_predict == X_AGENT.llm_max_tokens


def test_indexa_pdf_y_recupera_con_fuente_pagina_y_similitud(fake_embeddings, coleccion):
    from RAG3 import RAG

    rag = RAG(str(SYNTHETIC / "normativa_sintetica_pd.pdf"), embeddings_func=fake_embeddings)
    chunks = rag.split(rag.clean(rag.read_document()), chunk_ss=800, chunk_so=200)
    assert rag.qdrant_record(chunks, collection_name=coleccion) == len(chunks)

    consulta = rag.split_query("Comparamos cada año la PD con el test binomial por grado de rating", 600, 200)
    res = rag.process_query_chunks_v2(consulta, collection_name=coleccion)[0]
    assert len(res["relevant_chunks"]) == 3
    assert "binomial" in res["relevant_chunks"][0]
    assert res["scores"] == sorted(res["scores"], reverse=True)
    assert all(s["filename"] == "normativa_sintetica_pd.pdf" and s["page"] in (1, 2) for s in res["sources"])
    assert "Documento: normativa_sintetica_pd.pdf" in res["context"]


def test_no_recalcula_embeddings_si_la_coleccion_ya_tiene_datos(fake_embeddings, coleccion):
    from RAG3 import RAG

    rag = RAG(str(SYNTHETIC / "normativa_sintetica_pd.pdf"), embeddings_func=fake_embeddings)
    chunks = rag.split(rag.clean(rag.read_document()), 800, 200)
    rag.qdrant_record(chunks, collection_name=coleccion)
    llamadas = len(fake_embeddings.calls)
    assert rag.qdrant_record(chunks, collection_name=coleccion) == len(chunks)
    assert len(fake_embeddings.calls) == llamadas


def test_embeddings_se_envian_por_lotes(fake_embeddings, coleccion):
    """Regresión: Ollama fallaba con peticiones de ~250+ textos; ahora se trocean en lotes."""
    from X_qderant import QdrantHandler
    from langchain_core.documents import Document

    handler = QdrantHandler(embeddings_func=fake_embeddings, collection_name=coleccion)
    handler.qdrant_record([Document(page_content=f"fragmento {i}") for i in range(70)])
    assert fake_embeddings.calls == [32, 32, 6]


def test_los_errores_de_embedding_se_propagan(coleccion):
    """Regresión: antes se imprimía el error y la API devolvía 200 con resultados vacíos."""
    from X_qderant import QdrantHandler
    from langchain_core.documents import Document

    handler = QdrantHandler(embeddings_func=FakeEmbeddings(fail=True), collection_name=coleccion)
    with pytest.raises(ConnectionError):
        handler.qdrant_record([Document(page_content="texto")])
