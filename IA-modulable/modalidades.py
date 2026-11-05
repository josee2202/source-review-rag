"""Las tres modalidades de la interfaz.

Comparten el mismo pipeline: solo cambian las etiquetas, el número de referencias admitidas y el
rol que recibe el análisis opcional (prompt.py). Este módulo no tiene dependencias para que también
pueda importarlo el front-end.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Modalidad:
    id: str
    nombre: str
    descripcion: str
    etiqueta_referencia: str
    ayuda_referencia: str
    etiqueta_revisado: str
    ayuda_revisado: str
    multiples_referencias: bool = False


MODALIDADES = {m.id: m for m in (
    Modalidad(
        id="text_vs_text",
        nombre="Texto vs. texto",
        descripcion="Revisa un texto frente a otro que eliges como referencia. Incluye noticia vs. noticia "
                    "y original vs. resumen. Ser la referencia no implica que ese texto sea el correcto.",
        etiqueta_referencia="Texto de referencia",
        ayuda_referencia="El texto o documento en el que se buscarán pasajes relacionados.",
        etiqueta_revisado="Texto a revisar",
        ayuda_revisado="Se divide en fragmentos; para cada uno se buscan pasajes en la referencia.",
    ),
    Modalidad(
        id="report_vs_news",
        nombre="Informe vs. noticia",
        descripcion="Revisa una noticia frente al informe, comunicado o documento original que comenta.",
        etiqueta_referencia="Informe o documento original",
        ayuda_referencia="El informe, comunicado o documento original.",
        etiqueta_revisado="Noticia a revisar",
        ayuda_revisado="Se divide en fragmentos; para cada uno se buscan pasajes en el informe.",
    ),
    Modalidad(
        id="news_vs_newsletter",
        nombre="Noticias vs. newsletter",
        descripcion="Revisa una newsletter frente a una o varias noticias. La búsqueda se hace directamente "
                    "en todas las noticias aportadas y cada pasaje indica de cuál procede.",
        etiqueta_referencia="Noticias de referencia",
        ayuda_referencia="Una o varias noticias. Sube un archivo por noticia para saber de cuál procede cada pasaje.",
        etiqueta_revisado="Newsletter a revisar",
        ayuda_revisado="Se divide en fragmentos; para cada uno se buscan pasajes en todas las noticias.",
        multiples_referencias=True,
    ),
)}
