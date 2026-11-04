from langchain_community.document_loaders import TextLoader

from pathlib import Path
from typing import Union


# * 1. - Clase diseñada para leer archivos SAS, basada en 'PythonLoader'
# * ===========================================================================
class SASLoader(TextLoader):
    """Load SAS files, respecting any non-default encoding if specified."""

    def __init__(self, file_path: Union[str, Path]):
        """Initialize with a file path.
        
        Args:
            file_path: The path to the SAS file to load.
        """
        self.file_path = file_path
        encoding = self.detect_encoding(file_path)
        
        # Llamar al constructor de la clase base TextLoader con la codificación detectada
        super().__init__(file_path=file_path, encoding=encoding)

    def detect_encoding(self, file_path: Union[str, Path]):
        """Detect encoding by trying multiple options."""
        encodings_to_try = ['utf-8', 'latin-1', 'ISO-8859-1']
        for encoding in encodings_to_try:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    f.read()  # Tratar de leer el archivo
                return encoding  # Si no lanza error, devolver la codificación exitosa
            except (UnicodeDecodeError, IOError):
                continue  # Si falla, intentar la siguiente codificación
        raise RuntimeError(f"Unable to detect encoding for {file_path}")  # Si todas fallan

    def load(self):
        """Override the load method to handle the SAS files."""
        return super().load()


# * 2. - Funciones para limpieza 
# * ===========================================================================

import re
import unicodedata

def normalize_unicode(text: str) -> str:
    # Unicode normalization NFC
    return unicodedata.normalize('NFC', text)

def remove_non_printables(text: str) -> str:
    # Eliminar caracteres de control (excepto \n, \t)
    return re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]+", "", text)

def fix_hyphenation(text: str) -> str:
    # Une palabras partidas al final de línea: 'informa-\nción' -> 'información'
    return re.sub(r"(\w+)-\n(\w+)", lambda m: m.group(1) + m.group(2), text)

def normalize_punctuation(text: str) -> str:
    # Unificar comillas y guiones
    replacements = {
        '“': '"', '”': '"', '‘': "'", '’': "'",
        '—': '-', '–': '-', '…': '...'
    }
    for src, target in replacements.items():
        text = text.replace(src, target)
    return text

def collapse_whitespace(text: str) -> str:
    # Normalizar saltos de línea y espacios
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    # Colapsar espacios múltiples
    text = re.sub(r"[ \t]+", " ", text)
    # Eliminar líneas vacías excesivas (más de 2)
    text = re.sub(r"(\n\s*){3,}", "\n\n", text)
    return text.strip()

# * 2.1 - Función completa de limpieza 
# * ===========================================================================
def clean_text(text: str) -> str:
    """
    Pipeline completo de limpieza de texto:
    - Unicode NFC
    - Eliminación de control chars
    - Corrección de hyphenation
    - Normalización de puntuación
    - Colapso de whitespace
    - Eliminación de headers y footers automáticos
    """


    text = normalize_unicode(text)
    text = remove_non_printables(text)
    text = fix_hyphenation(text)
    text = normalize_punctuation(text)
    text = collapse_whitespace(text)
    
    return text 


# * 3. - Exportar a excel 
# * ===========================================================================

from pathlib import Path
from openpyxl import Workbook, load_workbook
from openpyxl.utils import get_column_letter

def export_to_excel(
    results: list[dict],
    output_path: str,
    sheet_name: str = "Resultados"
) -> str:

    path = Path(output_path)
    # Fuerza .xlsx
    if path.suffix.lower() not in (".xlsx", ".xlsm", ".xltx", ".xltm"):
        path = path.with_suffix(".xlsx")

    include_resp = any("respuesta" in r for r in results)
    headers = ["Query Chunk", "Chunk 1", "Chunk 2", "Chunk 3"]
    if include_resp:
        headers.append("Respuesta")

    # Crear o cargar
    if not path.exists():
        wb = Workbook()
        ws = wb.active
        ws.title = sheet_name
        ws.append(headers)
    else:
        wb = load_workbook(path)
        ws = wb[sheet_name] if sheet_name in wb.sheetnames else wb.create_sheet(sheet_name)
        if ws.max_row == 0:
            ws.append(headers)
        else:
            ws.append([""] * len(headers))

    # Añadir filas
    for res in results:
        chunk = res.get("chunk", "")
        rel   = res.get("relevant_chunks", [])
        row   = [
            chunk,
            rel[0] if len(rel) > 0 else "",
            rel[1] if len(rel) > 1 else "",
            rel[2] if len(rel) > 2 else ""
        ]
        if include_resp:
            row.append(res.get("respuesta", ""))
        ws.append(row)

    # Ajuste de anchos
    for idx, title in enumerate(headers, start=1):
        col = get_column_letter(idx)
        ws.column_dimensions[col].width = max(len(title), 20)

    wb.save(path)
    print(f"✅ Resultados añadidos exitosamente a: {path}")
    return str(path)





# ? FUTURAS PROPUESTAS --- Eliminar Headers y Footers 
# ? ===========================================================================

def detect_repetitive_lines(lines: list[str], min_occurrences: int = 3) -> set[str]:
    """Detecta líneas completas que aparecen al menos min_occurrences veces."""
    
    from collections import Counter

    filtered = [line.strip() for line in lines if len(line.strip()) < 100 and line.strip()]
    counts = Counter(filtered)
    return {line for line, cnt in counts.items() if cnt >= min_occurrences}

def estimate_page_size(lines: list[str], min_pages: int = 3) -> int:
    """Estima líneas por página usando índices de números de página y mediana."""
    idxs = [i for i, line in enumerate(lines) if re.fullmatch(r"\d+", line.strip())]
    if len(idxs) < min_pages:
        total = len(lines)
        return max(1, total // min_pages)
    diffs = [j - i for i, j in zip(idxs, idxs[1:])]
    mid = len(diffs) // 2
    return sorted(diffs)[mid] if diffs else total // min_pages

def remove_headers_footers(text: str) -> str:
    """
    Elimina:
      1. Números de página (líneas solo dígitos)
      2. Líneas anteriores y posteriores a los números de página
      3. Líneas genéricas repetitivas
    """
    lines = text.split('\n')
    # Detección repetitiva genérica
    repetitive = detect_repetitive_lines(lines)
    # Índices de números de página
    page_idxs = [i for i, line in enumerate(lines) if re.fullmatch(r"\d+", line.strip())]
    # Índices a eliminar por posición
    remove_idxs = set()
    for idx in page_idxs:
        remove_idxs.add(idx)
        if idx - 1 >= 0:
            remove_idxs.add(idx - 1)
        if idx + 1 < len(lines):
            remove_idxs.add(idx + 1)
    # Reconstruir excluyendo líneas ruidosas
    cleaned = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if i in remove_idxs:
            continue
        if stripped in repetitive:
            continue
        if not stripped:
            continue
        cleaned.append(line)
    return "\n".join(cleaned)