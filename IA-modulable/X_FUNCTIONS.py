import re
import unicodedata


class EntradaInvalida(ValueError):
    """Error causado por lo que aporta el usuario; la API lo devuelve como 400."""


# * Funciones para limpieza
# * ===========================================================================

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
    # Una línea con solo espacios cuenta como línea vacía: mantiene el límite de párrafo
    text = re.sub(r" \n", "\n", text)
    text = re.sub(r"\n ", "\n", text)
    # Eliminar líneas vacías excesivas (más de 2)
    text = re.sub(r"(\n\s*){3,}", "\n\n", text)
    return text.strip()

# * Función completa de limpieza
# * ===========================================================================
def clean_text(text: str) -> str:
    """
    Pipeline completo de limpieza de texto:
    - Unicode NFC
    - Eliminación de control chars
    - Corrección de hyphenation
    - Normalización de puntuación
    - Colapso de whitespace
    """
    text = normalize_unicode(text)
    text = remove_non_printables(text)
    text = fix_hyphenation(text)
    text = normalize_punctuation(text)
    text = collapse_whitespace(text)

    return text
