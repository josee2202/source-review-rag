import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]

# Solo se lee el .env de este proyecto: no se busca en carpetas superiores
load_dotenv(ROOT / ".env")

# Descargar la configuración (ruta relativa a este archivo, no al directorio de trabajo)
with open(Path(__file__).resolve().parent / 'config.yaml', 'r') as config_file:
    cfg_param = yaml.safe_load(config_file)

# Variables de entorno opcionales que sobrescriben config.yaml (ver .env.example)
if os.getenv("OLLAMA_HOST_NAME"):
    cfg_param['ollama']['host'] = os.getenv("OLLAMA_HOST_NAME")
if os.getenv("OLLAMA_PORT"):
    cfg_param['ollama']['port'] = int(os.getenv("OLLAMA_PORT"))
if os.getenv("EMBED_MODEL"):
    cfg_param['llm']['embed']['name'] = os.getenv("EMBED_MODEL")
if os.getenv("LLM_MODEL"):
    cfg_param['llm']['text']['name'] = os.getenv("LLM_MODEL")
if os.getenv("LLM_MAX_TOKENS"):
    cfg_param['llm']['text']['max_tokens'] = int(os.getenv("LLM_MAX_TOKENS"))
if os.getenv("LLM_TIMEOUT_S"):
    cfg_param['llm']['text']['timeout_s'] = int(os.getenv("LLM_TIMEOUT_S"))
if os.getenv("LLM_NUM_THREAD"):
    cfg_param['llm']['text']['num_thread'] = int(os.getenv("LLM_NUM_THREAD"))
if os.getenv("QDRANT_LOCATION"):
    cfg_param['qdrant']['location'] = os.getenv("QDRANT_LOCATION")

# Una ruta local relativa se resuelve siempre dentro de este proyecto, sea cual sea el directorio de trabajo
_location = cfg_param['qdrant']['location']
if _location != ":memory:" and not _location.startswith(("http://", "https://")) and not os.path.isabs(_location):
    cfg_param['qdrant']['location'] = str(ROOT / _location)
