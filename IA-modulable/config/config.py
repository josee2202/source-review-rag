import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

# Cargar las variables de entorno del archivo .env
load_dotenv()

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
if os.getenv("QDRANT_LOCATION"):
    cfg_param['qdrant']['location'] = os.getenv("QDRANT_LOCATION")
