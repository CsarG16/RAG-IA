import os
from pathlib import Path
from dotenv import load_dotenv

# Cargar variables de entorno desde .env
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Rutas de datos procesados
CHUNKS_FILE = PROCESSED_DATA_DIR / "chunks.json"
FAISS_INDEX_FILE = PROCESSED_DATA_DIR / "faiss.index"
METADATA_FILE = PROCESSED_DATA_DIR / "metadata.json"
CHATS_DB_FILE = DATA_DIR / "chats.db"

# Credenciales de API (Principal y Respaldo de emergencia)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_API_KEY_BACKUP = os.getenv("GEMINI_API_KEY_BACKUP", "")

# Modelos en orden de prioridad para tolerancia a fallos (Ultra-estables y rápidos)
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
FALLBACK_MODELS = list(dict.fromkeys([
    DEFAULT_MODEL,
    "gemini-3.5-flash-lite",
    "gemini-flash-lite-latest"
]))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")

# Parámetros de Recuperación
TOP_K_RESULTS = 6
MAX_HISTORY_MESSAGES = 6          # memoria de corto plazo: últimos 3 turnos (pregunta + respuesta)
HISTORY_USER_CHARS = 400          # recorte por mensaje del estudiante dentro del prompt
HISTORY_ASSISTANT_CHARS = 700     # recorte por respuesta previa del tutor dentro del prompt
