import os
from pathlib import Path
from dotenv import load_dotenv

# Cargar variables de entorno desde .env
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Archivos de datos procesados
CHUNKS_FILE = PROCESSED_DATA_DIR / "chunks.json"
FAISS_INDEX_FILE = PROCESSED_DATA_DIR / "faiss.index"
METADATA_FILE = PROCESSED_DATA_DIR / "metadata.json"

# Claves y modelos
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_API_KEY_BACKUP = os.getenv("GEMINI_API_KEY_BACKUP", "")

# Soporte para Streamlit Community Cloud Secrets
try:
    import streamlit as st
    if not GEMINI_API_KEY and hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
        GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    if not GEMINI_API_KEY_BACKUP and hasattr(st, "secrets") and "GEMINI_API_KEY_BACKUP" in st.secrets:
        GEMINI_API_KEY_BACKUP = st.secrets["GEMINI_API_KEY_BACKUP"]
except Exception:
    pass

DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
FALLBACK_MODELS = [
    DEFAULT_MODEL,
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite"
]
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")

# Configuración de recuperación
TOP_K_RESULTS = 4
MAX_HISTORY_MESSAGES = 10
