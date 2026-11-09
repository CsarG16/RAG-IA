import json
import logging
import os
import re
import sys
import unicodedata
import time
import zlib
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union
import faiss
from rank_bm25 import BM25Okapi

# Asegurar path de src
sys.path.append(str(Path(__file__).resolve().parent))

from config import CHUNKS_FILE, FAISS_INDEX_FILE, METADATA_FILE, EMBEDDING_MODEL, GEMINI_API_KEY

# Embeddings semánticos de Gemini (gemini-embedding-001). Si no hay API key o falla la red al
# construir el índice, se usa el vectorizador local de contingencia.
USE_GEMINI_EMBEDDINGS = os.getenv("USE_GEMINI_EMBEDDINGS", "true").lower() == "true"
EMBED_BATCH_SIZE = 45          # textos por petición al indexar
EMBED_TEXTS_PER_MINUTE = 90    # plan gratuito: 100 textos/minuto (cada diapositiva cuenta)
EMBED_MAX_CHARS = 6000         # recorte de seguridad por diapositiva
QUERY_CACHE_SIZE = 256         # consultas repetidas (demo) no gastan cuota

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SPANISH_STOP_WORDS = {
    "de", "la", "que", "el", "en", "y", "a", "los", "del", "se", "las", "por", "un", "para", "con",
    "no", "una", "su", "al", "lo", "como", "mas", "más", "pero", "sus", "le", "ya", "o", "este", "si",
    "sí", "porque", "esta", "son", "entre", "esta", "está", "cuando", "muy", "sin", "sobre", "ser",
    "tiene", "también", "tambien", "me", "hasta", "hay", "donde", "quien", "desde", "todo", "nos",
    "durante", "todos", "uno", "les", "ni", "contra", "otros", "ese", "eso", "ante", "ellos", "e",
    "esto", "antes", "algunos", "unos", "yo", "otro", "otras", "otra", "él", "tanto",
    "esa", "estos", "mucho", "quienes", "nada", "muchos", "cual", "cuál", "cuales", "cuáles", "sea",
    "poco", "ella", "estas", "es", "son", "unas", "cómo"
}


def _strip_accents(s: str) -> str:
    """Normaliza texto eliminando tildes y diacríticos de forma rápida."""
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn').lower()


# Sinónimos y siglas del curso: las diapositivas usan "AFN" mientras los alumnos escriben "AFND", etc.
TOKEN_EXPANSIONS = {
    "afnd": ["afn"],
    "afn": ["afnd"],
    "nfa": ["afn", "afnd"],
    "dfa": ["afd"],
    "pda": ["pila"],
    "ap": ["pila"],
    "mt": ["turing"],
    "ε": ["epsilon"],
    "epsilon": ["ε"],
    "afne": ["afn", "epsilon"],
}


def _stable_hash(token: str) -> int:
    """Hash determinista (hash() de Python cambia en cada proceso por PYTHONHASHSEED)."""
    return zlib.crc32(token.encode("utf-8"))


def _tokenize(text: str) -> List[str]:
    """Tokeniza texto sin tildes, filtrando stop words y expandiendo siglas del curso."""
    clean = re.sub(r"[^\w\sε\-\+]", " ", _strip_accents(text))
    tokens: List[str] = []
    for w in clean.split():
        parts = [w] + ([p for p in w.split("-") if p] if "-" in w else [])
        for part in parts:
            if (len(part) > 1 or part == "ε") and part not in SPANISH_STOP_WORDS:
                tokens.append(part)
                tokens.extend(TOKEN_EXPANSIONS.get(part, []))
    return tokens


def _normalize_vectors(v: np.ndarray) -> np.ndarray:
    """Normaliza vectores a norma L2 unitaria para similitud de coseno."""
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return v / norms


def _simple_text_vector(text: str, idf_weights: Dict[str, float] = None, dim: int = 512) -> np.ndarray:
    """Vectorizador de contingencia con filtrado de stop-words y ponderación IDF."""
    tokens = _tokenize(text)
    vec = np.zeros(dim, dtype=np.float32)
    for tok in tokens:
        w = idf_weights.get(tok, 1.5) if idf_weights else 1.0
        idx = _stable_hash(tok) % dim
        vec[idx] += float(w)
    for i in range(len(tokens) - 1):
        bg = f"{tokens[i]}_{tokens[i+1]}"
        idx = _stable_hash(bg) % dim
        vec[idx] += 2.0
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


# Peso de cada búsqueda en la fusión (RRF ponderado). Calibrado con tests/eval_rag.py.
DENSE_WEIGHT = float(os.getenv("RRF_DENSE_WEIGHT", "1.0"))
LEXICAL_WEIGHT = float(os.getenv("RRF_LEXICAL_WEIGHT", "1.0"))


def reciprocal_rank_fusion(dense_indices: List[int], lexical_indices: List[int], k: int = 60, top_n: int = 4) -> List[Tuple[int, float]]:
    """
    Combina rankings de búsqueda densa y léxica usando Reciprocal Rank Fusion (RRF).
    score = sum(1 / (k + rank_i))
    """
    scores = {}
    for rank, doc_id in enumerate(dense_indices):
        scores[doc_id] = scores.get(doc_id, 0.0) + DENSE_WEIGHT / (k + rank + 1)
    for rank, doc_id in enumerate(lexical_indices):
        scores[doc_id] = scores.get(doc_id, 0.0) + LEXICAL_WEIGHT / (k + rank + 1)

    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_docs[:top_n]


class VectorStore:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or GEMINI_API_KEY or os.getenv("GEMINI_API_KEY", "")
        self.index: Optional[faiss.Index] = None
        self.bm25: Optional[BM25Okapi] = None
        self.metadata: List[Dict[str, Any]] = []
        self.tokenized_corpus: List[List[str]] = []
        self.use_gemini_embeddings = False
        self.embedding_model_name: Optional[str] = None
        self.dim = 768
        self._query_cache: Dict[str, np.ndarray] = {}
        self._client = None

    def _get_gemini_client(self):
        if not self.api_key:
            return None
        if self._client is not None:
            return self._client
        try:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
            return self._client
        except Exception as e:
            logger.warning(f"No se pudo inicializar genai.Client: {e}")
            return None

    def _embed_batch(self, client, texts: List[str], task_type: str) -> List[List[float]]:
        """Una petición de embeddings con reintentos ante límite de cuota (429) o saturación (503)."""
        from google.genai import types
        cfg = types.EmbedContentConfig(task_type=task_type)
        for attempt in range(4):
            try:
                res = client.models.embed_content(model=EMBEDDING_MODEL, contents=texts, config=cfg)
                return [e.values for e in res.embeddings]
            except Exception as e:
                msg = str(e)
                if attempt < 3 and ("429" in msg or "503" in msg or "RESOURCE_EXHAUSTED" in msg):
                    m = re.search(r"retry in ([0-9.]+)s", msg)
                    wait = min(65.0, float(m.group(1)) + 2.0) if m else 20.0 * (attempt + 1)
                    logger.warning(f"Límite de embeddings alcanzado, reintentando en {wait:.0f}s...")
                    time.sleep(wait)
                    continue
                raise

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embeddings de las diapositivas (RETRIEVAL_DOCUMENT) con Gemini, o vectorizador local si no es posible."""
        client = self._get_gemini_client() if USE_GEMINI_EMBEDDINGS else None
        if client:
            try:
                embeddings: List[List[float]] = []
                window_start, window_count = time.time(), 0
                for i in range(0, len(texts), EMBED_BATCH_SIZE):
                    # Ritmo: no superar ~90 textos por minuto (límite del plan gratuito)
                    if window_count + EMBED_BATCH_SIZE > EMBED_TEXTS_PER_MINUTE:
                        wait = 61.0 - (time.time() - window_start)
                        if wait > 0:
                            logger.info(f"Esperando {wait:.0f}s por el límite de embeddings por minuto...")
                            time.sleep(wait)
                        window_start, window_count = time.time(), 0
                    window_count += min(EMBED_BATCH_SIZE, len(texts) - i)
                    batch = [t[:EMBED_MAX_CHARS] for t in texts[i:i + EMBED_BATCH_SIZE]]
                    embeddings.extend(self._embed_batch(client, batch, "RETRIEVAL_DOCUMENT"))
                    logger.info(f"Embeddings Gemini: {len(embeddings)}/{len(texts)}")
                if len(embeddings) == len(texts):
                    self.use_gemini_embeddings = True
                    self.embedding_model_name = EMBEDDING_MODEL
                    return _normalize_vectors(np.array(embeddings, dtype=np.float32))
            except Exception as e:
                logger.warning(f"Falla al generar embeddings con Gemini ({e}). Usando vectorizador local.")

        self.use_gemini_embeddings = False
        self.embedding_model_name = "local-hash-512"
        weights = getattr(self, "idf_weights", {})
        vectors = [_simple_text_vector(t, idf_weights=weights, dim=512) for t in texts]
        return np.array(vectors, dtype=np.float32)

    def embed_query(self, query: str) -> Optional[np.ndarray]:
        """
        Embedding de la consulta (RETRIEVAL_QUERY), con caché.
        Si el índice es de Gemini y la API falla, devuelve None: la búsqueda continúa solo con BM25
        (nunca se mezcla un vector local de otra dimensión con el índice de Gemini).
        """
        if not self.use_gemini_embeddings:
            weights = getattr(self, "idf_weights", {})
            return np.array([_simple_text_vector(query, idf_weights=weights, dim=512)], dtype=np.float32)

        key = query.strip().lower()
        if key in self._query_cache:
            return self._query_cache[key]

        client = self._get_gemini_client()
        if not client:
            return None
        try:
            vec = _normalize_vectors(np.array(self._embed_batch(client, [query], "RETRIEVAL_QUERY")[:1], dtype=np.float32))
            if len(self._query_cache) >= QUERY_CACHE_SIZE:
                self._query_cache.pop(next(iter(self._query_cache)))
            self._query_cache[key] = vec
            return vec
        except Exception as e:
            logger.warning(f"Embedding de consulta no disponible ({e}). Continuando solo con BM25.")
            return None

    def build_and_save(self, chunks_path: Path = CHUNKS_FILE):
        """Construye tanto el índice denso (FAISS) como el índice léxico (BM25)."""
        if not chunks_path.exists():
            raise FileNotFoundError(f"No existe {chunks_path}. Ejecuta src/ingest.py primero.")

        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        if not chunks:
            raise ValueError("chunks.json está vacío.")

        # Construir corpus tokenizado para BM25
        tokenized_corpus = []
        doc_freq = {}
        total_docs = len(chunks)

        for c in chunks:
            full_text = f"{c.get('source', '')} {c.get('topic', '')} {c.get('text', '')}"
            tokens = _tokenize(full_text)
            tokenized_corpus.append(tokens)
            for w in set(tokens):
                doc_freq[w] = doc_freq.get(w, 0) + 1

        self.bm25 = BM25Okapi(tokenized_corpus)
        self.tokenized_corpus = tokenized_corpus

        self.idf_weights = {
            w: float(np.log((total_docs + 1.0) / (df + 1.0)) + 1.0)
            for w, df in doc_freq.items()
        }

        texts = [
            f"{c.get('source', '')} {c.get('topic', '')} Diapositiva {c.get('page_or_slide', '')}\n{c['text']}"
            for c in chunks
        ]
        logger.info(f"Generando vectores para {len(texts)} fragmentos...")
        vectors = self.embed_texts(texts)

        dim = vectors.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(vectors)

        FAISS_INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(FAISS_INDEX_FILE))

        with open(METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump({
                "use_gemini": self.use_gemini_embeddings,
                "embedding_model": self.embedding_model_name,
                "dimension": dim,
                "idf_weights": self.idf_weights,
                "tokenized_corpus": tokenized_corpus,
                "chunks": chunks
            }, f, ensure_ascii=False, indent=2)

        self.index = index
        self.metadata = chunks
        logger.info(f"Índice Híbrido (FAISS + BM25) guardado con {index.ntotal} vectores en {FAISS_INDEX_FILE}")

    def load(self) -> bool:
        """Carga FAISS, BM25 y metadatos desde disco."""
        if not FAISS_INDEX_FILE.exists() or not METADATA_FILE.exists():
            return False

        try:
            self.index = faiss.read_index(str(FAISS_INDEX_FILE))
            with open(METADATA_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.metadata = data.get("chunks", [])
                self.use_gemini_embeddings = data.get("use_gemini", False)
                self.embedding_model_name = data.get("embedding_model") or (
                    EMBEDDING_MODEL if self.use_gemini_embeddings else "local-hash-512")
                self.idf_weights = data.get("idf_weights", {})
                self.tokenized_corpus = data.get("tokenized_corpus", [])
                if self.tokenized_corpus:
                    self.bm25 = BM25Okapi(self.tokenized_corpus)
            return True
        except Exception as e:
            logger.error(f"Error cargando índice híbrido: {e}")
            return False

    def search(self, query: str, top_k: int = 4, topic_filter: Optional[Union[str, List[str]]] = None,
               mode: str = "hybrid") -> List[Dict[str, Any]]:
        """
        Búsqueda Híbrida: Combina la búsqueda densa en FAISS con BM25 usando RRF.
        mode: "hybrid" (por defecto), "dense" (solo FAISS) o "bm25" (solo léxica). Los dos últimos se usan en la evaluación.
        """
        if self.index is None or self.bm25 is None:
            if not self.load():
                self.build_and_save()

        # Filtro por lista de presentaciones (checkboxes): se rankea todo el corpus y luego se filtra,
        # así siempre llegan top_k diapositivas de las clases elegidas.
        allowed_sources = None
        if isinstance(topic_filter, (list, tuple, set)):
            allowed_sources = {str(t) for t in topic_filter if t}
            topic_filter = None
            if not allowed_sources:
                allowed_sources = None

        k_search = self.index.ntotal if allowed_sources else min(top_k * 4, self.index.ntotal)

        # 1. Búsqueda Densa en FAISS
        dense_list = []
        if mode in ("hybrid", "dense"):
            q_vec = self.embed_query(query)
            if q_vec is not None:
                scores_dense, indices_dense = self.index.search(q_vec, k_search)
                dense_list = [int(idx) for idx in indices_dense[0] if 0 <= idx < len(self.metadata)]

        # 2. Búsqueda Léxica en BM25
        q_tokens = _tokenize(query)
        lexical_list = []
        if mode in ("hybrid", "bm25") and self.bm25 and q_tokens:
            bm25_scores = self.bm25.get_scores(q_tokens)
            lexical_list = list(np.argsort(bm25_scores)[::-1][:k_search])

        # 3. Fusión con Reciprocal Rank Fusion (RRF)
        fused_ranks = reciprocal_rank_fusion(
            dense_list, lexical_list, k=60,
            top_n=len(self.metadata) if allowed_sources else top_k * 3
        )

        # 4. Formatear y filtrar resultados
        results = []
        for doc_id, rrf_score in fused_ranks:
            if doc_id >= len(self.metadata):
                continue
            chunk = dict(self.metadata[doc_id])
            chunk["score"] = float(rrf_score)

            if allowed_sources is not None and chunk.get("source") not in allowed_sources:
                continue

            if topic_filter:
                src_norm = _strip_accents(chunk.get("source", ""))
                top_norm = _strip_accents(chunk.get("topic", ""))
                filt_norm = _strip_accents(topic_filter)
                filt_words = [w for w in filt_norm.split() if len(w) > 3]
                if filt_words and not any(w in src_norm or w in top_norm for w in filt_words):
                    continue

            results.append(chunk)
            if len(results) >= top_k:
                break

        return results


if __name__ == "__main__":
    store = VectorStore()
    store.build_and_save()
    test_q = "¿Qué es una máquina de Turing y cuáles son sus componentes?"
    hits = store.search(test_q, top_k=2)
    print(f"\n--- Prueba de Búsqueda Híbrida (FAISS + BM25 RRF) para: '{test_q}' ---")
    for h in hits:
        print(f"[{h['source']} - Diapositiva {h['page_or_slide']}] (RRF Score: {h['score']:.4f})")
        print(h['text'][:140] + "...\n")
