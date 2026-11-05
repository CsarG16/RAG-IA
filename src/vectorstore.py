import json
import logging
import os
import re
import sys
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import faiss
from rank_bm25 import BM25Okapi

# Asegurar path de src
sys.path.append(str(Path(__file__).resolve().parent))

from config import CHUNKS_FILE, FAISS_INDEX_FILE, METADATA_FILE, EMBEDDING_MODEL, GEMINI_API_KEY

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


def _tokenize(text: str) -> List[str]:
    """Tokeniza texto normalizando caracteres y filtrando stop words."""
    clean = re.sub(r"[^\w\sáéíóúüñÁÉÍÓÚÜÑε\-\+]", " ", text.lower())
    words = clean.split()
    return [w for w in words if len(w) > 1 and w not in SPANISH_STOP_WORDS]


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
        idx = hash(tok) % dim
        vec[idx] += float(w)
    for i in range(len(tokens) - 1):
        bg = f"{tokens[i]}_{tokens[i+1]}"
        idx = hash(bg) % dim
        vec[idx] += 2.0
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


def reciprocal_rank_fusion(dense_indices: List[int], lexical_indices: List[int], k: int = 60, top_n: int = 4) -> List[Tuple[int, float]]:
    """
    Combina rankings de búsqueda densa y léxica usando Reciprocal Rank Fusion (RRF).
    score = sum(1 / (k + rank_i))
    """
    scores = {}
    for rank, doc_id in enumerate(dense_indices):
        scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (k + rank + 1))
    for rank, doc_id in enumerate(lexical_indices):
        scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (k + rank + 1))

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
        self.dim = 768

    def _get_gemini_client(self):
        if not self.api_key:
            return None
        try:
            from google import genai
            return genai.Client(api_key=self.api_key)
        except Exception as e:
            logger.warning(f"No se pudo inicializar genai.Client: {e}")
            return None

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Obtiene embeddings usando Gemini API si está disponible, o el vectorizador local."""
        client = self._get_gemini_client()
        if client:
            try:
                embeddings = []
                batch_size = 20
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i + batch_size]
                    res = client.models.embed_content(
                        model=EMBEDDING_MODEL,
                        contents=batch
                    )
                    if hasattr(res, "embeddings"):
                        for emb in res.embeddings:
                            embeddings.append(emb.values)
                    elif hasattr(res, "embedding"):
                        embeddings.append(res.embedding.values)

                if len(embeddings) == len(texts):
                    self.use_gemini_embeddings = True
                    arr = np.array(embeddings, dtype=np.float32)
                    return _normalize_vectors(arr)
            except Exception as e:
                logger.warning(f"Falla al generar embeddings con Gemini ({e}). Usando vectorizador local.")

        self.use_gemini_embeddings = False
        weights = getattr(self, "idf_weights", {})
        vectors = [_simple_text_vector(t, idf_weights=weights, dim=512) for t in texts]
        return np.array(vectors, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Obtiene el embedding de una sola consulta."""
        if self.use_gemini_embeddings:
            client = self._get_gemini_client()
            if client:
                try:
                    res = client.models.embed_content(
                        model=EMBEDDING_MODEL,
                        contents=query
                    )
                    if hasattr(res, "embedding"):
                        vec = np.array([res.embedding.values], dtype=np.float32)
                        return _normalize_vectors(vec)
                    elif hasattr(res, "embeddings") and len(res.embeddings) > 0:
                        vec = np.array([res.embeddings[0].values], dtype=np.float32)
                        return _normalize_vectors(vec)
                except Exception as e:
                    logger.warning(f"Error generando embedding de consulta con Gemini: {e}")

        weights = getattr(self, "idf_weights", {})
        vec = _simple_text_vector(query, idf_weights=weights, dim=512)
        return np.array([vec], dtype=np.float32)

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
                self.idf_weights = data.get("idf_weights", {})
                self.tokenized_corpus = data.get("tokenized_corpus", [])
                if self.tokenized_corpus:
                    self.bm25 = BM25Okapi(self.tokenized_corpus)
            return True
        except Exception as e:
            logger.error(f"Error cargando índice híbrido: {e}")
            return False

    def search(self, query: str, top_k: int = 4, topic_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Búsqueda Híbrida: Combina la búsqueda densa en FAISS con BM25 usando RRF.
        """
        if self.index is None or self.bm25 is None:
            if not self.load():
                self.build_and_save()

        # 1. Búsqueda Densa en FAISS
        q_vec = self.embed_query(query)
        k_search = min(top_k * 4, self.index.ntotal)
        scores_dense, indices_dense = self.index.search(q_vec, k_search)
        dense_list = [int(idx) for idx in indices_dense[0] if 0 <= idx < len(self.metadata)]

        # 2. Búsqueda Léxica en BM25
        q_tokens = _tokenize(query)
        lexical_list = []
        if self.bm25 and q_tokens:
            bm25_scores = self.bm25.get_scores(q_tokens)
            lexical_list = list(np.argsort(bm25_scores)[::-1][:k_search])

        # 3. Fusión con Reciprocal Rank Fusion (RRF)
        fused_ranks = reciprocal_rank_fusion(dense_list, lexical_list, k=60, top_n=top_k * 3)

        # 4. Formatear y filtrar resultados
        results = []
        for doc_id, rrf_score in fused_ranks:
            if doc_id >= len(self.metadata):
                continue
            chunk = dict(self.metadata[doc_id])
            chunk["score"] = float(rrf_score)

            if topic_filter:
                import unicodedata
                def _strip_accents(s):
                    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn').lower()
                src_norm = _strip_accents(chunk.get("source", ""))
                top_norm = _strip_accents(chunk.get("topic", ""))
                filt_norm = _strip_accents(topic_filter)
                # Si alguna palabra del filtro coincide
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
