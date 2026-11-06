import os
import sys
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

# Asegurar path de src
sys.path.append(str(Path(__file__).resolve().parent))

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel, Field

from config import GEMINI_API_KEY, BASE_DIR
from rag_engine import RAGEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("aulaia.api")

# Inicializar motor RAG
rag_engine = RAGEngine(api_key=GEMINI_API_KEY)
STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Precarga el índice híbrido FAISS + BM25 en memoria RAM al arrancar el servidor."""
    logger.info("Iniciando AulaIA Backend... Precargando índice híbrido en memoria.")
    if rag_engine.vector_store.index is None:
        rag_engine.vector_store.load()
    logger.info(f"Índice activo en RAM con {len(rag_engine.vector_store.metadata)} diapositivas indexadas.")
    yield
    logger.info("Apagando AulaIA Backend.")


app = FastAPI(
    title="AulaIA API",
    description="Backend FastAPI para el tutor académico RAG con SSE y FAISS/BM25",
    version="2.0.0",
    lifespan=lifespan
)

# 1. Configuración de CORS amplia para pruebas locales
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =======================================================
# MODELOS PYDANTIC DE SOLICITUD Y RESPUESTA
# =======================================================
class ChatStreamRequest(BaseModel):
    pregunta: str = Field(..., description="Pregunta del estudiante")
    nivel: str = Field("formal", description="Nivel pedagógico: 'formal', 'intuitivo' o 'codigo'")
    tema_filtro: Optional[str] = Field(None, description="Filtro opcional por nombre de presentación")
    historial: List[Dict[str, str]] = Field(default_factory=list, description="Historial de mensajes previos")


class QuizRequest(BaseModel):
    tema: str = Field("Autómatas Finitos No Deterministas (AFND)", description="Tema a evaluar")
    num_preguntas: int = Field(3, ge=1, le=5, description="Número de preguntas")


# =======================================================
# ENDPOINTS REST & SSE
# =======================================================

@app.get("/api/health")
async def health_check():
    """Verifica que el servidor, el índice FAISS y el corpus BM25 estén activos en memoria."""
    if rag_engine.vector_store.index is None:
        rag_engine.vector_store.load()

    index_loaded = rag_engine.vector_store.index is not None
    total_chunks = len(rag_engine.vector_store.metadata) if index_loaded else 0

    return {
        "status": "healthy",
        "index_loaded": index_loaded,
        "total_chunks": total_chunks,
        "hybrid_search": "FAISS + BM25Okapi (RRF)",
        "models": rag_engine.models,
        "active_key_configured": bool(rag_engine.api_key)
    }


@app.post("/api/chat/stream")
async def chat_stream(req: ChatStreamRequest):
    """
    Endpoint SSE que emite eventos:
    - event: sources (JSON con las diapositivas citadas)
    - event: token (cada token generado por el LLM)
    - event: done (fin del streaming)
    - event: error (ante fallas de cuota u operativas)
    """
    pregunta = req.pregunta.strip()
    if not pregunta:
        raise HTTPException(status_code=400, detail="La pregunta no puede estar vacía.")

    logger.info(f"Petición de chat stream: '{pregunta}' | Nivel: {req.nivel} | Filtro: {req.tema_filtro}")

    async def sse_event_generator():
        try:
            # 1. Recuperar fragmentos y abrir generador en streaming
            stream_gen, sources = rag_engine.ask_stream(
                query=pregunta,
                history=req.historial,
                level=req.nivel,
                topic_filter=req.tema_filtro
            )

            # 2. Formatear y emitir las fuentes recuperadas primero
            formatted_sources = [
                {
                    "archivo": s.get("source", "Desconocido"),
                    "slide": s.get("page_or_slide", 1),
                    "tema": s.get("topic", "General"),
                    "extracto": s.get("text", "")[:280]
                }
                for s in sources
            ]

            yield {
                "event": "sources",
                "data": json.dumps(formatted_sources, ensure_ascii=False)
            }

            # 3. Emitir token a token en tiempo real
            for token in stream_gen:
                if token:
                    yield {
                        "event": "token",
                        "data": json.dumps({"text": token}, ensure_ascii=False)
                    }

            # 4. Emitir evento de finalización
            yield {
                "event": "done",
                "data": json.dumps({"status": "complete"})
            }

        except Exception as e:
            logger.error(f"Error en stream SSE: {e}", exc_info=True)
            yield {
                "event": "error",
                "data": json.dumps({
                    "error": f"Error procesando la respuesta: {str(e)}"
                }, ensure_ascii=False)
            }

    return EventSourceResponse(sse_event_generator())


@app.post("/api/quiz/generate")
async def generate_quiz_endpoint(req: QuizRequest):
    """Genera un cuestionario de opción múltiple estructurado en JSON."""
    try:
        logger.info(f"Generando quiz para tema: '{req.tema}', preguntas: {req.num_preguntas}")
        questions = rag_engine.generate_quiz(topic=req.tema, num_questions=req.num_preguntas)
        return questions
    except Exception as e:
        logger.error(f"Error generando quiz: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/topics")
async def get_available_topics():
    """Retorna la lista de presentaciones disponibles para filtrar en el frontend."""
    from config import RAW_DATA_DIR
    if not RAW_DATA_DIR.exists():
        return []
    files = [f.name for f in RAW_DATA_DIR.iterdir() if f.is_file() and f.suffix.lower() in [".pptx", ".pdf", ".md"]]
    return sorted(files)


# =======================================================
# MONTAJE DE ARCHIVOS ESTÁTICOS Y RUTA RAÍZ
# =======================================================
@app.get("/favicon.ico", include_in_schema=False)
async def serve_favicon():
    fav_file = STATIC_DIR / "favicon.svg"
    if fav_file.exists():
        return FileResponse(fav_file, media_type="image/svg+xml")
    return JSONResponse(status_code=404, content={"message": "Not found"})

@app.get("/")
async def serve_index():
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        return JSONResponse(
            status_code=200,
            content={"message": "AulaIA FastAPI backend activo. Coloca index.html en /static para la interfaz."}
        )
    return FileResponse(index_file)

# Montar directorio /static para scripts, estilos o recursos
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


if __name__ == "__main__":
    import uvicorn
    # Inicializar índice si aún no está en memoria
    if rag_engine.vector_store.index is None:
        rag_engine.vector_store.load()
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
