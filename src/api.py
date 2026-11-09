import os
import sys
import json
import uuid
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
from starlette.concurrency import iterate_in_threadpool, run_in_threadpool
from pydantic import BaseModel, Field

from config import GEMINI_API_KEY, BASE_DIR, MAX_HISTORY_MESSAGES
from rag_engine import RAGEngine, STREAM_RESET
import database

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
    temas_filtro: Optional[List[str]] = Field(None, description="Lista de presentaciones seleccionadas (checkboxes)")
    historial: List[Dict[str, str]] = Field(default_factory=list, description="Historial de mensajes previos")
    conversation_id: Optional[str] = Field(None, description="ID único de la conversación en SQLite")


class QuizRequest(BaseModel):
    tema: str = Field("Autómatas Finitos No Deterministas (AFND)", description="Tema a evaluar")
    num_preguntas: int = Field(5, ge=1, le=15, description="Número de preguntas (5, 10 o 15)")
    dificultad: str = Field("medio", description="Nivel de dificultad: 'facil', 'medio' o 'dificil'")


class RenameChatRequest(BaseModel):
    title: str = Field(..., description="Nuevo título de la conversación")


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
        "embedding_model": rag_engine.vector_store.embedding_model_name,
        "semantic_embeddings": rag_engine.vector_store.use_gemini_embeddings,
        "models": rag_engine.models,
        "active_key_configured": bool(rag_engine.api_key)
    }


# =======================================================
# GESTIÓN DE HISTORIAL DE CHATS (SQLITE)
# =======================================================

@app.get("/api/chats")
async def list_chats():
    """Retorna la lista cronológica de conversaciones guardadas en SQLite."""
    try:
        return database.list_conversations()
    except Exception as e:
        logger.error(f"Error listando conversaciones: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chats")
async def create_chat():
    """Crea una nueva conversación explícita en SQLite."""
    try:
        return database.create_conversation()
    except Exception as e:
        logger.error(f"Error creando conversación: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/chats/{conversation_id}")
async def get_chat(conversation_id: str):
    """Retorna todos los mensajes y fuentes de una conversación específica."""
    conv = database.get_conversation(conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversación no encontrada.")
    return conv


@app.delete("/api/chats/{conversation_id}")
async def delete_chat(conversation_id: str):
    """Elimina una conversación y sus mensajes asociados."""
    success = database.delete_conversation(conversation_id)
    if not success:
        raise HTTPException(status_code=404, detail="Conversación no encontrada.")
    return {"status": "deleted", "id": conversation_id}


@app.patch("/api/chats/{conversation_id}")
async def rename_chat(conversation_id: str, req: RenameChatRequest):
    """Renombra el título de una conversación."""
    success = database.rename_conversation(conversation_id, req.title)
    if not success:
        raise HTTPException(status_code=404, detail="Conversación no encontrada.")
    return {"status": "renamed", "id": conversation_id, "title": req.title}


@app.post("/api/chat/stream")
async def chat_stream(req: ChatStreamRequest):
    """
    Endpoint SSE que emite eventos:
    - event: conversation (ID de la conversación persistida)
    - event: sources (JSON con las diapositivas citadas)
    - event: token (cada token generado por el LLM)
    - event: done (fin del streaming con confirmación de persistencia)
    - event: error (ante fallas de cuota u operativas)
    """
    pregunta = req.pregunta.strip()
    if not pregunta:
        raise HTTPException(status_code=400, detail="La pregunta no puede estar vacía.")

    # Asegurar o generar ID de conversación
    cid = req.conversation_id or str(uuid.uuid4())

    # Memoria de la conversación: la fuente de verdad es SQLite (servidor).
    # El historial enviado por el navegador solo se usa si la conversación aún no existe en la BD.
    history = []
    if req.conversation_id:
        try:
            history = database.get_recent_messages(cid, limit=MAX_HISTORY_MESSAGES)
        except Exception as db_err:
            logger.warning(f"No se pudo leer el historial desde SQLite: {db_err}")
    if not history:
        history = req.historial[-MAX_HISTORY_MESSAGES:]

    # Persistir mensaje del estudiante en SQLite
    try:
        database.add_message(cid, "user", pregunta)
    except Exception as db_err:
        logger.warning(f"No se pudo guardar el mensaje de usuario en SQLite: {db_err}")

    logger.info(f"Petición de chat stream: '{pregunta}' | ConvID: {cid} | Nivel: {req.nivel} | Memoria: {len(history)} msgs | Filtro: {req.temas_filtro or req.tema_filtro}")

    async def sse_event_generator():
        try:
            # Emitir ID de conversación inmediatamente para sincronizar frontend
            yield {
                "event": "conversation",
                "data": json.dumps({"conversation_id": cid}, ensure_ascii=False)
            }

            # 1. Recuperar fragmentos y abrir generador en streaming
            # Búsqueda (embedding + FAISS + BM25) en un hilo aparte para no bloquear a otros usuarios
            stream_gen, sources = await run_in_threadpool(
                rag_engine.ask_stream,
                query=pregunta,
                history=history,
                level=req.nivel,
                topic_filter=req.temas_filtro or req.tema_filtro
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

            # 3. Emitir token a token en tiempo real y acumular para SQLite
            tokens_accumulated = []
            # Los tokens de Gemini se leen en un hilo aparte: el servidor sigue atendiendo otras peticiones
            async for token in iterate_in_threadpool(stream_gen):
                if token == STREAM_RESET:
                    # Gemini falló a mitad de la respuesta y se reintenta: el cliente descarta lo parcial
                    tokens_accumulated = []
                    yield {"event": "reset", "data": json.dumps({"reason": "retry"})}
                    continue
                if token:
                    tokens_accumulated.append(token)
                    yield {
                        "event": "token",
                        "data": json.dumps({"text": token}, ensure_ascii=False)
                    }

            # 4. Guardar respuesta final en SQLite
            full_response_text = "".join(tokens_accumulated).strip()
            if full_response_text:
                try:
                    database.add_message(cid, "assistant", full_response_text, sources=formatted_sources)
                except Exception as save_err:
                    logger.warning(f"Error al guardar respuesta en SQLite: {save_err}")

            # 5. Emitir evento de finalización
            yield {
                "event": "done",
                "data": json.dumps({"status": "complete", "conversation_id": cid})
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
    """Genera un cuestionario de opción múltiple estructurado en JSON con dificultad parametrizada."""
    try:
        logger.info(f"Generando quiz para tema: '{req.tema}', preguntas: {req.num_preguntas}, dificultad: {req.dificultad}")
        questions = await run_in_threadpool(
            rag_engine.generate_quiz,
            topic=req.tema,
            num_questions=req.num_preguntas,
            difficulty=req.dificultad
        )
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
@app.get("/landing")
async def serve_landing():
    """Sirve la Landing Page interactiva de AulaIA."""
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        return JSONResponse(
            status_code=200,
            content={"message": "AulaIA FastAPI backend activo. Coloca index.html en /static para la interfaz."}
        )
    return FileResponse(index_file)

@app.get("/chat")
@app.get("/app")
async def serve_chat():
    """Sirve la aplicación interactiva de Chat y Quizzes de AulaIA."""
    chat_file = STATIC_DIR / "chat.html"
    if not chat_file.exists():
        chat_file = STATIC_DIR / "index.html"
    return FileResponse(chat_file)

# Montar directorio /static para scripts, estilos o recursos
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


if __name__ == "__main__":
    import uvicorn
    # Inicializar índice si aún no está en memoria
    if rag_engine.vector_store.index is None:
        rag_engine.vector_store.load()
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
