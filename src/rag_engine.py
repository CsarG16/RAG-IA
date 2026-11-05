import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Generator, Tuple

# Asegurar path de src
sys.path.append(str(Path(__file__).resolve().parent))

from google.genai import errors, types
from config import (
    GEMINI_API_KEY,
    GEMINI_API_KEY_BACKUP,
    DEFAULT_MODEL,
    FALLBACK_MODELS,
    TOP_K_RESULTS
)
from vectorstore import VectorStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Eres AulaIA, un tutor académico universitario de élite para la asignatura de Inteligencia Artificial y Teoría de la Computación.
Tu misión es guiar al estudiante de manera clara, pedagógica y rigurosa, basándote EXCLUSIVAMENTE en las diapositivas oficiales del curso proporcionadas en el contexto.

REGLAS CRÍTICAS DE RESPUESTA:
1. FIDELIDAD DOCUMENTAL (ANTI-ALUCINACIÓN):
   - Basa tus respuestas únicamente en los fragmentos recuperados de las diapositivas.
   - Si la consulta del estudiante NO está cubierta en los fragmentos proporcionados, responde con honestidad: "Esta consulta no se encuentra explicada en las diapositivas disponibles del curso (Clases 01 a 07.1). Te sugiero consultarlo en clase o revisar la bibliografía recomendada."
   - No inventes definiciones que contradigan las diapositivas.

2. CITAS OBLIGATORIAS:
   - Al final de tu explicación o junto a afirmaciones clave, indica explícitamente la procedencia exacta usando este formato:
     `📚 [Fuente: NombreDelArchivo.pptx - Diapositiva X]`

3. DIAGRAMAS MERMAID VISUALES (IMPRESCINDIBLE):
   - Cuando la explicación involucre Autómatas (AFD, AFND, Transiciones Épsilon, Autómatas a Pila, Máquinas de Turing) o Árboles, DEBES generar un diagrama en formato Mermaid (usando `stateDiagram-v2` o `flowchart LR`).
   - El estudiante necesita ver gráficamente los estados, transiciones y condiciones de aceptación.

4. NIVELES DE EXPLICACIÓN:
   - Si el modo es "intuitivo": Usa analogías cotidianas, paso a paso accesible y lenguaje llano.
   - Si el modo es "formal": Emplea la notación matemática rigurosa de las diapositivas (tuplas formalizadas, alfabetos, deltas, conjuntos).
   - Si el modo es "codigo": Incluye un snippet en Python limpio y comentado que simule el algoritmo o autómata.
"""


class RAGEngine:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or GEMINI_API_KEY or os.getenv("GEMINI_API_KEY", "")
        self.backup_key = GEMINI_API_KEY_BACKUP or os.getenv("GEMINI_API_KEY_BACKUP", "")
        self.vector_store = VectorStore(api_key=self.api_key)
        self.models = FALLBACK_MODELS

    def _get_client(self, key: Optional[str] = None):
        target_key = key or self.api_key
        if not target_key:
            return None
        try:
            from google import genai
            return genai.Client(api_key=target_key)
        except Exception as e:
            logger.error(f"Error iniciando cliente Gemini: {e}")
            return None

    def retrieve_context(self, query: str, top_k: int = TOP_K_RESULTS, topic_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Recupera los fragmentos más relevantes usando Búsqueda Híbrida (FAISS + BM25 RRF)."""
        return self.vector_store.search(query, top_k=top_k, topic_filter=topic_filter)

    def format_context_prompt(self, chunks: List[Dict[str, Any]]) -> str:
        """Formatea los fragmentos recuperados para inyectarlos en el prompt del LLM."""
        if not chunks:
            return "No se encontraron fragmentos relevantes en las diapositivas."

        context_str = "--- FRAGMENTOS OFICIALES DE LAS DIAPOSITIVAS DEL CURSO ---\n\n"
        for i, c in enumerate(chunks, 1):
            context_str += f"[Documento #{i}]\n"
            context_str += f"Archivo: {c.get('source', 'Desconocido')}\n"
            context_str += f"Diapositiva/Página: {c.get('page_or_slide', 'N/A')}\n"
            context_str += f"Tema: {c.get('topic', 'General')}\n"
            context_str += f"Contenido:\n{c.get('text', '')}\n"
            context_str += "-" * 40 + "\n\n"
        return context_str

    def _build_user_prompt(self, query: str, context_text: str, history: List[Dict[str, str]] = None, level: str = "formal") -> str:
        """Construye el contenido del usuario separado del system prompt."""
        conversation_context = ""
        if history:
            conversation_context = "HISTORIAL RECIENTE DE LA CONVERSACIÓN:\n"
            for msg in history[-4:]:
                role = "Estudiante" if msg["role"] == "user" else "AulaIA"
                conversation_context += f"{role}: {msg['content']}\n"
            conversation_context += "\n"

        return f"""MODO DE EXPLICACIÓN SELECCIONADO POR EL ESTUDIANTE: {level.upper()}

{context_text}

{conversation_context}
CONSULTA DEL ESTUDIANTE:
{query}

RESPUESTA PEDAGÓGICA (Incluye citas y diagramas Mermaid si corresponde):
"""

    def ask_stream(self, query: str, history: List[Dict[str, str]] = None, level: str = "formal", topic_filter: Optional[str] = None) -> Tuple[Generator[str, None, None], List[Dict[str, Any]]]:
        """
        Streaming reactivo con doble blindaje de cuota:
        1. Fallback automático de modelo: gemini-2.5-flash -> gemini-2.5-flash-lite -> gemini-1.5-flash
        2. Fallback de llave secundaria si la principal se satura.
        """
        history = history or []
        chunks = self.retrieve_context(query, top_k=TOP_K_RESULTS, topic_filter=topic_filter)
        context_text = self.format_context_prompt(chunks)
        client = self._get_client()

        if not client:
            def offline_stream():
                offline_msg = (
                    "⚠️ **Modo Demostración Local (Sin API Key de Gemini)**\n\n"
                    "Para obtener respuestas generativas con IA en tiempo real, ingresa tu `GEMINI_API_KEY` en la barra lateral o en el archivo `.env`.\n\n"
                    "**Fragmentos más relevantes encontrados en tus diapositivas:**\n\n"
                )
                for word in offline_msg.split(" "):
                    yield word + " "
                    time.sleep(0.015)
                for c in chunks:
                    chunk_text = f"- **{c['source']} (Diapositiva {c['page_or_slide']}):**\n> {c['text'][:180]}...\n\n"
                    yield chunk_text
                    time.sleep(0.03)

            return offline_stream(), chunks

        user_content = self._build_user_prompt(query, context_text, history, level)

        def resilient_gemini_stream():
            # Probar llaves: principal y luego respaldo si existe
            keys_to_try = [self.api_key]
            if self.backup_key and self.backup_key != self.api_key:
                keys_to_try.append(self.backup_key)

            gen_config = types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.2  # Rigor formal, minimiza varianza en teoremas
            )

            success = False
            for key_idx, current_api_key in enumerate(keys_to_try):
                active_client = self._get_client(current_api_key)
                for model_name in self.models:
                    for attempt in range(2):
                        try:
                            logger.info(f"Enviando consulta a {model_name} (Intento {attempt + 1})...")
                            response_stream = active_client.models.generate_content_stream(
                                model=model_name,
                                contents=user_content,
                                config=gen_config
                            )
                            has_yielded = False
                            for chunk in response_stream:
                                if hasattr(chunk, "text") and chunk.text:
                                    has_yielded = True
                                    yield chunk.text

                            if has_yielded:
                                success = True
                                return

                        except errors.APIError as e:
                            logger.warning(f"Error {e.code} en {model_name}: {e.message}. Reintentando o pasando a modelo de respaldo...")
                            if e.code in [429, 503]:
                                time.sleep(1.5)
                                continue  # Siguiente intento o siguiente modelo
                            else:
                                break  # Otro error, pasar a siguiente modelo

                        except Exception as e:
                            logger.warning(f"Falla inesperada en {model_name}: {e}")
                            break

                if success:
                    return

            if not success:
                yield "\n\n❌ **Aviso:** Se agotó la cuota por minuto del servicio o hubo interrupción de red. Espera unos segundos y vuelve a consultar."

        return resilient_gemini_stream(), chunks

    def ask(self, query: str, history: List[Dict[str, str]] = None, level: str = "formal", topic_filter: Optional[str] = None) -> Dict[str, Any]:
        """Versión sincrónica para pruebas por consola."""
        stream_gen, sources = self.ask_stream(query, history, level, topic_filter)
        full_text = "".join(list(stream_gen))
        return {
            "answer": full_text,
            "sources": sources
        }

    def generate_quiz(self, topic: Optional[str] = None, num_questions: int = 3) -> List[Dict[str, Any]]:
        """Genera un cuestionario interactivo basado en las diapositivas con fallback blindado."""
        query_topic = topic or "Autómatas Finitos y Máquinas de Turing"
        chunks = self.retrieve_context(f"Conceptos clave, definiciones y ejercicios de {query_topic}", top_k=6, topic_filter=topic)
        context_text = self.format_context_prompt(chunks)

        client = self._get_client()
        if not client:
            # Fallback offline para pruebas sin clave API
            return [
                {
                    "id": 1,
                    "pregunta": f"¿Cuál de las siguientes afirmaciones sobre {query_topic} es correcta según las diapositivas?",
                    "opciones": [
                        "A) Un AFD tiene exactamente una transición por cada símbolo del alfabeto en cada estado.",
                        "B) Un AFND no permite transiciones con la cadena vacía épsilon.",
                        "C) Las Máquinas de Turing solo pueden moverse hacia la derecha en la cinta.",
                        "D) Los autómatas a pila no disponen de memoria adicional."
                    ],
                    "respuesta_correcta": "A",
                    "explicacion": "Un AFD es determinista porque su función de transición delta asigna a cada par (estado, símbolo) exactamente un único estado siguiente.",
                    "fuente": "03 Introducción a la Teoría de los Autómatas 2026-2.pptx - Diapositiva 9"
                }
            ]

        prompt = f"""Actúa como un evaluador académico para el curso de Inteligencia Artificial.
Basándote EXCLUSIVAMENTE en el siguiente contexto de las diapositivas, genera un examen de autoevaluación de {num_questions} preguntas de opción múltiple.

{context_text}

INSTRUCCIONES IMPORTANTES:
Responde ÚNICAMENTE con una lista JSON válida (sin formato Markdown adicional, solo el array JSON puro []):
[
  {{
    "id": 1,
    "pregunta": "¿Texto de la pregunta?",
    "opciones": ["A) Opción 1", "B) Opción 2", "C) Opción 3", "D) Opción 4"],
    "respuesta_correcta": "A",
    "explicacion": "Explicación detallada de por qué es correcta.",
    "fuente": "NombreDelArchivo.pptx - Diapositiva X"
  }}
]
"""
        config = types.GenerateContentConfig(temperature=0.1)

        for model_name in self.models:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=config
                )
                raw_text = response.text.strip()
                if raw_text.startswith("```"):
                    raw_text = raw_text.split("```")[1]
                    if raw_text.startswith("json"):
                        raw_text = raw_text[4:]
                data = json.loads(raw_text.strip())

                # Normalizar si vino en {"questions": [...]} o como lista directa
                questions_list = data if isinstance(data, list) else data.get("questions", [])
                normalized = []
                for idx, q in enumerate(questions_list, start=1):
                    normalized.append({
                        "id": q.get("id", idx),
                        "pregunta": q.get("pregunta") or q.get("question", "Pregunta sin título"),
                        "opciones": q.get("opciones") or q.get("options", []),
                        "respuesta_correcta": q.get("respuesta_correcta") or q.get("correct_answer", "A"),
                        "explicacion": q.get("explicacion") or q.get("explanation", ""),
                        "fuente": q.get("fuente") or q.get("source_slide", "Diapositivas del curso")
                    })
                return normalized
            except errors.APIError as e:
                logger.warning(f"Error {e.code} en quiz con {model_name}: {e.message}. Cambiando de modelo...")
                time.sleep(1.0)
                continue
            except Exception as e:
                logger.error(f"Error parseando quiz con {model_name}: {e}")
                continue

        return []
