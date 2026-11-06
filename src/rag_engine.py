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

SYSTEM_PROMPT = """Eres "AulaIA", tutor académico de Inteligencia Artificial y Teoría de la Computación (VII Ciclo). Tu misión es resolver dudas estudiantiles garantizando que cada afirmación esté estrictamente respaldada por las diapositivas oficiales provistas en el bloque <contexto_documental>.

---

### REGLAS DE ORO Y ANTI-ALUCINACIÓN
1. FIDELIDAD ESTRICTA: Responde explicando con claridad y rigor técnico los conceptos solicitados a partir de los fragmentos en <contexto_documental>. Comprende la equivalencia de términos técnicos académicos del curso (por ejemplo, "método de subconjuntos", "construcción de subconjuntos" o "algoritmo de subconjuntos" se refiere al procedimiento de conversión de AFND a AFD detallado en las diapositivas). Sintetiza los pasos, definiciones formales, tablas de transición y ejemplos que aparezcan en los fragmentos.
2. FUERA DE DOMINIO: ÚNICAMENTE si la consulta del estudiante trata sobre materias totalmente ajenas al curso (como redes neuronales, deep learning, visión computacional, historia universal, cocina o programación web) y no guarda relación con los temas de teoría de la computación, autómatas o lenguajes del material, responde exactamente:
   "Ese tema no se encuentra registrado en las diapositivas oficiales del curso (Clases 01 a 07.1). Por favor, formula una duda relacionada con autómatas, lenguajes, máquinas de Turing o el temario del ciclo."
3. CITAS OBLIGATORIAS: Cada respuesta debe concluir con un bloque de fuentes en este formato exacto:
   **Fuentes consultadas:**
   - [Fuente: {nombre_archivo.pptx} - Diapositiva {numero_slide}]

---

### MODOS PEDAGÓGICOS (Aplica según el parámetro recibido)
- Nivel "formal": Emplea rigor matemático formal con notación limpia.
  * REGLA DIDÁCTICA OBLIGATORIA: NUNCA coloques una tupla o fórmula formal (como la 5-tupla de un AFD, la 7-tupla de un Autómata a Pila o la 7-tupla de una Máquina de Turing M = (Q, Σ, Γ, δ, q0, B, F)) de forma aislada sin explicarla. Siempre debes desglosar inmediatamente después qué significa cada letra o símbolo en español claro (por ejemplo: Q = conjunto de estados, Σ = alfabeto de entrada, Γ = símbolos de cinta/pila, δ = función de transición, q0 = estado de inicio, B = símbolo blanco, Z0 = fondo de pila, F = estados de aceptación).
- Nivel "intuitivo": Emplea analogías sencillas de la vida diaria, lenguaje natural amigable y explicaciones directas (por ejemplo: la memoria de pila LIFO explicada como una torre de platos; la cinta de Turing como una tira infinita de casete con un cabezal de lectura/escritura). NUNCA sueltes tuplas matemáticas abstractas como M = (Q, Σ, Γ, ...) en modo intuitivo; en su lugar, explica en palabras sencillas cuáles son sus componentes (los estados, la cinta, las reglas, etc.). El objetivo es que cualquier estudiante comprenda el concepto a la primera sin confusión.
- Nivel "codigo": Proporciona implementaciones funcionales y limpias en Python 3 que simulen el comportamiento del autómata, pila o máquina de Turing discutida, explicando con comentarios cada función.

---

### FORMATO DE DIAGRAMAS MERMAID
Cuando la consulta implique el diseño o la transición de un autómata finito, autómata a pila o máquina de Turing, incluye un bloque de código Mermaid con la sintaxis exacta:

```mermaid
stateDiagram-v2
    direction LR
    [*] --> q0
    q0 --> q1 : 0
    q1 --> q1 : 1
    q1 --> [*]
```
Usa únicamente sintaxis válida para grafos de estados (stateDiagram-v2), indicando claramente el estado inicial [*] y los estados de aceptación.

---

### REGLA ESTRICTA ANTI-SALUDOS REPETITIVOS (CERO RELLENO)
1. VE DIRECTO AL GRANO: Comienza tu respuesta de inmediato explicando la respuesta conceptual, técnica o práctica.
2. PROHIBIDO SALUDAR Y PRESENTARSE EN CADA TURNO:
   - NUNCA digas frases robóticas de relleno como: "¡Hola! Qué gusto saludarte", "Soy AulaIA, tu tutor académico...", "Como me pediste el nivel...", "Vamos a revisar este tema...", "¡Hola! Qué bueno que quieras...".
   - Si el estudiante hace una pregunta (ej. "¿Cómo funciona un AFND?"), responde DIRECTAMENTE: "Un Autómata Finito No Determinista (AFND) es..." sin ningún preámbulo ni saludo.
   - Solo se tolera un breve saludo inicial ("¡Hola!") si es estrictamente el primer mensaje de la sesión. Si ya hay conversación previa o preguntas en el historial, jamás saludes.

---

### ESTRUCTURA DE LA RESPUESTA
1. Explicación directa y estructurada de la duda según el nivel solicitado (sin saludos de relleno).
2. Diagrama visual Mermaid o bloque de código (si aplica).
3. Bloque final de fuentes consultadas con archivo y diapositiva.
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
        """Formatea los fragmentos recuperados para inyectarlos en el bloque <contexto_documental>."""
        if not chunks:
            return "No se encontraron fragmentos relevantes en las diapositivas."

        context_lines = []
        for c in chunks:
            context_lines.append(
                f"[Archivo: {c.get('source', 'Desconocido')} | Diapositiva: {c.get('page_or_slide', 'N/A')}]\n{c.get('text', '')}"
            )
        return "\n\n".join(context_lines)

    def _build_user_prompt(self, query: str, context_text: str, history: List[Dict[str, str]] = None, level: str = "formal") -> str:
        """Construye el contenido del usuario separado del system prompt usando etiquetas XML."""
        conversation_context = ""
        is_followup = False
        if history and len(history) > 0:
            is_followup = True
            conversation_context = "HISTORIAL PREVIO:\n"
            for msg in history[-6:]:
                role = "Estudiante" if msg.get("role") == "user" else "AulaIA"
                clean_content = msg.get("content", "").strip()[:250]
                conversation_context += f"{role}: {clean_content}\n"
            conversation_context += "\n"

        flow_instruction = (
            "[IMPORTANTE: Conversación en curso. NO saludes, NO digas '¡Hola!', NO te presentes como AulaIA. Ve DIRECTAMENTE a responder la duda.]\n"
            if is_followup else
            "[IMPORTANTE: Ve directo a la materia. Si decides incluir un saludo inicial por ser la primera vez, que sea de máximo 3 palabras (ej. '¡Hola!'). No digas 'Soy tu tutor AulaIA...' ni te presentes largamente.]\n"
        )

        return f"""Nivel solicitado: {level}

<contexto_documental>
{context_text}
</contexto_documental>

{conversation_context}{flow_instruction}Pregunta del estudiante:
{query}"""

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
                temperature=0.15  # Máxima consistencia técnica y rigor formal
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
                            logger.warning(f"Error {e.code} en {model_name}: {e.message}. Pasando a modelo de respaldo...")
                            err_str = str(e).lower()
                            if e.code == 429 and ("perday" in err_str or "exceeded your current quota" in err_str):
                                break  # Límite diario del modelo agotado, cambiar de modelo inmediatamente
                            elif e.code in [429, 503]:
                                time.sleep(0.5)
                                continue
                            else:
                                break

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
