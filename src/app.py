import os
import sys
import json
from pathlib import Path

# Asegurar que el directorio src esté en sys.path
sys.path.append(str(Path(__file__).resolve().parent))

import streamlit as st
import streamlit.components.v1 as components

from config import RAW_DATA_DIR, CHUNKS_FILE, GEMINI_API_KEY, DEFAULT_MODEL
from ingest import process_all_documents
from vectorstore import VectorStore
from rag_engine import RAGEngine

# Configuración de página
st.set_page_config(
    page_title="AulaIA - Tutor Académico Inteligente",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados
st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 800;
        color: #0d47a1;
        margin-bottom: 2px;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #455a64;
        margin-bottom: 18px;
    }
    .source-box {
        background-color: #f8f9fa;
        border-left: 4px solid #1E88E5;
        padding: 10px 14px;
        margin-top: 8px;
        border-radius: 4px;
        font-size: 0.9rem;
    }
    .badge-topic {
        background-color: #e3f2fd;
        color: #0d47a1;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .quick-prompt-btn {
        margin-bottom: 6px;
    }
</style>
""", unsafe_allow_html=True)


def render_content_with_mermaid(content: str):
    """Detecta bloques ```mermaid y los renderiza visualmente en HTML usando Mermaid.js."""
    if "```mermaid" not in content:
        st.markdown(content)
        return

    parts = content.split("```mermaid")
    if parts[0].strip():
        st.markdown(parts[0])

    for p in parts[1:]:
        subparts = p.split("```", 1)
        mermaid_code = subparts[0].strip()
        remaining_text = subparts[1] if len(subparts) > 1 else ""

        html_code = f"""
        <div style="background: white; padding: 15px; border-radius: 8px; border: 1px solid #e0e0e0; margin: 10px 0; overflow-x: auto;">
            <pre class="mermaid" style="text-align: center;">
{mermaid_code}
            </pre>
        </div>
        <script type="module">
            import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
            mermaid.initialize({{ startOnLoad: true, theme: 'default' }});
        </script>
        """
        components.html(html_code, height=320, scrolling=True)

        if remaining_text.strip():
            st.markdown(remaining_text)


# Cachear la instancia del motor RAG en memoria RAM
@st.cache_resource(show_spinner=False)
def get_cached_rag_engine(api_key: str):
    return RAGEngine(api_key=api_key)


# Estado de la sesión
if "messages" not in st.session_state:
    st.session_state.messages = []

if "quiz_data" not in st.session_state:
    st.session_state.quiz_data = None

if "user_answers" not in st.session_state:
    st.session_state.user_answers = {}

if "quiz_submitted" not in st.session_state:
    st.session_state.quiz_submitted = False

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

# Sidebar
with st.sidebar:
    st.title("🎓 AulaIA")
    st.caption("Asistente Académico Inteligente")
    st.markdown("---")

    # Modo de Trabajo
    app_mode = st.radio(
        "🎯 Modo de Trabajo",
        ["💬 Tutor de Estudio (Streaming)", "📝 Simulador de Quiz"],
        index=0
    )

    st.markdown("---")

    # Nivel de Explicación
    st.subheader("🎚️ Nivel de Explicación")
    explanation_level = st.select_slider(
        "Ajusta el nivel pedagógico:",
        options=["intuitivo", "formal", "codigo"],
        value="formal",
        format_func=lambda x: {
            "intuitivo": "💡 Intuitivo (Analogías)",
            "formal": "📚 Formal (Nivel Examen)",
            "codigo": "💻 Práctico (Python)"
        }[x]
    )

    # Filtro por Diapositiva / Tema
    available_files = []
    if RAW_DATA_DIR.exists():
        available_files = [f.name for f in RAW_DATA_DIR.iterdir() if f.is_file() and f.suffix.lower() in [".pptx", ".pdf", ".md"]]

    filter_options = ["Todos los temas"] + available_files
    selected_filter = st.selectbox("📂 Filtrar por presentación:", filter_options)
    topic_filter = None if selected_filter == "Todos los temas" else selected_filter

    st.markdown("---")

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("🔄 Reiniciar Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.quiz_data = None
            st.session_state.user_answers = {}
            st.session_state.quiz_submitted = False
            st.session_state.pending_prompt = None
            st.rerun()

    with col_btn2:
        if st.button("📥 Reindexar PPTs", use_container_width=True):
            with st.spinner("Procesando diapositivas..."):
                process_all_documents()
                vstore = VectorStore(api_key=GEMINI_API_KEY)
                vstore.build_and_save()
                st.cache_resource.clear()
                st.success("¡Base actualizada!")
                st.rerun()

    # Contador de diapositivas cargadas
    chunks_count = 0
    if CHUNKS_FILE.exists():
        try:
            with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
                chunks_count = len(json.load(f))
        except Exception:
            pass
    st.caption(f"📊 **Base de Conocimiento:** {chunks_count} diapositivas indexadas.")


# Instancia del motor RAG
rag_engine = get_cached_rag_engine(GEMINI_API_KEY)


# ==========================================
# MODO 1: TUTOR DE ESTUDIO (STREAMING)
# ==========================================
if "Tutor" in app_mode:
    st.markdown('<div class="main-header">🎓 AulaIA: Tutor de Estudio</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Respuestas en tiempo real con respaldo documental riguroso, citas a diapositivas y diagramas visuales automáticos.</div>', unsafe_allow_html=True)

    # Mostrar historial de mensajes anteriores
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            render_content_with_mermaid(msg["content"])
            if "sources" in msg and msg["sources"]:
                with st.expander(f"🔍 Fuentes consultadas ({len(msg['sources'])} diapositivas)"):
                    for s in msg["sources"]:
                        st.markdown(f"""
                        <div class="source-box">
                            <b>📄 {s.get('source')}</b> — <i>Diapositiva {s.get('page_or_slide')}</i> 
                            <span class="badge-topic">{s.get('topic')}</span><br>
                            <small>{s.get('text')}</small>
                        </div>
                        """, unsafe_allow_html=True)

    # Si no hay mensajes, mostrar preguntas sugeridas de 1 clic
    active_prompt = None
    if len(st.session_state.messages) == 0:
        st.markdown("##### 💡 Preguntas sugeridas para iniciar:")
        q_cols = st.columns(2)
        suggestions = [
            "¿Cuáles son los componentes de la 5-tupla formal de un autómata finito?",
            "¿Cómo funciona el método de subconjuntos para convertir un AFND a AFD?",
            "¿Qué diferencia hay entre un autómata a pila y uno finito?",
            "¿Cómo opera la cinta y el cabezal de una Máquina de Turing?"
        ]
        for i, sugg in enumerate(suggestions):
            col = q_cols[i % 2]
            if col.button(sugg, key=f"sugg_{i}", use_container_width=True):
                st.session_state.pending_prompt = sugg
                st.rerun()

    # Manejar prompt pendiente por botón sugerido
    if st.session_state.pending_prompt:
        active_prompt = st.session_state.pending_prompt
        st.session_state.pending_prompt = None

    # Input manual del chat
    chat_input = st.chat_input("Escribe tu duda sobre autómatas, máquinas de Turing o el temario...")
    if chat_input:
        active_prompt = chat_input

    # Si hay una consulta a procesar
    if active_prompt:
        # Registrar y mostrar mensaje de usuario
        st.session_state.messages.append({"role": "user", "content": active_prompt})
        with st.chat_message("user"):
            st.markdown(active_prompt)

        # Generar respuesta con streaming en vivo
        with st.chat_message("assistant"):
            stream_gen, sources = rag_engine.ask_stream(
                query=active_prompt,
                history=st.session_state.messages[:-1],
                level=explanation_level,
                topic_filter=topic_filter
            )

            # Efecto streaming en vivo palabra por palabra
            full_response = st.write_stream(stream_gen)

            # Si la respuesta contiene código Mermaid, renderizar el diagrama gráfico
            if "```mermaid" in full_response:
                st.markdown("---")
                st.caption("📊 **Diagrama de Estados Renderizado:**")
                render_content_with_mermaid(full_response)

            # Mostrar acordeón con las fuentes
            if sources:
                with st.expander(f"🔍 Fuentes consultadas ({len(sources)} diapositivas)"):
                    for s in sources:
                        st.markdown(f"""
                        <div class="source-box">
                            <b>📄 {s.get('source')}</b> — <i>Diapositiva {s.get('page_or_slide')}</i> 
                            <span class="badge-topic">{s.get('topic')}</span><br>
                            <small>{s.get('text')}</small>
                        </div>
                        """, unsafe_allow_html=True)

        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "sources": sources
        })


# ==========================================
# MODO 2: SIMULADOR DE QUIZ
# ==========================================
elif "Quiz" in app_mode:
    st.markdown('<div class="main-header">📝 Simulador de Examen y Autoevaluación</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Preguntas dinámicas extraídas estrictamente de tus diapositivas con calificación y retroalimentación inmediata.</div>', unsafe_allow_html=True)

    col_q1, col_q2 = st.columns([3, 1])
    with col_q1:
        quiz_topic = st.selectbox(
            "Selecciona el tema a evaluar:",
            [
                "Autómatas Finitos No Deterministas (AFND)",
                "Autómatas Finitos con Transiciones Épsilon",
                "Autómatas a Pila",
                "Máquinas de Turing",
                "Lenguajes y Máquinas (Unidad 1)"
            ]
        )
    with col_q2:
        num_q = st.number_input("Número de preguntas", min_value=1, max_value=5, value=3)

    if st.button("🚀 Iniciar Autoevaluación", type="primary"):
        with st.spinner(f"Elaborando cuestionario sobre '{quiz_topic}'..."):
            quiz_result = rag_engine.generate_quiz(topic=quiz_topic, num_questions=num_q)
            if "error" in quiz_result:
                st.error(quiz_result["error"])
            else:
                st.session_state.quiz_data = quiz_result
                st.session_state.user_answers = {}
                st.session_state.quiz_submitted = False
                st.rerun()

        # Renderizar Quiz Activo
        if st.session_state.quiz_data and "questions" in st.session_state.quiz_data:
            quiz = st.session_state.quiz_data
            st.markdown(f"### Cuestionario: {quiz.get('topic', 'Evaluación')}")

            with st.form("quiz_form"):
                for q in quiz["questions"]:
                    st.markdown(f"**Pregunta {q['id']}:** {q['question']}")
                    choice = st.radio(
                        "Selecciona tu respuesta:",
                        q["options"],
                        key=f"q_{q['id']}",
                        index=None
                    )
                    st.markdown("---")

                submitted = st.form_submit_button("✅ Enviar y Calificar Respuestas")
                if submitted:
                    st.session_state.quiz_submitted = True

            # Calificación y retroalimentación
            if st.session_state.quiz_submitted:
                st.markdown("## 📊 Resultados de la Evaluación")
                correct_count = 0
                total = len(quiz["questions"])

                for q in quiz["questions"]:
                    user_val = st.session_state.get(f"q_{q['id']}", "")
                    selected_letter = user_val[:1] if user_val else ""
                    is_correct = (selected_letter.upper() == q["correct_answer"].upper())

                    if is_correct:
                        correct_count += 1
                        st.success(f"**Pregunta {q['id']}: ¡Correcto! ({selected_letter})**")
                    else:
                        st.error(f"**Pregunta {q['id']}: Incorrecto.** Tu respuesta: {user_val or 'Sin responder'}. Respuesta correcta: **{q['correct_answer']}**")

                    st.markdown(f"💡 **Explicación:** {q.get('explanation')}")
                    st.caption(f"📚 **Fuente:** {q.get('source_slide')}")
                    st.markdown("---")

                score_pct = (correct_count / total) * 100
                if score_pct >= 70:
                    st.balloons()
                    st.success(f"🎉 **¡Aprobado! Puntaje: {correct_count}/{total} ({score_pct:.0f}%)**")
                else:
                    st.warning(f"⚠️ **Puntaje: {correct_count}/{total} ({score_pct:.0f}%).** Te recomendamos repasar las diapositivas indicadas.")
