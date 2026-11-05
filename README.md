# AulaIA 🎓🤖

Asistente de estudio inteligente para el curso de **Inteligencia Artificial y Teoría de la Computación**. Arquitectura cliente-servidor desacoplada con **FastAPI** (asíncrono con Server-Sent Events / SSE), **RAG Híbrido** (FAISS + BM25 con Reciprocal Rank Fusion), **Google Gemini** y una SPA moderna en **HTML5/Tailwind CSS/Mermaid.js**.

---

## 🚀 Arquitectura y Características Principales

1. **⚡ Backend Asíncrono de Alto Rendimiento (FastAPI + SSE):**
   - **Streaming en Tiempo Real:** Emisión token a token mediante Server-Sent Events (`/api/chat/stream`).
   - **RAG Híbrido Multicapa:** Combina búsqueda vectorial densa (FAISS) con recuperación léxica exacta (BM25Okapi) mediante **RRF (Reciprocal Rank Fusion)** para 100% de acierto en términos técnicos (`5-tupla`, `ε-clausura`, `AFND`, `q0`, `LIFO`, `Turing`).
   - **Ingesta Profunda de Tablas y Notas:** Extrae matrices de transición $\delta$ de PowerPoint convertidas a Markdown y notas del orador.
   - **Tolerancia a Fallos y Multi-Modelo:** Fallback automático entre `gemini-2.5-flash`, `gemini-2.5-flash-lite` y soporte de llave secundaria de contingencia.
   - **Endpoint de Autoevaluación:** `/api/quiz/generate` para generación estructurada en JSON.

2. **💻 Frontend SPA Profesional (Dark Mode):**
   - Interfaz técnica minimalista (estilo Vercel / Claude UI) sin bundlers pesados (`Tailwind CSS`, `Marked.js`, `Mermaid.js`).
   - **Diagramas Reactivos en Vivo:** Renderiza grafos de estados para autómatas y máquinas de Turing directamente con Mermaid en modo oscuro.
   - **Simulador de Quiz Modal:** Autoevaluación interactiva sin recargar la página con retroalimentación instantánea y citas de diapositivas.
   - **Selector de Nivel Pedagógico:** Ajusta las respuestas en Formal (examen), Intuitivo (analogías) o Código (Python).

---

## 🛠️ Estructura del Proyecto

```plaintext
BOT IA/
├── data/
│   ├── raw/                 # 8 archivos PPTX del curso (Clases 01 a 07.1)
│   └── processed/           # Chunks extraídos (161 slides) e índice FAISS
├── static/
│   └── index.html           # SPA Dark Mode (Tailwind + Marked + Mermaid)
├── src/
│   ├── __init__.py
│   ├── api.py               # Servidor FastAPI con CORS y SSE streaming
│   ├── config.py            # Variables de entorno y configuración
│   ├── ingest.py            # Extractor con soporte de tablas Markdown y notas
│   ├── vectorstore.py       # Búsqueda híbrida FAISS + BM25Okapi con RRF
│   └── rag_engine.py        # Motor RAG multi-modelo con streaming y citas
├── tests/
│   ├── test_cases.json      # Banco de validación
│   └── run_tests.py         # Script de pruebas automatizadas
├── iniciar_aulaia.bat       # Lanzador 1-clic (Uvicorn + navegador)
├── requirements.txt         # Dependencias sin Streamlit
├── DEMO_GUIDE.md            # Guion minuto a minuto para sustentar ante el jurado
└── README.md
```

---

## ⚡ Ejecución Rápida

1. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Lanzar la aplicación:**
   - **En Windows:** Haz doble clic en `iniciar_aulaia.bat`.
   - **Vía comando:**
     ```bash
     uvicorn api:app --app-dir src --host 127.0.0.1 --port 8000 --reload
     ```
   Se abrirá en `http://localhost:8000`.
