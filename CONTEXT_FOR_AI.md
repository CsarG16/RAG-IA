# CONTEXTO DEL PROYECTO: AULAIA (Para Asistente de IA)

> **Instrucción para la IA:** Actúa como mi arquitecto de software y asesor técnico de Inteligencia Artificial. Lee atentamente la siguiente ficha técnica y estado actual de mi proyecto para ayudarme a tomar decisiones técnicas, optimizar la arquitectura y preparar mi sustentación académica.

---

## 1. Resumen Ejecutivo del Proyecto
- **Nombre:** AulaIA
- **Propósito:** Chatbot y tutor académico interactivo con IA generativa para la asignatura universitaria de **Inteligencia Artificial y Teoría de la Computación** (VII Ciclo).
- **Problema que resuelve:** Los estudiantes tienen el material repartido en múltiples diapositivas y lecturas. AulaIA permite consultar dudas en lenguaje natural y garantiza que **cada respuesta esté respaldada estrictamente por las diapositivas oficiales de la cátedra**, citando el archivo y número exacto de diapositiva y dibujando diagramas visuales para evitar alucinaciones.

---

## 2. Stack Tecnológico y Arquitectura
- **Lenguaje:** Python 3.12 (Windows)
- **Modelo de Lenguaje (LLM):** Google Gemini (`gemini-2.5-flash`) consumido mediante el SDK oficial `google-genai` (versión 2.x).
- **Arquitectura Central:** RAG (Retrieval-Augmented Generation).
- **Base Vectorial:** FAISS (`faiss-cpu`) con persistencia local en disco.
- **Extracción Documental:** `python-pptx` (PowerPoint) y `pypdf` (PDF).
- **Frontend / UI:** Streamlit (`st.write_stream`, `st.chat_message`, `st.cache_resource`).
- **Visualización Gráfica:** Mermaid.js inyectado vía HTML component para renderizar autómatas finitos y máquinas de Turing.

---

## 3. Estado Actual de la Implementación (Ya Terminado)
El repositorio se encuentra estructurado y funcionalmente probado:

```plaintext
BOT IA/
├── data/
│   ├── raw/                 # 8 archivos PPTX reales del curso (Clases 01 a 07.1)
│   └── processed/           # chunks.json (160 diapositivas extraídas) y faiss.index
├── src/
│   ├── __init__.py
│   ├── config.py            # Carga de .env y constantes del sistema
│   ├── ingest.py            # Extractor con metadatos (archivo, tema, diapositiva)
│   ├── vectorstore.py       # FAISS + Cosine Similarity + TF-IDF español de contingencia
│   ├── rag_engine.py        # RAG con streaming, citas, 3 niveles y Quiz generator
│   └── app.py               # Streamlit UI con chat streaming, Mermaid y Quiz
├── tests/
│   ├── test_cases.json      # Banco de pruebas de evaluación
│   └── run_tests.py         # Script de validación de recuperación RAG
├── iniciar_aulaia.bat       # Lanzador ejecutable con 1 solo clic en Windows
├── requirements.txt         # Dependencias instaladas
├── .env.example             # Plantilla de GEMINI_API_KEY
└── README.md                # Documentación del sistema
```

---

## 4. Base Documental Cargada (8 PPTX Universitarios Reales)
Se han indexado **160 diapositivas** que cubren:
1. `01 Introducción al Curso IA 2026-II.pptx` (Sumilla, estructura de unidades).
2. `02 Máquinas, Lenguajes y Problemas 2026-II.pptx` (Alfabetos, cadenas, lenguajes, operaciones).
3. `03 Introducción a la Teoría de los Autómatas 2026-2.pptx` (AFD, 5-tupla formal, función de transición).
4. `04 Autómatas Finitos No Deterministas AFND 2026-2.pptx` (No determinismo, algoritmo de subconjuntos).
5. `05 Autómatas Finitos Con Transiciones épsilon 2026-II.pptx` (Transiciones vacías, epsilon-clausura).
6. `06 Autómatas a Pila 2026-II.pptx` (Memoria de pila LIFO, transiciones con push/pop).
7. `07 Máquinas de Turing 2026-2.pptx` (Cinta infinita, cabezal de lectura/escritura, formalismo).
8. `07.1 Ejercicios Maquinas de Turing 2026-2.pptx` (Ejercicios y simulaciones prácticas).

---

## 5. Funcionalidades Diferenciales Implementadas
1. **Streaming en Vivo:** Conexión con `generate_content_stream` y `st.write_stream` para salida token por token sin tiempos de espera.
2. **Caché en Memoria (`@st.cache_resource`):** Índice FAISS precargado en RAM para búsquedas en < 2ms.
3. **Selector de Nivel de Explicación:**
   - *Intuitivo:* Analogías cotidianas para entender conceptos abstractos.
   - *Formal:* Definiciones matemáticas rigurosas (tuplas, alfabetos, deltas).
   - *Código Python:* Scripts ejecutables que simulan los autómatas.
4. **Diagramas Mermaid en Pantalla:** Generación y renderizado automático de grafos de estados.
5. **Modo Simulador de Quiz:** Generador de 3 a 5 preguntas de autoevaluación con validación y explicación inmediata basada en diapositivas.
6. **Trazabilidad y Control Anti-Alucinación:** Cita obligatoria `[Fuente: Archivo.pptx - Diapositiva X]` y rechazo amigable si la duda no está en las presentaciones.

---

## 6. Decisiones y Puntos Abiertos Donde Necesito Tu Consejo
1. **Extracción de Tablas:** En temas de autómatas, muchas funciones de transición $\delta$ vienen en tablas de PowerPoint (`shape.has_table`). ¿Cómo optimizar su extracción a Markdown tabular?
2. **Búsqueda Híbrida (Hybrid Search):** ¿Conviene fusionar BM25 léxico con vectores densos de Gemini para garantizar 100% de acierto en términos técnicos exactos (como $\epsilon$-clausura)?
3. **Estrategia de Demostración frente al Docente:** ¿Qué flujo de preguntas y demostración práctica causará el mayor impacto académico durante la exposición?
4. **Despliegue Web Gratuito:** ¿Recomiendas desplegarlo en Streamlit Community Cloud para que el docente pueda probarlo desde cualquier navegador sin instalar nada?

