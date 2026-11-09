# Evaluación RAG — AulaIA (`antes`)

- Fecha: 2026-11-09T02:53:36
- Embeddings: **vectorizador local (hash de palabras)**
- Diapositivas indexadas: 161
- LLM: gemini-3.5-flash-lite, gemini-flash-lite-latest

## 1. Recuperación (top-6)

### Todas las preguntas (n=22)

| Método | Hit@1 archivo | Hit@3 archivo | Hit@3 diapositiva | MRR diapositiva |
|---|---|---|---|---|
| bm25 | 73% | 100% | 91% | 0.77 |
| dense | 64% | 73% | 68% | 0.66 |
| hybrid | 73% | 100% | 86% | 0.77 |

### Preguntas directas (n=14)

| Método | Hit@1 archivo | Hit@3 archivo | Hit@3 diapositiva | MRR diapositiva |
|---|---|---|---|---|
| bm25 | 86% | 100% | 93% | 0.82 |
| dense | 79% | 86% | 79% | 0.76 |
| hybrid | 86% | 100% | 86% | 0.84 |

### Preguntas parafraseadas (sinónimos, sin términos técnicos) (n=6)

| Método | Hit@1 archivo | Hit@3 archivo | Hit@3 diapositiva | MRR diapositiva |
|---|---|---|---|---|
| bm25 | 67% | 100% | 83% | 0.75 |
| dense | 50% | 67% | 67% | 0.62 |
| hybrid | 67% | 100% | 83% | 0.75 |

### Preguntas de seguimiento (dependen del turno anterior) (n=2)

| Método | Hit@1 archivo | Hit@3 archivo | Hit@3 diapositiva | MRR diapositiva |
|---|---|---|---|---|
| bm25 | 0% | 100% | 100% | 0.42 |
| dense | 0% | 0% | 0% | 0.10 |
| hybrid | 0% | 100% | 100% | 0.33 |

### Detalle por pregunta (híbrido)

| # | Tipo | Pregunta | Top-3 | Diapositiva correcta en top-3 |
|---|---|---|---|---|
| 1 | directa | ¿Cuáles son los componentes de la 5-tupla de un autómata finito determinista? | 03 I#13, 04 A#17, 03 I#14 | ❌ |
| 2 | directa | ¿Cuál es la definición formal de un AFND? | 04 A#6, 04 A#5, 04 A#2 | ✅ |
| 3 | directa | ¿Cómo se convierte un AFN en un AFD? | 04 A#18, 04 A#3, 04 A#17 | ✅ |
| 4 | directa | ¿Qué es la función de transición extendida de un AFN? | 04 A#7, 04 A#6, 04 A#3 | ✅ |
| 5 | directa | ¿Qué es la clausura épsilon y cómo se calcula? | 05 A#14, 05 A#2, 05 A#6 | ✅ |
| 6 | directa | ¿Cuál es la notación formal de un AFN con transiciones épsilon? | 05 A#6, 05 A#2, 05 A#4 | ✅ |
| 7 | directa | ¿Cuáles son los siete elementos de un autómata a pila? | 06 A#5, 06 A#2, 06 A#9 | ✅ |
| 8 | directa | ¿Cómo funciona la función de transición de un autómata a pila? | 06 A#6, 06 A#5, 06 A#3 | ✅ |
| 9 | directa | ¿Cuál es la definición formal de la máquina de Turing? | 04 A#6, 07 M#10, 07 M#2 | ❌ |
| 10 | directa | ¿Qué es la jerarquía de Chomsky? | 06 A#4, 02 M#20, 07.1#5 | ✅ |
| 11 | directa | ¿Qué es la palabra vacía? | 02 M#23, 02 M#22, 02 M#27 | ✅ |
| 12 | directa | ¿Cómo se concatenan dos lenguajes? | 02 M#31, 02 M#29, 02 M#25 | ✅ |
| 13 | directa | ¿Cómo se calcula la nota final del curso? | 01 I#8, 01 I#13, 01 I#14 | ✅ |
| 14 | directa | ¿La máquina de Turing reconoce el lenguaje a^n b^n c^n? | 07.1#9, 07.1#3, 07.1#5 | ✅ |
| 15 | parafraseada | ¿Qué memoria auxiliar se le agrega a un autómata para poder reconocer cadenas como 0011 o aaabbb? | 06 A#11, 04 A#24, 06 A#10 | ✅ |
| 16 | parafraseada | Explícame el modelo teórico que lee y escribe símbolos sobre una cinta infinita moviendo un cabezal | 07 M#9, 07 M#12, 02 M#21 | ✅ |
| 17 | parafraseada | ¿Cómo diseño una máquina que encuentre las palabras web y ebay dentro de un texto? | 04 A#20, 04 A#24, 02 M#22 | ✅ |
| 18 | parafraseada | ¿Qué significa que una máquina pueda estar en varios estados al mismo tiempo? | 03 I#14, 04 A#3, 03 I#7 | ✅ |
| 19 | parafraseada | Dame un ejemplo de la vida real, como el control de la entrada de autos a un parqueo | 07 M#12, 03 I#30, 03 I#7 | ❌ |
| 20 | parafraseada | ¿Cómo simplifico una función lógica para usar menos compuertas? | 02 M#9, 02 M#4, 04 A#8 | ✅ |
| 24 | seguimiento | ¿Y cómo es su función de transición? | 04 A#7, 04 A#6, 06 A#6 | ✅ |
| 25 | seguimiento | ¿Cuál es su definición formal? | 04 A#6, 06 A#5, 07 M#12 | ✅ |
