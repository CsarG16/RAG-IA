# Evaluación RAG — AulaIA (`fase2_memoria`)

- Fecha: 2026-11-09T03:00:41
- Embeddings: **gemini-embedding-001**
- Diapositivas indexadas: 161
- LLM: gemini-3.5-flash-lite, gemini-flash-lite-latest

## 1. Recuperación (top-6)

### Todas las preguntas (n=22)

| Método | Hit@1 archivo | Hit@3 archivo | Hit@3 diapositiva | MRR diapositiva |
|---|---|---|---|---|
| bm25 | 77% | 100% | 91% | 0.80 |
| dense | 95% | 95% | 95% | 0.94 |
| hybrid | 95% | 100% | 95% | 0.92 |

### Preguntas directas (n=14)

| Método | Hit@1 archivo | Hit@3 archivo | Hit@3 diapositiva | MRR diapositiva |
|---|---|---|---|---|
| bm25 | 86% | 100% | 93% | 0.82 |
| dense | 100% | 100% | 100% | 1.00 |
| hybrid | 100% | 100% | 100% | 1.00 |

### Preguntas parafraseadas (sinónimos, sin términos técnicos) (n=6)

| Método | Hit@1 archivo | Hit@3 archivo | Hit@3 diapositiva | MRR diapositiva |
|---|---|---|---|---|
| bm25 | 67% | 100% | 83% | 0.75 |
| dense | 83% | 83% | 83% | 0.78 |
| hybrid | 83% | 100% | 83% | 0.72 |

### Preguntas de seguimiento (dependen del turno anterior) (n=2)

| Método | Hit@1 archivo | Hit@3 archivo | Hit@3 diapositiva | MRR diapositiva |
|---|---|---|---|---|
| bm25 | 50% | 100% | 100% | 0.75 |
| dense | 100% | 100% | 100% | 1.00 |
| hybrid | 100% | 100% | 100% | 1.00 |

### Detalle por pregunta (híbrido)

| # | Tipo | Pregunta | Top-3 | Diapositiva correcta en top-3 |
|---|---|---|---|---|
| 1 | directa | ¿Cuáles son los componentes de la 5-tupla de un autómata finito determinista? | 03 I#16, 03 I#21, 03 I#13 | ✅ |
| 2 | directa | ¿Cuál es la definición formal de un AFND? | 04 A#6, 04 A#5, 04 A#9 | ✅ |
| 3 | directa | ¿Cómo se convierte un AFN en un AFD? | 04 A#18, 04 A#17, 04 A#13 | ✅ |
| 4 | directa | ¿Qué es la función de transición extendida de un AFN? | 04 A#7, 04 A#10, 03 I#25 | ✅ |
| 5 | directa | ¿Qué es la clausura épsilon y cómo se calcula? | 05 A#14, 05 A#11, 05 A#12 | ✅ |
| 6 | directa | ¿Cuál es la notación formal de un AFN con transiciones épsilon? | 05 A#6, 05 A#7, 05 A#3 | ✅ |
| 7 | directa | ¿Cuáles son los siete elementos de un autómata a pila? | 06 A#5, 06 A#9, 06 A#7 | ✅ |
| 8 | directa | ¿Cómo funciona la función de transición de un autómata a pila? | 06 A#6, 06 A#5, 06 A#9 | ✅ |
| 9 | directa | ¿Cuál es la definición formal de la máquina de Turing? | 07 M#12, 07 M#10, 07 M#2 | ✅ |
| 10 | directa | ¿Qué es la jerarquía de Chomsky? | 02 M#20, 06 A#4, 02 M#18 | ✅ |
| 11 | directa | ¿Qué es la palabra vacía? | 02 M#23, 02 M#22, 05 A#3 | ✅ |
| 12 | directa | ¿Cómo se concatenan dos lenguajes? | 02 M#31, 02 M#29, 02 M#25 | ✅ |
| 13 | directa | ¿Cómo se calcula la nota final del curso? | 01 I#8, 01 I#14, 01 I#11 | ✅ |
| 14 | directa | ¿La máquina de Turing reconoce el lenguaje a^n b^n c^n? | 07.1#9, 07.1#8, 07.1#5 | ✅ |
| 15 | parafraseada | ¿Qué memoria auxiliar se le agrega a un autómata para poder reconocer cadenas como 0011 o aaabbb? | 06 A#11, 06 A#10, 03 I#12 | ✅ |
| 16 | parafraseada | Explícame el modelo teórico que lee y escribe símbolos sobre una cinta infinita moviendo un cabezal | 07 M#9, 07 M#12, 07 M#11 | ✅ |
| 17 | parafraseada | ¿Cómo diseño una máquina que encuentre las palabras web y ebay dentro de un texto? | 04 A#20, 04 A#24, 03 I#20 | ✅ |
| 18 | parafraseada | ¿Qué significa que una máquina pueda estar en varios estados al mismo tiempo? | 03 I#14, 07 M#11, 04 A#3 | ✅ |
| 19 | parafraseada | Dame un ejemplo de la vida real, como el control de la entrada de autos a un parqueo | 03 I#30, 03 I#7, 04 A#4 | ❌ |
| 20 | parafraseada | ¿Cómo simplifico una función lógica para usar menos compuertas? | 02 M#9, 02 M#4, 02 M#13 | ✅ |
| 24 | seguimiento | ¿Y cómo es su función de transición? | 06 A#6, 06 A#5, 06 A#9 | ✅ |
| 25 | seguimiento | ¿Cuál es su definición formal? | 07 M#12, 07 M#10, 07 M#2 | ✅ |
