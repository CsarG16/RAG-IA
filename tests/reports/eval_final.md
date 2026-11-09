# Evaluación RAG — AulaIA (`final`)

- Fecha: 2026-11-09T03:11:45
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

## 2. Respuestas completas (end-to-end con Gemini)

| Métrica | Valor |
|---|---|
| Casos ejecutados sin error | 25 / 25 |
| Rechazo correcto de preguntas fuera del temario | 100% |
| Falsos rechazos (preguntas del curso rechazadas) | 0% |
| Respuestas del curso con cita de fuente | 100% |
| Cobertura media de conceptos clave | 93% |
| Latencia hasta el primer token (mediana / p90) | 3.79 s / 5.65 s |
| Latencia total (mediana / p90) | 5.14 s / 6.00 s |

| # | Tipo | TTFT | Total | Rechazo | Cita | Conceptos | Inicio de la respuesta |
|---|---|---|---|---|---|---|---|
| 1 | directa | 4.46 s | 5.47 s | no | sí | 100% | Un Autómata Finito Determinista (AFD) se define formalmente como una 5-tupla: $$A = ( Q , \Sigma , \delta , q_0 , F )$$ Cada uno de sus componentes se desglosa  |
| 2 | directa | 3.30 s | 4.09 s | no | sí | 100% | Un Autómata Finito No Determinista (AFND o AFN) se define formalmente como una 5-tupla $A = (Q, \Sigma, \delta, q_0, F)$, donde cada componente se desglosa de l |
| 3 | directa | 3.84 s | 5.60 s | no | sí | 100% | La conversión de un Autómata Finito No Determinista (AFN) a un Autómata Finito Determinista (AFD) se realiza mediante el procedimiento conocido como **método de |
| 4 | directa | 4.35 s | 5.35 s | no | sí | 100% | La función de transición extendida de un Autómata Finito No Determinista (AFN), denotada formalmente como $\delta^*$, es una función que toma como argumentos un |
| 5 | directa | 3.75 s | 4.81 s | no | sí | 100% | La **clausura épsilon** (denotada formalmente como $C_\varepsilon(q)$ o $\varepsilon\text{-clausura}$) de un estado $q$ en un Autómata Finito No Determinista co |
| 6 | directa | 3.69 s | 5.15 s | no | sí | 100% | Un Autómata Finito No Determinista con transiciones $\varepsilon$ (AFN-$\varepsilon$ o AFND-$\varepsilon$) se define formalmente como una 5-tupla: $$E = (Q, \Si |
| 7 | directa | 4.11 s | 4.91 s | no | sí | 100% | Un Autómata a Pila (AP) se define formalmente como una 7-tupla: $$AP = (Q, \Sigma, \Gamma, \delta, q_0, Z_0, F)$$ Donde cada uno de sus elementos cumple la sigu |
| 8 | directa | 3.79 s | 5.44 s | no | sí | 100% | La función de transición de un Autómata a Pila (AP) formaliza su comportamiento dinámico, el cual depende simultáneamente del estado actual, del símbolo de entr |
| 9 | directa | 5.65 s | 7.52 s | no | sí | 100% | Una Máquina de Turing se define formalmente mediante una 7-tupla: $$M = (Q, \Sigma, \Gamma, \delta, q_0, B, F)$$ Donde cada uno de sus componentes se desglosa d |
| 10 | directa | 3.50 s | 4.76 s | no | sí | 50% | La jerarquía de Chomsky es una clasificación de los lenguajes formales introducida por el lingüista Avram Noam Chomsky en la década de 1950, la cual establece u |
| 11 | directa | 3.40 s | 4.10 s | no | sí | 100% | La palabra vacía es denotada por $\lambda$ (Lambda) y se define formalmente como aquella palabra que no contiene ninguna letra. Su longitud es igual a cero ($/\ |
| 12 | directa | 4.08 s | 5.14 s | no | sí | 100% | La concatenación de dos lenguajes $L_1$ y $L_2$ definidos sobre un alfabeto $\Sigma$ (es decir, $L_1, L_2 \subset \Sigma^*$), denotada como $L_1L_2$ o $L_1.L_2$ |
| 13 | directa | 5.01 s | 5.40 s | no | sí | 100% | Para calcular la nota final del curso se emplea la siguiente fórmula formal: $$PF = \frac{3 \cdot PE + EP + EF}{5}$$ Donde los componentes se desglosan de la si |
| 14 | directa | 4.57 s | 5.93 s | no | sí | 100% | Una Máquina de Turing formalmente se define como una 7-tupla: $$M = (Q, \Sigma, \Gamma, \delta, q_0, B, F)$$ Donde cada componente representa: - $Q$: Conjunto f |
| 15 | parafraseada | 3.23 s | 4.59 s | no | sí | 100% | Para reconocer lenguajes formales que requieren conteo ilimitado o balanceado, como $L = \{0^n 1^n \mid n \ge 1\}$, se le agrega al autómata una pila de tipo LI |
| 16 | parafraseada | 2.79 s | 4.04 s | no | sí | 100% | El modelo teórico que describe un autómata finito dotado de una cinta de longitud infinita con capacidad de lectura y escritura es la **Máquina de Turing (MT)** |
| 17 | parafraseada | 3.81 s | 4.48 s | no | sí | 100% | Para diseñar un mecanismo que reconozca las palabras "web" y "ebay" dentro de un texto, se emplea formalmente un Autómata Finito No Determinista (AFND). Un AFND |
| 18 | parafraseada | 2.67 s | 3.38 s | no | sí | 100% | En el contexto de los Autómatas Finitos No Deterministas (AFND o AFN), la afirmación de que el autómata «puede estar en varios estados a la vez» representa un m |
| 19 | parafraseada | 2.71 s | 4.61 s | no | sí | 0% | Un Autómata Finito Determinista (AFD) se define formalmente como una 5-tupla $A = (Q, \Sigma, \delta, q_0, F)$, cuyos componentes se desglozan de la siguiente m |
| 20 | parafraseada | 3.73 s | 5.55 s | no | sí | 100% | Para simplificar una función lógica y reducir el uso de compuertas, se emplea el álgebra de Boole. El proceso consiste en aplicar las leyes y teoremas booleanos |
| 21 | fuera_de_dominio | 6.12 s | 6.12 s | sí | no | — | Ese tema no se encuentra registrado en las diapositivas oficiales del curso (Clases 01 a 07.1). Por favor, formula una duda relacionada con autómatas, lenguajes |
| 22 | fuera_de_dominio | 6.00 s | 6.00 s | sí | no | — | Ese tema no se encuentra registrado en las diapositivas oficiales del curso (Clases 01 a 07.1). Por favor, formula una duda relacionada con autómatas, lenguajes |
| 23 | fuera_de_dominio | 5.59 s | 5.87 s | sí | no | — | Ese tema no se encuentra registrado en las diapositivas oficiales del curso (Clases 01 a 07.1). Por favor, formula una duda relacionada con autómatas, lenguajes |
| 24 | seguimiento | 3.47 s | 4.59 s | no | sí | 100% | La función de transición de un Autómata a Pila (AP) se define formalmente como: $$\delta: Q \times (\Sigma \cup \{\varepsilon\}) \times \Gamma \to \mathcal{P}_{ |
| 25 | seguimiento | 3.27 s | 4.10 s | no | sí | 100% | Una Máquina de Turing se define formalmente como una 7-tupla: $$M = (Q, \Sigma, \Gamma, \delta, q_0, B, F)$$ Donde cada componente se desglosa de la siguiente m |
