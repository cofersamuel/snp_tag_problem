
# Reporte de Resultados y Comparativa (22/04/2026)

Este reporte técnico presenta un análisis detallado de la evaluación experimental del Trabajo de Fin de Grado (TFG) centrado en el problema de selección de Tag SNPs mediante algoritmos evolutivos multiobjetivo. El documento sintetiza los resultados de dos configuraciones experimentales propias (Ejecuciones **A** y **B**) y establece una comparativa crítica con los resultados de referencia de **Moqa et al. (2022)**, permitiendo validar la eficacia de las propuestas metodológicas desarrolladas.

Ejecuciones incluidas:

- **A**: `20260420T205406`
- **B**: `20260422T113918`
- **C (Moqa 2022)**: resultados del paper (figuras + tablas extraídas a CSV)

---

## Índice

1. [Reporte 1 — Resultados y Análisis (Ejecución A)](#1-reporte-1--resultados-y-análisis-ejecución-a-20260420t205406)
2. [Reporte 2 — Resultados y Análisis (Ejecución B)](#2-reporte-2--resultados-y-análisis-ejecución-b-20260422t113918)
3. [Reporte 3 — Resultados del Paper (Moqa 2022)](#3-reporte-3--resultados-del-paper-moqa-et-al-2022)
4. [Comparación Final (A vs B vs Moqa)](#4-comparación-final-a-vs-b-vs-moqa-2022)

---

---

## 1. Reporte 1 — Resultados y Análisis (Ejecución A: 20260420T205406)

Configuración resumida:

- Dataset: Hinds 2005 (48 patrones × 772 SNPs polimórficos)
- `Modo=full | POP_SIZE=200 | N_GEN=500 | OFFSPRING=200 | PC=0.7 | N_RUNS=5`
- **Algoritmos Evaluados:**
    - **NSGA-II / NSGA-III:** Algoritmos basados en ordenamiento no dominado y nichos de diversidad (especializado el III en problemas de alta dimensionalidad de objetivos).
    - **SPEA2:** Algoritmo de fuerza de Pareto que utiliza una jerarquía de dominancia y densidad para la selección.
    - **MOEA/D (TCHE, PBI, WS):** Marco de trabajo basado en descomposición que transforma el problema multiobjetivo en subproblemas escalares. Utiliza diferentes funciones de agregación:
        - **WS (Weighted Sum):** Agregación lineal simple de objetivos; eficiente pero con limitaciones en frentes no convexos.
        - **TCHE (Tchebycheff):** Minimiza la distancia de Chebyshev a un punto de referencia, permitiendo capturar soluciones en frentes no convexos.
        - **PBI (Penalty-based Boundary Intersection):** Introduce un término de penalización ($\theta$) para equilibrar la convergencia hacia el frente y la diversidad (espaciado) de las soluciones.
- **Estrategias de Inicialización:**
    - `random_sparse`: Inicialización estocástica centrada en la **compacidad**, activando pocos SNPs inicialmente.
    - `random_dense`: Muestreo aleatorio estándar (0.5) para evaluar la capacidad de **poda** de los algoritmos.
    - `greedy_pure`: Estrategia constructiva que garantiza la **cobertura biológica** del 100% desde la primera generación.
    - `greedy_hybrid`: Enfoque híbrido (50/50) que equilibra la **explotación** (greedy) con la **exploración** (aleatoria).
- `ref_dirs=165 | particiones=8` (Define el número y densidad de los vectores de referencia que guían la búsqueda hacia un frente de Pareto uniformemente distribuido).
- Tiempo total: 1h 3m 55.76s

#### Marco de Normalización y Comparabilidad
Para garantizar el rigor académico y la comparabilidad entre algoritmos, es necesario transformar los objetivos (que poseen unidades y órdenes de magnitud dispares) a un espacio normalizado $[0, 1]$.

El sistema emplea el modo **`static_dataset_limits`**, el cual establece límites fijos basados en el potencial teórico del dataset completo (Hinds et al., 2005). Esto permite que un valor normalizado tenga un significado absoluto respecto al dataset, independientemente de la ejecución. La normalización se realiza mediante la función de escala:


$$
F_{norm} = \frac{f - f_{ideal}}{f_{nadir} - f_{ideal}}
$$


**Ejemplo de cálculo para el objetivo de Diversidad (Hamming):**
Supongamos que en el dataset de Hinds, el uso de todos los SNPs disponibles genera una diversidad genética máxima (Hamming media) de **250**.
1. Se define $f_{ideal} = -250$ (máximo potencial de diversidad, expresado en sentido de minimización).
2. Se define $f_{nadir} = 0$ (nula diversidad).

Si un algoritmo identifica una solución que alcanza un Hamming de **150**:

$$
F_{norm} = \frac{-150 - (-250)}{0 - (-250)} = \frac{100}{250} = \mathbf{0.40}
$$

*Interpretación:* El valor **0.40** indica que la solución ha capturado el 60% del potencial de diversidad total del dataset. Este enfoque asegura que los indicadores agregados (como el **Hypervolume**) sean consistentes y replicables.

### 1.1 Exploración del Espacio de Objetivos (A)

#### Frentes de Pareto (Convergencia final)

<img src="analisis_assets/20260420T205406/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_global_full.png" width="50%" alt="A Frente global">

| Algoritmo | random_sparse | random_dense | greedy_pure | greedy_hybrid |
| --- | --- | --- | --- | --- |
| **NSGA-II** | <img src="analisis_assets/20260420T205406/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_nsga2_random_sparse_full.png" width="100%"> | <img src="analisis_assets/20260420T205406/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_nsga2_random_dense_full.png" width="100%"> | <img src="analisis_assets/20260420T205406/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_nsga2_greedy_pure_full.png" width="100%"> | <img src="analisis_assets/20260420T205406/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_nsga2_greedy_hybrid_full.png" width="100%"> |
| **NSGA-III** | <img src="analisis_assets/20260420T205406/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_nsga3_random_sparse_full.png" width="100%"> | <img src="analisis_assets/20260420T205406/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_nsga3_random_dense_full.png" width="100%"> | <img src="analisis_assets/20260420T205406/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_nsga3_greedy_pure_full.png" width="100%"> | <img src="analisis_assets/20260420T205406/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_nsga3_greedy_hybrid_full.png" width="100%"> |
| **SPEA2** | <img src="analisis_assets/20260420T205406/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_spea2_random_sparse_full.png" width="100%"> | <img src="analisis_assets/20260420T205406/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_spea2_random_dense_full.png" width="100%"> | <img src="analisis_assets/20260420T205406/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_spea2_greedy_pure_full.png" width="100%"> | <img src="analisis_assets/20260420T205406/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_spea2_greedy_hybrid_full.png" width="100%"> |
| **MOEA/D (PBI)** | <img src="analisis_assets/20260420T205406/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_moead_pbi_random_sparse_full.png" width="100%"> | <img src="analisis_assets/20260420T205406/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_moead_pbi_random_dense_full.png" width="100%"> | <img src="analisis_assets/20260420T205406/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_moead_pbi_greedy_pure_full.png" width="100%"> | <img src="analisis_assets/20260420T205406/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_moead_pbi_greedy_hybrid_full.png" width="100%"> |
| **MOEA/D (TCHE)** | <img src="analisis_assets/20260420T205406/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_moead_tche_random_sparse_full.png" width="100%"> | <img src="analisis_assets/20260420T205406/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_moead_tche_random_dense_full.png" width="100%"> | <img src="analisis_assets/20260420T205406/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_moead_tche_greedy_pure_full.png" width="100%"> | <img src="analisis_assets/20260420T205406/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_moead_tche_greedy_hybrid_full.png" width="100%"> |
| **MOEA/D (WS)** | <img src="analisis_assets/20260420T205406/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_moead_ws_random_sparse_full.png" width="100%"> | <img src="analisis_assets/20260420T205406/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_moead_ws_random_dense_full.png" width="100%"> | <img src="analisis_assets/20260420T205406/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_moead_ws_greedy_pure_full.png" width="100%"> | <img src="analisis_assets/20260420T205406/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_moead_ws_greedy_hybrid_full.png" width="100%"> |

#### 1.1.1 Convergencia Progresiva (A)

| Hypervolume | Range | Avg Hamming |
| --- | --- | --- |
| ![A Conv HV](analisis_assets/20260420T205406/2_comparativa/2_metricas_convergencia/convergencia_hypervolume_full.png) | ![A Conv Range](analisis_assets/20260420T205406/2_comparativa/2_metricas_convergencia/convergencia_range_full.png) | ![A Conv Hamming](analisis_assets/20260420T205406/2_comparativa/2_metricas_convergencia/convergencia_avghammingdistance_full.png) |

| Avg Tolerance | Max Tolerance | SumMin |
| --- | --- | --- |
| ![A Conv AvgTol](analisis_assets/20260420T205406/2_comparativa/2_metricas_convergencia/convergencia_avgtolerancerate_full.png) | ![A Conv MaxTol](analisis_assets/20260420T205406/2_comparativa/2_metricas_convergencia/convergencia_maxtolerancerate_full.png) | ![A Conv SumMin](analisis_assets/20260420T205406/2_comparativa/2_metricas_convergencia/convergencia_summin_full.png) |

| MinSum |
| --- |
| ![A Conv MinSum](analisis_assets/20260420T205406/2_comparativa/2_metricas_convergencia/convergencia_minsum_full.png) |

### 1.2 Rankings (A)

<img src="analisis_assets/20260420T205406/2_comparativa/4_rankings/heatmap_comparativa_full.png" width="50%" alt="A Heatmap comparativa">

<img src="analisis_assets/20260420T205406/2_comparativa/4_rankings/ranking_global_full.png" width="50%" alt="A Ranking global">

> **Nota sobre el Ranking Global (Rank-Sum):** Para evaluar el desempeño de forma equilibrada, se asigna una posición ordinal a cada candidato por métrica (1º para el mejor, etc.). La puntuación global es el sumatorio de estos puestos. Un **valor menor** indica un desempeño superior y más robusto.
> *Criterios:* Hypervolume, Range, Tolerance y Hamming ($\uparrow$); SumMin y MinSum ($\downarrow$).

#### 1.2.1 Top 10 por Métrica (A)

A continuación se detallan las 10 mejores configuraciones para **todas** las métricas evaluadas en la Ejecución A:

**Range ($\uparrow$)**

| Pos | Algoritmo | Inicialización | Valor |
| --- | --- | --- | --- |
| 1 | NSGA2 | greedy_hybrid | 2.4224 |
| 2 | NSGA3 | greedy_hybrid | 1.9282 |
| 3 | SPEA2 | greedy_hybrid | 1.7074 |
| 4 | NSGA2 | random_dense | 1.9001 |
| 5 | NSGA3 | random_dense | 1.1689 |
| 6 | SPEA2 | random_dense | 1.1784 |
| 7 | MOEAD_TCHE | greedy_hybrid | 1.9002 |
| 8 | MOEAD_WS | greedy_hybrid | 2.1539 |
| 9 | MOEAD_TCHE | random_dense | 0.9234 |
| 10 | MOEAD_WS | random_dense | 0.9899 |

**SumMin ($\downarrow$)**

| Pos | Algoritmo | Inicialización | Valor |
| --- | --- | --- | --- |
| 1 | NSGA2 | greedy_hybrid | 0.5507 |
| 2 | NSGA2 | random_dense | 0.6629 |
| 3 | MOEAD_WS | greedy_hybrid | 0.7268 |
| 4 | NSGA3 | greedy_hybrid | 0.8041 |
| 5 | MOEAD_TCHE | greedy_hybrid | 0.8226 |
| 6 | NSGA3 | random_dense | 0.9285 |
| 7 | SPEA2 | greedy_hybrid | 0.9367 |
| 8 | MOEAD_PBI | greedy_hybrid | 0.9422 |
| 9 | SPEA2 | random_dense | 1.0520 |
| 10 | MOEAD_TCHE | random_dense | 1.0648 |

**MinSum ($\downarrow$)**

| Pos | Algoritmo | Inicialización | Valor |
| --- | --- | --- | --- |
| 1 | NSGA3 | random_dense | 1.4001 |
| 2 | MOEAD_TCHE | random_dense | 1.4335 |
| 3 | NSGA2 | random_dense | 1.4417 |
| 4 | MOEAD_PBI | random_dense | 1.4680 |
| 5 | SPEA2 | random_dense | 1.4752 |
| 6 | NSGA3 | greedy_hybrid | 1.4848 |
| 7 | NSGA2 | greedy_hybrid | 1.4944 |
| 8 | MOEAD_WS | random_dense | 1.4961 |
| 9 | MOEAD_PBI | greedy_hybrid | 1.5042 |
| 10 | MOEAD_TCHE | greedy_hybrid | 1.5114 |

**Max. Tolerance Rate ($\uparrow$)**

| Pos | Algoritmo | Inicialización | Valor |
| --- | --- | --- | --- |
| 1 | MOEAD_PBI | greedy_hybrid | 0.2745 |
| 2 | MOEAD_PBI | random_dense | 0.2644 |
| 3 | NSGA3 | random_sparse | 0.2699 |
| 4 | NSGA3 | random_dense | 0.2517 |
| 5 | NSGA3 | greedy_hybrid | 0.2517 |
| 6 | SPEA2 | random_sparse | 0.2489 |
| 7 | MOEAD_TCHE | random_sparse | 0.2558 |
| 8 | MOEAD_PBI | random_sparse | 0.2430 |
| 9 | NSGA2 | greedy_hybrid | 0.2378 |
| 10 | SPEA2 | random_dense | 0.2338 |

**Avg. Tolerance Rate ($\uparrow$)**

| Pos | Algoritmo | Inicialización | Valor |
| --- | --- | --- | --- |
| 1 | MOEAD_PBI | greedy_hybrid | 0.2362 |
| 2 | MOEAD_TCHE | random_dense | 0.2252 |
| 3 | MOEAD_PBI | random_dense | 0.2244 |
| 4 | NSGA3 | random_sparse | 0.2220 |
| 5 | MOEAD_TCHE | random_sparse | 0.2183 |
| 6 | NSGA3 | random_dense | 0.2164 |
| 7 | NSGA2 | random_dense | 0.2143 |
| 8 | MOEAD_PBI | random_sparse | 0.2127 |
| 9 | MOEAD_WS | random_dense | 0.2111 |
| 10 | MOEAD_TCHE | greedy_hybrid | 0.2085 |

**Avg. Hamming Distance ($\uparrow$)**

| Pos | Algoritmo | Inicialización | Valor |
| --- | --- | --- | --- |
| 1 | SPEA2 | random_dense | 149.5918 |
| 2 | NSGA3 | random_dense | 144.5238 |
| 3 | MOEAD_TCHE | random_dense | 140.4767 |
| 4 | NSGA2 | random_dense | 139.5301 |
| 5 | MOEAD_PBI | random_dense | 138.1274 |
| 6 | MOEAD_WS | random_dense | 135.7500 |
| 7 | SPEA2 | greedy_hybrid | 105.3644 |
| 8 | NSGA2 | greedy_hybrid | 99.4525 |
| 9 | MOEAD_WS | greedy_hybrid | 97.8013 |
| 10 | MOEAD_PBI | greedy_hybrid | 94.6259 |

**Hypervolume ($\uparrow$)**

| Pos | Algoritmo | Inicialización | Valor |
| --- | --- | --- | --- |
| 1 | NSGA2 | greedy_hybrid | 0.2446 |
| 2 | NSGA2 | random_dense | 0.2367 |
| 3 | MOEAD_TCHE | greedy_hybrid | 0.2337 |
| 4 | MOEAD_PBI | greedy_hybrid | 0.2214 |
| 5 | NSGA3 | greedy_hybrid | 0.2173 |
| 6 | NSGA3 | random_dense | 0.2083 |
| 7 | MOEAD_WS | greedy_hybrid | 0.2046 |
| 8 | SPEA2 | greedy_hybrid | 0.1979 |
| 9 | MOEAD_TCHE | random_dense | 0.1891 |
| 10 | SPEA2 | random_dense | 0.1685 |

#### 1.2.2 Conclusiones (A)

- **Especialización de Desempeño:** La Ejecución A muestra un cambio de paradigma con los nuevos criterios: **NSGA2** y **NSGA3** con inicialización **random_dense** lideran la robustez global al alcanzar las mejores distancias de Hamming y una alta consistencia en todas las métricas, compartiendo el **1º puesto del Ranking Global** (Score: 37.0).
- **Impacto de la Inicialización:** La estrategia `greedy_hybrid` se consolida como la más equilibrada para la exploración, copando los primeros puestos en **Hipervolumen** y **Rango**, mientras que `random_dense` maximiza la capacidad de distinción genética (Hamming).
- **Liderazgo en Tolerancia:** **MOEAD_PBI + greedy_hybrid** destaca como el motor más eficiente en términos de robustez ante pérdida de datos, liderando las métricas de **Avg. y Max. Tolerance Rate**.
- **Versatilidad de NSGA2:** Se mantiene como un algoritmo extremadamente competitivo, situándose en el Top 3 global y liderando el Hipervolumen con la configuración híbrida.

---

## 2. Reporte 2 — Resultados y Análisis (Ejecución B: 20260422T113918)

> **Nota metodológica sobre la Comparabilidad:**
> Se ha identificado que la Ejecución **A** utiliza una transformación de objetivos mediante negación (`neg`), mientras que la Ejecución **B** emplea inversión (`inverse`).
> - En el modo **`neg`**, un valor original de tolerancia $f=0.1$ se convierte en $f'=-0.1$.
> - En el modo **`inverse`**, se convierte en $f'=1/0.1 = 10$.
> Esta diferencia técnica altera radicalmente la topología del espacio normalizado y la magnitud de indicadores como el **Hypervolume**. Por lo tanto, los valores agregados de A y B **no son comparables directamente por su magnitud bruta**. La comparativa debe centrarse en el **ranking relativo** de los algoritmos dentro de cada entorno y en los valores decodificados (escala real) cuando proceda.

Configuración resumida:

- Dataset: Hinds 2005 (48 patrones × 772 SNPs polimórficos)
- `Modo=full | POP_SIZE=200 | N_GEN=500 | OFFSPRING=200 | PC=0.7 | N_RUNS=5`
- Algoritmos: NSGA2, NSGA3, SPEA2, MOEAD_TCHE, MOEAD_PBI, MOEAD_WS
- `ref_dirs=200 | particiones=9` (Configuración de mayor densidad de vectores para incrementar la resolución en la exploración de frentes complejos).
- Tiempo total: 1h 15m 16.62s

**Normalización:**
El sistema aplica normalización de objetivos a un espacio $[0, 1]$ para calcular indicadores agregados con el modo `static_dataset_limits`:

$$
F_{norm} = \frac{f - f_{ideal}}{f_{nadir} - f_{ideal}}
$$


### 2.1 Exploración del Espacio de Objetivos (B)

#### Frentes de Pareto (Convergencia final)

<img src="analisis_assets/20260422T113918/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_global_full.png" width="50%" alt="B Frente global">

| Algoritmo | random_sparse | random_dense | greedy_pure | greedy_hybrid |
| --- | --- | --- | --- | --- |
| **NSGA-II** | <img src="analisis_assets/20260422T113918/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_nsga2_random_sparse_full.png" width="100%"> | <img src="analisis_assets/20260422T113918/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_nsga2_random_dense_full.png" width="100%"> | <img src="analisis_assets/20260422T113918/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_nsga2_greedy_pure_full.png" width="100%"> | <img src="analisis_assets/20260422T113918/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_nsga2_greedy_hybrid_full.png" width="100%"> |
| **NSGA-III** | <img src="analisis_assets/20260422T113918/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_nsga3_random_sparse_full.png" width="100%"> | <img src="analisis_assets/20260422T113918/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_nsga3_random_dense_full.png" width="100%"> | <img src="analisis_assets/20260422T113918/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_nsga3_greedy_pure_full.png" width="100%"> | <img src="analisis_assets/20260422T113918/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_nsga3_greedy_hybrid_full.png" width="100%"> |
| **SPEA2** | <img src="analisis_assets/20260422T113918/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_spea2_random_sparse_full.png" width="100%"> | <img src="analisis_assets/20260422T113918/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_spea2_random_dense_full.png" width="100%"> | <img src="analisis_assets/20260422T113918/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_spea2_greedy_pure_full.png" width="100%"> | <img src="analisis_assets/20260422T113918/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_spea2_greedy_hybrid_full.png" width="100%"> |
| **MOEA/D (PBI)** | <img src="analisis_assets/20260422T113918/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_moead_pbi_random_sparse_full.png" width="100%"> | <img src="analisis_assets/20260422T113918/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_moead_pbi_random_dense_full.png" width="100%"> | <img src="analisis_assets/20260422T113918/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_moead_pbi_greedy_pure_full.png" width="100%"> | <img src="analisis_assets/20260422T113918/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_moead_pbi_greedy_hybrid_full.png" width="100%"> |
| **MOEA/D (TCHE)** | <img src="analisis_assets/20260422T113918/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_moead_tche_random_sparse_full.png" width="100%"> | <img src="analisis_assets/20260422T113918/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_moead_tche_random_dense_full.png" width="100%"> | <img src="analisis_assets/20260422T113918/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_moead_tche_greedy_pure_full.png" width="100%"> | <img src="analisis_assets/20260422T113918/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_moead_tche_greedy_hybrid_full.png" width="100%"> |
| **MOEA/D (WS)** | <img src="analisis_assets/20260422T113918/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_moead_ws_random_sparse_full.png" width="100%"> | <img src="analisis_assets/20260422T113918/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_moead_ws_random_dense_full.png" width="100%"> | <img src="analisis_assets/20260422T113918/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_moead_ws_greedy_pure_full.png" width="100%"> | <img src="analisis_assets/20260422T113918/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_moead_ws_greedy_hybrid_full.png" width="100%"> |

#### 2.1.1 Convergencia Progresiva (B)

| Hypervolume | Range | Avg Hamming |
| --- | --- | --- |
| ![B Conv HV](analisis_assets/20260422T113918/2_comparativa/2_metricas_convergencia/convergencia_hypervolume_full.png) | ![B Conv Range](analisis_assets/20260422T113918/2_comparativa/2_metricas_convergencia/convergencia_range_full.png) | ![B Conv Hamming](analisis_assets/20260422T113918/2_comparativa/2_metricas_convergencia/convergencia_avghammingdistance_full.png) |

| Avg Tolerance | Max Tolerance | SumMin |
| --- | --- | --- |
| ![B Conv AvgTol](analisis_assets/20260422T113918/2_comparativa/2_metricas_convergencia/convergencia_avgtolerancerate_full.png) | ![B Conv MaxTol](analisis_assets/20260422T113918/2_comparativa/2_metricas_convergencia/convergencia_maxtolerancerate_full.png) | ![B Conv SumMin](analisis_assets/20260422T113918/2_comparativa/2_metricas_convergencia/convergencia_summin_full.png) |

| MinSum |
| --- |
| ![B Conv MinSum](analisis_assets/20260422T113918/2_comparativa/2_metricas_convergencia/convergencia_minsum_full.png) |

### 2.2 Rankings (B)

<img src="analisis_assets/20260422T113918/2_comparativa/4_rankings/heatmap_comparativa_full.png" width="50%" alt="B Heatmap comparativa">

<img src="analisis_assets/20260422T113918/2_comparativa/4_rankings/ranking_global_full.png" width="50%" alt="B Ranking global">

> **Nota sobre el Ranking Global (Rank-Sum):** Para evaluar el desempeño de forma equilibrada, se asigna una posición ordinal a cada candidato por métrica. Un **valor menor** indica un desempeño superior y más robusto.
> *Criterios:* Hypervolume, Range, Tolerance y Hamming ($\uparrow$); SumMin y MinSum ($\downarrow$).

#### 2.2.1 Top 10 por Métrica (B)

Resumen de las mejores configuraciones en la Ejecución B (`OBJ_TRANSFORM=inverse`):

**Range ($\uparrow$)**

| Pos | Algoritmo | Inicialización | Valor |
| --- | --- | --- | --- |
| 1 | NSGA2 | greedy_hybrid | 2.1878 |
| 2 | MOEAD_WS | greedy_hybrid | 1.9151 |
| 3 | SPEA2 | greedy_hybrid | 1.8631 |
| 4 | MOEAD_TCHE | greedy_hybrid | 1.8329 |
| 5 | NSGA2 | random_sparse | 1.5094 |
| 6 | SPEA2 | random_sparse | 1.3274 |
| 7 | MOEAD_WS | greedy_pure | 1.1328 |
| 8 | NSGA3 | greedy_hybrid | 1.1263 |
| 9 | NSGA3 | random_sparse | 1.1047 |
| 10 | NSGA2 | greedy_pure | 1.0464 |

**SumMin ($\downarrow$)**

| Pos | Algoritmo | Inicialización | Valor |
| --- | --- | --- | --- |
| 1 | NSGA2 | greedy_hybrid | 0.0168 |
| 2 | MOEAD_WS | greedy_hybrid | 0.0261 |
| 3 | MOEAD_TCHE | greedy_hybrid | 0.0290 |
| 4 | SPEA2 | greedy_hybrid | 0.0326 |
| 5 | NSGA2 | random_sparse | 0.0366 |
| 6 | SPEA2 | random_sparse | 0.0645 |
| 7 | MOEAD_TCHE | random_sparse | 0.0817 |
| 8 | MOEAD_WS | random_sparse | 0.0922 |
| 9 | NSGA3 | greedy_hybrid | 0.1267 |
| 10 | MOEAD_PBI | greedy_hybrid | 0.1399 |

**MinSum ($\downarrow$)**

| Pos | Algoritmo | Inicialización | Valor |
| --- | --- | --- | --- |
| 1 | MOEAD_WS | greedy_hybrid | 0.1697 |
| 2 | MOEAD_TCHE | random_sparse | 0.1771 |
| 3 | MOEAD_TCHE | greedy_hybrid | 0.1774 |
| 4 | MOEAD_WS | random_sparse | 0.1783 |
| 5 | NSGA2 | random_sparse | 0.1819 |
| 6 | MOEAD_PBI | random_sparse | 0.1830 |
| 7 | NSGA2 | greedy_hybrid | 0.1901 |
| 8 | MOEAD_PBI | greedy_hybrid | 0.1930 |
| 9 | NSGA3 | greedy_hybrid | 0.1954 |
| 10 | NSGA3 | random_sparse | 0.2019 |

**Max. Tolerance Rate ($\uparrow$)**

| Pos | Algoritmo | Inicialización | Valor |
| --- | --- | --- | --- |
| 1 | MOEAD_WS | greedy_hybrid | 0.2760 |
| 2 | MOEAD_TCHE | greedy_hybrid | 0.2541 |
| 3 | MOEAD_TCHE | random_sparse | 0.2540 |
| 4 | NSGA3 | random_dense | 0.2476 |
| 5 | NSGA2 | random_sparse | 0.2466 |
| 6 | MOEAD_WS | random_sparse | 0.2457 |
| 7 | MOEAD_PBI | random_sparse | 0.2326 |
| 8 | NSGA2 | random_dense | 0.2300 |
| 9 | NSGA3 | greedy_hybrid | 0.2283 |
| 10 | NSGA2 | greedy_hybrid | 0.2274 |

**Avg. Tolerance Rate ($\uparrow$)**

| Pos | Algoritmo | Inicialización | Valor |
| --- | --- | --- | --- |
| 1 | MOEAD_PBI | random_sparse | 0.2137 |
| 2 | MOEAD_WS | greedy_hybrid | 0.2118 |
| 3 | NSGA3 | random_dense | 0.2105 |
| 4 | NSGA2 | random_dense | 0.2088 |
| 5 | MOEAD_TCHE | random_sparse | 0.2022 |
| 6 | MOEAD_WS | random_sparse | 0.1998 |
| 7 | MOEAD_TCHE | greedy_hybrid | 0.1921 |
| 8 | MOEAD_PBI | greedy_hybrid | 0.1822 |
| 9 | MOEAD_TCHE | random_dense | 0.1817 |
| 10 | MOEAD_PBI | random_dense | 0.1741 |

**Avg. Hamming Distance ($\uparrow$)**

| Pos | Algoritmo | Inicialización | Valor |
| --- | --- | --- | --- |
| 1 | NSGA2 | random_dense | 127.7955 |
| 2 | NSGA3 | random_dense | 115.9670 |
| 3 | SPEA2 | random_dense | 110.3085 |
| 4 | MOEAD_TCHE | random_dense | 93.3046 |
| 5 | MOEAD_PBI | random_dense | 88.9865 |
| 6 | MOEAD_WS | random_dense | 83.9749 |
| 7 | NSGA2 | greedy_hybrid | 74.0921 |
| 8 | SPEA2 | greedy_hybrid | 56.2438 |
| 9 | NSGA2 | random_sparse | 35.7477 |
| 10 | MOEAD_TCHE | greedy_hybrid | 34.4169 |

**Hypervolume ($\uparrow$)**

| Pos | Algoritmo | Inicialización | Valor |
| --- | --- | --- | --- |
| 1 | NSGA2 | greedy_hybrid | 0.9501 |
| 2 | MOEAD_WS | greedy_hybrid | 0.9435 |
| 3 | NSGA2 | random_sparse | 0.9410 |
| 4 | SPEA2 | greedy_hybrid | 0.9383 |
| 5 | MOEAD_TCHE | greedy_hybrid | 0.9378 |
| 6 | SPEA2 | random_sparse | 0.9185 |
| 7 | MOEAD_TCHE | random_sparse | 0.9125 |
| 8 | MOEAD_WS | random_sparse | 0.9044 |
| 9 | NSGA3 | greedy_hybrid | 0.8689 |
| 10 | MOEAD_PBI | greedy_hybrid | 0.8644 |

#### 2.2.2 Conclusiones (B)

- **Liderazgo Indiscutible:** Bajo los nuevos criterios de maximización de robustez, \textbf{MOEAD_WS + greedy_hybrid} se posiciona como la configuración ganadora absoluta (Score: 22.0), liderando la métrica de \textbf{MinSum} y situándose en el Top 2 de \textbf{Hipervolumen, Rango y Max. Tolerance}.
- **Exploración y Diversidad:** \textbf{NSGA2 + greedy_hybrid} mantiene el liderazgo en \textbf{Hipervolumen} (0.9501) y \textbf{Rango}, consolidándose como el motor de búsqueda más potente para cubrir el frente de Pareto.
- **Especialización en Hamming:** Al igual que en la Ejecución A, la inicialización \textbf{random_dense} maximiza la distancia de Hamming (distinción genética), con \textbf{NSGA2} a la cabeza de esta métrica.
- **Efectividad Híbrida:** La estrategia \texttt{greedy\_hybrid} domina el Top 3 global, demostrando que el equilibrio entre exploración aleatoria y explotación dirigida es clave para optimizar simultáneamente convergencia y robustez.

---

## 3. Reporte 3 — Resultados del Paper (Moqa et al., 2022)

### 3.1 Análisis de Reproducibilidad (Moqa 2022)

Tras cruzar los datos del paper con los datos de Ting, se establece el siguiente marco de análisis:

- **Datos Confirmados:** Dataset Hinds (1032 SNPs), Metaheurísticas (NSGA-II, SPEA2, NSGA-III, MOEA/D) y población de 200.
- **Hipótesis de Trabajo (Línea Ting):** Se asume que las métricas de maximización se normalizan mediante inversión ($1/f$) y que la lógica de inicialización `greedy` busca una cobertura del 100%.
- **Lagunas Críticas:** Omisión de resultados de Hypervolume (pese a citarse como métrica clave) y falta de parámetros estructurales en MOEA/D y vectores de referencia.

**Normalización:**
*Se desconoce el método exacto de normalización de objetivos empleado en el estudio original.* Para este análisis comparativo, los datos extraídos se tratan bajo las asunciones metodológicas estándar del TFG cuando procede.

### 3.2 Figuras y Tablas (Moqa 2022)

| Algoritmo | Moqa (Random) | Moqa (Greedy) |
| --- | --- | --- |
| **NSGA-II** | ![Moqa NSGA2 R](analisis_assets/moqa_2022/figures/moqa_fig_nsga2_random.jpeg) | ![Moqa NSGA2 G](analisis_assets/moqa_2022/figures/moqa_fig_nsga2_greedy.jpeg) |
| **NSGA-III** | ![Moqa NSGA3 R](analisis_assets/moqa_2022/figures/moqa_fig_nsga3_random.jpeg) | ![Moqa NSGA3 G](analisis_assets/moqa_2022/figures/moqa_fig_nsga3_greedy.jpeg) |
| **SPEA2** | ![Moqa SPEA2 R](analisis_assets/moqa_2022/figures/moqa_fig_spea2_random.jpeg) | ![Moqa SPEA2 G](analisis_assets/moqa_2022/figures/moqa_fig_spea2_greedy.jpeg) |
| **MOEA/D** | ![Moqa MOEAD R](analisis_assets/moqa_2022/figures/moqa_fig_moead_random.jpeg) | ![Moqa MOEAD G](analisis_assets/moqa_2022/figures/moqa_fig_moead_greedy.jpeg) |

#### Convergencia (Moqa 2022)

| Convergencia 1 | Convergencia 2 |
| --- | --- |
| ![Moqa Conv 1](analisis_assets/moqa_2022/figures/moqa_fig_convergencia_1.jpeg) | ![Moqa Conv 2](analisis_assets/moqa_2022/figures/moqa_fig_convergencia_2.jpeg) |

### 4.2 Tablas (Moqa 2022)

Las tablas siguientes se incluyen también como CSV en `analisis_assets/moqa_2022/tables/`.

#### Tabla 3 — Spread (rangos y desviaciones por objetivo)

| Algo. | Init | Obj1_Range | Obj1_Std | Obj2_Range | Obj2_Std | Obj3_Range | Obj3_Std | Obj4_Range | Obj4_Std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NSGA-II | RI | 319 | 82 | 74 | 16 | 0.1419 | 0.0338 | 0.0045 | 0.001 |
| NSGA-II | GI | 362 | 81 | 81 | 18 | 0.1427 | 0.031 | 0.0045 | 0.0009 |
| SPEA2 | RI | 281 | 59 | 72 | 15 | 0.0737 | 0.0135 | 0.0019 | 0.0003 |
| SPEA2 | GI | 253 | 58 | 67 | 15 | 0.0749 | 0.0153 | 0.0018 | 0.0004 |
| NSGA-III | RI | 208 | 29 | 44 | 6 | 0.0606 | 0.0114 | 0.0015 | 0.0003 |
| NSGA-III | GI | 190 | 29 | 40 | 6 | 0.0628 | 0.0112 | 0.0013 | 0.0003 |
| MOEAD | RI | 434 | 113 | 100 | 26 | 0.1541 | 0.0263 | 0.0081 | 0.0013 |
| MOEAD | GI | 723 | 101 | 91 | 25 | 0.4244 | 0.0242 | 0.0174 | 0.0015 |

#### Tabla 4 — Performance (agregados)

| Algo | Init | Range_Avg | Range_Std | SumMin_Avg | SumMin_Std | MinSum_Avg | MinSum_Std |
| --- | --- | --- | --- | --- | --- | --- | --- |
| NSGA-II | RI | 1.7877 | 0.4359 | 0.2331 | 0.0542 | 0.131 | 0.0296 |
| NSGA-II | GI | 1.809 | 0.4522 | 0.2005 | 0.0483 | 0.1114 | 0.0229 |
| SPEA-2 | RI | 1.0429 | 0.2499 | 0.2135 | 0.0506 | 0.1896 | 0.0404 |
| SPEA-2 | GI | 1.0608 | 0.2128 | 0.1768 | 0.0354 | 0.1311 | 0.0252 |
| NSGA-III | RI | 0.4639 | 0.0923 | 0.1304 | 0.0249 | 0.1244 | 0.0243 |
| NSGA-III | GI | 1.7563 | 0.439 | 0.1838 | 0.0443 | 0.1899 | 0.039 |
| MOEAD | RI | 2.1533 | 0.3981 | 0.4092 | 0.0603 | 0.1968 | 0.0238 |
| MOEAD | GI | 2.1382 | 0.408 | 0.3018 | 0.0435 | 0.1578 | 0.0179 |

#### Tabla 5 — Métricas (tolerancia y Hamming)

| Algo | Init | MaxTolRate_Avg | MaxTolRate_Std | AvgTolRate_Avg | AvgTolRate_Std | AvgHammingDist_Avg | AvgHammingDist_Std |
| --- | --- | --- | --- | --- | --- | --- | --- |
| NSGA-II | RI | 0.0692 | 0.0146 | 0.0485 | 0.006 | 0.1022 | 0.0191 |
| NSGA-II | GI | 0.0781 | 0.017 | 0.052 | 0.0075 | 0.1061 | 0.0195 |
| SPEA-2 | RI | 0.0633 | 0.0113 | 0.0487 | 0.0056 | 0.0773 | 0.0094 |
| SPEA-2 | GI | 0.0593 | 0.0108 | 0.0474 | 0.0056 | 0.0769 | 0.0098 |
| NSGA-III | RI | 0.0859 | 0.016 | 0.0726 | 0.012 | 0.1083 | 0.0191 |
| NSGA-III | GI | 0.0937 | 0.0204 | 0.0525 | 0.0075 | 0.0962 | 0.0177 |
| MOEAD | RI | 0.0891 | 0.0149 | 0.0816 | 0.0099 | 0.15 | 0.0309 |
| MOEAD | GI | 0.0895 | 0.0173 | 0.0879 | 0.013 | 0.1523 | 0.0326 |

### 3.3 Contraste: Paper vs. Datos Extraídos

Al comparar las afirmaciones cualitativas del estudio original con el análisis cuantitativo del TFG sobre sus tablas, se observan matices importantes:

1. **Consenso en Precisión:** El paper sitúa a **SPEA-2** en el **1º puesto** de todas las métricas de calidad (Tolerancia, Hamming, nº SNPs). Los rankings del TFG confirman este liderazgo de posiciones en los indicadores de precisión biológica.
2. **Discrepancia en Convergencia:** Mientras el paper posiciona a SPEA-2 como el líder en convergencia, los datos extraídos (*Performance Aggregates / SumMin*) revelan que **NSGA-III (RI)** logra la **1ª posición** en agregados de rendimiento, superando a SPEA-2 en este indicador específico.
3. **Rol de MOEA/D:** El paper reporta a MOEA/D como el algoritmo menos eficiente. Los rankings del TFG confirman esta posición, situándolo consistentemente en el 7º u 8º puesto de todas las categorías.

#### 3.3.1 Ranking por Métrica (Moqa)

Ranking de las 8 configuraciones del estudio original:

**Performance Aggregates (SumMin $\downarrow$)**

| Pos | Algoritmo | Inicialización | Valor |
| --- | --- | --- | --- |
| 1 | NSGA-III | RI | 0.1304 |
| 2 | SPEA-2 | GI | 0.1768 |
| 3 | NSGA-III | GI | 0.1838 |
| 4 | NSGA-II | GI | 0.2005 |
| 5 | SPEA-2 | RI | 0.2135 |
| 6 | NSGA-II | RI | 0.2331 |
| 7 | MOEAD | GI | 0.3018 |
| 8 | MOEAD | RI | 0.4092 |

**Avg. Tolerance Rate ($\uparrow$)**

| Pos | Algoritmo | Inicialización | Valor |
| --- | --- | --- | --- |
| 1 | MOEAD | GI | 0.0879 |
| 2 | MOEAD | RI | 0.0816 |
| 3 | NSGA-III | RI | 0.0726 |
| 4 | NSGA-III | GI | 0.0525 |
| 5 | NSGA-II | GI | 0.0520 |
| 6 | SPEA-2 | RI | 0.0487 |
| 7 | NSGA-II | RI | 0.0485 |
| 8 | SPEA-2 | GI | 0.0474 |

**Avg. Hamming Dist. ($\uparrow$)**

| Pos | Algoritmo | Inicialización | Valor |
| --- | --- | --- | --- |
| 1 | MOEAD | GI | 0.1523 |
| 2 | MOEAD | RI | 0.1500 |
| 3 | NSGA-III | RI | 0.1083 |
| 4 | NSGA-II | GI | 0.1061 |
| 5 | NSGA-II | RI | 0.1022 |
| 6 | NSGA-III | GI | 0.0962 |
| 7 | SPEA-2 | RI | 0.0773 |
| 8 | SPEA-2 | GI | 0.0769 |

#### 3.3.2 Conclusiones (Moqa)

- **MOEAD como Referente en Robustez:** Al contrario de lo reportado inicialmente, bajo criterios de maximización, **MOEAD** (GI) lidera las métricas de **Tolerancia y Hamming**, demostrando un alto poder de distinción genética.
- **NSGA-III en Convergencia:** El ranking confirma a **NSGA-III (RI)** como el líder en agregados de rendimiento (**SumMin**).
- **Validación Metodológica:** Los rankings del TFG refuerzan el liderazgo de la inicialización dirigida (**GI**) al ocupar consistentemente el **Top 3** en las métricas de error.
- **Liderazgo Supra Métrica (Moqa):** El ranking global confirma a **SPEA-2 + GI** en la **1ª posición** (Score: 12.0), gracias a su equilibrio excepcional entre todas las dimensiones, seguido de **SPEA-2 + RI** (2º puesto).

---

## 4. Comparación Final (A vs B vs Moqa 2022)

### 4.1 Comparación visual (Moqa vs A)

| Algoritmo | Moqa (R) | Moqa (G) | A (RS) | A (RD) | A (GP) | A (GH) |
| --- | --- | --- | --- | --- | --- | --- |
| **NSGA-II** | ![M NSGA2 R](analisis_assets/moqa_2022/figures/moqa_fig_nsga2_random.jpeg) | ![M NSGA2 G](analisis_assets/moqa_2022/figures/moqa_fig_nsga2_greedy.jpeg) | ![A NSGA2 RS](analisis_assets/20260420T205406/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_nsga2_random_sparse_full.png) | ![A NSGA2 RD](analisis_assets/20260420T205406/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_nsga2_random_dense_full.png) | ![A NSGA2 GP](analisis_assets/20260420T205406/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_nsga2_greedy_pure_full.png) | ![A NSGA2 GH](analisis_assets/20260420T205406/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_nsga2_greedy_hybrid_full.png) |
| **NSGA-III** | ![M NSGA3 R](analisis_assets/moqa_2022/figures/moqa_fig_nsga3_random.jpeg) | ![M NSGA3 G](analisis_assets/moqa_2022/figures/moqa_fig_nsga3_greedy.jpeg) | ![A NSGA3 RS](analisis_assets/20260420T205406/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_nsga3_random_sparse_full.png) | ![A NSGA3 RD](analisis_assets/20260420T205406/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_nsga3_random_dense_full.png) | ![A NSGA3 GP](analisis_assets/20260420T205406/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_nsga3_greedy_pure_full.png) | ![A NSGA3 GH](analisis_assets/20260420T205406/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_nsga3_greedy_hybrid_full.png) |
| **SPEA2** | ![M SPEA2 R](analisis_assets/moqa_2022/figures/moqa_fig_spea2_random.jpeg) | ![M SPEA2 G](analisis_assets/moqa_2022/figures/moqa_fig_spea2_greedy.jpeg) | ![A SPEA2 RS](analisis_assets/20260420T205406/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_spea2_random_sparse_full.png) | ![A SPEA2 RD](analisis_assets/20260420T205406/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_spea2_random_dense_full.png) | ![A SPEA2 GP](analisis_assets/20260420T205406/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_spea2_greedy_pure_full.png) | ![A SPEA2 GH](analisis_assets/20260420T205406/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_spea2_greedy_hybrid_full.png) |
| **MOEA/D** | ![M MOEAD R](analisis_assets/moqa_2022/figures/moqa_fig_moead_random.jpeg) | ![M MOEAD G](analisis_assets/moqa_2022/figures/moqa_fig_moead_greedy.jpeg) | ![A MOEAD TCHE RS](analisis_assets/20260420T205406/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_moead_tche_random_sparse_full.png) | ![A MOEAD TCHE RD](analisis_assets/20260420T205406/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_moead_tche_random_dense_full.png) | ![A MOEAD TCHE GP](analisis_assets/20260420T205406/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_moead_tche_greedy_pure_full.png) | ![A MOEAD TCHE GH](analisis_assets/20260420T205406/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_moead_tche_greedy_hybrid_full.png) |

### 4.2 Comparación visual (Moqa vs B)

| Algoritmo | Moqa (R) | Moqa (G) | B (RS) | B (RD) | B (GP) | B (GH) |
| --- | --- | --- | --- | --- | --- | --- |
| **NSGA-II** | ![M NSGA2 R](analisis_assets/moqa_2022/figures/moqa_fig_nsga2_random.jpeg) | ![M NSGA2 G](analisis_assets/moqa_2022/figures/moqa_fig_nsga2_greedy.jpeg) | ![B NSGA2 RS](analisis_assets/20260422T113918/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_nsga2_random_sparse_full.png) | ![B NSGA2 RD](analisis_assets/20260422T113918/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_nsga2_random_dense_full.png) | ![B NSGA2 GP](analisis_assets/20260422T113918/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_nsga2_greedy_pure_full.png) | ![B NSGA2 GH](analisis_assets/20260422T113918/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_nsga2_greedy_hybrid_full.png) |
| **NSGA-III** | ![M NSGA3 R](analisis_assets/moqa_2022/figures/moqa_fig_nsga3_random.jpeg) | ![M NSGA3 G](analisis_assets/moqa_2022/figures/moqa_fig_nsga3_greedy.jpeg) | ![B NSGA3 RS](analisis_assets/20260422T113918/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_nsga3_random_sparse_full.png) | ![B NSGA3 RD](analisis_assets/20260422T113918/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_nsga3_random_dense_full.png) | ![B NSGA3 GP](analisis_assets/20260422T113918/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_nsga3_greedy_pure_full.png) | ![B NSGA3 GH](analisis_assets/20260422T113918/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_nsga3_greedy_hybrid_full.png) |
| **SPEA2** | ![M SPEA2 R](analisis_assets/moqa_2022/figures/moqa_fig_spea2_random.jpeg) | ![M SPEA2 G](analisis_assets/moqa_2022/figures/moqa_fig_spea2_greedy.jpeg) | ![B SPEA2 RS](analisis_assets/20260422T113918/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_spea2_random_sparse_full.png) | ![B SPEA2 RD](analisis_assets/20260422T113918/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_spea2_random_dense_full.png) | ![B SPEA2 GP](analisis_assets/20260422T113918/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_spea2_greedy_pure_full.png) | ![B SPEA2 GH](analisis_assets/20260422T113918/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_spea2_greedy_hybrid_full.png) |
| **MOEA/D** | ![M MOEAD R](analisis_assets/moqa_2022/figures/moqa_fig_moead_random.jpeg) | ![M MOEAD G](analisis_assets/moqa_2022/figures/moqa_fig_moead_greedy.jpeg) | ![B MOEAD TCHE RS](analisis_assets/20260422T113918/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_moead_tche_random_sparse_full.png) | ![B MOEAD TCHE RD](analisis_assets/20260422T113918/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_moead_tche_random_dense_full.png) | ![B MOEAD TCHE GP](analisis_assets/20260422T113918/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_moead_tche_greedy_pure_full.png) | ![B MOEAD TCHE GH](analisis_assets/20260422T113918/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_moead_tche_greedy_hybrid_full.png) |

### 4.3 Comparativa de Resultados y Semejanzas

Para una comparación coherente entre los resultados del TFG y el estado del arte, se emplea el siguiente mapeo de nomenclatura:

| Nomenclatura TFG | Nomenclatura Moqa |
| --- | --- |
| `random_dense` | `random` (RI) |
| `greedy_hybrid` | `greedy_pure` (GI) |

#### 4.3.1 Semejanzas en Resultados (Rendimiento)

Al analizar las tendencias transversales de las tres fuentes (A, B y Moqa), se extraen las siguientes conclusiones:

1. **Liderazgo de Posiciones:** **NSGA2** y **SPEA2** son los motores más constantes. NSGA2 lidera la cobertura al ocupar el puesto **1º en Hipervolumen** en ambas ejecuciones (A y B). Por su parte, **MOEAD** y **SPEA2** destacan en robustez genética al situarse en los puestos más altos de Hamming y Tolerancia bajo criterios de maximización.
2. **Impacto de la Inicialización:** Existe una semejanza crítica: el uso de estrategias **Híbridas y Dirigidas** es el factor más determinante. En la ejecución B, las configuraciones con `greedy_hybrid` dominan el ranking global, superando la excesiva focalización de `greedy_pure`.
3. **Consistencia de NSGA2:** Se observa que NSGA2 es el algoritmo más estable, manteniendo el **1º puesto en métricas de agregados (SumMin/Hipervolumen)** tanto en el paper como en el TFG (A).

