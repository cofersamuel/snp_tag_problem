# Documentación del Experimento: Selección Evolutiva de Tag SNPs

Este documento recoge los resultados, la configuración y el análisis de la ejecución general del problema de optimización multiobjetivo para la selección de Tag SNPs.

## 1. Configuración Técnica

A continuación, se detallan los parámetros y configuraciones bajo los que se ha ejecutado el motor evolutivo para este experimento.

### Detalles del Dataset
* **Origen de datos**: Hinds et al. 2005 (Perlegen)
* **Archivo**: `snp_tag/data/datasets/hinds2005_1032.txt`
* **Número de SNPs**: 772 SNPs polimórficos (tras el filtrado inicial sobre 1032 originales).
* **Patrones Alélicos**: 48 haplotipos.

### Configuración del Motor Evolutivo
* **Tamaño de Población (`POP_SIZE`)**: 220
* **Generaciones (`N_GEN`)**: 500
* **Descendencia (`OFFSPRING`)**: 220
* **Probabilidad de Cruce (`PC`)**: 0.7
* **Probabilidad de Mutación (`PM`)**: 0.001295 ($\approx 1/L$)
* **Número de Ejecuciones (`N_RUNS`)**: 5 ejecuciones independientes por configuración.
* **Modo de Evaluación (`EVALUATION_MODE`)**: `proportional`
* **Modo de Normalización**: `static_proportional_limits`
* **Transformación de Objetivos (`OBJ_TRANSFORM`)**: `neg` (Negación lineal $f' = -f$, conservando el espacio relativo biológico).
* **Modo de Semillas (`SEED_MODE`)**: `non_deterministic`

### Diseño Experimental (280 Ejecuciones en total)

**7 Algoritmos Multiobjetivo:**
1. `NSGA-II`
2. `NSGA-III` (9 particiones, 220 direcciones de referencia)
3. `SPEA2`
4. `MOEA/D` (Agregación Tchebycheff `TCHE`)
5. `AGE-MOEA-II`
6. `SMS-EMOA`
7. `RVEA`

**2 Estrategias de Inicialización:**
1. `random_dense`: Inicialización aleatoria densa ($\approx 50\%$ de los SNPs seleccionados).
2. `greedy_multi`: Inicialización que inyecta conocimiento heurístico y asegura representatividad biológica desde la generación 0.

**4 Operadores de Cruce:**
1. `1P`: Cruce de un punto (One-Point).
2. `2P`: Cruce de dos puntos (Two-Point).
3. `UX`: Cruce Uniforme (Uniform Crossover).
4. `HUX`: Cruce Medio Uniforme (Half Uniform Crossover).

---

## 2. Resultados

### Frentes de Pareto

Para facilitar la comparativa visual del rendimiento de cada algoritmo según su configuración, se presentan a continuación los Frentes de Pareto consolidados (tras las 5 ejecuciones independientes).

#### Inicialización: `greedy_multi`

| Algoritmo | 1P | 2P | UX | HUX |
| :--- | :--- | :--- | :--- | :--- |
| **NSGA-II** | ![1P](20260512T183448/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_nsga2_greedy_multi+1p_full.png) | ![2P](20260512T183448/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_nsga2_greedy_multi+2p_full.png) | ![UX](20260512T183448/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_nsga2_greedy_multi+ux_full.png) | ![HUX](20260512T183448/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_nsga2_greedy_multi+hux_full.png) |
| **NSGA-III** | ![1P](20260512T183448/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_nsga3_greedy_multi+1p_full.png) | ![2P](20260512T183448/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_nsga3_greedy_multi+2p_full.png) | ![UX](20260512T183448/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_nsga3_greedy_multi+ux_full.png) | ![HUX](20260512T183448/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_nsga3_greedy_multi+hux_full.png) |
| **SPEA2** | ![1P](20260512T183448/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_spea2_greedy_multi+1p_full.png) | ![2P](20260512T183448/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_spea2_greedy_multi+2p_full.png) | ![UX](20260512T183448/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_spea2_greedy_multi+ux_full.png) | ![HUX](20260512T183448/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_spea2_greedy_multi+hux_full.png) |
| **MOEA/D-TCHE** | ![1P](20260512T183448/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_moead_tche_greedy_multi+1p_full.png) | ![2P](20260512T183448/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_moead_tche_greedy_multi+2p_full.png) | ![UX](20260512T183448/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_moead_tche_greedy_multi+ux_full.png) | ![HUX](20260512T183448/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_moead_tche_greedy_multi+hux_full.png) |
| **AGE-MOEA-II** | ![1P](20260512T183448/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_agemoea2_greedy_multi+1p_full.png) | ![2P](20260512T183448/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_agemoea2_greedy_multi+2p_full.png) | ![UX](20260512T183448/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_agemoea2_greedy_multi+ux_full.png) | ![HUX](20260512T183448/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_agemoea2_greedy_multi+hux_full.png) |
| **SMS-EMOA** | ![1P](20260512T183448/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_smsemoa_greedy_multi+1p_full.png) | ![2P](20260512T183448/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_smsemoa_greedy_multi+2p_full.png) | ![UX](20260512T183448/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_smsemoa_greedy_multi+ux_full.png) | ![HUX](20260512T183448/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_smsemoa_greedy_multi+hux_full.png) |
| **RVEA** | ![1P](20260512T183448/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_rvea_greedy_multi+1p_full.png) | ![2P](20260512T183448/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_rvea_greedy_multi+2p_full.png) | ![UX](20260512T183448/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_rvea_greedy_multi+ux_full.png) | ![HUX](20260512T183448/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_rvea_greedy_multi+hux_full.png) |

#### Inicialización: `random_dense`

| Algoritmo | 1P | 2P | UX | HUX |
| :--- | :--- | :--- | :--- | :--- |
| **NSGA-II** | ![1P](20260512T183448/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_nsga2_random_dense+1p_full.png) | ![2P](20260512T183448/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_nsga2_random_dense+2p_full.png) | ![UX](20260512T183448/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_nsga2_random_dense+ux_full.png) | ![HUX](20260512T183448/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_nsga2_random_dense+hux_full.png) |
| **NSGA-III** | ![1P](20260512T183448/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_nsga3_random_dense+1p_full.png) | ![2P](20260512T183448/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_nsga3_random_dense+2p_full.png) | ![UX](20260512T183448/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_nsga3_random_dense+ux_full.png) | ![HUX](20260512T183448/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_nsga3_random_dense+hux_full.png) |
| **SPEA2** | ![1P](20260512T183448/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_spea2_random_dense+1p_full.png) | ![2P](20260512T183448/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_spea2_random_dense+2p_full.png) | ![UX](20260512T183448/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_spea2_random_dense+ux_full.png) | ![HUX](20260512T183448/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_spea2_random_dense+hux_full.png) |
| **MOEA/D-TCHE** | ![1P](20260512T183448/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_moead_tche_random_dense+1p_full.png) | ![2P](20260512T183448/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_moead_tche_random_dense+2p_full.png) | ![UX](20260512T183448/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_moead_tche_random_dense+ux_full.png) | ![HUX](20260512T183448/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_moead_tche_random_dense+hux_full.png) |
| **AGE-MOEA-II** | ![1P](20260512T183448/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_agemoea2_random_dense+1p_full.png) | ![2P](20260512T183448/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_agemoea2_random_dense+2p_full.png) | ![UX](20260512T183448/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_agemoea2_random_dense+ux_full.png) | ![HUX](20260512T183448/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_agemoea2_random_dense+hux_full.png) |
| **SMS-EMOA** | ![1P](20260512T183448/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_smsemoa_random_dense+1p_full.png) | ![2P](20260512T183448/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_smsemoa_random_dense+2p_full.png) | ![UX](20260512T183448/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_smsemoa_random_dense+ux_full.png) | ![HUX](20260512T183448/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_smsemoa_random_dense+hux_full.png) |
| **RVEA** | ![1P](20260512T183448/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_rvea_random_dense+1p_full.png) | ![2P](20260512T183448/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_rvea_random_dense+2p_full.png) | ![UX](20260512T183448/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_rvea_random_dense+ux_full.png) | ![HUX](20260512T183448/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_rvea_random_dense+hux_full.png) |


#### Frente Global
Una agregación de todas las soluciones no dominadas descubiertas por todos los algoritmos en el experimento:
![Frente Global](20260512T183448/2_comparativa/1_frentes/frentes_pareto/frentes_pareto_global_full.png)

---

### Métricas de Convergencia
Evolución del desempeño generacional durante las 500 iteraciones del proceso de optimización, comparando algoritmos, inicializaciones y operadores.

#### 1. Hypervolume (Diversidad y Convergencia)
![Hypervolume](20260512T183448/2_comparativa/2_metricas_convergencia/convergencia_hypervolume_full.png)

#### 2. Distancia Promedio de Hamming (Diversidad Biológica)
![Avg Hamming Distance](20260512T183448/2_comparativa/2_metricas_convergencia/convergencia_avghammingdistance_full.png)

#### 3. Range (Extensión del Frente de Pareto)
![Range](20260512T183448/2_comparativa/2_metricas_convergencia/convergencia_range_full.png)

#### 4. Tasa de Tolerancia Promedio
![Avg Tolerance Rate](20260512T183448/2_comparativa/2_metricas_convergencia/convergencia_avgtolerancerate_full.png)

#### 5. SumMin y MinSum (Esparcimiento)
| SumMin | MinSum |
| :---: | :---: |
| ![SumMin](20260512T183448/2_comparativa/2_metricas_convergencia/convergencia_summin_full.png) | ![MinSum](20260512T183448/2_comparativa/2_metricas_convergencia/convergencia_minsum_full.png) |


---

## 3. Análisis de Resultados

Tras evaluar todas las combinaciones experimentales y la evolución de sus frentes de Pareto, se extraen las siguientes conclusiones fundamentales:

* **Mejor Inicialización (`greedy_multi` vs `random_dense`)**: La inicialización `greedy_multi` demuestra una ventaja sustancial. Los algoritmos que parten de una población enriquecida con información heurística biológica (como el Balance Alélico y las firmas únicas) convergen de manera mucho más abrupta y estable desde las primeras 50 generaciones. Además, logran frentes mucho más extensos (mayor `Range`) y una mejor Distancia de Hamming Promedio.
* **Mejores Operadores**: Si bien el impacto del cruce es menor comparado con la inicialización, el uso de cruces uniformes (`UX` y `HUX`) ha demostrado ser más idóneo para este problema combinatorio denso que los cruces tradicionales de 1 y 2 puntos (`1P`, `2P`), ya que promueven una mezcla genética más fina en cromosomas muy largos.
* **Rendimiento Algorítmico**:
    * **SMS-EMOA y AGE-MOEA-II** emergen como los algoritmos más robustos globalmente cuando se asocian con `greedy_multi`. Muestran frentes de Pareto densos y bien distribuidos, superando con creces a la competencia en métricas de cobertura y tamaño de panel.
    * **MOEA/D-TCHE** destaca enormemente en el descubrimiento de los extremos del frente (excelente `Range`), logrando soluciones muy compactas o muy representativas, aunque su densidad interior a veces es menor.
    * **RVEA**, a pesar de su gran velocidad de ejecución, muestra una severa convergencia prematura en este espacio, quedando atrapado en óptimos locales con paneles SNP demasiado extensos.

---

## 4. Multi-Criteria Decision Making (MCDM)

El paso final del pipeline ha consistido en aplicar técnicas estadísticas y de MCDM para elegir las configuraciones de Tag SNPs óptimas y justificar matemáticamente la decisión.

### Soluciones Propuestas (Top Global)

A partir del análisis en `mcdm_recomendaciones_full.csv`, el sistema propone unánimemente soluciones provenientes de inicializaciones `greedy_multi` mediante los mejores algoritmos:

1. **Mejor "Knee Point" (Punto de inflexión óptimo)**
   * **Algoritmo**: AGE-MOEA-II (`greedy_multi`)
   * **Tamaño del panel**: 13 SNPs
   * **Justificación**: Representa el mejor balance costo-beneficio biológico, logrando un panel ultra-compacto que mantiene una tolerancia competitiva ($0.23$) y buena distancia de Hamming ($0.505$).

2. **Pseudo-Weights Match (Preferencia Equilibrada)**
   * **Algoritmo**: AGE-MOEA-II (`greedy_multi`)
   * **Tamaño del panel**: 16 SNPs
   * **Justificación**: Aporta 3 SNPs adicionales sobre el Knee Point, incrementando el equilibrio general de las métricas biológicas ($Tol: 0.187$, $Hamming: 0.469$).

3. **Compromise ASF (Achievement Scalarization Function)**
   * **Algoritmo**: SMS-EMOA (`greedy_multi`)
   * **Tamaño del panel**: 17 SNPs
   * **Justificación**: Elegida como la solución global de menor riesgo que maximiza simultáneamente todos los objetivos normalizados frente a un punto ideal.

### Análisis Visual MCDM

Los siguientes gráficos ilustran el rendimiento y balance de la agregación de todos los algoritmos bajo criterios MCDM, reforzando la elección de las soluciones listadas.

#### Diagrama de Radar y Pétalos (Global Agregado)
| Radar | Pétalos |
| :---: | :---: |
| ![Radar](20260512T183448/2_comparativa/5_decision_mcdm/mcdm_radar_objetivos_agregado_full.png) | ![Pétalos](20260512T183448/2_comparativa/5_decision_mcdm/mcdm_petal_objetivos_agregado_full.png) |

#### Tabla de Recomendaciones Resumen
![Recomendaciones](20260512T183448/2_comparativa/5_decision_mcdm/mcdm_tabla_recomendaciones_full.png)

#### Gráfico de Dispersión MCDM
La posición de los Knee points y las soluciones ASF en el espacio objetivo consolidado:
![Scatter MCDM](20260512T183448/2_comparativa/5_decision_mcdm/mcdm_scatter_agregado_full.png)
