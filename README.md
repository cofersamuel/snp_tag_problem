La adquisición del dataset Hinds et al. (2005) fue un proceso de "arqueología de datos" para asegurar que estuviéramos replicando exactamente el experimento del artículo de Moqa et al. (2022). Aquí tienes la traza técnica de cómo lo conseguimos:

1. Identificación de la Fuente Original
El artículo de Moqa et al. (2022) cita el dataset Hinds et al. (2005) [2], pero no proporciona un enlace directo al bloque específico. Sin embargo, mencionan que siguen la metodología de Ting et al. (2010) [9], el artículo precursor que definió el uso del bloque de 1032 SNPs.

2. Rastreo del Artículo de Ting (2010)
Al investigar el artículo de Ting, encontramos que el software original y los datos fueron depositados en el sitio web del laboratorio de los autores en la Universidad Nacional Chung Cheng (CCU) de Taiwán.

3. Localización y Descarga
Localizamos un servidor activo (cilab.cs.ccu.edu.tw) que contenía un archivo comprimido titulado Code_MoTagSNPsSel.zip.

Descargamos el archivo directamente desde el servidor del laboratorio.
Dentro del ZIP, encontramos un fichero llamado input.txt.
4. Verificación de los Datos
Al abrir el fichero input.txt, confirmamos que cumplía exactamente con las especificaciones del problema:

Dimensiones: 48 filas por 1032 columnas.
Formato: Texto binario plano (0s y 1s), donde cada fila representa un patrón haplotípico (clase alélica) y cada columna un SNP.
Contenido: El README del código de Ting confirmaba explícitamente: "The example input.txt file, including 1032 SNPs, is used in the paper 'Multi-Objective Tag SNPs Selection Using Evolutionary Algorithms'".
5. Integración en el Proyecto
Finalmente, movimos ese fichero a nuestra carpeta local como data/hinds2005_1032.txt y desarrollamos el cargador específico (load_hinds2005_block) para interpretar este formato de cadenas binarias, que es muy distinto al formato comprimido de HapMap Phase II.

En resumen: No se descargó de una base de datos genómica general (donde el bloque de 1032 podría haber variado), sino directamente del paquete de replicación oficial de los autores del algoritmo original.

# 🧬 Selección de Tag SNPs con pymoo


**Trabajo:** Selección de Tag SNPs con algoritmos evolutivos multiobjetivo



**Autor:** Samuel Corrionero Fernández  

**Tutor:** José M. Granado-Criado

## 🌍 Objetivo



Este cuaderno implementa un experimento de selección de **Tag SNPs** con varios algoritmos evolutivos multiobjetivo. La meta es encontrar un subconjunto de SNPs que sea pequeño pero informativo para distinguir haplotipos.

En este notebook se implementa la reproduccion experimental del paper *Assessing effectiveness of many-objective evolutionary algorithms for selection of tag SNPs* usando `pymoo`.


## 👉 Índice

1. Introducción
   - Objetivo del cuaderno
   - Conceptos biológicos clave

2. Preparación del experimento
   - Temporizador
   - Imports y entorno
   - Configuración global

3. Datos sintéticos y verificación LD
   - Generación de haplotipos sintéticos con LD
   - Gráficas exploratorias (matriz H, bloques, zoom, frecuencias, variabilidad)
   - Verificación rápida de estructura LD

4. Formulación y optimización
   - Preparación de pares de haplotipos
   - Evaluación de candidatos (núcleo del problema)
   - Definición formal del problema en pymoo
   - Heurística greedy y muestreador híbrido
   - Construcción de algoritmos y direcciones de referencia
   - Bucle principal de experimentos

5. Métricas y análisis de resultados
   - Normalización global y métricas finales
   - Ensamblado de métricas por ejecución
   - Inspección inicial de resultados
   - Frentes de Pareto por algoritmo
   - Correlación de objetivos y coordenadas paralelas

6. Comparativa final
   - Agregación estadística por método
   - Análisis de convergencia generacional
   - Heatmap de métricas medias
   - Ranking global
   - Visualizaciones finales (boxplots, violin, media ± std)

7. Exportación y cierre
   - Exportación de resultados a CSV
   - Temporizador final

## 🧪 Conceptos biológicos clave



### 🧫 ¿Qué es el ADN?

El **ADN** es la molécula que almacena la información biológica. Se puede imaginar como un texto largo escrito con cuatro letras químicas.



### 🧱 ¿Qué es un gen?

Un **gen** es una región concreta del ADN que contiene instrucciones funcionales (por ejemplo, para fabricar una proteína o regular procesos celulares).



### 🧵 ¿Qué es un cromosoma?

Un **cromosoma** es una estructura donde el ADN se empaqueta. Los genes se distribuyen a lo largo de los cromosomas.



### 📍 ¿Qué es un locus y qué es un alelo?

- **Locus:** posición concreta del ADN.

- **Alelo:** variante que aparece en ese locus (por ejemplo, una base distinta).



### 👤 ¿Qué son genotipo y haplotipo?

- **Genotipo:** combinación de variantes de un individuo en varios loci.

- **Haplotipo:** conjunto de variantes heredadas juntas en una región del cromosoma.



En este notebook, cada haplotipo se representa como una secuencia binaria de 0/1.

Porque en SNPs, la mayoría de posiciones son bialélicas (solo hay 2 alelos posibles), y por eso se codifican como binario:

0 = un alelo (normalmente el de referencia o el mayoritario)
1 = el otro alelo (alternativo o minoritario)
En un haplotipo (una sola copia cromosómica), en cada SNP solo puede haber una de esas dos opciones, así que 0/1 es una representación natural y compacta.

Además, usar 0/1 facilita mucho los cálculos (distancias, optimización, modelos de ML, etc.).
Importante: el 0 y 1 son una convención de codificación, no significa “bueno/malo” ni dominancia biológica.



### 🔎 ¿Qué es un SNP?

Un **SNP** (Single Nucleotide Polymorphism) es una variación de una sola posición del ADN entre individuos.



### 🏷️ ¿Qué es un Tag SNP?

Un **Tag SNP** es un SNP representativo que permite capturar información de otros SNPs correlacionados. Así se reduce el número de marcadores necesarios sin perder demasiada información.



### 🔗 ¿Qué es el desequilibrio de ligamiento (LD)?

Hay **LD** cuando varios SNPs tienden a heredarse juntos. Si la correlación es alta, medir todos puede ser redundante: basta con una selección representativa.



### 📏 ¿Qué es la distancia de Hamming?

La **distancia de Hamming** cuenta cuántas posiciones difieren entre dos secuencias binarias. Aquí se usa para medir cuánto se distinguen dos haplotipos según los SNPs seleccionados.

## ⏱️ Temporizador

Propósito de la siguiente celda de código: medir/registrar los resultados del análisis o realizar cálculos relacionados con el experimento.


## 📦 Imports y entorno



En la siguiente celda se importan todas las librerías necesarias para ejecutar el experimento completo.



### ¿Qué se carga?

- Librerías científicas (`numpy`, `pandas`) para cálculo y tablas.

- Librerías de visualización (`matplotlib`, `seaborn`) para gráficas comparativas.

- Componentes de `pymoo` para definir el problema, los algoritmos multiobjetivo y los operadores genéticos.



### ¿Por qué es importante?

Sin estos imports no se puede construir ni evaluar soluciones. Esta celda prepara la “caja de herramientas” que usará todo el notebook.

## ⚙️ Configuración global del experimento

La siguiente celda define los parámetros principales de simulación y optimización.

### Parámetros que se fijan
- Semilla de aleatoriedad para reproducibilidad.
- Tamaño del problema: número de SNPs y haplotipos.
- Parámetros evolutivos: población, generaciones, descendencia, probabilidad de cruce y mutación.
- Selector `EXECUTION_MODE` para alternar entre distintos niveles de intensidad.

### ¿Por qué es crucial?
Estos valores controlan coste computacional y calidad de resultados. Pequeños cambios pueden alterar significativamente el comportamiento de los algoritmos.

### Modos de ejecución
- `EXECUTION_MODE = 'fast'`: validación rápida (segundos).
- `EXECUTION_MODE = 'medium'`: modo intermedio (3-5 minutos aproximadamente).
- `EXECUTION_MODE = 'full'`: experimento final completo (45 - 60').

A continuación, se definen el resto de parámetros:

## Carga de datos (Sintéticos o Hinds 2005)

En esta sección se construye la matriz de haplotipos `H` (0/1) que alimenta toda la optimización. El tamaño del problema se mantiene estrictamente en **N_SNPS = 1032**, respetando la estructura del marco experimental del TFG.

- **Opción A — `synthetic`**: genera datos sintéticos con estructura de LD (rápido y controlado).
- **Opción B — `hinds2005`**: carga el bloque real de Hinds et al. (2005) desde `data/hinds2005_1032.txt`.

El selector se controla con la variable `DATA_SOURCE` en la configuración global.

### Control de Densidad de Bloques LD (`NUM_BLOCKS`)

El parámetro `NUM_BLOCKS` (recomendado de 1 a 10) permite alterar la **densidad interna** de desequilibrio de ligamiento dentro de la ventana de 1032 SNPs en datos sintéticos:

1. **En datos Sintéticos:** El generador matemático divide estrictamente los 1032 SNPs en exactamente la cantidad de bloques dictada por `NUM_BLOCKS`. Por ejemplo, `NUM_BLOCKS=10` genera 10 sub-regiones independientes dentro de los 1032 SNPs. Esto es ideal para probar los límites matemáticos del algoritmo.

## 🧬 Generación/carga de haplotipos (`H`)

La siguiente celda construye la matriz de datos `H` que alimenta toda la optimización.

- Si `DATA_SOURCE == "synthetic"`: se generan haplotipos sintéticos con LD (bloques ligados).
- Si `DATA_SOURCE == "hinds2005"`: se carga el bloque real de Hinds et al. (2005) ya proporcionado en `data/hinds2005_1032.txt`.

### Resultado
Una matriz `H` con forma `(n_haplotypes, n_snps)` donde cada fila representa un haplotipo binario (0/1).

### Gráfica 0 — Mapa de la matriz de haplotipos `H`

Esta figura muestra directamente la matriz binaria de entrada (filas = haplotipos, columnas = SNPs).

Muestra visualmente los valores binarios (0/1) de todos los haplotipos y SNPs. Permite ver de un vistazo patrones por bloques, zonas homogéneas y cambios de alelo.

### Gráfica 0b — Estructura de bloques de herencia (LD)

Esta figura resume `H` por bloques de SNPs para hacer visible la herencia conjunta.

Cada columna representa un bloque (por defecto 64 SNPs) y cada celda indica la fracción de alelo `1` en ese bloque para cada haplotipo.

¿Por qué así se ve mejor? Porque al promediar varios SNPs en una sola celda se reduce el ruido local (flips puntuales), se mantiene la tendencia global del bloque y baja la dimensionalidad visual (de muchas columnas a pocas). Eso hace más fáciles de detectar las bandas/patrones de LD.

La relación con los Tag SNPs: el algoritmo de optimización puede elegir SNPs representativos de cada bloque correlacionado, reduciendo el número total de marcadores sin perder demasiada información.

### Ejemplo sencillo (2 bloques de 4 SNPs)

Supón 3 haplotipos y 8 SNPs, agrupados en 2 bloques:

- **Bloque 1**: SNP1–SNP4  
- **Bloque 2**: SNP5–SNP8

| Haplotipo | SNP1 | SNP2 | SNP3 | SNP4 | SNP5 | SNP6 | SNP7 | SNP8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| H1 | 1 | 1 | 0 | 1 | 0 | 0 | 1 | 0 |
| H2 | 1 | 1 | 1 | 1 | 0 | 1 | 1 | 0 |
| H3 | 0 | 0 | 0 | 1 | 1 | 1 | 0 | 1 |

Si resumimos por bloque (fracción de alelo `1`):

| Haplotipo | Bloque 1 (SNP1–4) | Bloque 2 (SNP5–8) |
|---|---:|---:|
| H1 | 0.75 | 0.25 |
| H2 | 1.00 | 0.50 |
| H3 | 0.25 | 0.75 |

Así se ve más claro el patrón global de LD (menos ruido SNP a SNP).

**Relación con Tag SNPs:**  
Si en cada bloque los SNPs están muy correlacionados, puede bastar **1 Tag SNP por bloque** (por ejemplo SNP2 para bloque 1 y SNP6 para bloque 2), reduciendo de 8 SNPs a 2 con pérdida limitada de información.

### Gráfica 0c — Zoom de la matriz de haplotipos (configurable)

Para facilitar la inspección visual, esta figura muestra una submatriz pequeña de `H` usando `N_HAP_VIEW` y `N_SNP_VIEW`.

### Gráfica 1 — Distribución de frecuencia alélica por SNP

Esta figura muestra la proporción de alelo `1` por SNP 

frecuencia = conteo de `1` en un SNP / el número total de haplotipos

Muestra cómo se reparten las frecuencias del alelo `1` entre SNPs. Valores alrededor de 0.5 indican SNPs más balanceados (más informativos), mientras que valores cercanos a 0 o 1 indican SNPs casi constantes.

#### Explicación de ejemplo

![image.png](attachment:image.png)

Esta gráfica muestra la distribución de la frecuencia del alelo 1 por SNP:

Eje X: frecuencia de alelo 1 en cada SNP (proporción de haplotipos con valor 1).
Eje Y: cuántos SNPs caen en cada rango de frecuencia.
La línea azul (KDE) suaviza la tendencia general.

**Lectura rápida del resultado:**

El pico principal está alrededor de 0.45–0.52 → muchos SNPs están relativamente balanceados.
Hay menos SNPs en extremos bajos/altos, pero aparece una cola hacia 0.60–0.72.
No se ve masa cerca de 0 o 1, así que casi no hay SNPs totalmente constantes.

**Interpretación para Tag SNPs:**

SNPs cerca de 0.5 suelen ser más informativos para distinguir haplotipos.
La forma es coherente con datos sintéticos con bloques LD + ruido (flip_prob), donde abundan SNPs útiles y algunos más sesgados por bloque.

### Gráfica 2 — Variabilidad por SNP (desviación típica)

Esta gráfica muestra, para cada SNP, **cuánto varían sus valores entre haplotipos**.  
Si un SNP tiene valores muy parecidos en todos los haplotipos, su desviación típica es baja; si mezcla muchos 0 y 1, es más alta.

Sea $x_{ij}\in\{0,1\}$ el valor del SNP $j$ en el haplotipo $i$, y $N$ el número de haplotipos:

$$
\mu_j = \frac{1}{N}\sum_{i=1}^{N} x_{ij}
$$

$$
\sigma_j = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(x_{ij}-\mu_j)^2}
$$

En SNPs binarios (0/1), si $f_j$ es la frecuencia del alelo 1, entonces:

$$
\sigma_j = \sqrt{f_j(1-f_j)}
$$

Por eso la variabilidad es máxima cerca de $f_j=0.5$ y mínima cuando $f_j\approx 0$ o $f_j\approx 1$.

### Gráfica 3 — Número de alelos 1 por haplotipo

Esta figura muestra la carga de alelos alternativos por haplotipo para detectar perfiles más o menos cargados.

#### Explicación gráfica de ejemplo

![image.png](attachment:image.png)

> Gráfico de barras: **Número de alelos 1 por haplotipo**

Esta figura visualiza la "carga" de variantes de cada individuo o cromosoma.

- **Eje X:** Índice del haplotipo (cada barra es un individuo/cromosoma distinto).
- **Eje Y:** Cantidad total de posiciones con valor `1` en ese haplotipo.

**Cómo interpretarla:**

- **Altura de la barra:** Indica cuántas variantes tiene ese haplotipo respecto a la referencia (asumiendo 0=ref, 1=alt).
- **Uniformidad:** Si todas las barras son parecidas, los haplotipos tienen una complejidad similar.
- **Desequilibrio:** Si hay barras muy bajas (pocos 1s) frente a otras muy altas (muchos 1s), indica que algunos haplotipos son muy parecidos a la secuencia base y otros muy divergentes.
- **Detección de outliers:** Una barra extremadamente baja o alta podría indicar un haplotipo anómalo, aunque en datos sintéticos suele reflejar la variabilidad natural simulada.

### Gráfica 4 — Distribución de distancias de Hamming entre pares de haplotipos

Esta figura resume cuán diferentes son los haplotipos entre sí considerando todos los SNPs.

#### Explicación de gráfica de ejemplo

![image.png](attachment:image.png)

> Histograma: **Distribución de distancia de Hamming entre haplotipos**

Esta figura resume cuántas diferencias de SNP hay entre cada par de haplotipos.

- **Eje X:** distancia de Hamming (número de SNPs distintos entre dos haplotipos).
- **Eje Y:** número de pares de haplotipos que presentan esa distancia.
- **Barras:** frecuencia observada por intervalos de distancia.
- **Línea KDE:** tendencia suavizada de la distribución.

**Cómo interpretarla rápidamente:**

- Un pico central indica que la mayoría de pares tienen una separación “típica” similar.
- Una distribución ancha sugiere heterogeneidad: conviven pares muy parecidos y otros muy diferentes.
- Si casi toda la masa estuviera en valores bajos, habría poca capacidad de discriminación.
- La presencia de distancias medias/altas respalda que existe señal útil para distinguir haplotipos y, por tanto, potencial para seleccionar Tag SNPs informativos.

## 🔍 Verificación rápida de estructura LD

Esta celda comprueba si los datos sintéticos realmente presentan correlación entre SNPs y, por tanto, si hay estructura útil para selección de Tag SNPs.

### ¿Qué es la correlación?

La **correlación** mide cuánto varían juntas dos variables. Aquí las variables son dos SNPs (dos columnas binarias 0/1 de la matriz `H`) observados en los mismos haplotipos.

- Correlación **positiva**: cuando un SNP toma 1, el otro tiende también a 1.
- Correlación **negativa**: cuando uno tiende a 1, el otro tiende a 0.
- Correlación cerca de **0**: no hay patrón lineal claro.

En práctica genética, esto se usa como aproximación rápida de estructura LD (aunque LD clásico suele reportarse también con medidas como `r²` o `D'`).

### ¿Por qué aquí se calcula correlación y no LD “directo”?

Se usa correlación en esta etapa por **practicidad** y porque funciona bien como diagnóstico preliminar de dependencia entre SNPs.

- En SNPs binarios bialélicos, la correlación de Pearson (`r`) ya mide asociación entre loci.
- Además, está directamente relacionada con el LD clásico: en práctica se usa mucho `r²` como medida de LD.
- Para una verificación rápida global, la correlación permite resumir, visualizar y comparar fácilmente (heatmap, histograma, CDF).

Qué implica esto en el notebook:

- Esta celda **no sustituye** un análisis formal de LD; actúa como chequeo de que hay señal estructurada antes de optimizar.
- Si se quisiera un reporte genético más canónico, se podría añadir cálculo explícito de `r²` y/o `D'` por pares de SNPs.

### Qué hace la celda de código (paso a paso)

1. **Calcula la matriz completa de correlación SNP×SNP** usando todos los SNPs.

2. **Filtra casos degenerados** (valores no finitos) para evitar correlaciones no informativas.

3. **Extrae todos los pares únicos de SNPs** (triángulo superior, sin diagonal).

4. **Resume la señal global** con la correlación media absoluta (`ld_mean`).

5. **Visualiza la estructura y la distribución** con:
   - Heatmap SNP×SNP completo.
   - Histograma de correlaciones de todos los pares.
   - CDF de `|correlación|` (porcentaje acumulado bajo cada umbral).

6. **Emite diagnóstico final** con umbrales explícitos:
   - valor de `ld_mean`,
   - `% de pares con `|corr| >= 0.20`,
   - `es_optimo` (booleano),
   - estado `ÓPTIMO / ACEPTABLE / NO ÓPTIMO` con icono.

### Cómo se interpretan los umbrales del diagnóstico

En el código se usan estos valores:
- `THRESH_OPT_MEAN = 0.15`
- `THRESH_OPT_ABS_CORR = 0.20`
- `THRESH_OPT_PCT = 30%`
- `THRESH_ACCEPT_MEAN = 0.10`

La lógica es:
- **ÓPTIMO (✅)**: `ld_mean >= 0.15` **y además** al menos un `30%` de pares tiene `|corr| >= 0.20`.
- **ACEPTABLE (⚠️)**: no llega a óptimo, pero `ld_mean >= 0.10`.
- **NO ÓPTIMO (❌)**: `ld_mean < 0.10`.

Interpretación práctica:
- `ld_mean` mide la intensidad media global de asociación entre SNPs.
- El `% con |corr|>=0.20` controla que no sea solo una media “inflada” por pocos pares extremos.
- Exigir ambas condiciones en “ÓPTIMO” da un criterio más robusto para validar estructura LD útil antes de optimizar.

### Por qué importa

Si la correlación fuese casi nula, el problema de Tag SNP perdería sentido práctico (habría poca redundancia para comprimir). Si es moderada/alta, sí hay dependencia aprovechable para seleccionar SNPs representativos.

**Lectura del mapa completo.** Este heatmap muestra el patrón de desequilibrio de ligamiento: los bloques rojos indican SNPs que cambian juntos (alta correlación) y los azules combinaciones inversas. Las diagonales intensas confirman que los bloques sintéticos conservan coherencia y justifican la existencia de Tag SNPs representativos.

### 📊 Gráfica — Distribución de correlaciones de Pearson entre pares de SNPs

Esta figura muestra cómo se reparten los valores de correlación lineal entre **todos los pares** de SNPs.

- Valores cerca de `0` indican pares casi independientes.
- Valores con magnitud alta (`|corr|` grande) indican dependencia fuerte.
- La forma global ayuda a estimar cuánta redundancia hay en el conjunto de SNPs antes de seleccionar Tag SNPs.

**Distribución de pares.** El histograma confirma que la mayoría de correlaciones se concentran cerca de 0, pero existe una cola positiva pronunciada y otra negativa suave. Esa mezcla implica que hay pares muy redundantes (|corr| alto) y otros casi independientes, lo que respalda trabajar con criterios de selección basados en cobertura y diversidad.

### 📈 Gráfica — CDF de la correlación absoluta |corr|

Esta figura muestra la **distribución acumulada** de `|corr|` entre pares de SNPs.

- Para cada umbral en el eje X, el eje Y indica qué fracción de pares tiene `|corr|` menor o igual a ese valor.
- Si la curva sube lentamente, hay más pares con dependencia alta (señal útil para Tag SNPs).
- Si sube muy rápido cerca de 0, predominan pares poco correlacionados.

### 🧭 Interpretación detallada de la CDF de |corr|

Esta curva responde a la pregunta: **¿qué fracción de pares de SNPs tiene una correlación absoluta menor o igual que un umbral dado?**

- Eje X: umbral de `|corr|` (fuerza de asociación, ignorando signo).
- Eje Y: proporción acumulada de pares con `|corr| <= X`.

#### Cómo leer un punto concreto

Si en la curva aparece aproximadamente el punto `X=0.25`, `Y=0.60`, significa:

- El **60%** de los pares tiene `|corr| <= 0.25`.
- Por complemento, el **40%** de los pares tiene `|corr| > 0.25`.

Ese 40% representa pares con asociación relativamente más fuerte (potencialmente aprovechables para compresión con Tag SNPs).

#### Regla rápida de interpretación global

- **Curva que sube muy pronto** (cerca de X pequeños): predominan pares con baja dependencia.
- **Curva que sube más tarde**: hay más pares con `|corr|` medio/alto, es decir, más estructura LD útil.
- **Curva muy desplazada a la derecha**: mayor proporción de dependencias fuertes entre SNPs.

#### Ejemplos mentales útiles

- Si en `X=0.10` ya estás en `Y=0.80`, entonces solo un 20% supera `|corr|=0.10` (dependencia global más débil).
- Si en `X=0.30` sigues en `Y=0.50`, entonces la mitad de pares supera `|corr|=0.30` (dependencia global más marcada).

#### Conexión con Tag SNPs

Cuanta mayor sea la fracción de pares con `|corr|` moderado/alto, mayor redundancia existe entre SNPs y más sentido tiene seleccionar un subconjunto representativo de Tag SNPs.

## 🧮 Preparación de pares de haplotipos



La siguiente celda precomputa todas las combinaciones de pares de haplotipos.



### ¿Para qué se usa?

Las funciones objetivo comparan haplotipos entre sí para medir capacidad de distinción. Reutilizar una lista de pares evita recalcular combinaciones en cada evaluación.



### Ventaja técnica

Mejora eficiencia y hace el código más limpio.



### Salida esperada

- `PAIR_IDX`: índices de pares.

- `N_PAIRS`: cantidad total de comparaciones posibles.

### Ejemplos de pares muy similares

En la siguiente celda se muestran los pares de haplotipos con **menor distancia de Hamming**.

Interpretación rápida:
- Menor distancia ⇒ haplotipos más parecidos.
- Suelen compartir gran parte de la secuencia en el prefijo mostrado.

### Ejemplos de pares muy distintos

En la siguiente celda se muestran los pares de haplotipos con **mayor distancia de Hamming**.

Interpretación rápida:
- Mayor distancia ⇒ haplotipos más diferentes.
- Estos pares ayudan a visualizar la máxima separación presente en los datos.

## 🎯 Evaluación de candidatos (núcleo del problema)



La siguiente celda define cómo puntuar una solución de selección de SNPs.



### Qué representa una solución

Un vector binario de longitud `N_SNPS`:

- `1` = SNP seleccionado como posible Tag SNP.

- `0` = SNP descartado.



### Objetivos que se calculan

Se evalúan 4 objetivos por cada solución (máscara de SNPs):

**1) Compactness (`f1`)**
Este objetivo mide cuántos SNPs se están usando en la solución. Cuanto menor sea, más compacta y barata es la selección (menos marcadores que genotipar, menos complejidad y mayor interpretabilidad). El reto es que reducir demasiado `f1` puede hacer que se pierda capacidad para distinguir haplotipos, por eso se optimiza junto al resto de objetivos y no de forma aislada.

**2) Tolerance real (`f2`, peor caso)**
La tolerancia real representa la separación del par de haplotipos más difícil de distinguir (el mínimo de todas las distancias entre pares, la distancia del par de haplotipos más similar). Es una métrica de robustez: si este valor sube, significa que incluso en el peor escenario hay discriminación suficiente. En términos prácticos evita soluciones que funcionen “de media” pero fallen en casos límite.

**3) Hamming media real (`f3`)**
Esta métrica resume la separación promedio entre todos los pares de haplotipos con los SNPs seleccionados. A diferencia de la tolerancia (que mira el peor caso), aquí se evalúa el comportamiento global del conjunto: valores altos indican que, en general, los haplotipos quedan bien diferenciados. Es útil para asegurar calidad global de discriminación y no solo en extremos.

**4) Balance var (`f4`)**
`f4` mide la variabilidad de esas distancias de Hamming entre pares, es decir, cuán homogénea es la discriminación. Una varianza baja indica comportamiento más estable y equilibrado (no depende de unos pocos pares muy separados y otros casi indistinguibles). En consecuencia, minimizar `f4` ayuda a evitar soluciones descompensadas y favorece un rendimiento más consistente.



### Aclarando algunos conceptos

#### ¿Qué significa “Tolerance real” en profundidad?

Sea `d_ij` la distancia de Hamming entre los haplotipos `i` y `j` usando solo los SNPs seleccionados.

- Se calcula el vector de distancias de todos los pares:  
  `d = [d_12, d_13, ..., d_(n-1,n)]`

- **Tolerance real** es el peor caso, es decir, el par más cercano (el mínimo de ese vector):  
  `tolerance_real = min(d)`

Interpretación:

- Si `tolerance_real` es **alta**, incluso el par más difícil (que más cuesta distinguir) sigue estando bien separado.

- Si `tolerance_real` es **baja** (o 0), existe al menos un par casi indistinguible con la selección actual.

Por eso biológicamente queremos **maximizar** esta métrica: mejora la robustez de discriminación en el peor caso.



### ¿Por qué se invierte el signo en algunos objetivos?

`pymoo` minimiza por defecto. Entonces:

- Lo que queremos minimizar se deja igual.

- Lo que queremos maximizar se multiplica por `-1` para convertirlo en minimización equivalente.

En este notebook:

- `f1 = compactness` (minimizar)

- `f2 = -tolerance_real` (equivale a maximizar `tolerance_real`)

- `f3 = -hamming_avg_real` (equivale a maximizar la separación media)

- `f4 = balance_var` (minimizar)



### Ejemplo numérico completo (con los 4 objetivos)

Supón que, para una máscara concreta, las distancias entre todos los pares de haplotipos son:

`d = [1, 3, 2, 4, 2, 5]`

Y que esa máscara selecciona `k = 7` SNPs.

1. **Compactness real**

- `compactness = k = 7`

- Objetivo: `f1 = 7` (cuanto menor, mejor).

2. **Tolerance real (peor caso)**

- `tolerance_real = min(d) = 1`

- En optimización: `f2 = -1`

- Si otra solución tiene `tolerance_real = 2`, entonces `f2 = -2` y es mejor porque `-2 < -1` (menor en minimización).

3. **Hamming media real**

- `hamming_avg_real = mean(d) = (1+3+2+4+2+5)/6 = 17/6 ≈ 2.833`

- En optimización: `f3 = -2.833`

- Una media mayor implica mejor separación global entre haplotipos.

4. **Balance var**

- `balance_var = var(d)` (varianza poblacional en el código, `numpy.var`).

- Con este `d`, `balance_var ≈ 1.806`.

- Objetivo: minimizarla para evitar soluciones con separaciones muy descompensadas.

Vector final que optimiza el algoritmo:

`F = [f1, f2, f3, f4] = [7, -1, -2.833, 1.806]`



### Lectura conjunta de los 4 objetivos

- `f1` bajo: selección compacta (menos SNPs).

- `f2` bajo (más negativo): mejor peor-caso entre pares.

- `f3` bajo (más negativo): mejor separación media global.

- `f4` bajo: distancias más homogéneas (menos desequilibrio).

No existe una única solución perfecta para todo; por eso se usa optimización multiobjetivo y se obtiene un frente de Pareto.



### Detalle importante

Si una solución sale vacía, se repara añadiendo un SNP para evitar errores.

Por ejemplo, si tras una mutación una solución queda completa de 0, se vuelve a mutar manualmente para que tenga al menos un 1, es decir, un SNP random seleccionado.

Este caso también puede ocurrir en la inicialización random

### 🧭 Cómo se evalúa una solución y una población

Esta celda define dos funciones clave para que el algoritmo pueda comparar candidatos.

#### `evaluate_candidate()`

- Recibe una solución (máscara de SNPs) y calcula qué tan buena es.
- Si la solución viene vacía, la corrige para que tenga al menos un SNP activo.
- Con los SNPs seleccionados, mide cómo de bien se separan los haplotipos.
- Devuelve los 4 objetivos en el formato que necesita `pymoo`, junto con métricas auxiliares para interpretar resultados.

#### `evaluate_population()`

- Toma una población completa de soluciones.
- Aplica la evaluación anterior a cada individuo.
- Devuelve una matriz con los objetivos de toda la población y una lista con información extra por solución.

En resumen: una función evalúa **una** solución y la otra evalúa **todas**; juntas conectan el problema biológico con la optimización multiobjetivo.

## 🧩 Definición formal del problema en pymoo



La siguiente celda encapsula el problema en una clase compatible con `pymoo`.



### ¿Qué hace esta clase?

- Declara número de variables (`n_var = N_SNPS`).

- Declara número de objetivos (`n_obj = 4`).

- Indica tipo binario (`vtype=bool`): para la máscara de 1s y 0s

- Implementa `_evaluate` para devolver la matriz de objetivos `F`.

##### ¿Qué es la matriz de objetivos `F`?

En `pymoo`, la matriz `F` contiene la evaluación de todas las soluciones candidatas de una población.

- **Filas (`n_individuos`)**: cada fila corresponde a una solución (un vector binario de selección de SNPs).
- **Columnas (`n_objetivos = 4`)**: cada columna corresponde a un objetivo del problema.

En este notebook, cada fila de `F` tiene esta forma:

$$
F_i = [f_1, f_2, f_3, f_4]
$$

donde:

- $f_1$: `compactness` (número de SNPs seleccionados, a minimizar).
- $f_2$: `-tolerance_real` (se niega para convertirlo en minimización).
- $f_3$: `-hamming_avg_real` (también negado para minimizar).
- $f_4$: `balance_var` (varianza de distancias, a minimizar).

Por eso en `_evaluate` se calcula `F` para toda la población y se asigna en `out['F']`: así el algoritmo puede comparar soluciones y construir el frente de Pareto.



### Por qué es necesaria esta clase `pymoo`

Los algoritmos de `pymoo` requieren una interfaz estándar para poder optimizar cualquier problema.



### Salida esperada

No imprime resultados por sí sola; deja listo el objeto problema para las celdas siguientes.

## 🛠️ Heurística greedy basada en distinguibilidad



Esta celda define cómo construir soluciones iniciales inteligentes.



### Idea principal

Se calcula cuántos pares de haplotipos distingue cada SNP y se priorizan los más informativos.



### Flujo simplificado

1. Medir distingibilidad por SNP.

2. Ordenar SNPs de mayor a menor contribución.

3. Añadir SNPs mientras mejoren la cobertura de pares.



### Ventaja

Lo que realmente importa es que el subconjunto de marcadores seleccionados mantenga la identidad única de cada secuencia genética (haplotipo) presente en la muestra original.

Desde un punto de vista genómico y computacional, la información que se busca distinguir se centra en tres pilares fundamentales:

1. La unicidad de los haplotipos
El objetivo principal es evitar el "colapso" de la información. Si dos individuos tienen secuencias de ADN diferentes en la matriz completa, deben seguir siendo distinguibles tras la reducción de datos. Si al seleccionar solo unos pocos marcadores (Tag SNPs) dos personas diferentes terminan con el mismo patrón de bits, se pierde la capacidad de realizar estudios de asociación precisos, ya que el sistema las trataría como si fueran genéticamente idénticas.

2. La representatividad de la población
Nos importa que los marcadores elegidos capturen la diversidad de la muestra. En una población, existen bloques de ADN que se heredan juntos debido al desequilibrio de ligamiento. La información relevante aquí es la capacidad de un marcador para actuar como "embajador" de un bloque entero. Si un SNP distingue bien entre los diferentes bloques existentes en la población, se considera que tiene un alto valor informativo.

3. La variabilidad biológica significativa
No todos los cambios en el ADN tienen el mismo peso. Interesa distinguir aquellas variaciones que separan grupos con características distintas, como la propensión a una enfermedad o la respuesta a un medicamento. La información que importa distinguir es aquella que permite reconstruir la estructura genética global con el menor error posible.

En los modelos de optimización matemática, esto se traduce en maximizar el número de pares de secuencias que presentan valores distintos en las posiciones elegidas. Al asegurar que la mayoría de los pares de individuos sean diferentes entre sí en el conjunto reducido, se garantiza que la integridad de la información biológica se mantenga a pesar de haber eliminado la mayor parte de los datos originales.



### Nota biológica

Este enfoque favorece SNPs con mayor capacidad de separar variabilidad haplotípica.

### Ejemplo de inicialización Greedy basada en la Capacidad de Distinción

En esta sección se detalla el proceso de generación de la población inicial propuesto en el artículo de referencia. Este método no se basa en la aleatoriedad total, sino que utiliza una estrategia codiciosa (greedy) para "sembrar" soluciones de alta calidad que aceleren la convergencia del algoritmo evolutivo.

#### 1. Matriz de Haplotipos de Ejemplo
Se considera un escenario simplificado con 4 individuos (haplotipos) y 3 marcadores potenciales (SNPs). El objetivo es distinguir los 6 pares posibles de individuos: (H1,H2), (H1,H3), (H1,H4), (H2,H3), (H2,H4) y (H3,H4).

| Individuo | SNP 1 | SNP 2 | SNP 3 |
| :--- | :---: | :---: | :---: |
| **H1** | 0 | 0 | 0 |
| **H2** | 1 | 0 | 0 |
| **H3** | 1 | 1 | 0 |
| **H4** | 1 | 1 | 1 |

#### 2. Cálculo de la Capacidad de Distinción
Se define la capacidad de distinción como el número de pares de individuos que un SNP puede diferenciar por sí solo (presentando valores distintos, 0 vs 1). Se procede al cálculo para cada marcador:

* **SNP 1 (0, 1, 1, 1):** Se diferencia a H1 de los demás individuos. Se obtienen **3 pares** distinguidos: (H1,H2), (H1,H3) y (H1,H4).
* **SNP 2 (0, 0, 1, 1):** Se diferencia al grupo {H1, H2} del grupo {H3, H4}. Se obtienen **4 pares** distinguidos: (H1,H3), (H1,H4), (H2,H3) y (H2,H4).
* **SNP 3 (0, 0, 0, 1):** Se diferencia a H4 del resto de individuos. Se obtienen **3 pares** distinguidos: (H1,H4), (H2,H4) y (H3,H4).



#### 3. Clasificación y Ranking de Marcadores
Se realiza una ordenación descendente de los SNPs basada en su puntuación de distinción. Este ranking actúa como una guía de "calidad informativa" para la construcción de los individuos:
1.  **SNP 2** (Puntuación: 4)
2.  **SNP 1** (Puntuación: 3)
3.  **SNP 3** (Puntuación: 3)

#### 4. Construcción de Individuos en la Población Inicial
Para evitar soluciones idénticas y cubrir el Frente de Pareto, se generan individuos con diferentes niveles de cobertura de forma escalonada. Se seleccionan los marcadores siguiendo estrictamente el orden del ranking:

* **Individuo A (Baja cobertura / Alta compacidad):** Se selecciona únicamente el primer marcador del ranking. El cromosoma resultante es `[0, 1, 0]`. Se logra distinguir 4 de los 6 pares.
* **Individuo B (Cobertura media):** Se seleccionan los dos primeros marcadores del ranking (SNP 2 y SNP 1). El cromosoma resultante es `[1, 1, 0]`. Se logra distinguir 5 de los 6 pares (se añade la distinción del par H1-H2).
* **Individuo C (Cobertura total / Alta precisión):** Se añaden marcadores del ranking hasta que todos los pares (6/6) quedan cubiertos. Se seleccionan SNP 2, SNP 1 y SNP 3. El cromosoma resultante es `[1, 1, 1]`.



#### 5. Justificación del Método
Se emplea esta técnica para garantizar que la población inicial no contenga únicamente ruido estadístico. Mediante la creación de esta "escalera" de soluciones:
1.  **Se asegura la diversidad:** Se obtienen puntos en diferentes regiones del espacio de búsqueda (soluciones con pocos SNPs y soluciones con muchos).
2.  **Se acelera la convergencia:** Al partir de marcadores con alta capacidad de distinción, se reduce drásticamente el número de generaciones necesarias para encontrar soluciones óptimas en términos de disimilitud y compacidad.

## 🌱 Muestreador híbrido random + greedy (`greedy_hybrid`)

La siguiente celda define la inicialización **híbrida** que combina construcción codiciosa con una fracción aleatoria.

### Qué resuelve

Un inicio puramente greedy puede perder diversidad, y uno puramente random puede ser lento. Esta estrategia mezcla ambos mundos.

### Estrategia

- Una fracción de individuos se construye con greedy (`greedy_construct`).
- La fracción aleatoria se controla con `random_fill_ratio` (por defecto 0.2 = 20% random, 80% greedy).
- Se fuerza que ninguna solución sea vacía.

### Resultado práctico

Se consigue una población inicial diversa pero con buena base de calidad.

### Nota de nomenclatura

A partir de este punto, esta estrategia se referirá como `greedy_hybrid` para distinguirla de la nueva inicialización `greedy_pure`.

## 🌿 Nueva inicialización 100% codiciosa (`greedy_pure`)

Se añade una segunda estrategia codiciosa, separada de la híbrida.

### Diferencia clave frente a `greedy_hybrid`

- `greedy_hybrid`: combina greedy + una fracción aleatoria.
- `greedy_pure`: genera **toda** la población mediante construcción greedy con distintos niveles de cobertura.

### Objetivo experimental

Permitir comparar explícitamente tres condiciones de arranque:
- `random`
- `greedy_hybrid`
- `greedy_pure`

Esto incrementa el diseño experimental y afecta tiempos, número de ejecuciones y visualizaciones.

## 🤖 Construcción de algoritmos y direcciones de referencia



La siguiente celda ensambla los cuatro algoritmos multiobjetivo con operadores y parámetros comunes.



### Puntos clave

- Se ajusta automáticamente `ref_dirs` para que no superen `POP_SIZE`.

- Se centraliza la creación de NSGA2, NSGA3, SPEA2 y MOEA/D.

- Se construyen versiones con inicialización `random` y `greedy`.

## 🧭 ¿Qué son las Direcciones de Referencia (`ref_dirs`)?

En algoritmos de optimización de muchos objetivos (como **NSGA-III** o **MOEA/D**), las **`ref_dirs`** son vectores unitarios distribuidos uniformemente que apuntan hacia el frente de Pareto ideal.

### ¿Por qué son necesarias? 
Cuando un problema tiene 3 o más objetivos (en este caso tenemos **4**), el concepto de "dominancia de Pareto" pierde eficacia: casi todas las soluciones se vuelven no dominadas entre sí. Las direcciones de referencia resuelven esto:
1.  **Guían la búsqueda**: Actúan como "puntos de anclaje" que el algoritmo intenta alcanzar.
2.  **Garantizan diversidad**: Obligan al algoritmo a encontrar soluciones en todas las regiones del frente, evitando que se agrupen solo en una zona.

### Implementación en este Notebook
Utilizamos el método de **Das-Dennis** para generar estas direcciones:
- Se crean particiones en un hiperplano para cubrir el espacio de objetivos de forma estructurada.
- La función `build_ref_dirs_for_pop` ajusta automáticamente el número de direcciones para que coincida lo mejor posible con nuestro `POP_SIZE`, asegurando que el algoritmo trabaje de forma eficiente con la población definida.

### Por qué esta celda es crítica

Garantiza comparabilidad justa entre algoritmos: todos comparten el mismo problema y los mismos operadores base.



### Salida esperada

- Número de configuraciones generadas.

- Número de direcciones de referencia usadas (especialmente relevante en NSGA3/MOEA/D).

## 🚀 Bucle principal de experimentos

La siguiente celda define la estructura de ejecución de todos los experimentos.

### Qué incluye

- Dataclass `RunResult` para guardar resultados de cada corrida.
- Función `run_all_experiments` que recorre algoritmos, inicializaciones y semillas.
- Medición de tiempo por ejecución y almacenamiento del frente final.

### Diseño experimental implementado

- 4 algoritmos.
- 3 tipos de inicialización: `random`, `greedy_hybrid`, `greedy_pure`.
- Varios runs por configuración (según `N_RUNS`).

### Por qué esta celda es esencial

Es el motor del notebook: sin ella no hay datos para comparar métodos.

### Qué salida esperar

Mensajes de progreso del tipo `[alg-init] run x/y ...` y número de soluciones finales.

## ▶️ Interruptor de ejecución pesada



Esta celda decide si se lanzan los experimentos o no.



### Variable clave

- `RUN_EXPERIMENTS = True`: ejecuta los runs y genera `run_results`.

- `RUN_EXPERIMENTS = False`: evita coste computacional y deja lista vacía.



### Uso recomendado

- Durante edición del notebook: mantener en `False`.

- Cuando esté todo validado: poner en `True`.



### Resultado esperado

- Si está activo: trazas de ejecución por algoritmo/run.

- Si no: mensaje de omisión segura.

## 📐 Normalización global y métricas finales



La siguiente celda implementa el bloque matemático de evaluación final.



### Qué se hace exactamente

1. Unir todos los frentes finales de todos los runs.

2. Calcular `ideal` y `nadir` globales.

3. Aplicar normalización Min-Max común.

4. Calcular métricas del paper sobre valores normalizados.

5. Calcular métricas auxiliares en escala original.

### 1. Unir todos los frentes de todos los runs

Debido a la naturaleza estocástica de los algoritmos evolutivos, los resultados pueden presentar variaciones entre distintas ejecuciones. Por este motivo, se realiza una unión de todos los frentes de Pareto obtenidos en las $N$ ejecuciones independientes para cada configuración experimental (combinación de algoritmo y método de inicialización).

Mediante este proceso, se agrupan todas las soluciones consideradas óptimas de forma local en cada ejecución para formar un conjunto global. El objetivo es evitar la pérdida de soluciones de alta calidad que hayan podido ser descubiertas únicamente en una ejecución específica. De este modo, se obtiene una visión completa del rendimiento de cada configuración antes de proceder a la identificación de las mejores soluciones globales.

#### Ejemplo de aplicación:

Supóngase que se han realizado **3 ejecuciones** para la configuración `NSGA-III + Greedy`:

1.  **Ejecución 1:** Se obtiene un frente con las soluciones $\{A, B\}$.
2.  **Ejecución 2:** Se obtiene un frente con las soluciones $\{C, D\}$.
3.  **Ejecución 3:** Se obtiene un frente con las soluciones $\{E, F\}$.

Tras aplicar este paso de consolidación, se genera un único conjunto que contiene la unión de todos los resultados: $\{A, B, C, D, E, F\}$. 



Este "mega-frente" temporal servirá como base para el siguiente paso, donde se descartarán aquellas soluciones que hayan quedado obsoletas o dominadas al compararlas con los hallazgos del resto de las ejecuciones.

### 2. Cálculo de los puntos ideal y nadir globales

Tras la consolidación de los frentes de todas las ejecuciones, **se procede** a determinar los puntos **ideal** (/i.ðe.ˈal/) y **nadir** (/na.ˈðir/) globales. Este proceso es fundamental para definir los límites físicos del espacio de los objetivos y permitir una normalización coherente de los datos antes de proceder al análisis de métricas.



* **Punto Ideal Global ($Z^*$)**: **Se calcula** identificando el mejor valor (mínimo en este problema) para cada uno de los 4 objetivos entre todas las soluciones del conjunto consolidado. Representa una solución utópica donde cada criterio alcanza su máximo rendimiento de forma simultánea.
* **Punto Nadir Global ($Z^{nad}$)**: **Se obtiene** al identificar el peor valor para cada objetivo dentro del conjunto de soluciones no dominadas globales. Este punto delimita el rango de compromiso y sirve como frontera superior para el escalado.

Al utilizar valores globales derivados de la totalidad de las ejecuciones, **se asegura** que la comparación entre los distintos algoritmos y métodos de inicialización **se realice** bajo un mismo marco de referencia, eliminando sesgos métricos producidos por las variaciones locales de cada experimento individual.

#### Ejemplo ilustrativo:

Considérese un escenario simplificado con dos objetivos donde **se han agrupado** soluciones de diversas pruebas:
* Conjunto de soluciones consolidadas: $S_1(2, 80)$, $S_2(5, 50)$, $S_3(8, 20)$.

Para determinar los puntos de referencia globales:
1.  **Identificación del Ideal**: **Se seleccionan** los mejores valores de cada columna $\rightarrow \min(2, 5, 8) = 2$ y $\min(80, 50, 20) = 20$. El **Ideal Global** resultante es $(2, 20)$.
2.  **Identificación del Nadir**: **Se seleccionan** los peores valores de cada columna $\rightarrow \max(2, 5, 8) = 8$ y $\max(80, 50, 20) = 80$. El **Nadir Global** resultante es $(8, 80)$.



Con estos límites establecidos, cualquier solución obtenida durante el proceso **se normaliza** matemáticamente al rango $[0, 1]$, permitiendo que objetivos con magnitudes muy dispares (como la precisión y el número de SNPs) puedan ser comparados de forma equitativa.

### 3. Normalización Min-Max común

Una vez determinados los puntos de referencia globales, se procede a realizar la **normalización min-max común**. Este paso es crítico en la optimización multiobjetivo, ya que permite transformar las métricas de todos los algoritmos y ejecuciones a un rango común entre **0 y 1**.



Al emplear los mismos valores de **ideal** ($Z^*$) y **nadir** ($Z^{nad}$) para todos los datos, se garantiza que las soluciones sean comparables entre sí de manera justa. Sin esta normalización «común», una mejora en un objetivo con magnitudes grandes (como el número de SNPs) podría parecer matemáticamente más importante que una mejora en un objetivo con magnitudes pequeñas (como la disimilitud), sesgando los resultados del experimento.

La transformación se realiza mediante la siguiente fórmula para cada objetivo $i$:

$$f_{i}^{norm} = \frac{f_i - z_{i}^{*}}{z_{i}^{nad} - z_{i}^{*}}$$

Donde:
* $f_{i}^{norm}$ es el valor resultante tras la normalización.
* $f_i$ es el valor original del objetivo.
* $z_{i}^{*}$ es el componente $i$ del punto ideal global.
* $z_{i}^{nad}$ es el componente $i$ del punto nadir global.

#### Ejemplo de aplicación:

Supóngase que el objetivo es minimizar el **Número de SNPs** y se han obtenido los siguientes límites globales:
* **Ideal global ($z^*$):** 5 SNPs.
* **Nadir global ($z^{nad}$):** 105 SNPs.

Si **se evalúan** dos soluciones de distintas ejecuciones:
1.  **Solución A (15 SNPs):** Se normaliza como $(15 - 5) / (105 - 5) = 10 / 100 = \mathbf{0.10}$.
2.  **Solución B (55 SNPs):** Se normaliza como $(55 - 5) / (105 - 5) = 50 / 100 = \mathbf{0.50}$.



Gracias a este procedimiento, todos los objetivos (independientemente de su unidad original) quedan representados en una escala donde el **0** representa el mejor rendimiento alcanzado en el experimento y el **1** el peor límite de compromiso. Esto permite calcular posteriormente métricas agregadas de calidad de forma robusta.

### 4. Cálculo de métricas de calidad sobre valores normalizados

Tras haber transformado las soluciones de todas las ejecuciones al rango $[0, 1]$, se procede al cálculo de las métricas de rendimiento. Este paso es fundamental para cuantificar de forma objetiva la calidad de los frentes de Pareto obtenidos por cada algoritmo y configuración experimental.

El uso de valores normalizados es estrictamente necesario en esta fase. Dado que las métricas de calidad se basan generalmente en distancias geométricas (euclídeas), si se utilizaran los valores originales, los objetivos con magnitudes mayores (como el número de SNPs) tendrían un peso desproporcionado en el resultado final. La normalización asegura que cada uno de los cuatro objetivos contribuya por igual a la evaluación del desempeño.



Siguiendo la metodología del estudio de referencia, se calculan las siguientes métricas:

#### 1. Range (Extensión o Amplitud)

La métrica **Range** se utiliza para cuantificar qué tan "ancho" es el frente de Pareto. En el contexto de la selección de Tag SNPs, esto es vital porque indica si el algoritmo ha sido capaz de encontrar todo el espectro de soluciones posibles.

* **Cómo se calcula:** Se mide la distancia euclídea entre las soluciones más extremas del frente normalizado. En un espacio de cuatro objetivos, el valor máximo teórico está relacionado con la diagonal del hipercubo unidad.
* **Qué evalúa:** La capacidad de exploración. Si el valor es alto, significa que se han obtenido desde conjuntos muy pequeños de SNPs (máxima compacidad) hasta conjuntos muy grandes y precisos (mínima disimilitud).
* **Interpretación:** Un Range bajo es señal de un frente "colapsado" o pobre, donde todas las soluciones son muy similares entre sí, lo que limita las opciones de elección para el investigador.


#### 2. SumMin (Cobertura desde los vectores)

Esta métrica se centra en la **distribución y la ausencia de huecos**. Se utiliza para verificar si la red de vectores de referencia (las guías de búsqueda) tiene soluciones asignadas cerca de cada dirección.

* **Cómo se calcula:** El proceso se inicia en los vectores de referencia. Para cada vector $v_j$, se identifica la solución $s_i$ del frente que se encuentra a la distancia mínima. Finalmente, se realiza la suma de todas esas distancias mínimas.
* **Perspectiva:** Es una búsqueda de **Vectores $\rightarrow$ Soluciones**.
* **Qué evalúa:** La cobertura uniforme. Un valor de SumMin bajo indica que el frente es continuo y que no existen regiones del espacio de los objetivos que hayan quedado vacías o desatendidas.
* **Importancia:** Un SumMin bajo garantiza que, sea cual sea el nivel de compromiso que busque el biólogo (por ejemplo, dar mucha importancia al equilibrio y poca a la compacidad), el algoritmo habrá encontrado una solución que se ajusta a ese perfil.


#### 3. MinSum (Convergencia desde las soluciones)

A diferencia de la anterior, la métrica **MinSum** se utiliza para medir la **nitidez y el ajuste** de las soluciones respecto a las direcciones ideales de búsqueda.

* **Cómo se calcula:** El proceso se inicia en las soluciones. Para cada solución $s_i$ encontrada por el algoritmo, se identifica el vector de referencia $v_j$ más cercano. Se realiza la suma de estas distancias.
* **Perspectiva:** Es una búsqueda de **Soluciones $\rightarrow$ Vectores**.
* **Qué evalúa:** La convergencia y la precisión del algoritmo. Si el valor es bajo, significa que las soluciones no están dispersas de forma caótica, sino que están bien alineadas con la estructura geométrica que se definió mediante las divisiones de Das-Dennis.
* **Diferencia clave:** Mientras que SumMin penaliza que falten soluciones en ciertas áreas, MinSum penaliza que las soluciones que existen estén "lejos" de ser óptimas o mal orientadas.


#### 4. Hypervolume (HV)

El **Hipervolumen** mide el tamaño del espacio (el "volumen") que está dominado por las soluciones del frente de Pareto, limitado en el otro extremo por un *Punto de Referencia*.

* **Cómo se calcula:** Dado que los objetivos ya están normalizados al rango $[0, 1]$, donde $0$ es el óptimo, se define un punto de referencia dinámico en **$1.01$** en todas las dimensiones. Se utiliza $1.01$ en lugar de $1.0$ para asegurar que las soluciones extremas que tocan el *nadir* sigan aportando algo de volumen y no se anulen por la penalización restrictiva pura.
* **Perspectiva:** Evalúa tanto la **convergencia** hacia el óptimo verdadero como la **diversidad** o expansión del frente (cobertura), resumiéndolos en un único indicador numérico.
* **Interpretación clave:** A diferencia de Range, SumMin y MinSum donde valores *menores* son mejores, en el Hypervolume **un valor MAYOR es estrcitamente mejor** (representa un área dominada más grande).



#### Interpretación de resultados

Al realizar múltiples ejecuciones, se calculan la media y la desviación típica de estas métricas para cada configuración. Por ejemplo, si una configuración presenta un `SumMin_mean` significativamente menor que otra, se puede afirmar que dicha combinación de algoritmo e inicialización es más eficaz para cubrir de forma uniforme todo el espacio de los objetivos.

### 5. Calcular métricas auxiliares en escala original

Este paso se implementa mediante la función `compute_raw_aux_metrics(F_raw)`, la cual actúa sobre los valores "crudos" (sin normalizar) de los objetivos del frente de Pareto final.

#### Proceso técnico

A diferencia de las métricas de calidad (como el Rango o SumMin) que requieren normalización para ser comparables, las métricas auxiliares se calculan sobre la escala real del problema biológico para facilitar su interpretación. Lo que se hace exactamente es:

1.  **Extracción de objetivos específicos**: Se toman las columnas de la matriz de objetivos `F_raw` que corresponden a la **Tolerancia** (índice 1) y a la **Disimilitud/Hamming** (índice 2).
2.  **Cálculo de la Tasa de Tolerancia (Tolerance Rate)**:
    * Se recuperan los valores de tolerancia de cada solución del frente.
    * **MaxToleranceRate**: Se identifica el valor máximo de tolerancia presente en el frente, indicando la capacidad máxima de distinguir individuos bajo ruido de ese conjunto de soluciones.
    * **AvgToleranceRate**: Se calcula la media aritmética (`np.nanmean`) de los valores de tolerancia de todas las soluciones del frente, lo que da una idea del rendimiento promedio en este objetivo.
3.  **Cálculo de la Distancia de Hamming promedio**:
    * **AvgHammingDistance**: Se extrae el objetivo de disimilitud (que en el problema se define como la distancia de Hamming acumulada o promedio entre pares).
    * Se calcula el promedio global de este valor para todo el frente de Pareto.

### Significado de los resultados
Estos cálculos permiten al investigador observar resultados en unidades tangibles:
* **Tolerancia**: Se expresa en la cantidad de pares de haplotipos que los Tag SNPs son capaces de diferenciar.
* **Hamming**: Indica cuántos bits (SNPs) de diferencia hay en promedio entre individuos usando solo los marcadores seleccionados.

Al final de este proceso, se integran estos valores en un `DataFrame` final para que puedan ser visualizados en las tablas de resumen estadístico junto a las medias y desviaciones típicas de cada ejecución.

#### ¿Por qué solo estos dos objetivos?

Se seleccionan únicamente los objetivos de **Tolerancia** (índice 1) y **Disimilitud/Hamming** (índice 2) para el cálculo de las métricas auxiliares debido a la naturaleza de los datos y a los objetivos del experimento.

A continuación se detallan los motivos específicos por los cuales se omiten los otros dos objetivos en este paso:

### 1. Exclusión de la Compacidad (Objetivo 0)
El objetivo de **Compacidad** representa simplemente el recuento total de SNPs seleccionados en cada solución. 
* En el notebook, este valor ya es un número entero directamente interpretable (por ejemplo, "12 SNPs"). 
* No se requiere calcular una "tasa" o un "promedio auxiliar" para entenderlo, ya que su escala original es la más simple posible y se incluye de forma nativa en los resúmenes de rendimiento.

### 2. Exclusión del Equilibrio (Objetivo 3)
El **Equilibrio** mide la distribución de los SNPs a lo largo del genoma. 
* En este experimento, se trata de una métrica de soporte estructural que utiliza índices estadísticos (como la desviación). 
* A diferencia de la Tolerancia o la Disimilitud, el valor crudo del Equilibrio no se traduce fácilmente en una "tasa de éxito" biológica o en una distancia física tangible. Por ello, se considera que su análisis dentro de las métricas de calidad normalizadas es suficiente para evaluar el desempeño del algoritmo.

En conclusión, se seleccionan estos dos objetivos específicos porque son los que requieren una transformación a términos biológicos (tasas y distancias promedio) para que los resultados sean útiles para un análisis clínico o genético, mientras que los demás objetivos ya cumplen su función informativa mediante los valores normalizados y los recuentos directos.

### Por qué es importante esta celda de código

La normalización global evita comparaciones injustas entre algoritmos y entre runs.



### Detalle técnico

Si una columna tiene rango cero, se maneja como caso degenerado para evitar `NaN/Inf`.

## Ensamblado de métricas por ejecución (Run)

Tras la finalización de los experimentos, se lleva a cabo la consolidación de los datos mediante una función de evaluación que procesa individualmente cada ejecución independiente. Este procedimiento transforma los objetos de resultados crudos en una estructura de datos tabular (`pandas.DataFrame`), permitiendo un análisis estadístico riguroso de cada configuración.

El proceso se ejecuta siguiendo estos pasos técnicos estrictos:

1.  **Normalización individual bajo marco global**: Para cada ejecución, se toma su frente de Pareto final ($F$) y se normaliza íntegramente utilizando los puntos **ideal** y **nadir** globales previamente calculados. Esta operación asegura que todas las métricas de calidad de todas las ejecuciones se midan en la misma escala $[0, 1]$.

2.  **Cálculo de métricas de calidad**: Sobre el frente ya normalizado de cada ejecución, se aplican los algoritmos de cálculo para obtener:
    * **Range**: Extensión del frente.
    * **SumMin**: Cobertura de los vectores de referencia.
    * **MinSum**: Convergencia hacia los vectores de referencia.



3.  **Cálculo de métricas auxiliares**: Utilizando los valores originales (sin normalizar) de la ejecución, se computan los indicadores de rendimiento biológico:
    * **MaxToleranceRate**: El valor máximo de la tasa de tolerancia.
    * **AvgToleranceRate**: El promedio de la tasa de tolerancia de todas las soluciones del frente.
    * **AvgHammingDistance**: El promedio de la distancia de Hamming.

4.  **Generación del registro de metadatos**: Por cada ejecución, se crea un registro que vincula las métricas obtenidas con sus parámetros de origen:
    * Identificadores: `algorithm` (NSGA-III o MOEA/D), `init` (Random o Greedy).
    * Control: `run` (número de ejecución) y `seed` (semilla aleatoria empleada).
    * Parámetros de diseño: `n_gen`, `pop_size` y `offspring`.



5.  **Construcción del DataFrame final**: Todos los registros individuales se concatenan para generar una tabla única donde cada fila representa una ejecución completa. Esta estructura permite realizar operaciones de agrupación para obtener las medias y desviaciones típicas que se presentan en los resultados finales del estudio.

### Estructura de la tabla de datos resultante

La tabla generada contiene las siguientes columnas técnicas para cada entrada:

| Columna | Descripción |
| :--- | :--- |
| `algorithm` | Motor de búsqueda evolutivo empleado. |
| `init` | Método de generación de la población inicial. |
| `run` | Índice de la ejecución (0 a $N-1$). |
| `Range`, `SumMin`, `MinSum` | Indicadores de calidad en el espacio normalizado. |
| `MaxToleranceRate` | Capacidad máxima de discriminación biológica en escala original. |
| `ExecutionTime` | Tiempo total empleado por la ejecución en segundos. |



Este proceso de ensamblado garantiza la trazabilidad total de los datos, permitiendo verificar la influencia de la semilla aleatoria o del método de inicialización en la calidad final de los conjuntos de Tag SNPs identificados.

## 📊 Primera inspección de resultados



La siguiente celda ejecuta el bloque métrico y muestra una vista preliminar.



### Qué verás

- Primeras filas de `df_runs`.

- Valores de `ideal_global` y `nadir_global`.



### Para qué sirve

Es una verificación rápida de sanidad: confirma que el pipeline produjo datos útiles antes de pasar a agregación y gráficas.



### Si aparece vacío

Suele indicar que `RUN_EXPERIMENTS` estaba en `False` o que no se ejecutaron celdas previas en orden.

### ¿Qué hace esta celda?

Esta celda define la función `plot_pareto_fronts(df_input, algorithm_name, init_name=None)`, que visualiza el frente de Pareto con **máxima separación** por configuración.

**Entrada esperada**
- `df_input`: DataFrame con objetivos por solución (`f1_compactness`, `f2_neg_tolerance`, `f3_neg_hamming_avg`, `f4_balance_var`) y columnas de contexto (`algorithm`, `init` o `init_type`).
- `algorithm_name`: algoritmo a analizar (por ejemplo, `NSGA3`, `MOEAD`, `NSGA2`, `SPEA2`).
- `init_name` (opcional): inicialización concreta. Si es `None`, se genera una figura separada por cada inicialización disponible.

**Qué hace paso a paso**
1. Filtra el DataFrame por algoritmo.
2. Para cada inicialización objetivo, crea una figura independiente 2x2.
3. Reconvierte objetivos negados (`Tolerance` y `Hamming`) a escala interpretable.
4. Dibuja puntos y tendencia dentro de esa única configuración.
5. Guarda cada figura con nombre `pareto_fronts_<algo>_<init>_<modo>.png`.

**Resultado**
- Una figura por par `(algoritmo, inicialización)`, sin mezclar inicializaciones en la misma gráfica.

### 8.1. Análisis del Frente de Pareto: NSGA-III (Many-Objective)

En esta sección se evalúa la distribución geométrica de las soluciones no dominadas descubiertas por **NSGA-III**. Al ser un algoritmo diseñado específicamente para espacios de alta dimensionalidad (mediante el uso de vectores de referencia), se espera observar un frente equilibrado. Las siguientes gráficas de dispersión 2D proyectan las relaciones bivariadas críticas entre los cuatro objetivos biológicos, permitiendo contrastar el impacto directo de la inicialización pre-calculada (`greedy`) frente a la aleatoria (`random`).

### 8.2. Análisis del Frente de Pareto: MOEA/D (Many-Objective)

A continuación se visualizan las soluciones obtenidas por **MOEA/D**. Este algoritmo aborda la optimización de muchos objetivos descomponiendo el problema global en múltiples subproblemas escalares agregados. Se analizará si esta estrategia de descomposición logra mantener la diversidad del frente y evitar el colapso geométrico al buscar Tag SNPs bajo las dos estrategias de inicialización.

### 8.3. Análisis del Frente de Pareto: NSGA-II (Línea de base clásica)

Se presenta el comportamiento de **NSGA-II**, el algoritmo estándar histórico en optimización multiobjetivo. Dado que su mecanismo de selección basado en la distancia de amontonamiento (*crowding distance*) suele perder eficacia al escalar a cuatro dimensiones, estas gráficas permitirán verificar empíricamente si el algoritmo logra cubrir los extremos del espacio de búsqueda o si converge hacia una subregión central del frente de Pareto.

### 8.4. Análisis del Frente de Pareto: SPEA2 (Línea de base clásica)

Finalmente, se evalúa **SPEA2**, el segundo algoritmo de referencia. Su mecanismo de preservación de la diversidad basado en la estimación de densidad del vecino más cercano ($k$-ésimo) se pone a prueba en este espacio de cuatro objetivos. Se comparará su capacidad para retener soluciones extremas (alta distancia de Hamming) frente al sesgo introducido por las inicializaciones.

### Matriz de Correlación de Objetivos en los Frentes
Usamos un mapa de calor para visualizar la matriz de correlación (coeficiente de Pearson) entre los 4 objetivos en las soluciones finales. Las correlaciones negativas altas indicarían un fuerte conflicto entre esos dos objetivos, lo que es propio de los trade-offs más agresivos de Pareto.

### 1. Concepto Funcional
En el contexto de la optimización multi-objetivo, esta gráfica sirve para cuantificar analíticamente las relaciones de conflicto o sinergia entre las metas:
*   **Correlación Negativa (Tonos azules / Valores menores a 0):** Indica un **conflicto (trade-off)** directo. Si una solución minimiza mejor un objetivo, el otro tiende a dispararse. Es la manifestación matemática de por qué existe una frontera de Pareto. Por ejemplo, al reducir drásticamente el número de SNPs ($f_1$ baja), las métricas negativizadas como la tolerancia o el Hamming tienden a empeorar ($f_2$ o $f_3$ suben).
*   **Correlación Positiva (Tonos rojos / Valores mayores a 0):** Indica una **sinergia**. Ambos objetivos apuntan en la misma dirección; al mejorar uno, estadísticamente también mejora el otro.
*   **Correlación Cercana a 0 (Tonos pálidos):** Señala una falta de relación lineal. Los objetivos son funcionalmente independientes dentro de las soluciones halladas.

### 2. Variables Analizadas
La matriz cruza los cuatro objetivos minimizables para el solucionador `pymoo`:
*   `f1_compactness`: Número de SNPs.
*   `f2_neg_tolerance`: Tolerancia (en negativo).
*   `f3_neg_hamming_avg`: Diferenciación media (en negativo).
*   `f4_balance_var`: Varianza o balanceo en la selección.

### 3. Interpretación de los Resultados (Diagonales y Cruces)
*   **Diagonal Principal (Unos):** Siempre es 1.00, ya que indica la correlación perfecta de cada variable consigo misma.
*   **Cruces Clave:** Si se observa un valor muy cercano a $-1$ entre `f1_compactness` y `f2_neg_tolerance`, confirmaría de modo concluyente lo que se observa en los diagramas de dispersión o *scatter plots*: ambas métricas son fuerzas opuestas en la toma de decisión del algoritmo.

### Coordenadas Paralelas (Objetivos Reales Normalizados)
Este gráfico de coordenadas paralelas ilustra las soluciones de los frentes de Pareto finales, mostrando cómo fluctúan los valores de los diferentes objetivos. 

Para facilitar su interpretación biológica, **se han revertido las métricas negativas** ($f_2$ y $f_3$), mostrando su **valor real positivo** (Tolerancia y Hamming). Dado que las métricas tienen diferentes unidades y escalas (ej. conteo de SNPs vs. distancias), todos los valores se han **normalizado entre 0 y 1** (donde 0 es el mínimo valor observado en el frente y 1 el máximo).

Permite observar visualmente los "trade-offs" globales (conflictos entre objetivos): los cruces en aspa (X) marcados entre dos ejes indican un fuerte conflicto entre esos objetivos (al mejorar uno, empeora el otro), mientras que los trazos horizontales sugieren sinergia. Se utiliza una muestra representativa de las soluciones para garantizar la legibilidad del gráfico.

#### Interpretación

Para interpretar una gráfica de coordenadas paralelas, se deben seguir los siguientes pasos:

1. **Identificación de ejes:**  
   Cada eje vertical representa una métrica u objetivo diferente. En la parte superior e inferior de cada eje se encuentran los valores máximo y mínimo (normalizados entre 0 y 1 en la mayoría de los casos).

2. **Lectura de líneas:**  
   Cada línea que cruza todos los ejes corresponde a una solución concreta del problema. El recorrido de la línea muestra el valor que esa solución obtiene en cada objetivo.

3. **Interpretación de valores:**  
   - Si el objetivo es minimizar una métrica, los valores bajos (cercanos a 0) en ese eje son preferibles.
   - Si el objetivo es maximizar, los valores altos (cercanos a 1) son mejores.

4. **Análisis de cruces:**  
   Cuando se observan cruces en forma de “X” entre dos ejes adyacentes, se indica un conflicto (trade-off) entre esos objetivos: mejorar uno implica empeorar el otro.  
   Si las líneas permanecen paralelas entre dos ejes, significa que no existe conflicto relevante entre esos objetivos.

5. **Comparación de grupos:**  
   Dado que se colorean o diferencian las líneas por grupo (por algoritmo y inicialización), se puede analizar cómo se comportan los distintos métodos respecto a los objetivos.

6. **Conclusión:**  
   Se debe buscar líneas que, en la medida de lo posible, permanezcan cerca de los valores óptimos en todos los ejes, aunque normalmente no existe una solución que sea óptima en todos a la vez. La gráfica permite visualizar el compromiso entre objetivos y seleccionar la solución más adecuada según las prioridades del estudio.

## 🧠 Agregación estadística por método



La siguiente celda resume resultados por algoritmo e inicialización.



### Qué calcula

Para cada métrica, obtiene media y desviación estándar agrupando por:

- `algorithm`

- `init` (random/greedy)



### Por qué es útil

Pasamos de ver runs sueltos a una comparación compacta y estadística, más adecuada para discusión en memoria.



### Resultado esperado

Tabla `df_summary` con columnas del tipo `Metric_mean` y `Metric_std`.

### 🐛 Análisis de Convergencia Generacional

Esta sección analiza la evolución temporal del rendimiento a lo largo de las generaciones con **máxima separación por configuración**.

Las gráficas se generan por cada par `(algoritmo, inicialización)`, evitando mezclar inicializaciones o algoritmos en la misma figura.

Indicadores incluidos:
* **Rango (Range)**: diversidad y extensión de soluciones.
* **SumMin y MinSum**: convergencia marginal y central.
* **Tasa de Tolerancia (máxima y promedio)**: robustez biológica.
* **Distancia Hamming Promedio**: separación genética capturada.

*(Nota: el Hipervolumen se mantiene para evaluación estática en frentes finales.)*

### 1) Mapa de calor (Heatmap) de métricas medias

Este mapa de calor muestra el valor promedio de cada métrica para cada combinación de algoritmo e inicialización. Los valores están normalizados por columna (métrica) para el color, permitiendo identificar rápidamente qué configuración destaca en cada área (claro para valores mínimos, oscuro para máximos del dataset actual).

### 2) Ranking global (Suma de Posiciones)

Evaluación del rendimiento conjunto mediante la **suma de sus posiciones (rankings)** en las métricas de minimización principales (`Range`, `SumMin`, `MinSum`). Una barra más baja indica que el algoritmo ha quedado en mejores puestos (más cerca del 1º) de forma consistente en las distintas métricas evaluadas.

## 📈 Visualización de métricas finales


## 📦 Diagramas de caja (Boxplots): rendimiento por configuración

Esta sección genera boxplots de métricas finales en **comparativa conjunta**: una única figura con todas las combinaciones `(algoritmo, inicialización)`.

### ¿Qué aporta esta visualización?
1. Resume la variabilidad entre runs para cada configuración en un mismo panel comparativo.
2. Facilita detectar outliers y estabilidad relativa entre configuraciones.
3. Permite comparar métricas sin tener que abrir múltiples figuras separadas.

### Detalles técnicos
- Se crea una cuadrícula dinámica según métricas disponibles en `df_runs`.
- Cada salida se guarda como `metrics_boxplots_all_configs_<modo>.png`.
- Se mantiene resolución de exportación a 200 DPI.


## 🎻 Visualización de la distribución: Violin + Strip plots

Esta sección analiza la distribución completa de las métricas finales en una **comparativa conjunta** por configuración.

Para cada métrica disponible, se genera una figura única con todas las combinaciones `(algoritmo, inicialización)`:
- **Violin plot** para densidad y forma de la distribución.
- **Strip plot** para visualizar cada run individual.

Este enfoque facilita comparar directamente configuraciones en el mismo gráfico.


## 📊 Gráficos de barras: media y desviación estándar por configuración

Esta celda genera barras de **media ± desviación estándar** para cada métrica en formato comparativo conjunto.

Cada figura incluye todas las combinaciones `(algoritmo, inicialización)`, permitiendo evaluar simultáneamente rendimiento promedio y estabilidad entre configuraciones.


## 💾 Exportación de resultados a CSV



La siguiente celda guarda resultados en disco para trazabilidad y comparación posterior.



### Archivos que puede generar

- Métricas por run (`runs_metrics_*.csv`).

- Resumen agregado (`summary_metrics_*.csv`).



### Convención utilizada

- Sufijo `_fast` para modo rápido.

- Sufijo `_medium` para modo medio.

- Sufijo `_full` para modo completo.



### Ventaja

Permite reproducir análisis, comparar fechas de ejecución y utilizar los datos fuera del notebook (R, Excel, informes).

## ⏱️ Temporizador

