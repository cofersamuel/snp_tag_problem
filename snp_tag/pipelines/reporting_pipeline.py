"""
Módulo de Orquestación de Reportes (reporting_pipeline.py)
---------------------------------------------------------
Gestiona la ejecución concurrente de las tareas de visualización y post-procesamiento.
"""

# =============================================================================
# LIBRERÍAS ESTÁNDAR
# =============================================================================
import concurrent.futures  # Módulo para ejecución concurrente usando hilos/procesos.
import os  # Interfaz con el sistema operativo para manipulación de rutas y carpetas.
import time  # Módulo estándar para el cálculo de tiempos y marcas temporales.
from typing import Any, Dict, List, Optional, Tuple  # Elementos para tipado estático y anotaciones.

# =============================================================================
# LIBRERÍAS DE TERCEROS
# =============================================================================
import numpy as np  # Biblioteca numérica para operaciones con vectores y matrices.
import pandas as pd  # Biblioteca para estructuración y análisis de datos mediante DataFrames.

# =============================================================================
# MÓDULOS LOCALES (snp_tag)
# =============================================================================
from snp_tag.config import ConfiguracionExperimento  # Clase contenedora de los parámetros del experimento.
from snp_tag.constants import (BASE_METRICS, HIGHER_IS_BETTER_METRICS,
                               PREFERRED_ALGORITHMS_ORDER)  # Constantes globales del pipeline.
from snp_tag.engine.metrics_logic import decodificar_objetivos_reales  # Utilidad para decodificar los objetivos a escala real.
from snp_tag.utils.logger import logger  # Instancia del registrador para salida de logs informativos.
from snp_tag.utils.runtime import calcular_max_workers_paralelo  # Cálculo del número de hilos o procesos paralelos.
from snp_tag.utils.terminal import (imprimir_encabezado,
                                    imprimir_grafico_guardado,
                                    imprimir_subseccion,
                                    obtener_enlace_terminal)  # Funciones para formateo en consola.
from snp_tag.visualization.convergence_plot import \
    graficar_evolucion_generacional  # Función para graficar convergencia generacional.
from snp_tag.visualization.fronts_plot import (
    graficar_coordenadas_paralelas_pareto,
    graficar_correlacion_objetivos_pareto, graficar_frentes_pareto,
    graficar_frentes_pareto_agregados)  # Funciones para graficar los frentes de Pareto.
from snp_tag.visualization.mcdm_plot import analizar_decision_mcdm  # Función del módulo de toma de decisiones multi-criterio.
from snp_tag.visualization.stats_plot import (graficar_analisis_estadistico,
                                              graficar_analisis_kruskal_dunn,
                                              graficar_boxplot_metricas,
                                              graficar_media_std_metricas,
                                              graficar_violin_metricas)  # Gráficas estadísticas y tests.


def _tarea_plot_frentes(
    tipo_tarea: str,
    df_fronts_total: pd.DataFrame,
    carpetas: Dict[str, str],
    etiqueta_modo: str,
    dpi: int,
    semilla: int,
    limites_ejes: Optional[Dict[str, Tuple[float, float]]] = None,
    modo_transformacion_objetivos: str = 'neg',
) -> Dict[str, Any]:
    """
    Dispatcher serializable para paralelizar tareas de graficado de frentes.
    
    Parámetros:
    -----------
    tipo_tarea : str
        Identificador del trabajo a realizar ('correlation', 'parallel', 'all_fronts').
    df_fronts_total : pd.DataFrame
        Dataset global de Pareto.
    carpetas : Dict[str, str]
        Diccionario con rutas de salida.
    etiqueta_modo : str
        Sufijo del experimento.
    dpi : int
        Calidad visual.
    semilla : int
        Semilla de control.
    limites_ejes : Optional[Dict]
        Mapeo de límites canónicos (min, max) de ejes.
    modo_transformacion_objetivos : str
        Dirección de optimización.

    Retorna:
    --------
    Dict[str, Any]
        Diccionario de la tarea junto con la lista de artefactos generados.
    """
    salida = {'tarea': tipo_tarea, 'artefactos': []}  # Inicializa el diccionario de salida con la tarea y lista de artefactos.
    if tipo_tarea == 'correlation':  # Verifica si la tarea actual es graficar correlaciones.
        salida['artefactos'] = graficar_correlacion_objetivos_pareto(  # Llama a la función de correlación y asigna los artefactos.
            df_fronts_total,  # Pasa el DataFrame con todos los frentes no dominados.
            carpetas,  # Directorio donde se guardará la gráfica generada.
            etiqueta_modo,  # Identificador de la ejecución (por ejemplo, 'medium').
            dpi=dpi,  # Resolución gráfica configurada para las imágenes.
            emitir_log=False,  # Evita que se emitan logs repetitivos durante el paralelismo.
            modo_transformacion_objetivos=modo_transformacion_objetivos,  # Indica el modo de transformación de objetivos.
        )  # Fin de la llamada a la función de correlación.
    elif tipo_tarea == 'parallel':  # Comprueba si la tarea solicita coordenadas paralelas.
        salida['artefactos'] = graficar_coordenadas_paralelas_pareto(  # Llama a la función de coordenadas paralelas.
            df_fronts_total,  # Pasa los datos consolidados de frentes.
            semilla,  # Semilla para asegurar consistencia cromática de las líneas.
            carpetas,  # Ruta donde guardar el archivo de imagen resultante.
            etiqueta_modo,  # Nombre/etiqueta de la ejecución.
            dpi=dpi,  # Calidad visual en DPI.
            emitir_log=False,  # Apaga los logs internos para no ensuciar la salida.
            modo_transformacion_objetivos=modo_transformacion_objetivos,  # Orientación matemática.
        )  # Fin de la llamada a coordenadas paralelas.
    elif tipo_tarea == 'all_fronts':  # Verifica si la tarea actual es generar todos los frentes.
        artefactos = []  # Inicializa la lista local de artefactos generados.
        # Obtiene una lista ordenada de todos los nombres únicos de algoritmos en el set de datos.
        algoritmos_presentes = sorted(df_fronts_total['algorithm'].dropna().unique().tolist())
        # Filtra y ordena los algoritmos según el orden preferido definido globalmente.
        algoritmos_ordenados = [a for a in PREFERRED_ALGORITHMS_ORDER if a in algoritmos_presentes]
        # Concatena los algoritmos presentes restantes que no estuviesen en el orden preferido.
        algoritmos_ordenados.extend([a for a in algoritmos_presentes if a not in algoritmos_ordenados])

        for algo in algoritmos_ordenados:  # Itera a través de los algoritmos en el orden de prioridad.
            df_algo = df_fronts_total[df_fronts_total['algorithm'] == algo].copy()  # Extrae y copia las filas del algoritmo.
            if not df_algo.empty:  # Procesa únicamente si se disponen de registros para dicho algoritmo.
                # Llama a graficar_frentes_pareto para las subconfiguraciones individuales de este algoritmo.
                artefactos.extend(  # Añade los artefactos individuales a la lista de salida.
                    graficar_frentes_pareto(  # Genera diagramas de dispersión del Pareto.
                        df_algo,  # Subconjunto de frentes del algoritmo actual.
                        algo,  # Nombre identificador del algoritmo.
                        carpetas=carpetas,  # Directorios de salida.
                        etiqueta_modo=etiqueta_modo,  # Sufijo del experimento.
                        dpi=dpi,  # Resolución para el guardado.
                        emitir_log=False,  # Desactiva la escritura redundante en consola.
                        limites_ejes=limites_ejes,  # Límites fijos globales de visualización.
                        modo_transformacion_objetivos=modo_transformacion_objetivos,  # Orientación de optimización.
                    )  # Fin del graficado de frentes individuales.
                )  # Fin de la extensión de la lista.
                # Construye el nombre físico para el gráfico de frentes agregados del algoritmo.
                nombre_agg = f'frentes_pareto_agregados_{algo.lower()}_{etiqueta_modo}.png'
                # Define diferenciar por estilo de trazo según operador de cruce (crossover) si existe.
                style_col = 'crossover' if 'crossover' in df_algo.columns else None
                # Llama al graficador agregado para superponer todas las inicializaciones de un algoritmo.
                artefactos.extend(  # Agrega las rutas de salida del gráfico superpuesto.
                    graficar_frentes_pareto_agregados(  # Llama a la función de frentes agregados.
                        df_algo,  # Datos de frentes asociados al algoritmo.
                        f"Frentes de Pareto Agregados: {algo}",  # Título formal del gráfico.
                        nombre_agg,  # Nombre de archivo final.
                        hue_col='init',  # Agrupa los colores de los puntos según inicialización.
                        style_col=style_col,  # Aplica marcadores diferentes por tipo de crossover.
                        carpetas=carpetas,  # Carpetas de destino configuradas.
                        dpi=dpi,  # Calidad visual en DPI.
                        emitir_log=False,  # No escribe logs en consola.
                        limites_ejes=limites_ejes,  # Límites unificados de los ejes.
                        modo_transformacion_objetivos=modo_transformacion_objetivos,  # Orientación matemática.
                    )  # Fin de frentes de Pareto agregados.
                )  # Fin de la extensión.
        
        if not df_fronts_total.empty:  # Procesa el gráfico global agregador si el set total contiene registros.
            df_global = df_fronts_total.copy()  # Copia el dataframe global de frentes de Pareto.
            # Agrupa algoritmo e inicialización en una etiqueta de configuración única.
            df_global['Configuración'] = df_global['algorithm'] + " (" + df_global['init'] + ")"
            nombre_global = f'frentes_pareto_global_{etiqueta_modo}.png'  # Nombre del archivo para el gráfico comparativo global.
            # Llama a graficar_frentes_pareto_agregados con el set consolidado de todas las configuraciones.
            artefactos.extend(  # Agrega la ruta de la visualización global resultante.
                graficar_frentes_pareto_agregados(  # Genera el gráfico agregativo global.
                    df_global,  # DataFrame total con todas las configuraciones mapeadas.
                    "Comparativa Global de Frentes de Pareto",  # Título principal del gráfico.
                    nombre_global,  # Nombre del archivo de imagen físico.
                    hue_col='Configuración',  # Diferencia por color las combinaciones algoritmo+inicialización.
                    style_col=None,  # Desactiva estilos de marcador para evitar confusión visual.
                    carpetas=carpetas,  # Directorios destino de salida.
                    dpi=dpi,  # Calidad visual del plot.
                    emitir_log=False,  # Apaga salida estándar a consola.
                    limites_ejes=limites_ejes,  # Límites unificados de objetivos.
                    modo_transformacion_objetivos=modo_transformacion_objetivos,  # Orientación del espacio.
                )  # Fin de la llamada global.
            )  # Fin de la extensión.
        salida['artefactos'] = artefactos  # Asocia todos los archivos de frentes creados al diccionario de salida.
    return salida  # Retorna el diccionario de salida de la tarea de frentes.


def _tarea_sintesis_estadistica(
    tipo_tarea: str,
    df_final: pd.DataFrame,
    dir_salida: str,
    etiqueta_modo: str,
    dpi: int,
) -> Dict[str, Any]:
    """
    Worker serializable para paralelizar la síntesis estadística de rendimiento.

    Parámetros:
    -----------
    tipo_tarea : str
        Identificador ('boxplots', 'violin', 'mean_std').
    df_final : pd.DataFrame
        Resultados finales del pipeline.
    dir_salida : str
        Ruta destino de escritura.
    etiqueta_modo : str
        Modificador global de corrida.
    dpi : int
        Resolución de imagen.

    Retorna:
    --------
    Dict[str, Any]
        Diccionario con identificador de la tarea y rutas creadas.
    """
    if tipo_tarea == 'boxplots':  # Comprueba si la tarea solicita diagramas de caja (boxplots).
        # Llama a la función de caja y bigotes para las métricas de rendimiento.
        artefactos = graficar_boxplot_metricas(df_final, dir_salida, etiqueta_modo, dpi=dpi, emitir_log=False)
    elif tipo_tarea == 'violin':  # Comprueba si la tarea solicita diagramas de violín.
        # Llama a la función que grafica la densidad de distribución mediante violín.
        artefactos = graficar_violin_metricas(df_final, dir_salida, etiqueta_modo, dpi=dpi, emitir_log=False)
    elif tipo_tarea == 'mean_std':  # Comprueba si la tarea solicita gráficos de tendencia central (media/std).
        # Llama al generador de gráficos de barra de media y desviación estándar.
        artefactos = graficar_media_std_metricas(df_final, dir_salida, etiqueta_modo, dpi=dpi, emitir_log=False)
    else:  # En caso de no coincidir con ninguna tarea mapeada.
        artefactos = []  # Asigna una lista vacía para evitar errores de tipo.
    return {'tarea': tipo_tarea, 'artefactos': artefactos}  # Retorna el mapeo de la tarea con sus rutas.


def ejecutar_reportes_visualizacion(
    cfg: ConfiguracionExperimento,
    df_final: pd.DataFrame,
    df_gen: pd.DataFrame,
    df_fronts_total: pd.DataFrame,
) -> None:
    """
    Ejecuta y orquesta exclusivamente la fase de reportes y generación de artefactos visuales.

    Parámetros:
    -----------
    cfg : ConfiguracionExperimento
        Contexto del ensayo experimental.
    df_final : pd.DataFrame
        Métricas por run independiente y algoritmo.
    df_gen : pd.DataFrame
        Dataset consolidado de trayectorias evolutivas y convergencia inter-generacional.
    df_fronts_total : pd.DataFrame
        Matrices enlazadas del frente no-dominado global.
    """
    imprimir_encabezado("REPORTES Y VISUALIZACIÓN")  # Imprime el banner del módulo en terminal.

    if df_fronts_total is None:  # Valida que el DataFrame consolidado de frentes no sea nulo.
        df_fronts_total = pd.DataFrame()  # Inicializa como DataFrame vacío si era None.

    df_fronts_mediano = pd.DataFrame()  # Inicializa el dataframe de frentes de la réplica mediana.
    df_gen_mediano = pd.DataFrame()  # Inicializa el dataframe de progreso de la réplica mediana.
    if not df_final.empty:  # Si los resultados finales contienen información histórica.
        median_runs = []  # Lista vacía temporal para acumular identificadores de corridas medianas.
        # Agrupa los resultados individuales por combinación de algoritmo, inicialización y cruce.
        for (algo, init, cx), group in df_final.groupby(['algorithm', 'init', 'crossover']):
            group_sorted = group.sort_values(by='Hypervolume')  # Ordena las réplicas de forma ascendente por Hypervolume.
            median_idx = len(group_sorted) // 2  # Obtiene el índice que representa la mediana matemática del grupo.
            median_run = group_sorted.iloc[median_idx]['run']  # Extrae el número identificador de esa ejecución mediana.
            # Registra la tupla de identificación en la lista de réplicas medianas.
            median_runs.append({'algorithm': algo, 'init': init, 'crossover': cx, 'run': median_run})
        
        df_median_runs = pd.DataFrame(median_runs)  # Convierte la colección de tuplas medianas a un DataFrame.
        
        if not df_fronts_total.empty:  # Si el conjunto consolidado de soluciones contiene datos.
            # Filtra las soluciones de frentes quedándose únicamente con la réplica central (mediana).
            df_fronts_mediano = pd.merge(df_fronts_total, df_median_runs, on=['algorithm', 'init', 'crossover', 'run'], how='inner')
        if df_gen is not None and not df_gen.empty:  # Si el progreso de convergencia no es nulo y contiene datos.
            # Filtra las trayectorias quedándose únicamente con las de la ejecución mediana.
            df_gen_mediano = pd.merge(df_gen, df_median_runs, on=['algorithm', 'init', 'crossover', 'run'], how='inner')
    else:  # Si no contamos con resultados previos para calcular las medianas.
        df_fronts_mediano = df_fronts_total.copy()  # Copia íntegramente todas las soluciones de frentes disponibles.
        # Asigna el progreso generacional original o inicializa vacío si fuese nulo.
        df_gen_mediano = df_gen.copy() if df_gen is not None else pd.DataFrame()

    imprimir_subseccion("Distribución y Correlación de Frentes de Pareto", icono="🔍")  # Encabezado en terminal.
    t0_frentes = time.time()  # Toma la marca temporal de inicio de procesamiento de frentes.

    cols_fronts = {  # Columnas canónicas obligatorias que debe poseer el set de frentes para ser válido.
        'algorithm', 'init', 'f1_compactness', 'f2_transformed_tolerance',
        'f3_transformed_hamming_avg', 'f4_balance_var'
    }

    if df_fronts_mediano.empty:  # Si no existen datos de frentes válidos.
        # Emite una advertencia formal y salta todo el bloque de dibujo de frentes de Pareto.
        logger.warning("      • ⚠️  Pareto omitido: no hay datos de frentes disponibles (CSV frentes ausente/vacío).")
    elif not cols_fronts.issubset(set(df_fronts_mediano.columns)):  # Si el esquema del DataFrame no coincide.
        # Calcula el conjunto de nombres de columnas requeridos que faltan en el archivo de frentes.
        faltantes = sorted(cols_fronts - set(df_fronts_mediano.columns))
        alias_disponibles = []  # Lista vacía temporal para registrar alias de compatibilidad alternativos.
        if 'f2_neg_tolerance' in df_fronts_mediano.columns:  # Evalúa la presencia del alias de tolerancia.
            alias_disponibles.append('f2_neg_tolerance')  # Registra que se detectó la columna histórica.
        if 'f3_neg_hamming_avg' in df_fronts_mediano.columns:  # Evalúa la presencia del alias de Hamming.
            alias_disponibles.append('f3_neg_hamming_avg')  # Registra la columna histórica.
        # Redacta un mensaje aclaratorio si se encontraron alias conocidos.
        mensaje_alias = (
            f" Alias detectados en CSV: {sorted(alias_disponibles)}."
            if alias_disponibles else ""
        )
        # Registra la advertencia detallada informando qué columnas faltaron.
        logger.warning(
            "      • ⚠️  Pareto omitido: el CSV de frentes no contiene columnas requeridas: "
            f"{faltantes}.{mensaje_alias}"
        )
    else:  # Si los datos de frentes de Pareto son completos y legibles.
        limites_ejes = {}  # Diccionario para almacenar los límites canónicos calculados.
        # Decodifica las puntuaciones matemáticas a la escala real de los objetivos originales.
        objetivos_reales = decodificar_objetivos_reales(
            df_fronts_mediano[
                ['f1_compactness', 'f2_transformed_tolerance', 'f3_transformed_hamming_avg', 'f4_balance_var']
            ].to_numpy(dtype=float),
            modo_transformacion_objetivos=cfg.modo_transformacion_objetivos,
        )
        # Mapea las claves conceptuales de visualización a las columnas físicas decodificadas.
        mapping_objetivos = {
            'Compacidad': pd.Series(objetivos_reales['compacidad']),
            'Tolerancia': pd.Series(objetivos_reales['tolerancia_real']),
            'Hamming': pd.Series(objetivos_reales['hamming_prom_real']),
            'Balance': pd.Series(objetivos_reales['balance_var']),
        }
        for nombre, serie in mapping_objetivos.items():  # Recorre cada objetivo físico.
            vmin, vmax = serie.min(), serie.max()  # Obtiene los valores mínimo y máximo absolutos de la serie.
            if not pd.isna(vmin) and not pd.isna(vmax):  # Procesa solo si las cotas obtenidas son números reales.
                rango = vmax - vmin  # Calcula el intervalo de dispersión del objetivo.
                # Establece un margen del 5% para que los marcadores no queden al borde del marco.
                margen = rango * 0.05 if rango > 0 else 0.5
                # Registra los límites formateados (mínimo con margen, máximo con margen).
                limites_ejes[nombre] = (vmin - margen, vmax + margen)

        max_workers = calcular_max_workers_paralelo()  # Obtiene el número óptimo de subprocesos concurrentes.
        tareas_plot = []  # Lista vacía para albergar identificadores de trabajos gráficos concurrentes.
        if 'frentes' in cfg.graficas_activas: tareas_plot.append('all_fronts')  # Agrega tarea de frentes de Pareto.
        if 'correlacion' in cfg.graficas_activas: tareas_plot.append('correlation')  # Agrega tarea de correlación.
        if 'paralelas' in cfg.graficas_activas: tareas_plot.append('parallel')  # Agrega tarea de coordenadas paralelas.
        
        if tareas_plot:  # Lanza el pool si hay al menos una tarea gráfica configurada.
            # Inicializa el gestor de paralelismo por procesos concurrentes.
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Envía los trabajos al pool para su ejecución en segundo plano.
                futures = [
                    executor.submit(
                        _tarea_plot_frentes,
                        t,
                        df_fronts_mediano,
                        cfg.carpetas,
                        cfg.modo_ejecucion,
                        cfg.report_plot_dpi,
                        cfg.semilla_maestra,
                        limites_ejes,
                        cfg.modo_transformacion_objetivos,
                    )
                    for t in tareas_plot
                ]
                resultados_tareas = {}  # Mapeo de salida local para los artefactos.
                for future in concurrent.futures.as_completed(futures):  # Itera conforme terminen los procesos.
                    try:
                        resultado = future.result()  # Recupera el diccionario devuelto por el proceso hijo.
                        # Registra el tipo de tarea y su lista de artefactos generados en disco.
                        resultados_tareas[resultado.get('tarea')] = resultado.get('artefactos', [])
                    except Exception as e:  # Captura fallos de ejecución en los procesos hijo.
                        logger.error(f"      • ⚠️  Error en graficado paralelo: {e}")  # Reporta la excepción.
        else:  # Si no hay gráficos configurados.
            resultados_tareas = {}  # Inicializa el mapeo como vacío.

        if 'correlacion' in cfg.graficas_activas:  # Si se habilitó el gráfico de correlación.
            imprimir_subseccion("Correlación de Objetivos", icono="🔗")  # Informa la sección actual en consola.
            for ruta, descripcion in resultados_tareas.get('correlation', []):  # Itera sobre los archivos generados.
                imprimir_grafico_guardado(ruta, descripcion)  # Imprime la ruta de la imagen en consola.

        if 'paralelas' in cfg.graficas_activas:  # Si se configuró coordenadas paralelas.
            imprimir_subseccion("Coordenadas Paralelas", icono="📈")  # Encabezado en terminal.
            for ruta, descripcion in resultados_tareas.get('parallel', []):  # Itera sobre las imágenes.
                imprimir_grafico_guardado(ruta, descripcion)  # Muestra en consola el aviso de guardado.

        if 'frentes' in cfg.graficas_activas:  # Si se habilitó la generación de frentes.
            imprimir_subseccion("Frentes de Pareto", icono="🎯")  # Encabezado en terminal.
            for ruta, descripcion in resultados_tareas.get('all_fronts', []):  # Recorre la lista de gráficos de frentes.
                imprimir_grafico_guardado(ruta, descripcion)  # Informa el archivo guardado en el logger.

    # Muestra en logs el tiempo exacto que tomó realizar todas las tareas gráficas de frentes de Pareto.
    logger.info(f"      • Tiempo bloque 'Frentes de Pareto': {time.time() - t0_frentes:.1f}s")

    imprimir_subseccion("Análisis de Convergencia Progresiva", icono="🔄")  # Encabezado en consola.
    if 'convergencia' in cfg.graficas_activas:  # Verifica si la gráfica de convergencia está activa.
        # Procesa únicamente si se disponen de registros generacionales y posee columnas clave.
        if not df_gen_mediano.empty and {'algorithm', 'init', 'crossover', 'generation'}.issubset(df_gen_mediano.columns):
            # Llama a la función gráfica para plasmar el historial generacional.
            artefactos_conv = graficar_evolucion_generacional(
                df_gen_mediano,  # Trayectorias de la réplica mediana seleccionada.
                dir_salida=cfg.carpetas['metricas_convergencia'],  # Carpeta física destino.
                etiqueta_modo=cfg.modo_ejecucion,  # Identificador del experimento.
                dpi=cfg.report_plot_dpi,  # Calidad gráfica.
                emitir_log=False,  # Apaga salida individual de logs.
            )
            for ruta, descripcion in artefactos_conv:  # Recorre los archivos resultantes.
                imprimir_grafico_guardado(ruta, descripcion)  # Muestra las rutas físicas guardadas.
        else:  # Si los datos históricos no eran válidos para la convergencia.
            # Escribe aviso de advertencia informando de la omisión del bloque.
            logger.warning("      • ⚠️  Convergencia omitida: no hay histórico generacional válido.")

    imprimir_subseccion("Síntesis Estadística Comparativa", icono="📊️")  # Encabezado en terminal.
    if not df_final.empty:  # Si disponemos de resultados históricos consolidados.
        tareas_estadisticas = []  # Lista vacía para registrar las tareas de resúmenes estadísticos.
        # Agrega boxplots si están habilitados en user_config.
        if 'boxplots' in cfg.graficas_activas: tareas_estadisticas.append(('boxplots', cfg.carpetas['sintesis_boxplots']))
        # Agrega violines si están habilitados en user_config.
        if 'violines' in cfg.graficas_activas: tareas_estadisticas.append(('violin', cfg.carpetas['sintesis_violines']))
        # Agrega gráficos de barra media/std si están activos.
        if 'media_std' in cfg.graficas_activas: tareas_estadisticas.append(('mean_std', cfg.carpetas['sintesis_barras']))
        
        resultados_est = {}  # Inicializa el mapeo local de salidas estadísticas.
        if tareas_estadisticas:  # Lanza el pool si hay tareas estadísticas requeridas.
            # Calcula el número de workers óptimo limitado por el total de trabajos.
            max_workers_est = min(calcular_max_workers_paralelo(), len(tareas_estadisticas))
            # Inicializa el ejecutor de subprocesos concurrentes.
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers_est) as executor:
                # Envía las tareas de síntesis estadística al pool concurrente.
                futures = [
                    executor.submit(
                        _tarea_sintesis_estadistica,
                        tarea,
                        df_final,
                        dir_salida,
                        cfg.modo_ejecucion,
                        cfg.report_plot_dpi,
                    )
                    for tarea, dir_salida in tareas_estadisticas
                ]
                for future in concurrent.futures.as_completed(futures):  # Recorre los subprocesos según terminen.
                    try:
                        resultado = future.result()  # Extrae el resultado.
                        # Almacena el resultado asociado al tipo de tarea estadística correspondiente.
                        resultados_est[resultado.get('tarea')] = resultado.get('artefactos', [])
                    except Exception as e:  # Controla excepciones de paralelización.
                        logger.error(f"      • ⚠️  Error en síntesis estadística paralela: {e}")  # Reporta error.

        if 'boxplots' in cfg.graficas_activas:  # Si boxplots estuvo activo.
            imprimir_subseccion("Resumen Global (Boxplots)", icono="📦")  # Encabezado en terminal.
            for ruta, descripcion in resultados_est.get('boxplots', []):  # Itera sobre los archivos boxplot creados.
                imprimir_grafico_guardado(ruta, descripcion)  # Informa el archivo guardado.

        if 'violines' in cfg.graficas_activas:  # Si violines estuvo activo.
            imprimir_subseccion("Distribuciones Detalladas (Violin Plots)", icono="🎻")  # Encabezado en terminal.
            for ruta, descripcion in resultados_est.get('violin', []):  # Itera sobre los gráficos de violín creados.
                imprimir_grafico_guardado(ruta, descripcion)  # Informa el archivo de imagen creado en los logs.

        if 'media_std' in cfg.graficas_activas:  # Si media y std estuvo activo.
            imprimir_subseccion("Análisis de Tendencia Central (Media ± Std)", icono="📉")  # Encabezado en terminal.
            for ruta, descripcion in resultados_est.get('mean_std', []):  # Itera sobre los gráficos de barra guardados.
                imprimir_grafico_guardado(ruta, descripcion)  # Informa el archivo guardado en el logger.

    if not df_final.empty:  # Si disponemos de resultados finales para calcular rankings.
        # Filtra las métricas básicas definidas de interés que estén en df_final.
        disponibles = [m for m in BASE_METRICS if m in df_final.columns]

        # Agrupa por algoritmo, inicialización y cruce para calcular la media aritmética de cada métrica.
        df_mean_config = df_final.groupby(['algorithm', 'init', 'crossover']).mean(numeric_only=True).reset_index()
        # Agrupa y calcula la desviación estándar por configuración.
        df_std_config = df_final.groupby(['algorithm', 'init', 'crossover']).std(numeric_only=True).reset_index()
        # Inicializa a cero la desviación estándar para la columna del promedio global de ranking.
        df_std_config['Average Ranking Overall'] = 0.0

        resumen_method = df_mean_config.copy()  # Copia el DataFrame de medias de configuración.
        rank_matrix_method = []  # Lista vacía temporal para acumular los rangos individuales de cada métrica.
        for m in disponibles:  # Itera a través de las métricas disponibles.
            asce = False if m in HIGHER_IS_BETTER_METRICS else True  # Invierte el sentido del ranking si la métrica es de maximizar.
            # Escribe la línea correspondiente.
            rank_matrix_method.append(resumen_method[m].rank(ascending=asce).values)
        # Calcula el promedio aritmético de los rankings acumulados a través de las métricas.
        resumen_method['Average Ranking Overall'] = np.mean(rank_matrix_method, axis=0)
        # Fusiona la columna del promedio de ranking al DataFrame principal de medias.
        df_mean_config = pd.merge(df_mean_config, resumen_method[['algorithm', 'init', 'crossover', 'Average Ranking Overall']], on=['algorithm', 'init', 'crossover'])

        # Agrupa por Algoritmo y calcula los valores medios para el ranking global por algoritmo.
        df_mean_algo = df_final.groupby(['algorithm']).mean(numeric_only=True).reset_index()
        rank_matrix_algo = []  # Inicializa lista temporal para acumular rankings por algoritmo.
        for m in disponibles:  # Recorre cada métrica.
            asce = False if m in HIGHER_IS_BETTER_METRICS else True  # Sentido del orden.
            # Registra los rangos relativos.
            rank_matrix_algo.append(df_mean_algo[m].rank(ascending=asce).values)
        # Calcula y asigna el promedio de rango por algoritmo.
        df_mean_algo['Average Ranking (Algorithm)'] = np.mean(rank_matrix_algo, axis=0)

        # Agrupa por Inicialización y calcula las medias de métricas.
        df_mean_init = df_final.groupby(['init']).mean(numeric_only=True).reset_index()
        rank_matrix_init = []  # Lista temporal para rankings por inicialización.
        for m in disponibles:  # Recorre las métricas.
            asce = False if m in HIGHER_IS_BETTER_METRICS else True  # Sentido del orden.
            # Registra los rangos relativos.
            rank_matrix_init.append(df_mean_init[m].rank(ascending=asce).values)
        # Asigna el promedio de rango por inicialización.
        df_mean_init['Average Ranking (Initialization)'] = np.mean(rank_matrix_init, axis=0)

        # Agrupa por Crossover y calcula medias.
        df_mean_cross = df_final.groupby(['crossover']).mean(numeric_only=True).reset_index()
        rank_matrix_cross = []  # Lista temporal para rankings por cruce.
        for m in disponibles:  # Recorre métricas.
            asce = False if m in HIGHER_IS_BETTER_METRICS else True  # Sentido del orden.
            # Registra los rangos relativos.
            rank_matrix_cross.append(df_mean_cross[m].rank(ascending=asce).values)
        # Asigna el promedio de rango por crossover.
        df_mean_cross['Average Ranking (Crossover)'] = np.mean(rank_matrix_cross, axis=0)

        imprimir_subseccion("Ranking por Métrica", icono="🏆")  # Subsección en la consola.
        
        # Mapeo estructurado para cada tipo de ranking
        # Cada tupla contiene: (nombre_métrica, ¿ascendente?, flecha_consola, tipo_de_dataframe)
        metricas_ranking = [
            ('Range', False, '↑', 'config'),
            ('MinSum', True, '↓', 'config'),
            ('SumMin', True, '↓', 'config'),
            ('MaxToleranceRate', False, '↑', 'config'),
            ('AvgToleranceRate', False, '↑', 'config'),
            ('AvgHammingDistance', False, '↑', 'config'),
            ('Hypervolume', False, '↑', 'config'),
            ('IGD+', True, '↓', 'config'),
            ('GD+', True, '↓', 'config'),
            ('Average Ranking Overall', True, '↓', 'config'),
            ('Average Ranking (Algorithm)', True, '↓', 'algorithm'),
            ('Average Ranking (Initialization)', True, '↓', 'init'),
            ('Average Ranking (Crossover)', True, '↓', 'crossover')
        ]
        
        for metrica, ascending, flecha, tipo in metricas_ranking:  # Itera sobre los rankings de métricas.
            if tipo == 'config':  # Si el tipo es configuración.
                df_act = df_mean_config  # Medias por configuración.
                df_std_act = df_std_config  # Desviaciones por configuración.
                cols_grp = ['algorithm', 'init', 'crossover']  # Columnas de agrupación.
            elif tipo == 'algorithm':  # Si el tipo es algoritmo.
                df_act = df_mean_algo  # Medias por algoritmo.
                df_std_act = None  # No aplica desviación estándar.
                cols_grp = ['algorithm']  # Columna de agrupación.
            elif tipo == 'init':  # Si el tipo es inicialización.
                df_act = df_mean_init  # Medias por inicialización.
                df_std_act = None  # No aplica desviación.
                cols_grp = ['init']  # Columna de agrupación.
            else: # crossover  # Si es crossover.
                df_act = df_mean_cross  # Medias por operador de cruce.
                df_std_act = None  # No aplica desviación.
                cols_grp = ['crossover']  # Columna de agrupación.

            if metrica in df_act.columns:  # Procesa si la métrica de interés está presente.
                logger.info(f"      • \033[1m{metrica}\033[0m ({flecha})")  # Resalta la métrica y optimalidad en logs.
                df_sorted = df_act.sort_values(by=metrica, ascending=ascending)  # Ordena el dataframe según optimalidad.
                
                # Ruta del archivo CSV físico donde exportar el ranking.
                ruta_csv = os.path.join(cfg.carpetas['rankings'], f"ranking_{metrica.replace(' ', '_')}_{cfg.modo_ejecucion}.csv")
                # Escribe la tabla de clasificación ordenada al archivo físico CSV.
                df_sorted[cols_grp + [metrica]].to_csv(ruta_csv, index=False)
                
                total = len(df_sorted)  # Número de elementos en el ranking.
                
                for idx, (_, row) in enumerate(df_sorted.head(10).iterrows(), 1):  # Recorre las 10 mejores configuraciones.
                    val = row[metrica]  # Valor de rendimiento medio.
                    str_std = " ± 0.0000"  # Inicializa la cadena de desviación.
                    if df_std_act is not None:  # Si aplica desviación para este nivel.
                        # Filtra la desviación estándar asociada.
                        std_v = df_std_act.loc[(df_std_act['algorithm'] == row['algorithm']) & (df_std_act['init'] == row['init']) & (df_std_act['crossover'] == row['crossover']), metrica].values
                        # Formatea el valor si está disponible y es válido.
                        str_std = f" ± {std_v[0]:.4f}" if len(std_v) > 0 and pd.notna(std_v[0]) else " ± 0.0000"
                    
                    # Genera la cadena del nombre de la configuración según el nivel de agregación.
                    nombre = f"{row['algorithm']} ({row['init']}+{row['crossover']})" if tipo == 'config' else row[tipo]
                    logger.info(f"         {idx:2d}. {nombre}: {val:.4f}{str_std}")  # Escribe la línea del ranking.
                
                if total > 10:  # Si existen más de 10 elementos.
                    logger.info(f"         ...")  # Escribe puntos suspensivos.
                    n_worst = min(5, total - 10)  # Obtiene el número de peores configuraciones a listar (máximo 5).
                    # Itera sobre los peores registros de la cola del ranking.
                    for idx, (_, row) in enumerate(df_sorted.tail(n_worst).iterrows(), total - n_worst + 1):
                        val = row[metrica]  # Valor medio.
                        str_std = " ± 0.0000"  # Cadena por defecto.
                        if df_std_act is not None:  # Si aplica desviación estándar.
                            # Extrae la desviación.
                            std_v = df_std_act.loc[(df_std_act['algorithm'] == row['algorithm']) & (df_std_act['init'] == row['init']) & (df_std_act['crossover'] == row['crossover']), metrica].values
                            str_std = f" ± {std_v[0]:.4f}" if len(std_v) > 0 and pd.notna(std_v[0]) else " ± 0.0000"
                        # Define nombre legible de la configuración.
                        nombre = f"{row['algorithm']} ({row['init']}+{row['crossover']})" if tipo == 'config' else row[tipo]
                        logger.info(f"         {idx:2d}. {nombre}: {val:.4f}{str_std}")  # Escribe la peor configuración.

                link_csv = obtener_enlace_terminal(ruta_csv)  # Genera el hipervínculo para la terminal.
                logger.info(f"         • \033[1mPara más detalles\033[0m: {link_csv}")  # Escribe en consola el acceso directo.

                graficar_hm = 'estadistica' in cfg.graficas_activas  # Booleano que indica si se guardan los heatmaps del test.
                if tipo == 'config' and 'Average Ranking' not in metrica:  # Kruskal-Wallis + Dunn para configuraciones.
                    # Llama al motor de validación estadística y post-hoc Dunn.
                    graficar_analisis_kruskal_dunn(df_final, cfg.carpetas['rankings'], metrica, cfg.modo_ejecucion, indent=9, graficar=graficar_hm)
                elif 'Average Ranking' in metrica:  # Friedman + Nemenyi para promedios de ranking.
                    if metrica == 'Average Ranking Overall':  # Agrupación config.
                        col_friedman = 'config'  # Configuración completa.
                    elif 'Algorithm' in metrica:  # Agrupación algorithm.
                        col_friedman = 'algorithm'  # Algoritmo.
                    elif 'Initialization' in metrica:  # Agrupación init.
                        col_friedman = 'init'  # Inicialización.
                    else:  # Agrupación crossover.
                        col_friedman = 'crossover'  # Operador de cruce.
                    # Llama al test no paramétrico de Friedman y post-hoc Nemenyi.
                    graficar_analisis_estadistico(df_final, cfg.carpetas['rankings'], cfg.modo_ejecucion, col_group=col_friedman, indent=9, graficar=graficar_hm)
                
                logger.info("") # Inserta línea en blanco de separación estética.

    if 'mcdm' in cfg.graficas_activas:  # Procesa la sección MCDM si está activa en la configuración.
        imprimir_subseccion("Análisis de Decisión Multi-Criterio (MCDM)", icono="🎯")  # Encabezado en terminal.
        if df_fronts_mediano is not None and not df_fronts_mediano.empty:  # Si los frentes medianos son correctos.
            try:
                # Llama a la función principal del análisis MCDM.
                analizar_decision_mcdm(
                    df_fronts_mediano,  # Datos del frente de Pareto de la réplica mediana.
                    dir_salida=cfg.carpetas['decision_mcdm'],  # Carpeta destino física.
                    etiqueta_modo=cfg.modo_ejecucion,  # Identificador del experimento.
                    modo_transformacion_objetivos=cfg.modo_transformacion_objetivos,  # Dirección matemática de optimización.
                    dpi=cfg.report_plot_dpi,  # Calidad gráfica.
                    emitir_log=True,  # Imprime logs y resúmenes de decisión en terminal.
                )  # Fin de la llamada a analizar_decision_mcdm.
                logger.info("      • Análisis MCDM completado.")  # Notifica en logs la finalización correcta.
            except Exception as e:  # Captura fallos internos del módulo de decisión.
                logger.error(f"      • ⚠️  Error en análisis MCDM: {e}")  # Registra el mensaje de error.
        else:  # Si no hay datos de frentes válidos.
            # Advierte que se omite el análisis por falta de información.
            logger.warning("      • ⚠️  MCDM omitido: no hay datos de frentes de Pareto disponibles.")
