"""
Módulo de Orquestación de Reportes (reporting_pipeline.py)
---------------------------------------------------------
Gestiona la ejecución concurrente de las tareas de visualización y post-procesamiento.
"""

import os
import time
import concurrent.futures
import pandas as pd
import numpy as np

from snp_tag.config import ConfiguracionExperimento
from snp_tag.utils.terminal import (
    imprimir_encabezado, imprimir_subseccion, 
    imprimir_grafico_guardado, obtener_enlace_terminal
)
from snp_tag.utils.logger import logger
from snp_tag.utils.runtime import calcular_max_workers_paralelo

from snp_tag.engine.metrics_logic import decodificar_objetivos_reales

from snp_tag.visualization.fronts_plot import (
    graficar_frentes_pareto, 
    graficar_correlacion_objetivos_pareto,
    graficar_coordenadas_paralelas_pareto,
    graficar_frentes_pareto_agregados
)
from snp_tag.visualization.convergence_plot import graficar_evolucion_generacional
from snp_tag.visualization.stats_plot import (
    graficar_boxplot_metricas,
    graficar_violin_metricas, graficar_media_std_metricas,
    graficar_analisis_estadistico,
    graficar_analisis_kruskal_dunn
)
from snp_tag.visualization.mcdm_plot import analizar_decision_mcdm

from typing import Dict, Any, Optional, Tuple, List

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
    salida = {'tarea': tipo_tarea, 'artefactos': []}
    if tipo_tarea == 'correlation':
        salida['artefactos'] = graficar_correlacion_objetivos_pareto(
            df_fronts_total,
            carpetas,
            etiqueta_modo,
            dpi=dpi,
            emitir_log=False,
            modo_transformacion_objetivos=modo_transformacion_objetivos,
        )
    elif tipo_tarea == 'parallel':
        salida['artefactos'] = graficar_coordenadas_paralelas_pareto(
            df_fronts_total,
            semilla,
            carpetas,
            etiqueta_modo,
            dpi=dpi,
            emitir_log=False,
            modo_transformacion_objetivos=modo_transformacion_objetivos,
        )
    elif tipo_tarea == 'all_fronts':
        artefactos = []
        # 1. Gráficos individuales por (algo, init) y agregados por algoritmo
        from snp_tag.constants import PREFERRED_ALGORITHMS_ORDER
        algoritmos_presentes = sorted(df_fronts_total['algorithm'].dropna().unique().tolist())
        algoritmos_ordenados = [a for a in PREFERRED_ALGORITHMS_ORDER if a in algoritmos_presentes]
        algoritmos_ordenados.extend([a for a in algoritmos_presentes if a not in algoritmos_ordenados])

        for algo in algoritmos_ordenados:
            df_algo = df_fronts_total[df_fronts_total['algorithm'] == algo].copy()
            if not df_algo.empty:
                # Mantener individuales
                artefactos.extend(
                    graficar_frentes_pareto(
                        df_algo, algo, carpetas=carpetas, etiqueta_modo=etiqueta_modo,
                        dpi=dpi,
                        emitir_log=False,
                        limites_ejes=limites_ejes,
                        modo_transformacion_objetivos=modo_transformacion_objetivos,
                    )
                )
                # NUEVO: Gráfico agregado por algoritmo (todas sus inits juntas)
                nombre_agg = f'frentes_pareto_agregados_{algo.lower()}_{etiqueta_modo}.png'
                style_col = 'crossover' if 'crossover' in df_algo.columns else None
                artefactos.extend(
                    graficar_frentes_pareto_agregados(
                        df_algo, f"Frentes de Pareto Agregados: {algo}", 
                        nombre_agg, hue_col='init', style_col=style_col, carpetas=carpetas, 
                        dpi=dpi,
                        emitir_log=False,
                        limites_ejes=limites_ejes,
                        modo_transformacion_objetivos=modo_transformacion_objetivos,
                    )
                )
        
        # 2. NUEVO: Gráfico global de todos los frentes
        if not df_fronts_total.empty:
            df_global = df_fronts_total.copy()
            df_global['Configuración'] = df_global['algorithm'] + " (" + df_global['init'] + ")"
            nombre_global = f'frentes_pareto_global_{etiqueta_modo}.png'
            artefactos.extend(
                graficar_frentes_pareto_agregados(
                    df_global, "Comparativa Global de Frentes de Pareto", 
                    nombre_global, hue_col='Configuración', style_col=None, carpetas=carpetas,
                    dpi=dpi,
                    emitir_log=False,
                    limites_ejes=limites_ejes,
                    modo_transformacion_objetivos=modo_transformacion_objetivos,
                )
            )
        salida['artefactos'] = artefactos
    return salida


def _tarea_sintesis_estadistica(tipo_tarea: str, df_final: pd.DataFrame, dir_salida: str, etiqueta_modo: str, dpi: int) -> Dict[str, Any]:
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
    if tipo_tarea == 'boxplots':
        artefactos = graficar_boxplot_metricas(df_final, dir_salida, etiqueta_modo, dpi=dpi, emitir_log=False)
    elif tipo_tarea == 'violin':
        artefactos = graficar_violin_metricas(df_final, dir_salida, etiqueta_modo, dpi=dpi, emitir_log=False)
    elif tipo_tarea == 'mean_std':
        artefactos = graficar_media_std_metricas(df_final, dir_salida, etiqueta_modo, dpi=dpi, emitir_log=False)
    else:
        artefactos = []
    return {'tarea': tipo_tarea, 'artefactos': artefactos}


def ejecutar_reportes_visualizacion(cfg: ConfiguracionExperimento, df_final: pd.DataFrame,
                                    df_gen: pd.DataFrame, df_fronts_total: pd.DataFrame) -> None:
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
    imprimir_encabezado("REPORTES Y VISUALIZACIÓN")

    if df_fronts_total is None:
        df_fronts_total = pd.DataFrame()

    # Frentes de Pareto
    imprimir_subseccion("Distribución y Correlación de Frentes de Pareto", icono="🔍")
    t0_frentes = time.time()

    cols_fronts = {
        'algorithm', 'init', 'f1_compactness', 'f2_transformed_tolerance',
        'f3_transformed_hamming_avg', 'f4_balance_var'
    }

    if df_fronts_total.empty:
        logger.warning("      • ⚠️  Pareto omitido: no hay datos de frentes disponibles (CSV frentes ausente/vacío).")
    elif not cols_fronts.issubset(set(df_fronts_total.columns)):
        faltantes = sorted(cols_fronts - set(df_fronts_total.columns))
        alias_disponibles = []
        if 'f2_neg_tolerance' in df_fronts_total.columns:
            alias_disponibles.append('f2_neg_tolerance')
        if 'f3_neg_hamming_avg' in df_fronts_total.columns:
            alias_disponibles.append('f3_neg_hamming_avg')
        mensaje_alias = (
            f" Alias detectados en CSV: {sorted(alias_disponibles)}."
            if alias_disponibles else ""
        )
        logger.warning(
            "      • ⚠️  Pareto omitido: el CSV de frentes no contiene columnas requeridas: "
            f"{faltantes}.{mensaje_alias}"
        )
    else:
        # Cálculo de límites globales para estandarizar ejes (unificar escala entre algoritmos)
        limites_ejes = {}
        objetivos_reales = decodificar_objetivos_reales(
            df_fronts_total[
                ['f1_compactness', 'f2_transformed_tolerance', 'f3_transformed_hamming_avg', 'f4_balance_var']
            ].to_numpy(dtype=float),
            modo_transformacion_objetivos=cfg.modo_transformacion_objetivos,
        )
        mapping_objetivos = {
            'Compacidad': pd.Series(objetivos_reales['compacidad']),
            'Tolerancia': pd.Series(objetivos_reales['tolerancia_real']),
            'Hamming': pd.Series(objetivos_reales['hamming_prom_real']),
            'Balance': pd.Series(objetivos_reales['balance_var']),
        }
        for nombre, serie in mapping_objetivos.items():
            vmin, vmax = serie.min(), serie.max()
            if not pd.isna(vmin) and not pd.isna(vmax):
                rango = vmax - vmin
                margen = rango * 0.05 if rango > 0 else 0.5
                limites_ejes[nombre] = (vmin - margen, vmax + margen)

        max_workers = calcular_max_workers_paralelo()
        tareas_plot = ['all_fronts', 'correlation', 'parallel']
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    _tarea_plot_frentes,
                    t,
                    df_fronts_total,
                    cfg.carpetas,
                    cfg.modo_ejecucion,
                    cfg.report_plot_dpi,
                    cfg.semilla_maestra,
                    limites_ejes,
                    cfg.modo_transformacion_objetivos,
                )
                for t in tareas_plot
            ]
            resultados_tareas = {}
            for future in concurrent.futures.as_completed(futures):
                try:
                    resultado = future.result()
                    resultados_tareas[resultado.get('tarea')] = resultado.get('artefactos', [])
                except Exception as e:
                    logger.error(f"      • ⚠️  Error en graficado paralelo: {e}")

        imprimir_subseccion("Correlación de Objetivos", icono="🔗")
        for ruta, descripcion in resultados_tareas.get('correlation', []):
            imprimir_grafico_guardado(ruta, descripcion)

        imprimir_subseccion("Coordenadas Paralelas", icono="📈")
        for ruta, descripcion in resultados_tareas.get('parallel', []):
            imprimir_grafico_guardado(ruta, descripcion)

        imprimir_subseccion("Frentes de Pareto", icono="🎯")
        for ruta, descripcion in resultados_tareas.get('all_fronts', []):
            imprimir_grafico_guardado(ruta, descripcion)

    logger.info(f"      • Tiempo bloque 'Frentes de Pareto': {time.time() - t0_frentes:.1f}s")

    imprimir_subseccion("Análisis de Convergencia Progresiva", icono="🔄")
    if not df_gen.empty and {'algorithm', 'init', 'crossover', 'generation'}.issubset(df_gen.columns):
        artefactos_conv = graficar_evolucion_generacional(
            df_gen,
            dir_salida=cfg.carpetas['metricas_convergencia'],
            etiqueta_modo=cfg.modo_ejecucion,
            dpi=cfg.report_plot_dpi,
            emitir_log=False,
        )
        for ruta, descripcion in artefactos_conv:
            imprimir_grafico_guardado(ruta, descripcion)
    else:
        logger.warning("      • ⚠️  Convergencia omitida: no hay histórico generacional válido.")

    imprimir_subseccion("Síntesis Estadística Comparativa", icono="📊️")
    if not df_final.empty:
        tareas_estadisticas = [
            ('boxplots', cfg.carpetas['sintesis_boxplots']),
            ('violin', cfg.carpetas['sintesis_violines']),
            ('mean_std', cfg.carpetas['sintesis_barras']),
        ]
        resultados_est = {}
        max_workers_est = min(calcular_max_workers_paralelo(), len(tareas_estadisticas))
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers_est) as executor:
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
            for future in concurrent.futures.as_completed(futures):
                try:
                    resultado = future.result()
                    resultados_est[resultado.get('tarea')] = resultado.get('artefactos', [])
                except Exception as e:
                    logger.error(f"      • ⚠️  Error en síntesis estadística paralela: {e}")

        imprimir_subseccion("Resumen Global (Boxplots)", icono="📦")
        for ruta, descripcion in resultados_est.get('boxplots', []):
            imprimir_grafico_guardado(ruta, descripcion)

        imprimir_subseccion("Distribuciones Detalladas (Violin Plots)", icono="🎻")
        for ruta, descripcion in resultados_est.get('violin', []):
            imprimir_grafico_guardado(ruta, descripcion)

        imprimir_subseccion("Análisis de Tendencia Central (Media ± Std)", icono="📉")
        for ruta, descripcion in resultados_est.get('mean_std', []):
            imprimir_grafico_guardado(ruta, descripcion)

    if not df_final.empty:
        # --- 1. Preparación de Datos y Rankings ---
        from snp_tag.constants import HIGHER_IS_BETTER_METRICS, BASE_METRICS
        disponibles = [m for m in BASE_METRICS if m in df_final.columns]

        # a) Medias y Desviaciones por Configuración (Algoritmo + Init + Crossover)
        df_mean_config = df_final.groupby(['algorithm', 'init', 'crossover']).mean(numeric_only=True).reset_index()
        df_std_config = df_final.groupby(['algorithm', 'init', 'crossover']).std(numeric_only=True).reset_index()
        df_std_config['Average Ranking Overall'] = 0.0

        # b) Cálculo de Average Ranking Overall (Método)
        resumen_method = df_mean_config.copy()
        rank_matrix_method = []
        for m in disponibles:
            asce = False if m in HIGHER_IS_BETTER_METRICS else True
            rank_matrix_method.append(resumen_method[m].rank(ascending=asce).values)
        resumen_method['Average Ranking Overall'] = np.mean(rank_matrix_method, axis=0)
        df_mean_config = pd.merge(df_mean_config, resumen_method[['algorithm', 'init', 'crossover', 'Average Ranking Overall']], on=['algorithm', 'init', 'crossover'])

        # c) Cálculo de Average Ranking (Algorithm)
        df_mean_algo = df_final.groupby(['algorithm']).mean(numeric_only=True).reset_index()
        rank_matrix_algo = []
        for m in disponibles:
            asce = False if m in HIGHER_IS_BETTER_METRICS else True
            rank_matrix_algo.append(df_mean_algo[m].rank(ascending=asce).values)
        df_mean_algo['Average Ranking (Algorithm)'] = np.mean(rank_matrix_algo, axis=0)

        # d) Cálculo de Average Ranking (Initialization)
        df_mean_init = df_final.groupby(['init']).mean(numeric_only=True).reset_index()
        rank_matrix_init = []
        for m in disponibles:
            asce = False if m in HIGHER_IS_BETTER_METRICS else True
            rank_matrix_init.append(df_mean_init[m].rank(ascending=asce).values)
        df_mean_init['Average Ranking (Initialization)'] = np.mean(rank_matrix_init, axis=0)

        # e) Cálculo de Average Ranking (Crossover)
        df_mean_cross = df_final.groupby(['crossover']).mean(numeric_only=True).reset_index()
        rank_matrix_cross = []
        for m in disponibles:
            asce = False if m in HIGHER_IS_BETTER_METRICS else True
            rank_matrix_cross.append(df_mean_cross[m].rank(ascending=asce).values)
        df_mean_cross['Average Ranking (Crossover)'] = np.mean(rank_matrix_cross, axis=0)

        # --- 2. Impresión Unificada de Rankings y Estadísticas ---
        imprimir_subseccion("Ranking por Métrica", icono="🏆")
        
        # Definición del orden y tipo de métrica
        # (Nombre, Ascendente?, Flecha, Tipo [config, algorithm, init])
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
        
        for metrica, ascending, flecha, tipo in metricas_ranking:
            # Seleccionar dataframe según tipo
            if tipo == 'config':
                df_act = df_mean_config
                df_std_act = df_std_config
                cols_grp = ['algorithm', 'init', 'crossover']
            elif tipo == 'algorithm':
                df_act = df_mean_algo
                df_std_act = None
                cols_grp = ['algorithm']
            elif tipo == 'init':
                df_act = df_mean_init
                df_std_act = None
                cols_grp = ['init']
            else: # crossover
                df_act = df_mean_cross
                df_std_act = None
                cols_grp = ['crossover']

            if metrica in df_act.columns:
                logger.info(f"      • \033[1m{metrica}\033[0m ({flecha})")
                df_sorted = df_act.sort_values(by=metrica, ascending=ascending)
                
                # Exportar CSV
                ruta_csv = os.path.join(cfg.carpetas['rankings'], f"ranking_{metrica.replace(' ', '_')}_{cfg.modo_ejecucion}.csv")
                df_sorted[cols_grp + [metrica]].to_csv(ruta_csv, index=False)
                
                total = len(df_sorted)
                
                # Imprimir Ranking (Top 10 + Peores 5)
                for idx, (_, row) in enumerate(df_sorted.head(10).iterrows(), 1):
                    val = row[metrica]
                    str_std = " ± 0.0000"
                    if df_std_act is not None:
                        std_v = df_std_act.loc[(df_std_act['algorithm'] == row['algorithm']) & (df_std_act['init'] == row['init']) & (df_std_act['crossover'] == row['crossover']), metrica].values
                        str_std = f" ± {std_v[0]:.4f}" if len(std_v) > 0 and pd.notna(std_v[0]) else " ± 0.0000"
                    
                    nombre = f"{row['algorithm']} ({row['init']}+{row['crossover']})" if tipo == 'config' else row[tipo]
                    logger.info(f"         {idx:2d}. {nombre}: {val:.4f}{str_std}")
                
                if total > 10:
                    logger.info(f"         ...")
                    n_worst = min(5, total - 10)
                    for idx, (_, row) in enumerate(df_sorted.tail(n_worst).iterrows(), total - n_worst + 1):
                        val = row[metrica]
                        str_std = " ± 0.0000"
                        if df_std_act is not None:
                            std_v = df_std_act.loc[(df_std_act['algorithm'] == row['algorithm']) & (df_std_act['init'] == row['init']) & (df_std_act['crossover'] == row['crossover']), metrica].values
                            str_std = f" ± {std_v[0]:.4f}" if len(std_v) > 0 and pd.notna(std_v[0]) else " ± 0.0000"
                        nombre = f"{row['algorithm']} ({row['init']}+{row['crossover']})" if tipo == 'config' else row[tipo]
                        logger.info(f"         {idx:2d}. {nombre}: {val:.4f}{str_std}")

                # Enlace al CSV detallado del ranking
                link_csv = obtener_enlace_terminal(ruta_csv)
                logger.info(f"         • \033[1mPara más detalles\033[0m: {link_csv}")

                # --- VALIDACIÓN ESTADÍSTICA ANIDADA ---
                if tipo == 'config' and 'Average Ranking' not in metrica:
                    graficar_analisis_kruskal_dunn(df_final, cfg.carpetas['rankings'], metrica, cfg.modo_ejecucion, indent=9)
                elif 'Average Ranking' in metrica:
                    if metrica == 'Average Ranking Overall':
                        col_friedman = 'config'
                    elif 'Algorithm' in metrica:
                        col_friedman = 'algorithm'
                    elif 'Initialization' in metrica:
                        col_friedman = 'init'
                    else:
                        col_friedman = 'crossover'
                    graficar_analisis_estadistico(df_final, cfg.carpetas['rankings'], cfg.modo_ejecucion, col_group=col_friedman, indent=9)
                
                logger.info("") # Espacio entre métricas

    # --- Análisis de Decisión Multi-Criterio (MCDM) ---
    imprimir_subseccion("Análisis de Decisión Multi-Criterio (MCDM)", icono="🎯")
    if df_fronts_total is not None and not df_fronts_total.empty:
        try:
            analizar_decision_mcdm(
                df_fronts_total,
                dir_salida=cfg.carpetas['decision_mcdm'],
                etiqueta_modo=cfg.modo_ejecucion,
                modo_transformacion_objetivos=cfg.modo_transformacion_objetivos,
                dpi=cfg.report_plot_dpi,
                emitir_log=True,
            )
            logger.info("      • Análisis MCDM completado.")
        except Exception as e:
            logger.error(f"      • ⚠️  Error en análisis MCDM: {e}")
    else:
        logger.warning("      • ⚠️  MCDM omitido: no hay datos de frentes de Pareto disponibles.")
