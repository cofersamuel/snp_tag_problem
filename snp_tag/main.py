"""
Módulo Principal (main.py)
--------------------------
Punto de entrada oficial para la ejecución del pipeline modular Tag SNP.
Coordina la configuración, carga de datos, diagnóstico, ejecución evolutiva
y síntesis de resultados.
"""

import os
import sys
import time
import shutil
import tempfile
import argparse
import concurrent.futures
from pathlib import Path
import pandas as pd
import numpy as np
import multiprocessing
from itertools import combinations

# Importaciones del paquete snp_tag
from snp_tag.config import ConfiguracionExperimento, construir_configuracion, MODOS_DISPONIBLES, FUENTES_DATOS_DISPONIBLES
from snp_tag.utils.terminal import (
    imprimir_encabezado, imprimir_subseccion, imprimir_paso, 
    imprimir_estado, imprimir_grafico_guardado, Tee, obtener_bit_string_estilizado,
    imprimir_metadato, obtener_enlace_terminal
)
from snp_tag.utils.filesystem import crear_arbol_directorios_dataset
from snp_tag.utils.runtime import calcular_max_workers_paralelo
from snp_tag.data.loader import cargar_dataset_objetivo, exportar_dataset
from snp_tag.data.diagnostics import ejecutar_diagnostico_ld, analizar_similitud_genotipica, calcular_ld_completo, detectar_bloques_ld
from snp_tag.core.algorithm import construir_direcciones_referencia
from snp_tag.engine.runner import ejecutar_suite_completa
from snp_tag.engine.metrics import (
    evaluar_metricas_finales,
    construir_metricas_generacionales,
    decodificar_objetivos_reales,
    filtrar_soluciones_factibles,
    obtener_referencias_estaticas_dataset,
)
from snp_tag.visualization.dataset import (
    graficar_mapa_calor_haplotipos, graficar_bloques_ld, 
    graficar_histograma_hamming, graficar_variabilidad_snps,
    graficar_conteo_alelos, graficar_histograma_alelico,
    graficar_ld_detallado
)
from snp_tag.visualization.fronts import (
    graficar_frentes_pareto, 
    graficar_correlacion_objetivos_pareto,
    graficar_coordenadas_paralelas_pareto,
    graficar_frentes_pareto_agregados
)
from snp_tag.visualization.convergence import graficar_evolucion_generacional
from snp_tag.visualization.reporting import (
    graficar_boxplot_metricas,
    graficar_rendimiento_tiempo, graficar_comparativa_objetivos,
    graficar_violin_metricas, graficar_media_std_metricas,
    graficar_analisis_estadistico,
    graficar_analisis_kruskal_dunn, graficar_diagrama_diferencia_critica
)
from snp_tag.visualization.decision import analizar_decision_mcdm


def _tarea_plot_frentes(
    tipo_tarea,
    df_fronts_total,
    carpetas,
    etiqueta_modo,
    dpi,
    semilla,
    limites_ejes=None,
    modo_transformacion_objetivos='neg',
):
    """Dispatcher picklable para paralelizar tareas de graficado de frentes."""
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
        algoritmos_preferidos = ['NSGA3', 'MOEAD_TCHE', 'MOEAD_PBI', 'MOEAD_WS', 'NSGA2', 'SPEA2']
        algoritmos_presentes = sorted(df_fronts_total['algorithm'].dropna().unique().tolist())
        algoritmos_ordenados = [a for a in algoritmos_preferidos if a in algoritmos_presentes]
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


def _tarea_sintesis_estadistica(tipo_tarea, df_final, dir_salida, etiqueta_modo, dpi):
    """Worker picklable para síntesis estadística (boxplot/violin/media±std)."""
    if tipo_tarea == 'boxplots':
        artefactos = graficar_boxplot_metricas(df_final, dir_salida, etiqueta_modo, dpi=dpi, emitir_log=False)
    elif tipo_tarea == 'violin':
        artefactos = graficar_violin_metricas(df_final, dir_salida, etiqueta_modo, dpi=dpi, emitir_log=False)
    elif tipo_tarea == 'mean_std':
        artefactos = graficar_media_std_metricas(df_final, dir_salida, etiqueta_modo, dpi=dpi, emitir_log=False)
    else:
        artefactos = []
    return {'tarea': tipo_tarea, 'artefactos': artefactos}

def inicializar_configuracion(modo='medium', data_source='hinds2005'):
    """
    Configura los parámetros iniciales del experimento.
    """
    return construir_configuracion(modo=modo, data_source=data_source)

def informar_configuracion(cfg: ConfiguracionExperimento):
    """Muestra un resumen jerárquico y condicional de user_config.ini en la terminal."""
    imprimir_encabezado("CONFIGURACIÓN (user_config.ini)")
    
    # [General]
    print(f"      • [General]")
    print(f"          - dir_salida_base: {cfg.dir_salida_base}")
    
    # [Dataset]
    print(f"      • [Dataset]")
    print(f"          - n_snps: {cfg.n_snps}")
    print(f"          - num_bloques: {cfg.num_bloques}")
    print(f"          - origen_datos: {cfg.origen_datos}")
    if cfg.origen_datos == 'synthetic':
        print(f"          - prob_flip_sintetico: {cfg.prob_flip_sintetico}")
        print(f"          - dif_min_pares_sintetico: {cfg.dif_min_pares_sintetico}")
        print(f"          - intentos_max_sintetico: {cfg.intentos_max_sintetico}")
    
    # [Objetivos]
    print(f"      • [Objetivos]")
    print(f"          - transform: {cfg.modo_transformacion_objetivos}")
    print(f"          - eval: {cfg.modo_evaluacion}")
    print(f"          - cap_tolerancia: {cfg.cap_tolerancia}")
    
    # [Algoritmos - General]
    print(f"      • [Algoritmos]")
    print(f"          - semilla_maestra: {cfg.semilla_maestra}")
    print(f"          - modo_semillas: {cfg.modo_semillas}")
    print(f"          - normalizacion: {cfg.modo_normalizacion}")
    print(f"          - pc: {cfg.pc}")
    print(f"          - pm: {cfg.pm:.6f}")
    print(f"          - cruces: {', '.join(cfg.crossover_operadores_activos)}")
    print(f"          - algoritmos: {', '.join(cfg.algoritmos_activos)}")
    print(f"          - inits: {', '.join(cfg.opciones_init)}")
    
    # [Algoritmos - Específicos] (Condicional)
    moead_activos = [a for a in cfg.algoritmos_activos if 'MOEAD' in a]
    if 'greedy_ting' in cfg.opciones_init or 'greedy_multi' in cfg.opciones_init or 'greedy_holistic' in cfg.opciones_init or moead_activos:
        print(f"      • [Parámetros Específicos de Algoritmo]")
        if 'greedy_ting' in cfg.opciones_init:
            print(f"          - ratio_greedy_ting: {cfg.ratio_greedy_ting}")
        if 'greedy_multi' in cfg.opciones_init:
            print(f"          - max_cobertura_objetivo: {cfg.max_cobertura_objetivo}")
        if 'greedy_holistic' in cfg.opciones_init:
            print(f"          - max_k_holistic: {cfg.max_k_holistic}")
        if moead_activos:
            print(f"          - moead_vecinos: {cfg.vecinos_moead}")
            print(f"          - moead_prob_vecindad: {cfg.prob_vecindad_moead}")
            if 'MOEAD_PBI' in cfg.algoritmos_activos:
                print(f"          - moead_theta_pbi: {cfg.theta_moead_pbi}")

    # [Reporting]
    print(f"      • [Sistema]")
    print(f"          - report_plot_dpi: {cfg.report_plot_dpi}")
    
    # [Resumen de Objetivos]
    suffix = " Prop." if cfg.modo_evaluacion == 'proportional' else ""
    print("      • Objetivos de optimización:")
    print(f"          - f1 (Compacidad): Minimizar")
    print(f"          - f2 (Tolerancia{suffix}): Maximizar")
    print(f"          - f3 (Hamming Medio{suffix}): Maximizar")
    print(f"          - f4 (Disimilitud{suffix}): Minimizar")


def _construir_df_fronts_desde_resultados(resultados, modo_transformacion_objetivos: str = 'neg'):
    """Construye un DataFrame tabular de soluciones de frentes finales para CSV/plots."""
    columnas = [
        'algorithm', 'init', 'crossover', 'run', 'seed',
        'f1_compactness',
        'f2_transformed_tolerance',
        'f3_transformed_hamming_avg',
        'f4_balance_var',
    ]
    filas = []
    for rr in resultados:
        if rr.F_final is None or len(rr.F_final) == 0:
            continue
        F_factibles = filtrar_soluciones_factibles(
            rr.F_final,
            modo_transformacion_objetivos=modo_transformacion_objetivos,
        )
        for f in F_factibles:
            filas.append({
                'algorithm': rr.algoritmo,
                'init': rr.inicializacion,
                'crossover': rr.crossover,
                'run': rr.replica,
                'seed': rr.semilla,
                'f1_compactness': f[0],
                'f2_transformed_tolerance': f[1],
                'f3_transformed_hamming_avg': f[2],
                'f4_balance_var': f[3],
            })
    if not filas:
        return pd.DataFrame(columns=columnas)
    df = pd.DataFrame(filas, columns=columnas)
    # Elimina duplicados exactos (puntos repetidos en el frente final)
    subset = [
        'algorithm', 'init', 'crossover', 'run', 'seed',
        'f1_compactness',
        'f2_transformed_tolerance',
        'f3_transformed_hamming_avg',
        'f4_balance_var',
    ]
    return df.drop_duplicates(subset=subset, keep='first').reset_index(drop=True)


def _inferir_modo_desde_nombre_csv(ruta_csv: Path, prefijo: str):
    """Extrae el modo desde un nombre tipo '<prefijo><modo>.csv'."""
    nombre = ruta_csv.name
    if not (nombre.startswith(prefijo) and nombre.endswith('.csv')):
        return None
    return nombre[len(prefijo):-4]


def _normalizar_columnas_frentes_csv(df_fronts: pd.DataFrame):
        """Normaliza nombres de columnas de frentes para compatibilidad en report-only.

        Esquema canónico actual:
            - f2_transformed_tolerance
            - f3_transformed_hamming_avg

        Alias compatibles (reportes antiguos):
            - f2_neg_tolerance
            - f3_neg_hamming_avg
        """
        if df_fronts is None or df_fronts.empty:
                return df_fronts, []

        df_norm = df_fronts.copy()
        columnas_renombradas = []

        alias_a_canonico = {
                'f2_neg_tolerance': 'f2_transformed_tolerance',
                'f3_neg_hamming_avg': 'f3_transformed_hamming_avg',
        }

        for alias, canonica in alias_a_canonico.items():
                if canonica not in df_norm.columns and alias in df_norm.columns:
                        df_norm.rename(columns={alias: canonica}, inplace=True)
                        columnas_renombradas.append((alias, canonica))

        return df_norm, columnas_renombradas


def _seleccionar_mas_reciente(ruta_input: Path, patron: str):
    """Devuelve el CSV más reciente por fecha de modificación."""
    candidatos = [p for p in ruta_input.glob(patron) if p.is_file()]
    if not candidatos:
        return None
    return max(candidatos, key=lambda p: p.stat().st_mtime)


def _cargar_dataframes_report_only(ruta_input: Path):
    """Carga CSVs desde snp_tag/input para ejecutar sólo reportes/visualización."""
    if not ruta_input.exists() or not ruta_input.is_dir():
        raise FileNotFoundError(f"No existe el directorio de entrada fijo: {ruta_input}")

    ruta_detallado = _seleccionar_mas_reciente(ruta_input, 'resultados_detallados_*.csv')
    if ruta_detallado is None:
        raise FileNotFoundError(
            f"No se encontró ningún 'resultados_detallados_*.csv' en {ruta_input}"
        )

    print(f"      • CSV detallado seleccionado: {ruta_detallado}")
    df_final = pd.read_csv(ruta_detallado)

    if df_final.empty:
        raise ValueError(f"El CSV detallado está vacío: {ruta_detallado}")

    modo_detectado = _inferir_modo_desde_nombre_csv(ruta_detallado, 'resultados_detallados_')

    ruta_hist = None
    if modo_detectado:
        candidato_modo = ruta_input / f"historico_generacional_{modo_detectado}.csv"
        if candidato_modo.exists() and candidato_modo.is_file():
            ruta_hist = candidato_modo
    if ruta_hist is None:
        ruta_hist = _seleccionar_mas_reciente(ruta_input, 'historico_generacional_*.csv')

    if ruta_hist is not None:
        print(f"      • CSV histórico seleccionado: {ruta_hist}")
        df_gen = pd.read_csv(ruta_hist)
    else:
        print("      • ⚠️  No se encontró CSV histórico; se omitirá el bloque de convergencia.")
        df_gen = pd.DataFrame()

    ruta_fronts = None
    if modo_detectado:
        candidato_fronts = ruta_input / f"frentes_pareto_{modo_detectado}.csv"
        if candidato_fronts.exists() and candidato_fronts.is_file():
            ruta_fronts = candidato_fronts
    if ruta_fronts is None:
        ruta_fronts = _seleccionar_mas_reciente(ruta_input, 'frentes_pareto_*.csv')

    if ruta_fronts is None:
        print("      • ⚠️  No hay CSV de frentes; se omitirá Pareto con aviso explícito.")
        df_fronts = pd.DataFrame()
    else:
        print(f"      • CSV frentes seleccionado: {ruta_fronts}")
        df_fronts = pd.read_csv(ruta_fronts)
        df_fronts, columnas_renombradas = _normalizar_columnas_frentes_csv(df_fronts)
        if columnas_renombradas:
            detalle = ', '.join([f"{src}→{dst}" for src, dst in columnas_renombradas])
            print(f"      • Compatibilidad report-only aplicada en frentes: {detalle}")

    return df_final, df_gen, df_fronts, modo_detectado


def ejecutar_reportes_visualizacion(cfg: ConfiguracionExperimento, df_final: pd.DataFrame,
                                    df_gen: pd.DataFrame, df_fronts_total: pd.DataFrame):
    """Ejecuta exclusivamente la fase de reportes y visualización."""
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
        print("      • ⚠️  Pareto omitido: no hay datos de frentes disponibles (CSV frentes ausente/vacío).")
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
        print(
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
                    print(f"      • ⚠️  Error en graficado paralelo: {e}")

        imprimir_subseccion("Correlación de Objetivos", icono="🔗")
        for ruta, descripcion in resultados_tareas.get('correlation', []):
            imprimir_grafico_guardado(ruta, descripcion)

        imprimir_subseccion("Coordenadas Paralelas", icono="📈")
        for ruta, descripcion in resultados_tareas.get('parallel', []):
            imprimir_grafico_guardado(ruta, descripcion)

        imprimir_subseccion("Frentes de Pareto", icono="🎯")
        for ruta, descripcion in resultados_tareas.get('all_fronts', []):
            imprimir_grafico_guardado(ruta, descripcion)

    print(f"      • Tiempo bloque 'Frentes de Pareto': {time.time() - t0_frentes:.1f}s")

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
        print("      • ⚠️  Convergencia omitida: no hay histórico generacional válido.")

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
                    print(f"      • ⚠️  Error en síntesis estadística paralela: {e}")

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
        higher_is_better = ['Hypervolume', 'Range', 'MaxToleranceRate', 'AvgToleranceRate', 'AvgHammingDistance']
        metricas_base = ['MinSum', 'Range', 'SumMin', 'MaxToleranceRate', 'AvgToleranceRate', 'AvgHammingDistance', 'Hypervolume', 'IGD+', 'GD+']
        disponibles = [m for m in metricas_base if m in df_final.columns]

        # a) Medias y Desviaciones por Configuración (Algoritmo + Init + Crossover)
        df_mean_config = df_final.groupby(['algorithm', 'init', 'crossover']).mean(numeric_only=True).reset_index()
        df_std_config = df_final.groupby(['algorithm', 'init', 'crossover']).std(numeric_only=True).reset_index()
        df_std_config['Average Ranking Overall'] = 0.0

        # b) Cálculo de Average Ranking Overall (Método)
        resumen_method = df_mean_config.copy()
        rank_matrix_method = []
        for m in disponibles:
            asce = False if m in higher_is_better else True
            rank_matrix_method.append(resumen_method[m].rank(ascending=asce).values)
        resumen_method['Average Ranking Overall'] = np.mean(rank_matrix_method, axis=0)
        df_mean_config = pd.merge(df_mean_config, resumen_method[['algorithm', 'init', 'crossover', 'Average Ranking Overall']], on=['algorithm', 'init', 'crossover'])

        # c) Cálculo de Average Ranking (Algorithm)
        df_mean_algo = df_final.groupby(['algorithm']).mean(numeric_only=True).reset_index()
        rank_matrix_algo = []
        for m in disponibles:
            asce = False if m in higher_is_better else True
            rank_matrix_algo.append(df_mean_algo[m].rank(ascending=asce).values)
        df_mean_algo['Average Ranking (Algorithm)'] = np.mean(rank_matrix_algo, axis=0)

        # d) Cálculo de Average Ranking (Initialization)
        df_mean_init = df_final.groupby(['init']).mean(numeric_only=True).reset_index()
        rank_matrix_init = []
        for m in disponibles:
            asce = False if m in higher_is_better else True
            rank_matrix_init.append(df_mean_init[m].rank(ascending=asce).values)
        df_mean_init['Average Ranking (Initialization)'] = np.mean(rank_matrix_init, axis=0)

        # e) Cálculo de Average Ranking (Crossover)
        df_mean_cross = df_final.groupby(['crossover']).mean(numeric_only=True).reset_index()
        rank_matrix_cross = []
        for m in disponibles:
            asce = False if m in higher_is_better else True
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
                print(f"    • \033[1m{metrica}\033[0m ({flecha})")
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
                    print(f"         {idx:2d}. {nombre}: {val:.4f}{str_std}")
                
                if total > 10:
                    print(f"         ...")
                    n_worst = min(5, total - 10)
                    for idx, (_, row) in enumerate(df_sorted.tail(n_worst).iterrows(), total - n_worst + 1):
                        val = row[metrica]
                        str_std = " ± 0.0000"
                        if df_std_act is not None:
                            std_v = df_std_act.loc[(df_std_act['algorithm'] == row['algorithm']) & (df_std_act['init'] == row['init']) & (df_std_act['crossover'] == row['crossover']), metrica].values
                            str_std = f" ± {std_v[0]:.4f}" if len(std_v) > 0 and pd.notna(std_v[0]) else " ± 0.0000"
                        nombre = f"{row['algorithm']} ({row['init']}+{row['crossover']})" if tipo == 'config' else row[tipo]
                        print(f"         {idx:2d}. {nombre}: {val:.4f}{str_std}")

                # Enlace al CSV detallado del ranking
                link_csv = obtener_enlace_terminal(ruta_csv)
                print(f"         • \033[1mPara más detalles\033[0m: {link_csv}")

                # --- VALIDACIÓN ESTADÍSTICA ANIDADA ---
                if tipo == 'config' and 'Average Ranking' not in metrica:
                    graficar_analisis_kruskal_dunn(df_final, cfg.carpetas['estadistica_hv'], metrica, cfg.modo_ejecucion, indent=9)
                    graficar_diagrama_diferencia_critica(df_final, cfg.carpetas['estadistica_hv'], metrica, cfg.modo_ejecucion, indent=9)
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
                
                print() # Espacio entre métricas

    # --- Análisis de Decisión Multi-Criterio (MCDM) ---
    imprimir_subseccion("Análisis de Decisión Multi-Criterio (MCDM)", icono="🎯")
    if df_fronts_total is not None and not df_fronts_total.empty:
        analizar_decision_mcdm(
            df_fronts_total,
            dir_salida=cfg.carpetas['decision_mcdm'],
            etiqueta_modo=cfg.modo_ejecucion,
            modo_transformacion_objetivos=cfg.modo_transformacion_objetivos,
            dpi=cfg.report_plot_dpi,
            emitir_log=True,
        )
    else:
        print("      • ⚠️  MCDM omitido: no hay datos de frentes de Pareto disponibles.")



def ejecutar_pipeline_report_only(args):
    """Ejecuta únicamente la fase de reportes usando CSVs preexistentes en snp_tag/input."""
    inicio_total = time.time()

    cfg = inicializar_configuracion(modo=args.mode, data_source=args.data_source)
    informar_configuracion(cfg)

    ruta_input = Path(__file__).parent / 'input'

    imprimir_subseccion("Carga de CSVs para Report-Only", icono="📥")
    df_final, df_gen, df_fronts_total, modo_detectado = _cargar_dataframes_report_only(
        ruta_input
    )

    if modo_detectado:
        cfg.modo_ejecucion = str(modo_detectado)
        print(f"      • Modo inferido desde CSV detallado: {cfg.modo_ejecucion}")

    ruta_base, carpetas = crear_arbol_directorios_dataset(cfg, cfg.origen_datos)
    cfg.carpetas = carpetas

    print(f"      • Carpeta de salida report-only: {ruta_base}")

    ejecutar_reportes_visualizacion(cfg, df_final, df_gen, df_fronts_total)

    duracion = time.time() - inicio_total
    hor, rem = divmod(duracion, 3600)
    minu, seg = divmod(rem, 60)
    print(f"\n{'='*40}")
    print(f"⏱️  TIEMPO TOTAL DE EJECUCIÓN (REPORT-ONLY): {int(hor)}h {int(minu)}m {seg:.2f}s")
    print(f"{'='*40}\n")

    return ruta_base

def ejecutar_pipeline(args):
    """
    Ejecuta el pipeline Tag SNP de principio a fin con el estilo visual original.
    """
    inicio_total = time.time()
    
    # 1. Configuración
    cfg = inicializar_configuracion(modo=args.mode, data_source=args.data_source)
    informar_configuracion(cfg)
    
    # Crear árbol de directorios
    ruta_base, carpetas = crear_arbol_directorios_dataset(cfg, cfg.origen_datos)
    cfg.carpetas = carpetas
    
    # 2. Carga de Datos
    imprimir_subseccion("Metadatos y Dimensiones del Dataset", icono="📊️")
    H, snp_ids, posiciones, hap_ids = cargar_dataset_objetivo(cfg)
    
    # Obtener ruta relativa del fichero si existe
    fichero_rel = os.path.relpath(cfg.ruta_hinds2005) if cfg.origen_datos == 'hinds2005' else "N/A"
    
    print(f"      • N_SNPS={len(snp_ids)} | N_PATRONES={len(hap_ids)} | FICHERO={fichero_rel}")
    
    # 3. Diagnóstico y EDA
    imprimir_encabezado("DIAGNÓSTICO DE DATOS Y DESEQUILIBRIO (LD)")
    
    # Análisis de Similitud (EDA Visual)
    imprimir_subseccion("Visualización de la Estructura de Haplotipos", icono="🧬")
    graficar_mapa_calor_haplotipos(H, cfg.carpetas, cfg.modo_ejecucion)
    graficar_bloques_ld(H, cfg.carpetas, cfg.modo_ejecucion, cfg=cfg)
    
    imprimir_subseccion("Análisis de Variabilidad y Frecuencia Alélica", icono="📈")
    graficar_histograma_alelico(H, cfg.carpetas, cfg.modo_ejecucion)
    graficar_variabilidad_snps(H, cfg.carpetas, cfg.modo_ejecucion)
    graficar_conteo_alelos(H, cfg.carpetas, cfg.modo_ejecucion)
    
    dvals, p33, p66, top_sim, top_dist = analizar_similitud_genotipica(H)
    graficar_histograma_hamming(dvals, cfg.carpetas, cfg.modo_ejecucion)
    
    # LD y Veredicto
    media_ld, corrs, corr_full = calcular_ld_completo(H)
    segmentos = detectar_bloques_ld(H) # Ya detectados antes en graficar_bloques_ld pero calculamos de nuevo o pasamos
    
    # Generar gráficos LD y capturar rutas
    rutas_ld = graficar_ld_detallado(corr_full, corrs, cfg.carpetas, cfg.modo_ejecucion)
    
    # Ejecutar reporte de diagnóstico con enlaces integrados
    ejecutar_diagnostico_ld(H, cfg, rutas_ld=rutas_ld)
    
    # 5. Ejecución Evolutiva
    imprimir_encabezado("MOTOR MULTIOBJETIVO")
    
    n_algoritmos = len(cfg.algoritmos_activos)
    n_inits = len(cfg.opciones_init)
    n_cross = len(cfg.crossover_operadores_activos)
    n_ejec_total = n_algoritmos * n_inits * n_cross * cfg.n_ejecuciones
    
    imprimir_subseccion("Planificación de Ejecución", icono="📅")
    print(
        f"      • Total: {n_algoritmos} alg. x {n_inits} inits. x {n_cross} cruces x {cfg.n_ejecuciones} runs = {n_ejec_total} ejecuciones"
    )
    
    dirs_ref, n_part = construir_direcciones_referencia(cfg.tam_poblacion)
    print(f"      • Referencias: {len(dirs_ref)} puntos | Particiones: {n_part}")
    
    pares_idx = np.array(list(combinations(range(cfg.n_haplotipos), 2)), dtype=np.int32)
    resultados = ejecutar_suite_completa(H, pares_idx, cfg)
    
    # 6. Síntesis y Métricas
    imprimir_encabezado("SÍNTESIS DE RESULTADOS")
    imprimir_metadato("Configuración de síntesis", f"DPI={cfg.report_plot_dpi}", sangria=2)
    
    # Análisis de Rendimiento y Tiempo
    imprimir_subseccion("Análisis de Rendimiento y Tiempo", icono="⏱️")
    t0_perf = time.time()
    df_perf = pd.DataFrame([
        {'algorithm': rr.algoritmo, 'init': rr.inicializacion, 'crossover': rr.crossover, 'time_seg': rr.tiempo_seg, 'run': rr.replica, 'frente_size': len(rr.F_final) if rr.F_final is not None else 0}
        for rr in resultados
    ])
    if not df_perf.empty:
        graficar_rendimiento_tiempo(df_perf, cfg.carpetas['tiempo'], cfg.modo_ejecucion)
    print(f"      • Tiempo bloque 'Análisis de Rendimiento y Tiempo': {time.time() - t0_perf:.1f}s")

    
    # Procesamiento de Métricas Progresivo
    imprimir_subseccion("Procesamiento de Métricas (con progreso)", icono="🧮")
    print(f"    • Iniciando métricas finales: {len(resultados)} ejecuciones")
    t0_fin = time.time()
    df_final, ideal_g, nadir_g = evaluar_metricas_finales(
        resultados, n_snps_total=cfg.n_snps, 
        modo_normalizacion=cfg.modo_normalizacion,
        hamming_pares=dvals,
        modo_transformacion_objetivos=cfg.modo_transformacion_objetivos,
    )
    print(f"      • Métricas finales completadas: {len(resultados)} ejecuciones en {time.time() - t0_fin:.1f}s")
    
    print(f"\n    • Iniciando métricas generacionales: {len(resultados)} ejecuciones con historial")
    t0_gen = time.time()
    df_gen = construir_metricas_generacionales(
        resultados, ideal_g, nadir_g, n_snps_total=cfg.n_snps,
        modo_normalizacion=cfg.modo_normalizacion,
        hamming_pares=dvals,
        modo_transformacion_objetivos=cfg.modo_transformacion_objetivos,
    )
    print(f"      • Métricas generacionales completadas: {len(resultados)} ejecuciones en {time.time() - t0_gen:.1f}s")

    df_fronts_total = _construir_df_fronts_desde_resultados(
        resultados,
        modo_transformacion_objetivos=cfg.modo_transformacion_objetivos,
    )
    
    imprimir_subseccion("Trazabilidad y Exportación de Datos (CSV)", icono="📊️")
    ruta_csv = os.path.join(cfg.carpetas['ejecuciones'], f"resultados_detallados_{cfg.modo_ejecucion}.csv")
    df_final.to_csv(ruta_csv, index=False)
    imprimir_grafico_guardado(ruta_csv, "Resultados detallados por ejecución")
    
    ruta_hist = os.path.join(cfg.carpetas['ejecuciones'], f"historico_generacional_{cfg.modo_ejecucion}.csv")
    df_gen.to_csv(ruta_hist, index=False)
    imprimir_grafico_guardado(ruta_hist, "Historial evolutivo generacional")

    ruta_fronts_csv = os.path.join(cfg.carpetas['ejecuciones'], f"frentes_pareto_{cfg.modo_ejecucion}.csv")
    df_fronts_total.to_csv(ruta_fronts_csv, index=False)
    imprimir_grafico_guardado(ruta_fronts_csv, "Soluciones de frentes finales (CSV)")
    
    imprimir_subseccion("Puntos Críticos del Espacio de Objetivos", icono="📍")
    imprimir_metadato("Punto Ideal Empírico (mejor)", str(ideal_g))
    imprimir_metadato("Punto Nadir Empírico (peor)", str(nadir_g))
    
    if cfg.modo_normalizacion in ('static_dataset_limits', 'static_proportional_limits'):
        ideal_teorico, denom_teorico = obtener_referencias_estaticas_dataset(
            n_snps_total=cfg.n_snps,
            hamming_pares=dvals,
            modo_transformacion_objetivos=cfg.modo_transformacion_objetivos,
            modo_normalizacion=cfg.modo_normalizacion,
        )
        nadir_teorico = denom_teorico + ideal_teorico - 1e-9
        ref_hv_teorico = ideal_teorico + 1.1 * denom_teorico
        
        imprimir_metadato("Punto Ideal Teórico (Ref)", str(ideal_teorico))
        imprimir_metadato("Punto Nadir Teórico (Ref)", str(nadir_teorico))
        imprimir_metadato("Punto Ref. Hipervolumen (1.1)", str(ref_hv_teorico))
    
    # 7. Reportes Visuales
    ejecutar_reportes_visualizacion(cfg, df_final, df_gen, df_fronts_total)
    
    duracion = time.time() - inicio_total
    hor, rem = divmod(duracion, 3600)
    minu, seg = divmod(rem, 60)
    print(f"\n{'='*40}")
    print(f"⏱️  TIEMPO TOTAL DE EJECUCIÓN: {int(hor)}h {int(minu)}m {seg:.2f}s")
    print(f"{'='*40}\n")
    
    return ruta_base

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tag SNP Modular Pipeline', add_help=False)
    parser.add_argument('-h', '--help', action='help', help='Mostrar este mensaje de ayuda y salir')
    parser.add_argument('--mode', '-m', choices=list(MODOS_DISPONIBLES), default='medium')
    parser.add_argument('--data-source', '-d', choices=list(FUENTES_DATOS_DISPONIBLES), default='hinds2005')
    parser.add_argument(
        '--report-only-csv',
        action='store_true',
        help=(
            "Activa el modo de reporte exclusivo. El sistema selecciona automáticamente los CSV "
            "más recientes en el directorio snp_tag/input/ para generar las visualizaciones."
        ),
    )
    args = parser.parse_args()
    
    # Configurar Tee para log
    temp_fd, t_log = tempfile.mkstemp(suffix='.log')
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    ruta_base = None
    
    try:
        # Usamos un bloque with para asegurar que el fichero temporal se cierre (y flushee) antes de copiarlo
        with open(t_log, 'w') as f:
            sys.stdout = Tee(orig_stdout, f)
            sys.stderr = Tee(orig_stderr, f)
            
            if args.report_only_csv:
                ruta_base = ejecutar_pipeline_report_only(args)
            else:
                ruta_base = ejecutar_pipeline(args)
            
            # Asegurar volcado total antes de salir del block with
            sys.stdout.flush()
            sys.stderr.flush()
        
        # Una vez cerrado el bloque with, el fichero está en disco completo. Lo movemos a su destino final.
        if ruta_base and os.path.exists(ruta_base):
            shutil.copyfile(t_log, os.path.join(ruta_base, 'terminal.log'))
            
    except Exception as e:
        # Si algo falla antes de devolver ruta_base, imprimimos el error en el stdout original
        orig_stdout.write(f"\n❌  Error fatal durante la ejecución: {e}\n")
        raise
    finally:
        # Restauración incondicional de los flujos originales
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
        
        # Cierre del descriptor y eliminación del temporal
        try:
            os.close(temp_fd)
        except OSError:
            pass
            
        if os.path.exists(t_log):
            os.remove(t_log)

