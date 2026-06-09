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
from snp_tag.config import ConfiguracionExperimento, construir_configuracion, MODOS_DISPONIBLES, FUENTES_DATOS_DISPONIBLES, informar_configuracion
from snp_tag.utils.logger import logger, add_file_handler
from snp_tag.utils.terminal import (
    imprimir_encabezado, imprimir_subseccion, imprimir_paso, 
    imprimir_estado, imprimir_grafico_guardado, obtener_bit_string_estilizado,
    imprimir_metadato, obtener_enlace_terminal
)
from snp_tag.utils.filesystem import crear_arbol_directorios_dataset
from snp_tag.utils.runtime import calcular_max_workers_paralelo
from snp_tag.data.loader import cargar_dataset_objetivo, exportar_dataset
from snp_tag.engine.diagnostics_logic import ejecutar_diagnostico_ld, analizar_similitud_genotipica, calcular_ld_completo, detectar_bloques_ld
from snp_tag.core.algorithm import construir_direcciones_referencia
from snp_tag.pipelines.evolution_pipeline import ejecutar_suite_completa
from snp_tag.engine.metrics_logic import (
    evaluar_metricas_finales,
    construir_metricas_generacionales,
    decodificar_objetivos_reales,
    filtrar_soluciones_factibles,
    obtener_referencias_estaticas_dataset,
)
from snp_tag.pipelines.diagnostics_pipeline import ejecutar_pipeline_diagnostico
from snp_tag.pipelines.reporting_pipeline import ejecutar_reportes_visualizacion
from snp_tag.visualization.stats_plot import graficar_rendimiento_tiempo


from typing import Optional, Dict, Any, List, Tuple

def inicializar_configuracion(modo: str = 'medium', data_source: str = 'hinds2005', overrides: Optional[Dict[str, Any]] = None) -> ConfiguracionExperimento:
    """
    Configura los parámetros iniciales del experimento mediante inyección de dependencias.
    """
    return construir_configuracion(modo=modo, data_source=data_source, overrides=overrides)




def _construir_df_fronts_desde_resultados(resultados: List[Any], modo_transformacion_objetivos: str = 'neg') -> pd.DataFrame:
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


def _inferir_modo_desde_nombre_csv(ruta_csv: Path, prefijo: str) -> Optional[str]:
    """Extrae el modo desde un nombre tipo '<prefijo><modo>.csv'."""
    nombre = ruta_csv.name
    if not (nombre.startswith(prefijo) and nombre.endswith('.csv')):
        return None
    return nombre[len(prefijo):-4]


def _normalizar_columnas_frentes_csv(df_fronts: pd.DataFrame) -> Tuple[pd.DataFrame, List[Tuple[str, str]]]:
        """
        Normaliza nombres de columnas de frentes para compatibilidad en report-only.

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


def _seleccionar_mas_reciente(ruta_input: Path, patron: str) -> Optional[Path]:
    """Devuelve el CSV más reciente por fecha de modificación."""
    candidatos = [p for p in ruta_input.glob(patron) if p.is_file()]
    if not candidatos:
        return None
    return max(candidatos, key=lambda p: p.stat().st_mtime)


def _cargar_dataframes_report_only_csv(ruta_input: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[str]]:
    """Carga CSVs desde snp_tag/input para ejecutar sólo reportes/visualización (modo report-only-csv)."""
    if not ruta_input.exists() or not ruta_input.is_dir():
        raise FileNotFoundError(f"No existe el directorio de entrada fijo: {ruta_input}")

    ruta_detallado = _seleccionar_mas_reciente(ruta_input, 'resultados_detallados_*.csv')
    if ruta_detallado is None:
        raise FileNotFoundError(
            f"No se encontró ningún 'resultados_detallados_*.csv' en {ruta_input}"
        )

    logger.info(f"      • CSV detallado seleccionado: {ruta_detallado}")
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
        logger.info(f"      • CSV histórico seleccionado: {ruta_hist}")
        df_gen = pd.read_csv(ruta_hist)
    else:
        logger.warning("      • ⚠️  No se encontró CSV histórico; se omitirá el bloque de convergencia.")
        df_gen = pd.DataFrame()

    ruta_fronts = None
    if modo_detectado:
        candidato_fronts = ruta_input / f"frentes_pareto_{modo_detectado}.csv"
        if candidato_fronts.exists() and candidato_fronts.is_file():
            ruta_fronts = candidato_fronts
    if ruta_fronts is None:
        ruta_fronts = _seleccionar_mas_reciente(ruta_input, 'frentes_pareto_*.csv')

    if ruta_fronts is None:
        logger.warning("      • ⚠️  No hay CSV de frentes; se omitirá Pareto con aviso explícito.")
        df_fronts = pd.DataFrame()
    else:
        logger.info(f"      • CSV frentes seleccionado: {ruta_fronts}")
        df_fronts = pd.read_csv(ruta_fronts)
        df_fronts, columnas_renombradas = _normalizar_columnas_frentes_csv(df_fronts)
        if columnas_renombradas:
            detalle = ', '.join([f"{src}→{dst}" for src, dst in columnas_renombradas])
            logger.info(f"      • Compatibilidad report-only aplicada en frentes: {detalle}")

    return df_final, df_gen, df_fronts, modo_detectado




def ejecutar_pipeline_report_only_csv(args: Any) -> str:
    """Ejecuta únicamente la fase de reportes usando CSVs preexistentes en snp_tag/input (modo report-only-csv)."""
    inicio_total = time.time()

    ruta_input = Path(__file__).parent / 'input'

    imprimir_subseccion("Carga de CSVs para Report-Only", icono="📥")
    df_final, df_gen, df_fronts_total, modo_detectado = _cargar_dataframes_report_only_csv(
        ruta_input
    )

    overrides = None
    candidato_ini = None
    if modo_detectado:
        candidato_ini_modo = ruta_input / f"user_config_{modo_detectado}.ini"
        if candidato_ini_modo.exists():
            candidato_ini = candidato_ini_modo
    
    if candidato_ini is None:
        candidatos = list(ruta_input.glob("user_config*.ini"))
        if candidatos:
            candidato_ini = max(candidatos, key=lambda p: p.stat().st_mtime)

    if candidato_ini:
        logger.info(f"      • Configuración cargada desde input: {candidato_ini.name}")
        from snp_tag.config import cargar_params_tunables_desde_ini
        try:
            overrides = cargar_params_tunables_desde_ini(candidato_ini)
        except Exception as e:
            logger.warning(f"      • ⚠️ Error cargando config de input, usando por defecto: {e}")
    else:
        logger.warning("      • ⚠️ No se encontró user_config en input, usando por defecto.")

    cfg = inicializar_configuracion(modo=args.mode, data_source=args.data_source, overrides=overrides)

    # Inyectar el modo de evaluación correcto en metrics.py ya que se evaluó estáticamente al importar
    import snp_tag.engine.metrics
    from snp_tag.config import resolver_modo_evaluacion
    snp_tag.engine.metrics._MODO_EVALUACION_GLOBAL = resolver_modo_evaluacion(cfg.modo_evaluacion)

    informar_configuracion(cfg)

    ruta_base, carpetas = crear_arbol_directorios_dataset(cfg, cfg.origen_datos, is_report_only_csv=True)
    cfg.carpetas = carpetas
    
    add_file_handler(logger, os.path.join(ruta_base, "ejecucion.log"))
    
    logger.info(f"      • Carpeta de salida report-only: {ruta_base}")

    ejecutar_reportes_visualizacion(cfg, df_final, df_gen, df_fronts_total)

    duracion = time.time() - inicio_total
    hor, rem = divmod(duracion, 3600)
    minu, seg = divmod(rem, 60)
    logger.info(f"\n{'='*40}")
    logger.info(f"⏱️  TIEMPO TOTAL DE EJECUCIÓN (REPORT-ONLY): {int(hor)}h {int(minu)}m {seg:.2f}s")
    logger.info(f"{'='*40}\n")

    return ruta_base

def ejecutar_pipeline(args: Any) -> str:
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
    
    add_file_handler(logger, os.path.join(ruta_base, "ejecucion.log"))
    
    # 2. Carga de Datos & 3. Diagnóstico y EDA
    H, snp_ids, posiciones, hap_ids, dvals = ejecutar_pipeline_diagnostico(cfg)
    
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
    print(
        f"      • Modo de ejecución: {cfg.modo_ejecucion.upper()} "
        f"(Población: {cfg.tam_poblacion}, Generaciones: {cfg.n_generaciones}, "
        f"Evaluación: {cfg.modo_evaluacion}, Transformación: {cfg.modo_transformacion_objetivos})"
    )
    
    dirs_ref, n_part = construir_direcciones_referencia(cfg.tam_poblacion)
    logger.info(f"      • Referencias (Das and Dennis): {len(dirs_ref)} puntos | Particiones: {n_part}")
    
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
        if 'tiempo' in cfg.graficas_activas:
            graficar_rendimiento_tiempo(df_perf, cfg.carpetas['tiempo'], cfg.modo_ejecucion)
        else:
            logger.info("      • ⚠️  Gráfica de tiempo omitida (user_config.ini).")
    logger.info(f"      • Tiempo bloque 'Análisis de Rendimiento y Tiempo': {time.time() - t0_perf:.1f}s")

    
    # Procesamiento de Métricas Progresivo
    imprimir_subseccion("Procesamiento de Métricas (con progreso)", icono="🧮")
    logger.info(f"    • Iniciando métricas finales: {len(resultados)} ejecuciones")
    t0_fin = time.time()
    df_final, ideal_g, nadir_g = evaluar_metricas_finales(
        resultados, n_snps_total=cfg.n_snps, 
        modo_normalizacion=cfg.modo_normalizacion,
        hamming_pares=dvals,
        modo_transformacion_objetivos=cfg.modo_transformacion_objetivos,
    )
    logger.info(f"      • Métricas finales completadas: {len(resultados)} ejecuciones en {time.time() - t0_fin:.1f}s")
    
    logger.info(f"\n    • Iniciando métricas generacionales: {len(resultados)} ejecuciones con historial")
    t0_gen = time.time()
    
    if getattr(cfg, 'paso_generacional_metricas', 10) == 0:
        logger.warning("      • ⚠️ Cálculo de métricas generacionales deshabilitado (paso = 0).")
        df_gen = pd.DataFrame()
    else:
        df_gen = construir_metricas_generacionales(
            resultados, ideal_g, nadir_g, n_snps_total=cfg.n_snps,
            modo_normalizacion=cfg.modo_normalizacion,
            hamming_pares=dvals,
            modo_transformacion_objetivos=cfg.modo_transformacion_objetivos,
            paso_generacional_metricas=cfg.paso_generacional_metricas,
        )
    logger.info(f"      • Métricas generacionales completadas: {len(resultados)} ejecuciones en {time.time() - t0_gen:.1f}s")

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
    
    ruta_config_exp = os.path.join(cfg.carpetas['ejecuciones'], f"user_config_{cfg.modo_ejecucion}.ini")
    from snp_tag.config import RUTA_USER_CONFIG
    if RUTA_USER_CONFIG.exists():
        shutil.copyfile(RUTA_USER_CONFIG, ruta_config_exp)
        imprimir_grafico_guardado(ruta_config_exp, "Configuración del experimento (INI)")
    
    imprimir_subseccion("Puntos Críticos del Espacio de Objetivos", icono="📍")
    
    def format_arr(arr):
        return "[" + ", ".join([f"{x:12.4f}" for x in arr]) + "]"
        
    imprimir_metadato("Punto Ideal Empírico (mejor)", format_arr(ideal_g))
    imprimir_metadato("Punto Nadir Empírico (peor)", format_arr(nadir_g))
    
    if cfg.modo_normalizacion in ('static_dataset_limits', 'static_proportional_limits'):
        ideal_teorico, denom_teorico = obtener_referencias_estaticas_dataset(
            n_snps_total=cfg.n_snps,
            hamming_pares=dvals,
            modo_transformacion_objetivos=cfg.modo_transformacion_objetivos,
            modo_normalizacion=cfg.modo_normalizacion,
        )
        nadir_teorico = denom_teorico + ideal_teorico - 1e-9
        ref_hv_teorico = ideal_teorico + 1.1 * denom_teorico
        
        imprimir_metadato("Punto Ideal Teórico (Ref)", format_arr(ideal_teorico))
        imprimir_metadato("Punto Nadir Teórico (Ref)", format_arr(nadir_teorico))
        imprimir_metadato("Punto Ref. Hipervolumen (1.1)", format_arr(ref_hv_teorico))
    
    # 7. Reportes Visuales
    ejecutar_reportes_visualizacion(cfg, df_final, df_gen, df_fronts_total)
    
    duracion = time.time() - inicio_total
    hor, rem = divmod(duracion, 3600)
    minu, seg = divmod(rem, 60)
    logger.info(f"\n{'='*40}")
    logger.info(f"⏱️  TIEMPO TOTAL DE EJECUCIÓN: {int(hor)}h {int(minu)}m {seg:.2f}s")
    logger.info(f"{'='*40}\n")
    
    return ruta_base



