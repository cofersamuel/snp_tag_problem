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
    imprimir_metadato
)
from snp_tag.utils.filesystem import crear_arbol_directorios_dataset
from snp_tag.utils.runtime import calcular_max_workers_paralelo
from snp_tag.data.loader import cargar_dataset_objetivo, exportar_dataset
from snp_tag.data.diagnostics import ejecutar_diagnostico_ld, analizar_similitud_genotipica, calcular_ld_completo, detectar_bloques_ld
from snp_tag.core.algorithm import construir_direcciones_referencia
from snp_tag.engine.runner import ejecutar_suite_completa
from snp_tag.engine.metrics import evaluar_metricas_finales, construir_metricas_generacionales
from snp_tag.visualization.dataset import (
    graficar_mapa_calor_haplotipos, graficar_bloques_ld, 
    graficar_histograma_hamming, graficar_variabilidad_snps,
    graficar_conteo_alelos, graficar_histograma_alelico,
    graficar_ld_detallado
)
from snp_tag.visualization.fronts import (
    graficar_frentes_pareto, 
    graficar_correlacion_objetivos_pareto,
    graficar_coordenadas_paralelas_pareto
)
from snp_tag.visualization.convergence import graficar_evolucion_generacional
from snp_tag.visualization.reporting import (
    graficar_boxplot_metricas, graficar_ranking_global,
    graficar_rendimiento_tiempo, graficar_comparativa_objetivos,
    graficar_violin_metricas, graficar_media_std_metricas
)


def _tarea_plot_frentes(tipo_tarea, df_fronts_total, carpetas, etiqueta_modo, dpi, semilla):
    """Dispatcher picklable para paralelizar tareas de graficado de frentes."""
    salida = {'tarea': tipo_tarea, 'artefactos': []}
    if tipo_tarea == 'correlation':
        salida['artefactos'] = graficar_correlacion_objetivos_pareto(
            df_fronts_total, carpetas, etiqueta_modo, dpi=dpi, emitir_log=False
        )
    elif tipo_tarea == 'parallel':
        salida['artefactos'] = graficar_coordenadas_paralelas_pareto(
            df_fronts_total, semilla, carpetas, etiqueta_modo, dpi=dpi, emitir_log=False
        )
    elif tipo_tarea == 'all_fronts':
        artefactos = []
        for algo in ['NSGA3', 'MOEAD', 'NSGA2', 'SPEA2']:
            df_algo = df_fronts_total[df_fronts_total['algorithm'] == algo]
            if not df_algo.empty:
                artefactos.extend(
                    graficar_frentes_pareto(
                        df_algo,
                        algo,
                        carpetas=carpetas,
                        etiqueta_modo=etiqueta_modo,
                        dpi=dpi,
                        emitir_log=False,
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
    """Muestra un resumen de la configuración en la terminal."""
    imprimir_subseccion("CONFIGURACIÓN", icono="⚙")
    print(f"      • Modo={cfg.modo_ejecucion} | POP_SIZE={cfg.tam_poblacion} | N_GEN={cfg.n_generaciones} | "
          f"OFFSPRING={cfg.n_descendencia} | PC={cfg.pc} | PM={cfg.pm:.6f} | N_RUNS={cfg.n_ejecuciones}")


def _construir_df_fronts_desde_resultados(resultados):
    """Construye un DataFrame tabular de soluciones de frentes finales para CSV/plots."""
    columnas = [
        'algorithm', 'init', 'run', 'seed',
        'f1_compactness', 'f2_neg_tolerance', 'f3_neg_hamming_avg', 'f4_balance_var'
    ]
    filas = []
    for rr in resultados:
        if rr.F_final is None or len(rr.F_final) == 0:
            continue
        for f in rr.F_final:
            filas.append({
                'algorithm': rr.algoritmo,
                'init': rr.inicializacion,
                'run': rr.replica,
                'seed': rr.semilla,
                'f1_compactness': f[0],
                'f2_neg_tolerance': f[1],
                'f3_neg_hamming_avg': f[2],
                'f4_balance_var': f[3],
            })
    if not filas:
        return pd.DataFrame(columns=columnas)
    return pd.DataFrame(filas, columns=columnas)


def _inferir_modo_desde_nombre_csv(ruta_csv: Path, prefijo: str):
    """Extrae el modo desde un nombre tipo '<prefijo><modo>.csv'."""
    nombre = ruta_csv.name
    if not (nombre.startswith(prefijo) and nombre.endswith('.csv')):
        return None
    return nombre[len(prefijo):-4]


def _seleccionar_mas_reciente(ruta_input: Path, patron: str):
    """Devuelve el CSV más reciente por fecha de modificación."""
    candidatos = [p for p in ruta_input.glob(patron) if p.is_file()]
    if not candidatos:
        return None
    return max(candidatos, key=lambda p: p.stat().st_mtime)


def _cargar_dataframes_report_only(ruta_input: Path, csv_detallado_arg: str):
    """Carga CSVs desde snp_tag/input para ejecutar sólo reportes/visualización."""
    if not ruta_input.exists() or not ruta_input.is_dir():
        raise FileNotFoundError(f"No existe el directorio de entrada fijo: {ruta_input}")

    if csv_detallado_arg:
        ruta_detallado = Path(csv_detallado_arg)
        if not ruta_detallado.is_absolute():
            ruta_detallado = ruta_input / ruta_detallado
        ruta_detallado = ruta_detallado.resolve()
        if not ruta_detallado.exists() or not ruta_detallado.is_file():
            raise FileNotFoundError(f"CSV detallado no encontrado: {ruta_detallado}")
    else:
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
        print("      • ⚠️ No se encontró CSV histórico; se omitirá el bloque de convergencia.")
        df_gen = pd.DataFrame()

    ruta_fronts = None
    if modo_detectado:
        candidato_fronts = ruta_input / f"frentes_pareto_{modo_detectado}.csv"
        if candidato_fronts.exists() and candidato_fronts.is_file():
            ruta_fronts = candidato_fronts
    if ruta_fronts is None:
        ruta_fronts = _seleccionar_mas_reciente(ruta_input, 'frentes_pareto_*.csv')

    if ruta_fronts is None:
        print("      • ⚠️ No hay CSV de frentes; se omitirá Pareto con aviso explícito.")
        df_fronts = pd.DataFrame()
    else:
        print(f"      • CSV frentes seleccionado: {ruta_fronts}")
        df_fronts = pd.read_csv(ruta_fronts)

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
        'algorithm', 'init', 'f1_compactness', 'f2_neg_tolerance',
        'f3_neg_hamming_avg', 'f4_balance_var'
    }

    if df_fronts_total.empty:
        print("      • ⚠️ Pareto omitido: no hay datos de frentes disponibles (CSV frentes ausente/vacío).")
    elif not cols_fronts.issubset(set(df_fronts_total.columns)):
        faltantes = sorted(cols_fronts - set(df_fronts_total.columns))
        print(
            "      • ⚠️ Pareto omitido: el CSV de frentes no contiene columnas requeridas: "
            f"{faltantes}"
        )
    else:
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
                )
                for t in tareas_plot
            ]
            resultados_tareas = {}
            for future in concurrent.futures.as_completed(futures):
                try:
                    resultado = future.result()
                    resultados_tareas[resultado.get('tarea')] = resultado.get('artefactos', [])
                except Exception as e:
                    print(f"      • ⚠️ Error en graficado paralelo: {e}")

        print("\n    🔗 Correlación de Objetivos")
        for ruta, descripcion in resultados_tareas.get('correlation', []):
            imprimir_grafico_guardado(ruta, descripcion)

        print("\n    📈 Coordenadas Paralelas")
        for ruta, descripcion in resultados_tareas.get('parallel', []):
            imprimir_grafico_guardado(ruta, descripcion)

        print("\n    🎯 Frentes de Pareto")
        for ruta, descripcion in resultados_tareas.get('all_fronts', []):
            imprimir_grafico_guardado(ruta, descripcion)

    print(f"      • Tiempo bloque 'Frentes de Pareto': {time.time() - t0_frentes:.1f}s")

    imprimir_subseccion("Análisis de Convergencia Progresiva", icono="🔄")
    if not df_gen.empty and {'algorithm', 'init', 'generation'}.issubset(df_gen.columns):
        artefactos_conv = graficar_evolucion_generacional(
            df_gen,
            dir_salida=cfg.carpetas['metricas_convergencia'],
            etiqueta_modo=cfg.modo_ejecucion,
            dpi=cfg.report_plot_dpi,
            emitir_log=False,
        )
        for ruta, descripcion in sorted(artefactos_conv, key=lambda x: (x[1], x[0])):
            imprimir_grafico_guardado(ruta, descripcion)
    else:
        print("      • ⚠️ Convergencia omitida: no hay histórico generacional válido.")

    imprimir_subseccion("Síntesis Estadística Comparativa", icono="📊")
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
                    print(f"      • ⚠️ Error en síntesis estadística paralela: {e}")

        print("\n    📦 \033[1mResumen Global (Boxplots)\033[0m")
        for ruta, descripcion in resultados_est.get('boxplots', []):
            imprimir_grafico_guardado(ruta, descripcion)

        print("\n    🎻 \033[1mDistribuciones Detalladas (Violin Plots)\033[0m")
        for ruta, descripcion in resultados_est.get('violin', []):
            imprimir_grafico_guardado(ruta, descripcion)

        print("\n    📉 \033[1mAnálisis de Tendencia Central (Media ± Std)\033[0m")
        for ruta, descripcion in resultados_est.get('mean_std', []):
            imprimir_grafico_guardado(ruta, descripcion)

    imprimir_subseccion("Resumen Estadístico de Métricas", icono="📊")
    if not df_final.empty:
        graficar_comparativa_objetivos(df_final, cfg.carpetas['rankings'], cfg.modo_ejecucion)

    imprimir_subseccion("Ranking Global por Suma de Posiciones", icono="🏆")
    if not df_final.empty:
        graficar_ranking_global(df_final, cfg.carpetas['rankings'], cfg.modo_ejecucion)


def ejecutar_pipeline_report_only(args):
    """Ejecuta únicamente la fase de reportes usando CSVs preexistentes en snp_tag/input."""
    inicio_total = time.time()

    cfg = inicializar_configuracion(modo=args.mode, data_source=args.data_source)
    informar_configuracion(cfg)

    ruta_input = Path(__file__).parent / 'input'

    imprimir_subseccion("Carga de CSVs para Report-Only", icono="📥")
    df_final, df_gen, df_fronts_total, modo_detectado = _cargar_dataframes_report_only(
        ruta_input,
        args.report_only_csv,
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
    
    print(f"      • ORIGEN_DATOS={cfg.origen_datos} ({'Hinds et al. 2005 / Perlegen' if cfg.origen_datos == 'hinds2005' else 'Simulación'}) | "
          f"N_SNPS={len(snp_ids)} | N_PATRONES={len(hap_ids)} | PM={cfg.pm:.6f}")
    print(f"      • FICHERO={fichero_rel} | GREEDY_MAX_COVERAGE={cfg.cobertura_max_greedy}")
    print(f"      • REPORT_DPI={cfg.report_plot_dpi} | NORMALIZATION_MODE={cfg.modo_normalizacion}")
    
    # 3. Diagnóstico y EDA
    imprimir_encabezado("DIAGNÓSTICO DE DATOS Y DESEQUILIBRIO (LD)")
    
    # Análisis de Similitud (EDA Visual)
    imprimir_subseccion("Visualización de la Estructura de Haplotipos", icono="🧬")
    graficar_mapa_calor_haplotipos(H, cfg.carpetas, cfg.modo_ejecucion)
    graficar_bloques_ld(H, cfg.carpetas, cfg.modo_ejecucion)
    
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
    print(f"  ⚙️ \033[1mConfiguración del Motor Evolutivo\033[0m")
    print("  " + "─" * 37)
    
    # Línea de resumen de parámetros ( must be exact)
    print(f"      • Modo={cfg.modo_ejecucion} | POP_SIZE={cfg.tam_poblacion} | N_GEN={cfg.n_generaciones} | "
          f"OFFSPRING={cfg.n_descendencia} | PC={cfg.pc} | PM={cfg.pm:.6f} | N_RUNS={cfg.n_ejecuciones}")
    
    n_ejec_total = 4 * len(cfg.opciones_init) * cfg.n_ejecuciones
    print(f"      • Desglose: 4 algoritmos x {len(cfg.opciones_init)} inicializaciones x {cfg.n_ejecuciones} runs = {n_ejec_total} ejecuciones")
    print(f"      • Configuraciones únicas (algoritmo-init): {4 * len(cfg.opciones_init)}")
    
    dirs_ref, n_part = construir_direcciones_referencia(cfg.tam_poblacion)
    print(f"      • Puntos de referencia (ref_dirs): {len(dirs_ref)} | Particiones: {n_part}")
    print(f"      • Tamaño de población (pop_size): {cfg.tam_poblacion}")
    
    pares_idx = np.array(list(combinations(range(cfg.n_haplotipos), 2)), dtype=np.int32)
    resultados = ejecutar_suite_completa(H, pares_idx, cfg)
    
    # 6. Síntesis y Métricas
    imprimir_encabezado("SÍNTESIS DE RESULTADOS")
    imprimir_metadato("Configuración de síntesis", f"DPI={cfg.report_plot_dpi}", sangria=2)
    
    # Análisis de Rendimiento y Tiempo
    imprimir_subseccion("Análisis de Rendimiento y Tiempo", icono="⏱️")
    t0_perf = time.time()
    df_perf = pd.DataFrame([
        {'algorithm': rr.algoritmo, 'init': rr.inicializacion, 'time_seg': rr.tiempo_seg, 'run': rr.replica, 'frente_size': len(rr.F_final) if rr.F_final is not None else 0}
        for rr in resultados
    ])
    if not df_perf.empty:
        graficar_rendimiento_tiempo(df_perf, cfg.carpetas['tiempo'], cfg.modo_ejecucion)
    print(f"      • Tiempo bloque 'Análisis de Rendimiento y Tiempo': {time.time() - t0_perf:.1f}s")

    
    # Procesamiento de Métricas Progresivo
    imprimir_subseccion("Procesamiento de Métricas (con progreso)", icono="🧮")
    print(f"    • Iniciando métricas finales: {len(resultados)} ejecuciones")
    
    df_final, ideal_g, nadir_g = evaluar_metricas_finales(
        resultados, n_snps_total=cfg.n_snps, 
        modo_normalizacion=cfg.modo_normalizacion,
        hamming_pares=dvals
    )
    
    print(f"\n    • Iniciando métricas generacionales: {len(resultados)} ejecuciones con historial")
    t0_gen = time.time()
    df_gen = construir_metricas_generacionales(
        resultados, ideal_g, nadir_g, n_snps_total=cfg.n_snps,
        modo_normalizacion=cfg.modo_normalizacion,
        hamming_pares=dvals
    )
    print(f"      • Métricas generacionales completadas: {len(resultados)} ejecuciones en {time.time() - t0_gen:.1f}s")

    df_fronts_total = _construir_df_fronts_desde_resultados(resultados)
    
    imprimir_subseccion("Trazabilidad y Exportación de Datos (CSV)", icono="📊")
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
    imprimir_metadato("Punto Ideal Global (mejor)", str(ideal_g))
    imprimir_metadato("Punto Nadir Global (peor)", str(nadir_g))
    
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
    parser = argparse.ArgumentParser(description='Tag SNP Modular Pipeline')
    parser.add_argument('--mode', '-m', choices=list(MODOS_DISPONIBLES), default='medium')
    parser.add_argument('--data-source', '-d', choices=list(FUENTES_DATOS_DISPONIBLES), default='hinds2005')
    parser.add_argument(
        '--report-only-csv',
        nargs='?',
        const='',
        default=None,
        metavar='CSV_DETALLADO',
        help=(
            "Activa modo report-only. Si se indica CSV_DETALLADO, se busca en snp_tag/input "
            "(o ruta absoluta). Si no se indica, se usa el más reciente en snp_tag/input."
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
            
            if args.report_only_csv is not None:
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
        orig_stdout.write(f"\n❌ Error fatal durante la ejecución: {e}\n")
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

