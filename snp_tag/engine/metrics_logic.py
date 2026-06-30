"""
Módulo de Cálculo de Métricas (metrics.py)
-----------------------------------------
Implementa las métricas de rendimiento para la evaluación de frentes de Pareto,
incluyendo Range, SumMin, MinSum e Hipervolumen, así como indicadores crudos.
"""

# =============================================================================
# LIBRERÍAS ESTÁNDAR
# =============================================================================
import concurrent.futures
import os
import time
from collections import defaultdict
from functools import lru_cache
from typing import Any, Dict, List, Tuple

# =============================================================================
# LIBRERÍAS DE TERCEROS
# =============================================================================
import numpy as np
import pandas as pd
from pymoo.indicators.gd_plus import GDPlus
from pymoo.indicators.hv import HV
from pymoo.indicators.igd_plus import IGDPlus
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

# =============================================================================
# MÓDULOS LOCALES (snp_tag)
# =============================================================================
from snp_tag.config import (cargar_params_tunables_desde_ini,
                            resolver_modo_evaluacion,
                            resolver_modo_normalizacion)


def decodificar_objetivos_reales(
    F_crudo: np.ndarray,
    modo_transformacion_objetivos: str = 'neg',
    modo_evaluacion: str = 'absoluta',
    epsilon: float = 1e-9,
):
    """
    Decodifica objetivos físicos (maximización) desde el espacio transformado.

    Devuelve compacidad, tolerancia real (min_cobertura-1), cobertura mínima,
    distancia Hamming promedio y balance.
    """
    F = np.asarray(F_crudo, dtype=float)
    if F.ndim == 1:
        F = F.reshape(1, -1)

    modo = str(modo_transformacion_objetivos or 'neg').strip().lower()
    if modo == 'inverse':
        min_cobertura = 1.0 / np.maximum(F[:, 1], epsilon)
        hamming_prom_real = 1.0 / np.maximum(F[:, 2], epsilon)
    else:
        min_cobertura = -F[:, 1]
        hamming_prom_real = -F[:, 2]

    # Restaurar métricas físicas reales si se corrió en modo proporcional
    compacidad = F[:, 0]
    balance_var = F[:, 3]

    tolerancia_real = min_cobertura

    return {
        'compacidad': compacidad,
        'tolerancia_real': tolerancia_real,
        'min_cobertura': min_cobertura,
        'hamming_prom_real': hamming_prom_real,
        'balance_var': balance_var,
    }


def mascara_soluciones_factibles(
    F_crudo: np.ndarray,
    modo_transformacion_objetivos: str = 'neg',
    modo_evaluacion: str = 'absoluta',
    umbral_min_cobertura: float = 1.0,
    epsilon: float = 1e-9,
) -> np.ndarray:
    """Devuelve máscara booleana de soluciones factibles (min_cobertura >= 1)."""
    objetivos = decodificar_objetivos_reales(
        F_crudo,
        modo_transformacion_objetivos=modo_transformacion_objetivos,
        modo_evaluacion=modo_evaluacion,
        epsilon=epsilon,
    )
    return objetivos['min_cobertura'] >= (umbral_min_cobertura - epsilon)


def filtrar_soluciones_factibles(
    F_crudo: np.ndarray,
    modo_transformacion_objetivos: str = 'neg',
    modo_evaluacion: str = 'absoluta',
    umbral_min_cobertura: float = 1.0,
    epsilon: float = 1e-9,
) -> np.ndarray:
    """Filtra filas factibles del frente; conserva forma (n,4)."""
    F = np.asarray(F_crudo, dtype=float)
    if F.ndim == 1:
        F = F.reshape(1, -1)
    if F.size == 0:
        return F.reshape(0, 4)
    mask = mascara_soluciones_factibles(
        F,
        modo_transformacion_objetivos=modo_transformacion_objetivos,
        modo_evaluacion=modo_evaluacion,
        umbral_min_cobertura=umbral_min_cobertura,
        epsilon=epsilon,
    )
    return F[mask]

def calcular_puntos_ideales_nadires_globales(resultados_ejecucion, modo_transformacion_objetivos: str = 'neg', modo_evaluacion: str = 'absoluta'):
    """
    Determina los límites globales (ideal y nadir) a partir de todos los frentes.
    """
    frentes_finales = []
    for rr in resultados_ejecucion:
        if rr.F_final is None or len(rr.F_final) == 0:
            continue
        F_factibles = filtrar_soluciones_factibles(
            rr.F_final,
            modo_transformacion_objetivos=modo_transformacion_objetivos,
            modo_evaluacion=modo_evaluacion,
        )
        if len(F_factibles) > 0:
            frentes_finales.append(F_factibles)
    if not frentes_finales:
        raise ValueError("Ausencia de frentes finales para la normalización.")
    pila_frentes = np.vstack(frentes_finales)
    return pila_frentes.min(axis=0), pila_frentes.max(axis=0)

def normalización_min_max(F, ideal, nadir):
    """
    Aplica el escalado Min-Max sobre un frente de Pareto.
    """
    denominador = nadir - ideal
    seguro = np.where(denominador == 0, 1.0, denominador)
    F_norm = (F - ideal) / seguro
    degradado = (denominador == 0)
    if np.any(degradado):
        F_norm[:, degradado] = 0.0
    return F_norm

def obtener_referencias_estaticas_dataset(
    n_snps_total: int,
    hamming_pares=None,
    modo_transformacion_objetivos: str = 'neg',
    modo_normalizacion: str = 'static_dataset_limits',
    epsilon: float = 1e-9,
):
    """
    Define límites de normalización fijos basados en la naturaleza del dataset.
    """
    L = float(max(1, n_snps_total))
    if hamming_pares is not None and len(hamming_pares) > 0:
        max_min_cobertura = float(hamming_pares.min())
        max_tol = max_min_cobertura
        max_ham = float(hamming_pares.mean())
        max_var = float(np.var(hamming_pares)) if len(hamming_pares) > 1 else (L**2)/4.0
    else:
        max_min_cobertura, max_tol, max_ham, max_var = L, L, L, (L**2)/4.0

    is_prop = (modo_normalizacion == 'static_proportional_limits')
    if is_prop:
        # La tolerancia ya no es proporcional (no se divide por k), por lo que
        # se conserva su límite absoluto calculado arriba.
        max_ham = 1.0
        max_var = 0.25

    modo = str(modo_transformacion_objetivos or 'neg').strip().lower()
    if modo == 'inverse':
        # Robustez: en inverse, los peores casos factibles para f2/f3 están en 1.
        # Forzamos cotas físicas mínimas para evitar ideal > nadir cuando
        # max_min_cobertura o max_ham vienen degenerados (p.ej., 0).
        max_min_cobertura_seguro = max(1.0, float(max_min_cobertura))
        max_ham_seguro = max(1.0, float(max_ham))
        ideal = np.array([
            1.0,
            1.0 / max(epsilon, max_min_cobertura_seguro),
            1.0 / max(epsilon, max_ham_seguro),
            0.0,
        ])
        nadir_f3 = 10.0 if is_prop else 1.0
        nadir = np.array([L, 1.0, nadir_f3, max_var])
    else:
        ideal = np.array([1.0, -max_tol, -max_ham, 0.0])
        nadir = np.array([L, 0.0, 0.0, max_var])
    return ideal, (nadir - ideal + 1e-9)

def calcular_metricas_convergencia(F_norm):
    """
    Computa los indicadores Range, SumMin y MinSum sobre un frente normalizado.
    """
    min_col = F_norm.min(axis=0)
    max_col = F_norm.max(axis=0)
    range_val = float((max_col - min_col).sum())
    summin_val = float(min_col.sum())
    minsum_val = float(F_norm.sum(axis=1).min())
    return range_val, summin_val, minsum_val

def calcular_metricas_crudas(F_crudo, n_snps_total: int = 1032, modo_transformacion_objetivos: str = 'neg', modo_evaluacion: str = 'absoluta'):
    """
    Extrae indicadores físicos en escala real (independientes de la normalización).
    """
    objetivos_reales = decodificar_objetivos_reales(
        F_crudo,
        modo_transformacion_objetivos=modo_transformacion_objetivos,
        modo_evaluacion=modo_evaluacion,
    )
    compacidad = objetivos_reales['compacidad']
    tolerancia_real = objetivos_reales['tolerancia_real']
    hamming_prom_real = objetivos_reales['hamming_prom_real']
    
    if str(modo_evaluacion).strip().lower() == 'proportional':
        # Restauramos su magnitud absoluta exclusivamente para la exportación de esta métrica
        hamming_absoluto = hamming_prom_real * compacidad
    else:
        hamming_absoluto = hamming_prom_real
    
    seguro_comp = np.where(compacidad <= 0, np.nan, compacidad)
    tasa_tol = tolerancia_real / seguro_comp
    
    return {
        'MaxToleranceRate': float(np.nanmax(tasa_tol)),
        'AvgToleranceRate': float(np.nanmean(tasa_tol)),
        'AvgHammingDistance': float(np.nanmean(hamming_absoluto))
    }

# Decorador de caché que almacena hasta 8 retornos para evitar instanciar repetidamente el indicador para las mismas dimensiones
@lru_cache(maxsize=8)
# Función auxiliar que inicializa (o recupera de caché) el objeto calculador de hipervolumen según la cantidad de objetivos
def _obtener_indicador_hv(n_obj: int):
    # Genera un vector NumPy unidimensional de unos, forzado a coma flotante, y lo escala por 1.1 (punto de referencia)
    punto_ref = np.ones(int(n_obj), dtype=float) * 1.1
    # Instancia y devuelve el indicador HV de la biblioteca pymoo usando el punto de referencia dinámico recién calculado
    return HV(ref_point=punto_ref)

# Función principal que recibe un frente de Pareto ya normalizado para computar el valor numérico de su hipervolumen
def calcular_hipervolumen(F_norm):
    """
    Calcula el Hipervolumen (HV) respecto al punto de referencia [1.1, ..., 1.1].
    """
    # Verifica de forma segura si el frente proporcionado es un objeto nulo o si carece de filas (soluciones)
    if F_norm is None or len(F_norm) == 0:
        # Retorna NaN (Not a Number) porque es matemáticamente imposible calcular volumen sin puntos
        return np.nan
    
    # Extrae la cantidad de columnas (objetivos) de F_norm e invoca la función cacheada para obtener el indicador
    hv = _obtener_indicador_hv(int(F_norm.shape[1]))
    
    # Llama al objeto indicador evaluando F_norm y fuerza explícitamente el retorno a un tipo flotante nativo de Python
    return float(hv(F_norm))


def _calcular_referencias_por_algoritmo(resultados_ejecucion, modo_transformacion_objetivos: str = 'neg', modo_evaluacion: str = 'absoluta'):
    """Calcula referencias ideal/denominador por algoritmo."""
    grupos = defaultdict(list)
    for rr in resultados_ejecucion:
        grupos[str(rr.algoritmo)].append(rr)

    refs = {}
    for algoritmo, rrs in grupos.items():
        ideal_a, nadir_a = calcular_puntos_ideales_nadires_globales(
            rrs,
            modo_transformacion_objetivos=modo_transformacion_objetivos,
            modo_evaluacion=modo_evaluacion,
        )
        refs[algoritmo] = (ideal_a, nadir_a - ideal_a + 1e-9)
    return refs


def _calcular_referencias_por_modo(
    resultados_ejecucion,
    modo_normalizacion='global_all_pairs',
    n_snps_total=1032,
    hamming_pares=None,
    modo_transformacion_objetivos='neg',
    modo_evaluacion='absoluta',
):
    """Devuelve referencias para Range/SumMin/MinSum según el modo configurado."""
    modo = resolver_modo_normalizacion(modo_normalizacion)
    if modo == 'per_algorithm':
        return _calcular_referencias_por_algoritmo(
            resultados_ejecucion,
            modo_transformacion_objetivos=modo_transformacion_objetivos,
            modo_evaluacion=modo_evaluacion,
        )

    if modo in ('static_dataset_limits', 'static_proportional_limits'):
        ideal_ref, denom_ref = obtener_referencias_estaticas_dataset(
            n_snps_total,
            hamming_pares,
            modo_transformacion_objetivos=modo_transformacion_objetivos,
            modo_normalizacion=modo,
        )
        algoritmos = sorted({str(rr.algoritmo) for rr in resultados_ejecucion})
        return {alg: (ideal_ref, denom_ref) for alg in algoritmos}

    ideal_g, nadir_g = calcular_puntos_ideales_nadires_globales(
        resultados_ejecucion,
        modo_transformacion_objetivos=modo_transformacion_objetivos,
        modo_evaluacion=modo_evaluacion,
    )
    denom_g = nadir_g - ideal_g + 1e-9
    algoritmos = sorted({str(rr.algoritmo) for rr in resultados_ejecucion})
    return {alg: (ideal_g, denom_g) for alg in algoritmos}


def extraer_frente_referencia_empirico(resultados_ejecucion, modo_transformacion_objetivos='neg', modo_evaluacion='absoluta'):
    """Construye el Frente de Referencia Empírico de forma incremental (por lotes) para evitar picos masivos de RAM."""
    todos_frentes = []
    for rr in resultados_ejecucion:
        if rr.F_final is None or len(rr.F_final) == 0:
            continue
        F_factibles = filtrar_soluciones_factibles(
            rr.F_final,
            modo_transformacion_objetivos=modo_transformacion_objetivos,
            modo_evaluacion=modo_evaluacion,
        )
        if len(F_factibles) > 0:
            if len(F_factibles) > 1:
                F_factibles = np.unique(F_factibles, axis=0)
            todos_frentes.append(F_factibles)
    
    if not todos_frentes:
        return None

    nds = NonDominatedSorting()
    lote_size = 50
    F_actual = None
    
    for i in range(0, len(todos_frentes), lote_size):
        bloque = todos_frentes[i:i+lote_size]
        if F_actual is not None:
            bloque.append(F_actual)
            
        F_combinado = np.vstack(bloque)
        if len(F_combinado) > 1:
            F_combinado = np.unique(F_combinado, axis=0)
            
        fronts = nds.do(F_combinado, only_non_dominated_front=True)
        F_actual = F_combinado[fronts]
        
    return F_actual

def evaluar_metricas_finales(resultados_ejecucion, n_snps_total=1032, 
                             modo_normalizacion='static_dataset_limits',
                             hamming_pares=None,
                             modo_transformacion_objetivos='neg',
                             modo_evaluacion='absoluta'):
    """
    Evalúa todas las métricas de rendimiento para un conjunto de resultados de ejecución.
    """
    if not resultados_ejecucion:
        return pd.DataFrame(), None, None

    modo_normalizacion = resolver_modo_normalizacion(modo_normalizacion)
    ideal_g, nadir_g = calcular_puntos_ideales_nadires_globales(
        resultados_ejecucion,
        modo_transformacion_objetivos=modo_transformacion_objetivos,
        modo_evaluacion=modo_evaluacion,
    )
    denom_g = nadir_g - ideal_g + 1e-9
    refs_algo = _calcular_referencias_por_modo(
        resultados_ejecucion,
        modo_normalizacion=modo_normalizacion,
        n_snps_total=n_snps_total,
        hamming_pares=hamming_pares,
        modo_transformacion_objetivos=modo_transformacion_objetivos,
        modo_evaluacion=modo_evaluacion,
    )

    F_referencia_global = extraer_frente_referencia_empirico(
        resultados_ejecucion, 
        modo_transformacion_objetivos=modo_transformacion_objetivos,
        modo_evaluacion=modo_evaluacion,
    )
    igd_plus_metric = None
    gd_plus_metric = None
    if F_referencia_global is not None:
        F_ref_norm_global = np.clip((F_referencia_global - ideal_g) / denom_g, 0, 1)
        igd_plus_metric = IGDPlus(F_ref_norm_global)
        gd_plus_metric = GDPlus(F_ref_norm_global)

    conteos_replica = defaultdict(int)
    for rr_aux in resultados_ejecucion:
        conteos_replica[(str(rr_aux.algoritmo), str(rr_aux.inicializacion), str(rr_aux.crossover))] += 1

    filas = []
    total_ejecuciones = len(resultados_ejecucion)
    w = len(str(max(1, total_ejecuciones)))
    procesadas = 0
    for rr in resultados_ejecucion:
        t0_ind = time.time()
        F_total = rr.F_final
        if F_total is None or len(F_total) == 0:
            continue
        n_total = len(F_total)
        F_crudo = filtrar_soluciones_factibles(
            F_total,
            modo_transformacion_objetivos=modo_transformacion_objetivos,
            modo_evaluacion=modo_evaluacion,
        )
        if len(F_crudo) > 1:
            # Evita frentes con puntos repetidos (muy común en colapsos de MOEA/D)
            F_crudo = np.unique(np.asarray(F_crudo, dtype=float), axis=0)
        n_factibles = len(F_crudo)
        n_infactibles = max(0, n_total - n_factibles)
        if len(F_crudo) == 0:
            continue
        
        # Normalización para métricas de convergencia
        ideal_ref, denom_ref = refs_algo[str(rr.algoritmo)]
        F_norm_ref = np.clip((F_crudo - ideal_ref) / denom_ref, 0, 1)
        rng, smn, msn = calcular_metricas_convergencia(F_norm_ref)
        
        # Hipervolumen consistente con el modo de normalización seleccionado
        hv_ref = calcular_hipervolumen(F_norm_ref)
        
        # IGD+ / GD+ (usando el espacio global normalizado)
        F_crudo_norm_global = np.clip((F_crudo - ideal_g) / denom_g, 0, 1)
        igd_plus_val = float(igd_plus_metric(F_crudo_norm_global)) if igd_plus_metric is not None else np.nan
        gd_plus_val = float(gd_plus_metric(F_crudo_norm_global)) if gd_plus_metric is not None else np.nan
        
        # Métricas crudas
        crudas = calcular_metricas_crudas(
            F_crudo,
            n_snps_total,
            modo_transformacion_objetivos=modo_transformacion_objetivos,
            modo_evaluacion=modo_evaluacion,
        )
        
        filas.append({
            'algorithm': rr.algoritmo, 'init': rr.inicializacion, 'crossover': rr.crossover, 'run': rr.replica,
            'seed': rr.semilla,
            'elapsed_sec': rr.tiempo_seg,
            'n_solutions_final_front': n_factibles,
            'n_infeasible_final_front': n_infactibles,
            'Range': rng, 'SumMin': smn, 'MinSum': msn,
            'Hypervolume': hv_ref,
            'IGD+': igd_plus_val,
            'GD+': gd_plus_val,
            **{f"{k}": v for k, v in crudas.items()}
        })
        procesadas += 1
        duracion_ind = time.time() - t0_ind
        total_config = conteos_replica[(str(rr.algoritmo), str(rr.inicializacion), str(rr.crossover))]
        print(f"      • [Progreso: {procesadas:>{w}}/{total_ejecuciones}] | [{rr.algoritmo}-{rr.inicializacion}-{rr.crossover}] ejecución {rr.replica}/{total_config} ({duracion_ind:.1f} s)")
        
    return pd.DataFrame(filas), ideal_g, nadir_g


def _construir_filas_generacionales_por_ejecucion(
    rr,
    ideal_algo,
    denom_algo,
    ideal_global,
    denom_global,
    igd_plus_metric,
    gd_plus_metric,
    n_snps_total=1032,
    modo_transformacion_objetivos='neg',
    paso_generacional_metricas=10,
    modo_evaluacion='absoluta',
):
    """Construye filas de métricas por generación para una ejecución individual."""
    filas = []
    ruta = getattr(rr, 'ruta_checkpoint', None)
    if not ruta or not os.path.exists(ruta):
        return filas

    try:
        with np.load(ruta) as data:
            indices = data['gen_indices']
            matrices = [data[f'hist_F_{i}'] for i in range(len(indices))]
    except Exception:
        return filas

    for idx_gen, F_crudo in zip(indices, matrices):
        if F_crudo is None or len(F_crudo) == 0:
            continue

        F_crudo = np.array(F_crudo, dtype=float)
        if F_crudo.ndim == 1:
            F_crudo = F_crudo.reshape(1, -1)

        F_crudo = filtrar_soluciones_factibles(
            F_crudo,
            modo_transformacion_objetivos=modo_transformacion_objetivos,
            modo_evaluacion=modo_evaluacion,
        )
        if len(F_crudo) > 1:
            F_crudo = np.unique(np.asarray(F_crudo, dtype=float), axis=0)
        if len(F_crudo) == 0:
            continue

        F_norm_ref = np.clip((F_crudo - ideal_algo) / denom_algo, 0, 1)
        rng, smn, msn = calcular_metricas_convergencia(F_norm_ref)

        hv_ref = calcular_hipervolumen(F_norm_ref)

        F_crudo_norm_global = np.clip((F_crudo - ideal_global) / denom_global, 0, 1)
        igd_plus_val = float(igd_plus_metric(F_crudo_norm_global)) if igd_plus_metric is not None else np.nan
        gd_plus_val = float(gd_plus_metric(F_crudo_norm_global)) if gd_plus_metric is not None else np.nan

        crudas = calcular_metricas_crudas(
            F_crudo,
            n_snps_total,
            modo_transformacion_objetivos=modo_transformacion_objetivos,
            modo_evaluacion=modo_evaluacion,
        )

        filas.append({
            'generation': idx_gen + 1,
            'algorithm': rr.algoritmo,
            'init': rr.inicializacion,
            'crossover': rr.crossover,
            'run': rr.replica,
            'seed': rr.semilla,
            'n_solutions': len(F_crudo),
            'Range': rng,
            'SumMin': smn,
            'MinSum': msn,
            'Hypervolume': hv_ref,
            'IGD+': igd_plus_val,
            'GD+': gd_plus_val,
            **{f"{k}": v for k, v in crudas.items()}
        })
    return filas


def _evaluar_metrica_generacional_individual(args):
    """Wrapper picklable para paralelizar métricas generacionales."""
    rr, ideal_algo, denom_algo, ideal_global, denom_global, igd_metric, gd_metric, n_snps_total, modo_transformacion_objetivos, paso_generacional_metricas, modo_evaluacion = args
    t0 = time.time()
    df = pd.DataFrame(
        _construir_filas_generacionales_por_ejecucion(
            rr,
            ideal_algo,
            denom_algo,
            ideal_global,
            denom_global,
            igd_metric,
            gd_metric,
            n_snps_total=n_snps_total,
            modo_transformacion_objetivos=modo_transformacion_objetivos,
            paso_generacional_metricas=paso_generacional_metricas,
            modo_evaluacion=modo_evaluacion,
        )
    )
    return df, time.time() - t0

def construir_metricas_generacionales(resultados_ejecucion, ideal_global, nadir_global,
                                    n_snps_total=1032,
                                    modo_normalizacion='static_dataset_limits',
                                    hamming_pares=None,
                                    modo_transformacion_objetivos='neg',
                                    paso_generacional_metricas=10,
                                    modo_evaluacion='absoluta'):
    """
    Procesa el historial de cada réplica para extraer métricas por generación.
    """
    if not resultados_ejecucion:
        return pd.DataFrame()

    modo_normalizacion = resolver_modo_normalizacion(modo_normalizacion)
    denom_global = nadir_global - ideal_global + 1e-9
    refs_algo = _calcular_referencias_por_modo(
        resultados_ejecucion,
        modo_normalizacion=modo_normalizacion,
        n_snps_total=n_snps_total,
        hamming_pares=hamming_pares,
        modo_transformacion_objetivos=modo_transformacion_objetivos,
        modo_evaluacion=modo_evaluacion,
    )

    ejecuciones_validas = [rr for rr in resultados_ejecucion if getattr(rr, 'ruta_checkpoint', None) is not None]
    if not ejecuciones_validas:
        return pd.DataFrame()

    F_referencia_global = extraer_frente_referencia_empirico(
        resultados_ejecucion, 
        modo_transformacion_objetivos=modo_transformacion_objetivos,
        modo_evaluacion=modo_evaluacion,
    )
    igd_plus_metric = None
    gd_plus_metric = None
    if F_referencia_global is not None:
        F_ref_norm_global = np.clip((F_referencia_global - ideal_global) / denom_global, 0, 1)
        igd_plus_metric = IGDPlus(F_ref_norm_global)
        gd_plus_metric = GDPlus(F_ref_norm_global)

    max_workers = max(1, (os.cpu_count() or 2) - 2)
    total_ejecuciones = len(ejecuciones_validas)
    ancho = len(str(total_ejecuciones))
    conteos_replica = defaultdict(int)
    for rr_aux in ejecuciones_validas:
        conteos_replica[(str(rr_aux.algoritmo), str(rr_aux.inicializacion), str(rr_aux.crossover))] += 1

    completadas = 0
    tablas = []
    batch_size = 50

    for i in range(0, total_ejecuciones, batch_size):
        lote = ejecuciones_validas[i:i + batch_size]
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_rr = {
                executor.submit(
                    _evaluar_metrica_generacional_individual,
                    (
                        rr,
                        refs_algo[str(rr.algoritmo)][0],
                        refs_algo[str(rr.algoritmo)][1],
                        ideal_global,
                        denom_global,
                        igd_plus_metric,
                        gd_plus_metric,
                        n_snps_total,
                        modo_transformacion_objetivos,
                        paso_generacional_metricas,
                        modo_evaluacion,
                    ),
                ): rr for rr in lote
            }

            for future in concurrent.futures.as_completed(future_to_rr):
                rr = future_to_rr[future]
                try:
                    df_res, duracion_ind = future.result()
                    if not df_res.empty:
                        tablas.append(df_res)
                except Exception as e:
                    print(f"      • ⚠️  Error en métricas generacionales [{rr.algoritmo}-{rr.inicializacion}-{rr.crossover}] ejecución {rr.replica}: {e}")
                    continue

                completadas += 1
                total_config = conteos_replica[(str(rr.algoritmo), str(rr.inicializacion), str(rr.crossover))]
                print(f"      • [Progreso: {completadas:>{ancho}}/{total_ejecuciones}] | [{rr.algoritmo}-{rr.inicializacion}-{rr.crossover}] ejecución {rr.replica}/{total_config} ({duracion_ind:.1f} s)")

    if not tablas:
        return pd.DataFrame()
    return pd.concat(tablas, ignore_index=True)
