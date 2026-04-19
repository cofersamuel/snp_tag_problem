"""
Módulo de Cálculo de Métricas (metrics.py)
-----------------------------------------
Implementa las métricas de rendimiento para la evaluación de frentes de Pareto,
incluyendo Range, SumMin, MinSum e Hipervolumen, así como indicadores crudos.
"""

import numpy as np
import pandas as pd
import time
import os
import concurrent.futures
from functools import lru_cache
from typing import List, Tuple, Dict, Any
from collections import defaultdict

from snp_tag.config import resolver_modo_normalizacion

def calcular_puntos_ideales_nadires_globales(resultados_ejecucion):
    """
    Determina los límites globales (ideal y nadir) a partir de todos los frentes.
    """
    frentes_finales = [rr.F_final for rr in resultados_ejecucion 
                       if rr.F_final is not None and len(rr.F_final) > 0]
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

def obtener_referencias_estaticas_dataset(n_snps_total: int, hamming_pares=None):
    """
    Define límites de normalización fijos basados en la naturaleza del dataset.
    """
    L = float(max(1, n_snps_total))
    if hamming_pares is not None and len(hamming_pares) > 0:
        max_tol = float(hamming_pares.min() - 1.0)
        max_ham = float(hamming_pares.mean())
        max_var = float(np.var(hamming_pares)) if len(hamming_pares) > 1 else (L**2)/4.0
    else:
        max_tol, max_ham, max_var = L - 1.0, L, (L**2)/4.0

    ideal = np.array([1.0, -max_tol, -max_ham, 0.0])
    nadir = np.array([L, 1.0, 0.0, max_var])
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

def calcular_metricas_crudas(F_crudo, n_snps_total: int = 1032):
    """
    Extrae indicadores físicos en escala real (independientes de la normalización).
    """
    compacidad = F_crudo[:, 0]
    tolerancia_real = -F_crudo[:, 1]
    hamming_prom_real = -F_crudo[:, 2]
    
    seguro_comp = np.where(compacidad <= 0, np.nan, compacidad)
    tasa_tol = tolerancia_real / seguro_comp
    
    return {
        'MaxToleranceRate': float(np.nanmax(tasa_tol)),
        'AvgToleranceRate': float(np.nanmean(tasa_tol)),
        'AvgHammingDistance': float(np.nanmean(hamming_prom_real)),
        'avg_h_norm': float(np.nanmean(hamming_prom_real)) / max(1, n_snps_total)
    }

@lru_cache(maxsize=8)
def _obtener_indicador_hv(n_obj: int):
    from pymoo.indicators.hv import HV
    punto_ref = np.ones(int(n_obj), dtype=float)
    return HV(ref_point=punto_ref)

def calcular_hipervolumen(F_norm):
    """
    Calcula el Hipervolumen (HV) respecto al punto de referencia [1, 1, 1, 1].
    """
    if F_norm is None or len(F_norm) == 0:
        return np.nan
    hv = _obtener_indicador_hv(int(F_norm.shape[1]))
    return float(hv(F_norm))


def _calcular_referencias_por_algoritmo(resultados_ejecucion):
    """Calcula referencias ideal/denominador por algoritmo."""
    grupos = defaultdict(list)
    for rr in resultados_ejecucion:
        grupos[str(rr.algoritmo)].append(rr)

    refs = {}
    for algoritmo, rrs in grupos.items():
        ideal_a, nadir_a = calcular_puntos_ideales_nadires_globales(rrs)
        refs[algoritmo] = (ideal_a, nadir_a - ideal_a + 1e-9)
    return refs


def _calcular_referencias_por_modo(resultados_ejecucion, modo_normalizacion='global_all_pairs', n_snps_total=1032, hamming_pares=None):
    """Devuelve referencias para Range/SumMin/MinSum según el modo configurado."""
    modo = resolver_modo_normalizacion(modo_normalizacion)
    if modo == 'per_algorithm':
        return _calcular_referencias_por_algoritmo(resultados_ejecucion)

    if modo == 'static_dataset_limits':
        ideal_ref, denom_ref = obtener_referencias_estaticas_dataset(n_snps_total, hamming_pares)
        algoritmos = sorted({str(rr.algoritmo) for rr in resultados_ejecucion})
        return {alg: (ideal_ref, denom_ref) for alg in algoritmos}

    ideal_g, nadir_g = calcular_puntos_ideales_nadires_globales(resultados_ejecucion)
    denom_g = nadir_g - ideal_g + 1e-9
    algoritmos = sorted({str(rr.algoritmo) for rr in resultados_ejecucion})
    return {alg: (ideal_g, denom_g) for alg in algoritmos}

def evaluar_metricas_finales(resultados_ejecucion, n_snps_total=1032, 
                             modo_normalizacion='static_dataset_limits',
                             hamming_pares=None):
    """
    Evalúa todas las métricas de rendimiento para un conjunto de resultados de ejecución.
    """
    if not resultados_ejecucion:
        return pd.DataFrame(), None, None

    modo_normalizacion = resolver_modo_normalizacion(modo_normalizacion)
    ideal_g, nadir_g = calcular_puntos_ideales_nadires_globales(resultados_ejecucion)
    denom_g = nadir_g - ideal_g + 1e-9
    refs_algo = _calcular_referencias_por_modo(
        resultados_ejecucion,
        modo_normalizacion=modo_normalizacion,
        n_snps_total=n_snps_total,
        hamming_pares=hamming_pares,
    )

    filas = []
    total_ejecuciones = len(resultados_ejecucion)
    w = len(str(max(1, total_ejecuciones)))
    procesadas = 0
    for rr in resultados_ejecucion:
        F_crudo = rr.F_final
        if F_crudo is None or len(F_crudo) == 0:
            continue
        
        # Normalización para métricas de convergencia
        ideal_ref, denom_ref = refs_algo[str(rr.algoritmo)]
        F_norm_ref = np.clip((F_crudo - ideal_ref) / denom_ref, 0, 1)
        rng, smn, msn = calcular_metricas_convergencia(F_norm_ref)
        
        # Normalización global para Hipervolumen
        F_norm_g = np.clip((F_crudo - ideal_g) / denom_g, 0, 1)
        hv = calcular_hipervolumen(F_norm_g)
        
        # Métricas crudas
        crudas = calcular_metricas_crudas(F_crudo, n_snps_total)
        
        filas.append({
            'algorithm': rr.algoritmo, 'init': rr.inicializacion, 'run': rr.replica,
            'seed': rr.semilla,
            'elapsed_sec': rr.tiempo_seg,
            'n_solutions_final_front': len(F_crudo),
            'Range': rng, 'SumMin': smn, 'MinSum': msn, 'Hypervolume': hv,
            **{f"{k}": v for k, v in crudas.items()}
        })
        procesadas += 1
        print(f"      • [Progreso: {procesadas:>{w}}/{total_ejecuciones}] | [{rr.algoritmo}-{rr.inicializacion}] ejecución {rr.replica}")
        
    return pd.DataFrame(filas), ideal_g, nadir_g


def _construir_filas_generacionales_por_ejecucion(rr, ideal_algo, denom_algo, ideal_global, denom_global, n_snps_total=1032):
    """Construye filas de métricas por generación para una ejecución individual."""
    filas = []
    historial = getattr(rr, 'historial_F', None)
    if historial is None or len(historial) == 0:
        return filas

    for idx_gen, F_crudo in enumerate(historial):
        if F_crudo is None or len(F_crudo) == 0:
            continue

        F_crudo = np.array(F_crudo, dtype=float)
        if F_crudo.ndim == 1:
            F_crudo = F_crudo.reshape(1, -1)

        F_norm_ref = np.clip((F_crudo - ideal_algo) / denom_algo, 0, 1)
        rng, smn, msn = calcular_metricas_convergencia(F_norm_ref)

        F_norm_global = np.clip((F_crudo - ideal_global) / denom_global, 0, 1)
        hv = calcular_hipervolumen(F_norm_global)

        crudas = calcular_metricas_crudas(F_crudo, n_snps_total)

        filas.append({
            'generation': idx_gen + 1,
            'algorithm': rr.algoritmo,
            'init': rr.inicializacion,
            'run': rr.replica,
            'seed': rr.semilla,
            'n_solutions': len(F_crudo),
            'Range': rng,
            'SumMin': smn,
            'MinSum': msn,
            'Hypervolume': hv,
            **{f"{k}": v for k, v in crudas.items()}
        })
    return filas


def _evaluar_metrica_generacional_individual(args):
    """Wrapper picklable para paralelizar métricas generacionales."""
    rr, ideal_algo, denom_algo, ideal_global, denom_global, n_snps_total = args
    return pd.DataFrame(
        _construir_filas_generacionales_por_ejecucion(
            rr,
            ideal_algo,
            denom_algo,
            ideal_global,
            denom_global,
            n_snps_total=n_snps_total,
        )
    )

def construir_metricas_generacionales(resultados_ejecucion, ideal_global, nadir_global,
                                    n_snps_total=1032,
                                    modo_normalizacion='static_dataset_limits',
                                    hamming_pares=None):
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
    )

    ejecuciones_validas = [rr for rr in resultados_ejecucion if getattr(rr, 'historial_F', None)]
    if not ejecuciones_validas:
        return pd.DataFrame()

    max_workers = max(1, (os.cpu_count() or 2) - 2)
    total_ejecuciones = len(ejecuciones_validas)
    ancho = len(str(total_ejecuciones))
    completadas = 0
    tablas = []

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
                    n_snps_total,
                ),
            ): rr for rr in ejecuciones_validas
        }

        for future in concurrent.futures.as_completed(future_to_rr):
            rr = future_to_rr[future]
            try:
                tablas.append(future.result())
            except Exception as e:
                print(f"      • ⚠ Error en métricas generacionales [{rr.algoritmo}-{rr.inicializacion}] ejecución {rr.replica}: {e}")
                continue

            completadas += 1
            print(f"      • [Progreso: {completadas:>{ancho}}/{total_ejecuciones}] | [{rr.algoritmo}-{rr.inicializacion}] ejecución {rr.replica}")

    if not tablas:
        return pd.DataFrame()
    return pd.concat(tablas, ignore_index=True)
