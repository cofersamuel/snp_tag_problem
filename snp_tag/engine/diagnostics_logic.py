"""
Módulo de Diagnóstico de Datos (diagnostics.py)
----------------------------------------------
Ejecuta el análisis estadístico y de ligamiento (LD) del dataset haplotípico.
Incluye rutinas para la detección de bloques genómicos y la caracterización
de la variabilidad alélica.
"""

# =============================================================================
# LIBRERÍAS ESTÁNDAR
# =============================================================================
import os
from typing import Dict, List, Optional, Tuple

# =============================================================================
# LIBRERÍAS DE TERCEROS
# =============================================================================
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# =============================================================================
# MÓDULOS LOCALES (snp_tag)
# =============================================================================
from snp_tag.config import ConfiguracionExperimento
from snp_tag.utils.terminal import (imprimir_estado, imprimir_grafico_guardado,
                                    imprimir_metadato, imprimir_subseccion,
                                    obtener_bit_string_estilizado)


def calcular_ld_completo(X: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Calcula el Desequilibrio de Ligamiento (LD) mediante correlación de Pearson.

    Calcula el coeficiente de Pearson para todos los pares de variantes (columnas)
    para mapear la fuerza de asociación genómica del bloque.

    Parámetros:
    -----------
    X : np.ndarray
        Matriz binaria representativa del bloque haplotípico (haplotipos x SNPs).

    Retorna:
    --------
    Tuple[float, np.ndarray, np.ndarray]
        - media_ld: Promedio de los valores absolutos de correlación fuera de la diagonal.
        - corrs: Vector unidimensional con los coeficientes del triángulo superior.
        - corr_full: Matriz bidimensional simétrica de correlación (n_snps x n_snps).
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        corr_full = np.corrcoef(X.T)
    corr_full = np.nan_to_num(corr_full, nan=0.0, posinf=0.0, neginf=0.0)
    tri_i, tri_j = np.triu_indices(corr_full.shape[0], k=1)
    corrs = corr_full[tri_i, tri_j]
    media_ld = float(np.mean(np.abs(corrs))) if corrs.size > 0 else 0.0
    return media_ld, corrs, corr_full

def detectar_bloques_ld(H: np.ndarray, ventana_suavizado: int = 11, 
                       umbral: float = 0.20, min_snps_bloque: int = 10) -> List[Tuple[int, int]]:
    """
    Algoritmo de detección de bloques LD basado en hotspots de recombinación.

    Calcula el r² adyacente entre SNPs consecutivos, lo suaviza con una ventana
    móvil y corta los bloques genómicos allí donde la correlación cae por debajo
    de un umbral, respetando un tamaño mínimo por bloque.

    Parámetros:
    -----------
    H : np.ndarray
        Matriz haplotípica binaria (n_haplotipos, n_snps).
    ventana_suavizado : int
        Tamaño de la ventana móvil para convolucionar el r². Por defecto 11.
    umbral : float
        Límite por debajo del cual se considera una ruptura de bloque. Por defecto 0.20.
    min_snps_bloque : int
        Tamaño mínimo requerido (en nº de SNPs) para consolidar un bloque.

    Retorna:
    --------
    List[Tuple[int, int]]
        Lista de tuplas (inicio, fin) indicando los índices de las columnas
        que demarcan cada bloque LD detectado.
    """
    n_snps = H.shape[1]
    n_hap = H.shape[0]
    H_f = H.astype(float)
    
    media = H_f.mean(axis=0)
    desvio = H_f.std(axis=0) + 1e-9
    H_norm = (H_f - media) / desvio
    
    r_adj = np.einsum('hi,hi->i', H_norm[:, :-1], H_norm[:, 1:]) / n_hap
    r2_adj = r_adj ** 2
    
    nucleo = np.ones(ventana_suavizado) / ventana_suavizado
    r2_suavizado = np.convolve(r2_adj, nucleo, mode='same')
    
    pos_fronteras = np.where(r2_suavizado < umbral)[0] + 1
    
    todos_cortes = np.concatenate([[0], pos_fronteras, [n_snps]])
    cortes_validos = [0]
    acumulado = 0
    for idx, (corte, anchura) in enumerate(zip(todos_cortes[1:], np.diff(todos_cortes))):
        acumulado += int(anchura)
        es_ultimo = (idx == len(todos_cortes) - 2)
        if acumulado >= min_snps_bloque or es_ultimo:
            cortes_validos.append(int(corte))
            acumulado = 0
    if cortes_validos[-1] != n_snps:
        cortes_validos[-1] = n_snps
        
    return [(cortes_validos[i], cortes_validos[i + 1]) for i in range(len(cortes_validos) - 1)]

def analizar_similitud_genotipica(H: np.ndarray) -> np.ndarray:
    """
    Calcula distancias Hamming por pares.

    Compara todos los pares posibles de haplotipos en la matriz para
    evaluar la diversidad poblacional.

    Parámetros:
    -----------
    H : np.ndarray
        Matriz haplotípica binaria.

    Retorna:
    --------
    np.ndarray
        - dvals: Vector de distancias Hamming para todos los pares.
    """
    n = H.shape[0]
    dlist = []
    for a in range(n):
        for b in range(a+1, n):
            dlist.append(((a, b), int((H[a] != H[b]).sum())))
            
    dvals = np.array([v for (_, v) in dlist]) if dlist else np.array([0])
    
    return dvals

def ejecutar_diagnostico_ld(H: np.ndarray, cfg: ConfiguracionExperimento, rutas_ld: Optional[Dict[str, str]] = None) -> Tuple[float, np.ndarray, np.ndarray, List[Tuple[int, int]]]:
    """
    Realiza un diagnóstico exhaustivo de la estructura LD del dataset.

    Parámetros:
    -----------
    H : np.ndarray
        Matriz haplotípica binaria.
    cfg : ConfiguracionExperimento
        Configuración global del experimento.
    rutas_ld : Optional[Dict[str, str]]
        Diccionario opcional con las rutas para renderizar gráficos de diagnóstico.

    Retorna:
    --------
    Tuple[float, np.ndarray, np.ndarray, List[Tuple[int, int]]]
        - media_ld: Correlación LD media absoluta.
        - corrs: Vector de correlaciones del triángulo superior.
        - corr_full: Matriz completa de correlaciones.
        - segmentos: Lista de límites de bloques detectados.
    """
    
    media_ld, corrs, corr_full = calcular_ld_completo(H)
    abs_corr = np.abs(corrs)
    
    segmentos = detectar_bloques_ld(H)
    n_bloques = len(segmentos)
    
    # 1. Caracterización Global
    imprimir_subseccion("Caracterización Global de Correlación (LD)", icono="🔗")
    print(f"      • \033[1mCorrelación media absoluta (global |r|)\033[0m: {media_ld:.4f}")
    print(f"      • \033[1mTotal de pares evaluados\033[0m: {len(corrs)}")
    
    n_hap = H.shape[0]
    n_pares = (n_hap * (n_hap - 1)) // 2
    print(f"      • \033[1mNúmero de pares de haplotipos\033[0m: {n_pares}")
    
    if rutas_ld:
        imprimir_grafico_guardado(rutas_ld['heatmap'], "Mapa de calor de correlación (LD)")
        imprimir_grafico_guardado(rutas_ld['histograma'], "Distribución de correlaciones (LD)")
        imprimir_grafico_guardado(rutas_ld['cdf'], "CDF de correlación LD")
    
    return media_ld, corrs, corr_full, segmentos
