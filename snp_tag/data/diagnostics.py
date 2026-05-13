"""
Módulo de Diagnóstico de Datos (diagnostics.py)
----------------------------------------------
Ejecuta el análisis estadístico y de ligamiento (LD) del dataset haplotípico.
Incluye rutinas para la detección de bloques genómicos y la caracterización
de la variabilidad alélica.
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from snp_tag.config import ConfiguracionExperimento
from snp_tag.utils.terminal import imprimir_subseccion, imprimir_estado, imprimir_metadato, obtener_bit_string_estilizado

def calcular_ld_completo(X):
    """
    Calcula la matriz de correlación de Pearson para caracterizar el Desequilibrio de Ligamiento.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        corr_full = np.corrcoef(X.T)
    corr_full = np.nan_to_num(corr_full, nan=0.0, posinf=0.0, neginf=0.0)
    tri_i, tri_j = np.triu_indices(corr_full.shape[0], k=1)
    corrs = corr_full[tri_i, tri_j]
    media_ld = float(np.mean(np.abs(corrs))) if corrs.size > 0 else 0.0
    return media_ld, corrs, corr_full

def detectar_bloques_ld(H: np.ndarray, ventana_suavizado: int = 11, 
                       umbral: float = 0.20, min_snps_bloque: int = 10):
    """
    Algoritmo de detección de bloques LD basado en hotspots de recombinación.
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

def analizar_similitud_genotipica(H):
    """Calcula distancias Hamming por pares e identifica extremos."""
    n = H.shape[0]
    dlist = []
    for a in range(n):
        for b in range(a+1, n):
            dlist.append(((a, b), int((H[a] != H[b]).sum())))
            
    dvals = np.array([v for (_, v) in dlist]) if dlist else np.array([0])
    p33 = float(np.percentile(dvals, 33))
    p66 = float(np.percentile(dvals, 66))
    
    sorted_pairs = sorted(dlist, key=lambda x: x[1])
    top_sim = sorted_pairs[:3]
    top_dist = sorted_pairs[-3:][::-1]
    
    return dvals, p33, p66, top_sim, top_dist

def ejecutar_diagnostico_ld(H, cfg: ConfiguracionExperimento, rutas_ld=None):
    """
    Realiza un diagnóstico exhaustivo de la estructura LD del dataset (Formato Legacy).
    """
    from snp_tag.utils.terminal import imprimir_grafico_guardado
    
    media_ld, corrs, corr_full = calcular_ld_completo(H)
    abs_corr = np.abs(corrs)
    
    segmentos = detectar_bloques_ld(H)
    n_bloques = len(segmentos)
    
    # 1. Caracterización Global
    imprimir_subseccion("Caracterización Global de Correlación (LD)", icono="🔗")
    print(f"      • \033[1mCorrelación media absoluta (global |r|)\033[0m: {media_ld:.4f}")
    print(f"      • \033[1mTotal de pares evaluados\033[0m: {len(corrs)}")
    
    if rutas_ld:
        imprimir_grafico_guardado(rutas_ld['heatmap'], "Mapa de calor de correlación (LD)")
        imprimir_grafico_guardado(rutas_ld['histograma'], "Distribución de correlaciones (LD)")
        imprimir_grafico_guardado(rutas_ld['cdf'], "CDF de correlación LD")
    
    # 3. Similitud Genotípica
    dvals, p33, p66, top_sim, top_dist = analizar_similitud_genotipica(H)
    imprimir_subseccion("Análisis de Similitud Genotípica (Pares de Haplotipos)", icono="📐")
    print(f"      • \033[1mNúmero de pares de haplotipos\033[0m: {len(dvals)}")
    print(f"      • \033[1mPares mostrados\033[0m: 3 similares / 3 distintos")
    print(f"      • \033[1mVista parcial\033[0m: primeros 32 SNPs")
    print(f"      • \033[1mPercentiles (Hamming)\033[0m: P33={p33:.2f}, P66={p66:.2f}")
    print(f"      • [Etiquetas: <=P33 -> muy similar | (P33,P66] -> intermedio | >P66 -> muy distinto]")

    imprimir_subseccion("Pares de mayor similitud genética", icono="🤝")
    for (ia, ib), val in top_sim:
        etiqueta = "muy similar" if val <= p33 else "intermedio"
        print(f"      •  \033[1mPar ({ia+1}, {ib+1})\033[0m | Hamming=\033[1m{val}\033[0m | {etiqueta}")
        print(f"        h{ia+1:03d}: {obtener_bit_string_estilizado(H[ia, :32])}...")
        print(f"        h{ib+1:03d}: {obtener_bit_string_estilizado(H[ib, :32])}...")

    imprimir_subseccion("Pares de mayor divergencia genética", icono="↔️")
    for (ia, ib), val in top_dist:
        etiqueta = "muy distinto" if val > p66 else "intermedio"
        print(f"      •  \033[1mPar ({ia+1}, {ib+1})\033[0m | Hamming=\033[1m{val}\033[0m | {etiqueta}")
        print(f"        h{ia+1:03d}: {obtener_bit_string_estilizado(H[ia, :32])}...")
        print(f"        h{ib+1:03d}: {obtener_bit_string_estilizado(H[ib, :32])}...")
    
    return media_ld, corrs, corr_full, segmentos
