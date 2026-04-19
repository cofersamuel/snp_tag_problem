"""
Módulo de Visualización del Dataset (dataset.py)
-----------------------------------------------
Implementa funciones para la inspección visual de la estructura haplotípica,
incluyendo mapas de calor, bloques LD y distribuciones de variabilidad.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from scipy.stats import cumfreq
from snp_tag.data.diagnostics import detectar_bloques_ld
from snp_tag.utils.terminal import imprimir_grafico_guardado

def graficar_mapa_calor_haplotipos(H, carpetas, etiqueta_modo, dpi=200):
    sns.set_theme(style='whitegrid')
    plt.figure(figsize=(16, 5))
    sns.heatmap(H, cmap='gray', vmin=0, vmax=1, cbar_kws={'label': 'Alelo (0/1)'})
    plt.title('Matriz de Haplotipos H (Filas: individuos, Columnas: SNPs)')
    plt.xlabel('SNP')
    plt.ylabel('Haplotipo')
    plt.tight_layout()
    ruta = os.path.join(carpetas['datos'], f'heatmap_haplotipos_{etiqueta_modo}.png')
    plt.savefig(ruta, dpi=dpi)
    imprimir_grafico_guardado(ruta, "Mapa de calor de haplotipos")
    plt.close()

def graficar_histograma_hamming(dvals, carpetas, etiqueta_modo, dpi=200):
    plt.figure(figsize=(10, 6))
    sns.histplot(dvals, kde=True, color='purple', bins=20)
    plt.title('Distribución de Distancias de Hamming entre Pares')
    plt.xlabel('Distancia de Hamming')
    plt.ylabel('Frecuencia')
    plt.tight_layout()
    ruta = os.path.join(carpetas['datos'], f'histograma_hamming_{etiqueta_modo}.png')
    plt.savefig(ruta, dpi=dpi)
    imprimir_grafico_guardado(ruta, "Distribución de distancias de Hamming")
    plt.close()

def graficar_variabilidad_snps(H, carpetas, etiqueta_modo, dpi=200):
    desvios = H.std(axis=0)
    plt.figure(figsize=(16, 4))
    plt.plot(desvios, color='teal', linewidth=1)
    plt.title('Variabilidad por SNP (Desviación Estándar)')
    plt.xlabel('Índice SNP')
    plt.ylabel('Desviación Estándar')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    ruta = os.path.join(carpetas['datos'], f'variabilidad_snps_{etiqueta_modo}.png')
    plt.savefig(ruta, dpi=dpi)
    imprimir_grafico_guardado(ruta, "Variabilidad por SNP")
    plt.close()

def graficar_conteo_alelos(H, carpetas, etiqueta_modo, dpi=200):
    conteos = H.sum(axis=1)
    plt.figure(figsize=(12, 5))
    sns.barplot(x=list(range(len(conteos))), y=conteos, color='salmon')
    plt.title('Conteo de Alelos "1" por Haplotipo')
    plt.xlabel('Índice Haplotipo')
    plt.ylabel('Cantidad de Alelos "1"')
    plt.tight_layout()
    ruta = os.path.join(carpetas['datos'], f'conteo_alelos_{etiqueta_modo}.png')
    plt.savefig(ruta, dpi=dpi)
    imprimir_grafico_guardado(ruta, "Alelos dominantes por haplotipo")
    plt.close()

def graficar_histograma_alelico(H, carpetas, etiqueta_modo, dpi=200):
    freqs = H.mean(axis=0)
    plt.figure(figsize=(10, 6))
    sns.histplot(freqs, bins=30, kde=True, color='green')
    plt.title('Distribución de Frecuencia Alélica (MAF aproximado)')
    plt.xlabel('Frecuencia del Alelo "1"')
    plt.ylabel('Número de SNPs')
    plt.tight_layout()
    ruta = os.path.join(carpetas['datos'], f'histograma_alelico_{etiqueta_modo}.png')
    plt.savefig(ruta, dpi=dpi)
    imprimir_grafico_guardado(ruta, "Distribución de frecuencia alélica")
    plt.close()

def graficar_ld_detallado(corr_full, corrs, carpetas, etiqueta_modo, dpi=200):
    # 1. Heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(np.abs(corr_full), cmap='vlag', center=0, cbar_kws={'label': 'Correlación'})
    plt.title('Mapa de Calor de Correlación LD (|r|)')
    plt.tight_layout()
    ruta_hm = os.path.join(carpetas['datos'], f'heatmap_correlacion_completa_{etiqueta_modo}.png')
    plt.savefig(ruta_hm, dpi=dpi)
    
    # 2. Histograma
    plt.figure(figsize=(10, 6))
    sns.histplot(np.abs(corrs), bins=50, color='slateblue', kde=True)
    plt.title('Distribución de Correlaciones Absolutas LD')
    plt.xlabel('|r|')
    plt.ylabel('Frecuencia')
    plt.tight_layout()
    ruta_hist = os.path.join(carpetas['datos'], f'histograma_correlaciones_ld_{etiqueta_modo}.png')
    plt.savefig(ruta_hist, dpi=dpi)
    
    # 3. CDF
    abs_c = np.sort(np.abs(corrs))
    y = np.arange(len(abs_c)) / float(len(abs_c))
    plt.figure(figsize=(10, 6))
    plt.plot(abs_c, y, color='darkgreen', linewidth=2)
    plt.title('CDF de Correlación Absoluta LD')
    plt.xlabel('|r|')
    plt.ylabel('Probabilidad Acumulada')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    ruta_cdf = os.path.join(carpetas['datos'], f'cdf_correlacion_absoluta_ld_{etiqueta_modo}.png')
    plt.savefig(ruta_cdf, dpi=dpi)
    plt.close('all')
    
    return {
        'heatmap': ruta_hm,
        'histograma': ruta_hist,
        'cdf': ruta_cdf
    }

def graficar_bloques_ld(H, carpetas, etiqueta_modo, dpi=200):
    """Visualiza los bloques de ligamiento detectados sobre la matriz."""
    segmentos = detectar_bloques_ld(H)
    n_bloques = len(segmentos)
    
    H_f = H.astype(float)
    n_hap = H_f.shape[0]
    medias_bloque = np.zeros((n_hap, n_bloques), dtype=float)
    for b, (s, e) in enumerate(segmentos):
        medias_bloque[:, b] = H_f[:, s:e].mean(axis=1)
        
    pesos = np.linspace(1.0, 2.0, n_bloques)
    orden = np.argsort(medias_bloque @ pesos)
    H_ord = medias_bloque[orden, :]
    
    bordes_x = np.array([s for s, _ in segmentos] + [segmentos[-1][1]], dtype=float)
    bordes_y = np.arange(n_hap + 1, dtype=float)
    
    fig, ax = plt.subplots(figsize=(16, 6))
    malla = ax.pcolormesh(bordes_x, bordes_y, H_ord, cmap='viridis', vmin=0, vmax=1, shading='flat')
    fig.colorbar(malla, ax=ax, label='Fracción de alelo 1 (Media del bloque)')
    
    for b, (s, e) in enumerate(segmentos):
        if s > 0: ax.axvline(x=s, color='red', linewidth=1.2, linestyle='--', alpha=0.85)
        
    ax.set_title(f'Estructura Estructural: {n_bloques} bloques LD detectados')
    ax.set_xlabel('Posición SNP (Escala real)')
    ax.set_ylabel('Haplotipo (Ordenado por patrones)')
    ax.set_xlim(bordes_x[0], bordes_x[-1])
    ax.set_ylim(0, n_hap)
    ax.invert_yaxis()
    plt.tight_layout()
    
    ruta = os.path.join(carpetas['datos'], f'bloques_ld_haplotipos_{etiqueta_modo}.png')
    plt.savefig(ruta, dpi=dpi)
    imprimir_grafico_guardado(ruta, f"Estructura LD ({n_bloques} bloques)")
    plt.close()
