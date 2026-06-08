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
from typing import Optional, Dict, Any

from snp_tag.engine.diagnostics_logic import detectar_bloques_ld
from snp_tag.utils.terminal import imprimir_grafico_guardado

def graficar_mapa_calor_haplotipos(H: np.ndarray, carpetas: Dict[str, str], etiqueta_modo: str, dpi: int = 200) -> None:
    """
    Genera un mapa de calor visualizando la matriz binaria haplotípica completa.

    Parámetros:
    -----------
    H : np.ndarray
        Matriz haplotípica binaria (filas=haplotipos, columnas=SNPs).
    carpetas : Dict[str, str]
        Diccionario con las rutas de los directorios de salida.
    etiqueta_modo : str
        Sufijo identificador del modo de ejecución.
    dpi : int
        Resolución de la imagen generada (por defecto 200).
    """
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

def graficar_histograma_hamming(dvals: np.ndarray, carpetas: Dict[str, str], etiqueta_modo: str, dpi: int = 200) -> None:
    """
    Genera un histograma de la distribución de distancias Hamming por pares.

    Parámetros:
    -----------
    dvals : np.ndarray
        Array unidimensional de distancias Hamming calculadas.
    carpetas : Dict[str, str]
        Diccionario con las rutas de los directorios de salida.
    etiqueta_modo : str
        Sufijo identificador del modo de ejecución.
    dpi : int
        Resolución de la imagen generada.
    """
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

def graficar_variabilidad_snps(H: np.ndarray, carpetas: Dict[str, str], etiqueta_modo: str, dpi: int = 200) -> None:
    """
    Grafica la variabilidad (desviación estándar) de cada SNP en el bloque.

    Parámetros:
    -----------
    H : np.ndarray
        Matriz haplotípica binaria.
    carpetas : Dict[str, str]
        Diccionario con las rutas de salida.
    etiqueta_modo : str
        Sufijo identificador del experimento.
    dpi : int
        Resolución de la imagen generada.
    """
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

def graficar_conteo_alelos(H: np.ndarray, carpetas: Dict[str, str], etiqueta_modo: str, dpi: int = 200) -> None:
    """
    Representa un diagrama de barras con el conteo de alelos "1" por cada haplotipo.

    Parámetros:
    -----------
    H : np.ndarray
        Matriz haplotípica binaria.
    carpetas : Dict[str, str]
        Diccionario con rutas de salida.
    etiqueta_modo : str
        Sufijo identificador.
    dpi : int
        Resolución de la imagen generada.
    """
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

def graficar_histograma_alelico(H: np.ndarray, carpetas: Dict[str, str], etiqueta_modo: str, dpi: int = 200) -> None:
    """
    Genera un histograma de la distribución de las frecuencias alélicas (aproximación MAF).

    Parámetros:
    -----------
    H : np.ndarray
        Matriz haplotípica binaria.
    carpetas : Dict[str, str]
        Diccionario con rutas de salida.
    etiqueta_modo : str
        Sufijo identificador.
    dpi : int
        Resolución de la imagen generada.
    """
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

def graficar_ld_detallado(corr_full: np.ndarray, corrs: np.ndarray, carpetas: Dict[str, str], etiqueta_modo: str, dpi: int = 200) -> Dict[str, str]:
    """
    Genera un conjunto de tres gráficos (Heatmap, Histograma y CDF) sobre el LD.

    Parámetros:
    -----------
    corr_full : np.ndarray
        Matriz bidimensional simétrica de correlaciones de Pearson (|r|).
    corrs : np.ndarray
        Vector unidimensional de coeficientes del triángulo superior.
    carpetas : Dict[str, str]
        Diccionario con rutas de salida.
    etiqueta_modo : str
        Sufijo identificador.
    dpi : int
        Resolución de la imagen generada.

    Retorna:
    --------
    Dict[str, str]
        Diccionario con las rutas de las tres imágenes creadas.
    """
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

def graficar_bloques_ld(H: np.ndarray, carpetas: Dict[str, str], etiqueta_modo: str, dpi: int = 200, cfg: Optional[Any] = None) -> None:
    """
    Visualiza los bloques de ligamiento (LD) sobre la matriz haplotípica y la renderiza.

    Para datos sintéticos, los segmentos se reconstruyen directamente desde `cfg.num_bloques`.
    Para datos reales (Hinds), se usa la detección estadística basada en hotspots.

    Parámetros:
    -----------
    H : np.ndarray
        Matriz haplotípica binaria (n_haplotipos, n_snps).
    carpetas : Dict[str, str]
        Diccionario con las rutas de directorios de salida.
    etiqueta_modo : str
        Sufijo identificador del experimento.
    dpi : int
        Resolución de la imagen exportada.
    cfg : Optional[Any]
        Configuración global (instancia de ConfiguracionExperimento) opcional.
    """
    # Seleccionar estrategia de segmentación según el origen del dataset
    if cfg is not None and getattr(cfg, 'origen_datos', None) == 'synthetic':
        # Reconstruir bloques estructurales desde la configuración
        n_snps_real = H.shape[1]
        num_bloques = max(1, int(cfg.num_bloques))
        tam_bloque = max(1, n_snps_real // num_bloques)
        segmentos = []
        inicio = 0
        for _ in range(num_bloques):
            fin = min(inicio + tam_bloque, n_snps_real)
            if inicio >= n_snps_real:
                break
            segmentos.append((inicio, fin))
            inicio = fin
        # Absorber SNPs residuales en el último bloque (resto de la división entera)
        if segmentos and segmentos[-1][1] < n_snps_real:
            segmentos[-1] = (segmentos[-1][0], n_snps_real)
    else:
        # Detección estadística estándar para datos reales
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
