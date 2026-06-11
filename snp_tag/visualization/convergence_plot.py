"""
Módulo de Análisis de Convergencia (convergence.py)
--------------------------------------------------
Visualiza la evolución de las métricas de rendimiento a lo largo de las
generaciones para cada configuración experimental.
"""

# =============================================================================
# LIBRERÍAS ESTÁNDAR
# =============================================================================
import math
import os
from typing import Dict, List, Optional, Tuple

# =============================================================================
# LIBRERÍAS DE TERCEROS
# =============================================================================
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# =============================================================================
# MÓDULOS LOCALES (snp_tag)
# =============================================================================
from snp_tag.constants import METRICS_DISPLAY_NAMES, PREFERRED_ALGORITHMS_ORDER
from snp_tag.utils.terminal import imprimir_grafico_guardado


def graficar_evolucion_generacional(df_gen: pd.DataFrame, dir_salida: Optional[str] = None, 
                                   etiqueta_modo: Optional[str] = None, 
                                   figsize: Tuple[int, int] = (16, 12), dpi: int = 300,
                                   emitir_log: bool = True) -> List[Tuple[str, str]]:
    """
    Genera una figura de convergencia por cada algoritmo, con subgráficas (3x3)
    para cada métrica. Muestra la evolución completa de cada configuración
    (Init + Crossover).

    Parámetros:
    -----------
    df_gen : pd.DataFrame
        Dataset consolidado con las métricas inter-generacionales por iteración
        (típicamente filtrado al *run* mediano).
    dir_salida : Optional[str]
        Directorio absoluto destino para exportación.
    etiqueta_modo : Optional[str]
        Sufijo identificador del experimento.
    figsize : Tuple[int, int]
        Dimensiones de la figura (ancho, alto).
    dpi : int
        Calidad visual.
    emitir_log : bool
        Logs por terminal tras la exportación.

    Retorna:
    --------
    List[Tuple[str, str]]
        Tuplas de la ruta generada y el título de la figura.
    """
    metricas = list(METRICS_DISPLAY_NAMES.items())

    cols_base = {'algorithm', 'init', 'crossover', 'generation'}
    if df_gen.empty or not cols_base.issubset(df_gen.columns):
        return []

    df_plot = df_gen.copy()
    df_plot = df_plot.dropna(subset=['algorithm', 'init', 'crossover', 'generation'])
    if df_plot.empty:
        return []

    df_plot['generation'] = pd.to_numeric(df_plot['generation'], errors='coerce')
    df_plot = df_plot.dropna(subset=['generation'])
    if df_plot.empty:
        return []

    df_plot['algorithm'] = df_plot['algorithm'].astype(str)
    df_plot['init'] = df_plot['init'].astype(str)
    df_plot['crossover'] = df_plot['crossover'].astype(str)
    
    # Crear la columna del método combinando las características
    df_plot['metodo'] = df_plot['algorithm'] + "+" + df_plot['init'] + "+" + df_plot['crossover']

    # Estandarización de ejes Y globales por métrica (misma escala para todas las curvas de la métrica en todos los algoritmos)
    limites_y = {}
    for m_col, _ in metricas:
        if m_col not in df_plot.columns:
            continue
        datos = pd.to_numeric(df_plot[m_col], errors='coerce').dropna()
        if datos.empty:
            continue
        y_min, y_max = float(datos.min()), float(datos.max())
        pad = (y_max - y_min) * 0.05 if y_max > y_min else 1.0
        limites_y[m_col] = (y_min - pad, y_max + pad)

    g_min, g_max = float(df_plot['generation'].min()), float(df_plot['generation'].max())

    algoritmos_presentes = sorted(df_plot['algorithm'].dropna().unique().tolist())
    algoritmos_ordenados = [a for a in PREFERRED_ALGORITHMS_ORDER if a in algoritmos_presentes]
    algoritmos_ordenados.extend([a for a in algoritmos_presentes if a not in algoritmos_ordenados])
    if not algoritmos_ordenados:
        return []

    sns.set_theme(style='whitegrid')
    artefactos = []
    
    for algoritmo in algoritmos_ordenados:
        sub_algo = df_plot[df_plot['algorithm'] == algoritmo].copy()
        if sub_algo.empty:
            continue
            
        fig, axes = plt.subplots(3, 3, figsize=figsize)
        axes = axes.ravel()
        
        for idx, (m_col, titulo) in enumerate(metricas):
            ax = axes[idx]
            if m_col not in sub_algo.columns:
                ax.axis('off')
                continue
                
            datos_metric = sub_algo[['generation', 'metodo', m_col]].copy()
            datos_metric[m_col] = pd.to_numeric(datos_metric[m_col], errors='coerce')
            datos_metric = datos_metric.dropna(subset=[m_col])
            
            if datos_metric.empty:
                ax.axis('off')
                continue
                
            # Ya no se usa estimator='mean' pues los datos ya están filtrados por el run mediano
            sns.lineplot(
                data=datos_metric,
                x='generation',
                y=m_col,
                hue='metodo',
                style='metodo',
                ax=ax,
                linewidth=2,
            )

            if m_col in limites_y:
                ax.set_ylim(limites_y[m_col])
            ax.set_xlim(g_min, g_max)
            ax.set_title(titulo, fontsize=12, fontweight='bold')
            ax.set_xlabel('Generación')
            ax.set_ylabel('Valor')
            
            # Quitar leyenda individual de cada subplot para evitar solapamientos
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()

        # Ocultar ejes vacíos si hay menos de 9 métricas
        for j in range(len(metricas), len(axes)):
            axes[j].axis('off')

        # Buscar la primera leyenda disponible para usarla como global
        handles, labels = [], []
        for ax in axes:
            if hasattr(ax, 'get_legend_handles_labels') and ax.get_legend_handles_labels()[0]:
                handles, labels = ax.get_legend_handles_labels()
                break
                
        if handles and labels:
            fig.legend(handles, labels, loc='center right', title='Configuración', 
                       bbox_to_anchor=(1.18, 0.5), fontsize=10, title_fontsize=12)

        fig.suptitle(f'Convergencia Generacional | {algoritmo}', fontsize=16, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        if dir_salida and etiqueta_modo:
            ruta = os.path.join(dir_salida, f'convergencia_{algoritmo.lower()}_{etiqueta_modo}.png')
            fig.savefig(ruta, dpi=dpi, bbox_inches='tight')
            artefactos.append((ruta, f"Convergencia por métrica - {algoritmo}"))
            if emitir_log:
                imprimir_grafico_guardado(ruta, f"Convergencia por métrica - {algoritmo}")
        plt.close(fig)

    return artefactos
