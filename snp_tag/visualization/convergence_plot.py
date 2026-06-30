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
from snp_tag.utils.ai_exports import exportar_gemelo_ia_csv
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
                
            datos_metric = sub_algo[['generation', 'init', 'crossover', m_col]].copy()
            datos_metric[m_col] = pd.to_numeric(datos_metric[m_col], errors='coerce')
            datos_metric = datos_metric.dropna(subset=[m_col])
            
            if datos_metric.empty:
                ax.axis('off')
                continue
                
            INIT_COLORS = {'greedy_multi': '#1f77b4', 'greedy_ting': '#d62728', 'random_dense': '#2ca02c'}
            CROSS_STYLES = {'1P': '-', '2P': '--', 'UX': ':', 'HUX': '-.'}
            CROSS_MARKERS = {'1P': 'o', '2P': 's', 'UX': '^', 'HUX': 'D'}
            
            for init_name, color in INIT_COLORS.items():
                for cross_idx, (cross_name, style) in enumerate(CROSS_STYLES.items()):
                    subset = datos_metric[(datos_metric['init'] == init_name) & (datos_metric['crossover'] == cross_name)]
                    if subset.empty:
                        continue
                    subset = subset.sort_values('generation')
                    
                    # Calcular el markevery desfasado para evitar solapamientos
                    step = max(1, len(subset) // 10)
                    offset = int((cross_idx / len(CROSS_STYLES)) * step)
                    
                    ax.plot(
                        subset['generation'], subset[m_col],
                        color=color, linestyle=style,
                        marker=CROSS_MARKERS.get(cross_name, 'o'),
                        markevery=(offset, step),
                        markersize=5,
                        linewidth=2.5, alpha=0.85
                    )

            if m_col in limites_y:
                ax.set_ylim(limites_y[m_col])
            ax.set_xlim(g_min, g_max)
            ax.set_title(titulo, fontsize=12, fontweight='bold')
            ax.set_xlabel('Generación')
            ax.set_ylabel('Valor')
            
        # Ocultar ejes vacíos si hay menos de 9 métricas
        for j in range(len(metricas), len(axes)):
            axes[j].axis('off')

        from matplotlib.lines import Line2D
        INIT_COLORS = {'greedy_multi': '#1f77b4', 'greedy_ting': '#d62728', 'random_dense': '#2ca02c'}
        CROSS_STYLES = {'1P': '-', '2P': '--', 'UX': ':', 'HUX': '-.'}
        CROSS_MARKERS = {'1P': 'o', '2P': 's', 'UX': '^', 'HUX': 'D'}
        
        legend_elements_init = [
            Line2D([0], [0], color=c, lw=4, label=i) 
            for i, c in INIT_COLORS.items() if i in sub_algo['init'].unique()
        ]
        legend_elements_cross = [
            Line2D([0], [0], color='gray', linestyle=s, marker=CROSS_MARKERS.get(c, 'o'), lw=2, markersize=7, label=c) 
            for c, s in CROSS_STYLES.items() if c in sub_algo['crossover'].unique()
        ]
        
        if legend_elements_init:
            fig.legend(handles=legend_elements_init, title='Inicialización', loc='center right', bbox_to_anchor=(1.15, 0.6), fontsize=10, title_fontsize=12)
        if legend_elements_cross:
            fig.legend(handles=legend_elements_cross, title='Cruce', loc='center right', bbox_to_anchor=(1.15, 0.4), fontsize=10, title_fontsize=12)

        fig.suptitle(f'Convergencia Generacional | {algoritmo}', fontsize=16, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        if dir_salida and etiqueta_modo:
            algoritmo_safe = algoritmo.lower().replace('/', '-')
            ruta = os.path.join(dir_salida, f'convergencia_{algoritmo_safe}_{etiqueta_modo}.png')
            fig.savefig(ruta, dpi=dpi, bbox_inches='tight')
            exportar_gemelo_ia_csv(ruta, df_datos=sub_algo[['generation', 'metodo'] + [m for m, _ in metricas if m in sub_algo.columns]])
            artefactos.append((ruta, f"Convergencia por métrica - {algoritmo}"))
            if emitir_log:
                imprimir_grafico_guardado(ruta, f"Convergencia por métrica - {algoritmo}")
        plt.close(fig)

    return artefactos

def graficar_convergencia_hipervolumen(df_gen: pd.DataFrame, dir_salida: Optional[str] = None, 
                                     etiqueta_modo: Optional[str] = None, 
                                     figsize: Tuple[int, int] = (10, 6), dpi: int = 300,
                                     emitir_log: bool = True) -> List[Tuple[str, str]]:
    """
    Genera una figura individual mostrando únicamente la evolución del Hipervolumen 
    para cada algoritmo.
    """
    if 'Hypervolume' not in df_gen.columns:
        return []

    df_plot = df_gen.copy()
    df_plot = df_plot.dropna(subset=['algorithm', 'init', 'crossover', 'generation', 'Hypervolume'])
    if df_plot.empty:
        return []

    df_plot['generation'] = pd.to_numeric(df_plot['generation'], errors='coerce')
    df_plot['Hypervolume'] = pd.to_numeric(df_plot['Hypervolume'], errors='coerce')
    df_plot = df_plot.dropna(subset=['generation', 'Hypervolume'])

    g_min, g_max = float(df_plot['generation'].min()), float(df_plot['generation'].max())
    y_min, y_max = float(df_plot['Hypervolume'].min()), float(df_plot['Hypervolume'].max())
    pad = (y_max - y_min) * 0.05 if y_max > y_min else 1.0

    algoritmos_presentes = sorted(df_plot['algorithm'].dropna().unique().tolist())
    algoritmos_ordenados = [a for a in PREFERRED_ALGORITHMS_ORDER if a in algoritmos_presentes]
    algoritmos_ordenados.extend([a for a in algoritmos_presentes if a not in algoritmos_ordenados])

    sns.set_theme(style='whitegrid')
    artefactos = []
    
    INIT_COLORS = {'greedy_multi': '#1f77b4', 'greedy_ting': '#d62728', 'random_dense': '#2ca02c'}
    CROSS_STYLES = {'1P': '-', '2P': '--', 'UX': ':', 'HUX': '-.'}
    CROSS_MARKERS = {'1P': 'o', '2P': 's', 'UX': '^', 'HUX': 'D'}

    for algoritmo in algoritmos_ordenados:
        sub_algo = df_plot[df_plot['algorithm'] == algoritmo].copy()
        if sub_algo.empty:
            continue
            
        fig, ax = plt.subplots(figsize=figsize)
        
        for init_name, color in INIT_COLORS.items():
            for cross_idx, (cross_name, style) in enumerate(CROSS_STYLES.items()):
                subset = sub_algo[(sub_algo['init'] == init_name) & (sub_algo['crossover'] == cross_name)]
                if subset.empty:
                    continue
                subset = subset.sort_values('generation')
                
                # Calcular el markevery desfasado para evitar solapamientos
                step = max(1, len(subset) // 10)
                offset = int((cross_idx / len(CROSS_STYLES)) * step)
                
                ax.plot(
                    subset['generation'], subset['Hypervolume'],
                    color=color, linestyle=style, 
                    marker=CROSS_MARKERS.get(cross_name, 'o'),
                    markevery=(offset, step),
                    markersize=6,
                    linewidth=2.5, alpha=0.85
                )

        ax.set_ylim(y_min - pad, y_max + pad)
        ax.set_xlim(g_min, g_max)
        ax.set_title(f'Evolución del Hipervolumen - {algoritmo}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Generación', fontsize=12)
        ax.set_ylabel('Hipervolumen', fontsize=12)

        from matplotlib.lines import Line2D
        legend_elements_init = [
            Line2D([0], [0], color=c, lw=4, label=i) 
            for i, c in INIT_COLORS.items() if i in sub_algo['init'].unique()
        ]
        legend_elements_cross = [
            Line2D([0], [0], color='gray', linestyle=s, marker=CROSS_MARKERS.get(c, 'o'), lw=2, markersize=8, label=c) 
            for c, s in CROSS_STYLES.items() if c in sub_algo['crossover'].unique()
        ]
        
        if legend_elements_init:
            leg1 = fig.legend(handles=legend_elements_init, title='Inicialización', loc='center left', bbox_to_anchor=(1.02, 0.6), fontsize=10, title_fontsize=12)
        if legend_elements_cross:
            fig.legend(handles=legend_elements_cross, title='Cruce', loc='center left', bbox_to_anchor=(1.02, 0.4), fontsize=10, title_fontsize=12)

        fig.tight_layout(rect=[0, 0, 0.9, 1])

        if dir_salida and etiqueta_modo:
            algoritmo_safe = algoritmo.lower().replace('/', '-')
            ruta = os.path.join(dir_salida, f'convergencia_hv_{algoritmo_safe}_{etiqueta_modo}.png')
            fig.savefig(ruta, dpi=dpi, bbox_inches='tight')
            artefactos.append((ruta, f"Convergencia Hipervolumen - {algoritmo}"))
            if emitir_log:
                imprimir_grafico_guardado(ruta, f"Convergencia Hipervolumen - {algoritmo}")
        plt.close(fig)

    return artefactos
