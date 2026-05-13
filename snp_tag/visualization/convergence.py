"""
Módulo de Análisis de Convergencia (convergence.py)
--------------------------------------------------
Visualiza la evolución de las métricas de rendimiento a lo largo de las
generaciones para cada configuración experimental.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import math
from typing import Optional, Dict, List, Tuple

def graficar_evolucion_generacional(df_gen: pd.DataFrame, dir_salida: Optional[str] = None, 
                                   etiqueta_modo: Optional[str] = None, 
                                   figsize=(14, 10), dpi=300,
                                   emitir_log: bool = True) -> List[Tuple[str, str]]:
    """
    Genera una figura por métrica con subplots por algoritmo.
    En cada subplot se muestran únicamente las inicializaciones del algoritmo,
    evitando solapamiento entre algoritmos.
    """
    metricas = [
        # Pareto Front Geometric Metrics
        ('Range', 'Rango (Range): Diversidad Geométrica'),
        ('MinSum', 'MinSum: Convergencia Central'),
        ('SumMin', 'SumMin: Convergencia Marginal'),
        # Domain-Specific (Physical SNP Metrics)
        ('MaxToleranceRate', 'Tasa de Tolerancia Máxima'),
        ('AvgToleranceRate', 'Tasa de Tolerancia Promedio'),
        ('AvgHammingDistance', 'Distancia Hamming Promedio'),
        # Performance Indicators (Convergence & Spread)
        ('Hypervolume', 'Hipervolumen (HV)'),
        ('IGD+', 'Distancia Generacional Invertida Plus (IGD+)'),
        ('GD+', 'Distancia Generacional Plus (GD+)')
    ]

    cols_base = {'algorithm', 'init', 'generation'}
    if df_gen.empty or not cols_base.issubset(df_gen.columns):
        return []

    df_plot = df_gen.copy()
    df_plot = df_plot.dropna(subset=['algorithm', 'init', 'generation'])
    if df_plot.empty:
        return []

    df_plot['generation'] = pd.to_numeric(df_plot['generation'], errors='coerce')
    df_plot = df_plot.dropna(subset=['generation'])
    if df_plot.empty:
        return []

    df_plot['algorithm'] = df_plot['algorithm'].astype(str)
    df_plot['init'] = df_plot['init'].astype(str)

    # Estandarización de ejes Y globales por métrica (misma escala para todas las curvas de la métrica)
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

    algoritmos_preferidos = ['NSGA2', 'NSGA3', 'SPEA2', 'MOEAD_TCHE', 'MOEAD_PBI', 'MOEAD_WS']
    algoritmos_presentes = sorted(df_plot['algorithm'].dropna().unique().tolist())
    algoritmos_ordenados = [a for a in algoritmos_preferidos if a in algoritmos_presentes]
    algoritmos_ordenados.extend([a for a in algoritmos_presentes if a not in algoritmos_ordenados])
    if not algoritmos_ordenados:
        return []

    sns.set_theme(style='whitegrid')
    artefactos = []
    for m_col, titulo in metricas:
        if m_col not in df_plot.columns:
            continue

        sub = df_plot[['generation', 'algorithm', 'init', m_col]].copy()
        sub[m_col] = pd.to_numeric(sub[m_col], errors='coerce')
        sub = sub.dropna(subset=[m_col])
        if sub.empty:
            continue

        n_alg = len(algoritmos_ordenados)
        ncols = 2 if n_alg > 1 else 1
        nrows = int(math.ceil(n_alg / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        axes = axes.ravel() if hasattr(axes, 'ravel') else [axes]

        for idx, algoritmo in enumerate(algoritmos_ordenados):
            ax = axes[idx]
            sub_algo = sub[sub['algorithm'] == algoritmo]
            if sub_algo.empty:
                ax.axis('off')
                continue

            sns.lineplot(
                data=sub_algo,
                x='generation',
                y=m_col,
                hue='init',
                ax=ax,
                estimator='mean',
                errorbar=None,
                linewidth=2,
            )

            if m_col in limites_y:
                ax.set_ylim(limites_y[m_col])
            ax.set_xlim(g_min, g_max)
            ax.set_title(f'{algoritmo}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Generación')
            ax.set_ylabel('Valor')
            ax.legend(title='Inicialización', loc='best')

        for j in range(n_alg, len(axes)):
            axes[j].axis('off')

        fig.suptitle(f'Convergencia Generacional | {titulo}', fontsize=14, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        if dir_salida and etiqueta_modo:
            ruta = os.path.join(dir_salida, f'convergencia_{m_col.lower()}_{etiqueta_modo}.png')
            fig.savefig(ruta, dpi=dpi, bbox_inches='tight')
            artefactos.append((ruta, f"Convergencia por métrica ({m_col}) con subplots por algoritmo"))
            if emitir_log:
                from snp_tag.utils.terminal import imprimir_grafico_guardado
                imprimir_grafico_guardado(ruta, f"Convergencia por métrica ({m_col}) con subplots por algoritmo")
        plt.close(fig)

    return artefactos
