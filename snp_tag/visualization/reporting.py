"""
Módulo de Reportes Sintéticos (reporting.py)
-------------------------------------------
Genera comparaciones estadísticas globales, rankings y resúmenes de rendimiento
entre los diferentes algoritmos e inicializaciones.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import pandas as pd
from typing import List, Tuple
from snp_tag.utils.terminal import imprimir_grafico_guardado

def graficar_rendimiento_tiempo(df_runs: pd.DataFrame, dir_salida: str, etiqueta_modo: str, dpi=300):
    """Visualiza el tiempo de ejecución y su relación con el tamaño del frente."""
    df_plot = df_runs.copy()
    df_plot['config'] = df_plot['algorithm'] + '-' + df_plot['init']
    
    # 1. Boxplot Tiempo
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df_plot, x='config', y='time_seg', hue='config', palette='Set2', legend=False)
    plt.title('Distribución del Tiempo de Ejecución por Configuración')
    plt.xticks(rotation=35, ha='right')
    plt.tight_layout()
    ruta_t = os.path.join(dir_salida, f'boxplot_tiempo_{etiqueta_modo}.png')
    plt.savefig(ruta_t, dpi=dpi)
    imprimir_grafico_guardado(ruta_t, "Boxplot tiempo de ejecución")
    
    # 2. Media +- Std (Barplot)
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_plot, x='config', y='time_seg', hue='config', palette='muted', legend=False, errorbar='sd', capsize=.2)
    plt.title('Media ± Desviación Estándar del Tiempo de Ejecución')
    plt.xticks(rotation=35, ha='right')
    plt.tight_layout()
    ruta_std = os.path.join(dir_salida, f'media_std_tiempo_{etiqueta_modo}.png')
    plt.savefig(ruta_std, dpi=dpi)
    imprimir_grafico_guardado(ruta_std, "Media ± std tiempo ejecución")
    
    # 3. Tiempo vs Tamaño de Frente
    if 'frente_size' in df_plot.columns:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df_plot, x='time_seg', y='frente_size', hue='config', palette='Set2')
        plt.title('Relación entre Tiempo de Ejecución y Tamaño del Frente')
        plt.tight_layout()
        ruta_scatter = os.path.join(dir_salida, f'tiempo_vs_tamano_frente_{etiqueta_modo}.png')
        plt.savefig(ruta_scatter, dpi=dpi)
        imprimir_grafico_guardado(ruta_scatter, "Tiempo vs Tamaño de frente")
        
    plt.close('all')

def graficar_comparativa_objetivos(df_runs: pd.DataFrame, dir_salida: str, etiqueta_modo: str, dpi=300):
    """Genera un heatmap comparativo de las métricas (Réplica Legacy: Rojo=Peor, Verde=Mejor)."""
    # 1. Agregación y preparación
    cols_met = ['Range', 'SumMin', 'MinSum', 'MaxToleranceRate', 'AvgToleranceRate', 'AvgHammingDistance', 'Hypervolume']
    disponibles = [c for c in cols_met if c in df_runs.columns]
    
    resumen = df_runs.groupby(['algorithm', 'init'])[disponibles].mean().reset_index()
    resumen['method'] = resumen['algorithm'] + '-' + resumen['init']
    
    heat_df_plot = resumen.set_index('method')[disponibles].copy()
    heat_norm_better = heat_df_plot.copy()
    
    # 2. Lógica de Normalización de Calidad (Legacy)
    higher_is_better = ['Hypervolume', 'Range', 'MaxToleranceRate', 'AvgToleranceRate', 'AvgHammingDistance']
    lower_is_better = ['SumMin', 'MinSum']
    
    for col in disponibles:
        c_min, c_max = heat_df_plot[col].min(), heat_df_plot[col].max()
        c_range = c_max - c_min
        if c_range == 0:
            heat_norm_better[col] = 1.0
        elif any(k in col for k in higher_is_better):
            heat_norm_better[col] = (heat_df_plot[col] - c_min) / c_range
        else:
            # Para métricas donde menos es mejor, invertimos el rango [0, 1]
            heat_norm_better[col] = (c_max - heat_df_plot[col]) / c_range
    
    # 3. Graficado exacto
    plt.figure(figsize=(14, 8))
    sns.heatmap(heat_norm_better, annot=heat_df_plot, fmt='.3f', cmap='RdYlGn', linewidths=0.5)
    plt.title('Comparativa de Algoritmos: Rojo (Peor) vs Verde (Mejor)', fontsize=15, pad=20)
    plt.tight_layout()
    
    ruta_h = os.path.join(dir_salida, f"heatmap_comparativa_{etiqueta_modo}.png")
    plt.savefig(ruta_h, dpi=dpi, bbox_inches='tight')
    imprimir_grafico_guardado(ruta_h, "Mapa de calor comparativo (Benchmark)")
    plt.close()

def graficar_violin_metricas(df_runs: pd.DataFrame, dir_salida: str, etiqueta_modo: str,
                             dpi=300, emitir_log: bool = True) -> List[Tuple[str, str]]:
    """Genera panel general y diagramas de violín individuales para las métricas primarias."""
    metricas = ['Range', 'SumMin', 'MinSum', 'MaxToleranceRate', 'AvgToleranceRate', 'AvgHammingDistance', 'Hypervolume']
    disponibles = [m for m in metricas if m in df_runs.columns]
    if not disponibles:
        return []
    df_plot = df_runs.copy()
    df_plot['config'] = df_plot['algorithm'] + '-' + df_plot['init']
    artefactos = []
    
    # 1. Panel General
    ncols = 3
    nrows = int(np.ceil(len(disponibles) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for i, m in enumerate(disponibles):
        ax = axes[i]
        sns.violinplot(data=df_plot, x='config', y=m, ax=ax, inner="quart", palette="Pastel1", hue='config', legend=False)
        ax.set_title(f'Detalle Violin: {m}', fontsize=12, fontweight='bold')
        ax.tick_params(axis='x', rotation=35)
    for j in range(i + 1, len(axes)): axes[j].axis('off')
    fig.tight_layout()
    ruta_p = os.path.join(dir_salida, f'violin_panel_metricas_finales_{etiqueta_modo}.png')
    fig.savefig(ruta_p, dpi=dpi)
    artefactos.append((ruta_p, "Panel de Distribución Violin"))
    if emitir_log:
        imprimir_grafico_guardado(ruta_p, "Panel de Distribución Violin")

    # 2. Individuales
    for m in disponibles:
        plt.figure(figsize=(12, 6))
        sns.violinplot(data=df_plot, x='config', y=m, inner="quart", hue='config', palette="Pastel1", legend=False)
        sns.stripplot(data=df_plot, x='config', y=m, color="black", alpha=0.3, size=3)
        plt.title(f'Distribución Detallada: {m} (Violin Plot)')
        plt.xticks(rotation=35, ha='right')
        plt.tight_layout()
        ruta_i = os.path.join(dir_salida, f'violin_metricas_{m}_{etiqueta_modo}.png')
        plt.savefig(ruta_i, dpi=dpi)
        artefactos.append((ruta_i, f"Distribución {m} (Violin)"))
        if emitir_log:
            imprimir_grafico_guardado(ruta_i, f"Distribución {m} (Violin)")
    plt.close('all')
    return artefactos

def graficar_media_std_metricas(df_runs: pd.DataFrame, dir_salida: str, etiqueta_modo: str,
                                dpi=300, emitir_log: bool = True) -> List[Tuple[str, str]]:
    """Genera panel general y pointplots (media +- std) individuales."""
    metricas = ['Range', 'SumMin', 'MinSum', 'MaxToleranceRate', 'AvgToleranceRate', 'AvgHammingDistance', 'Hypervolume']
    disponibles = [m for m in metricas if m in df_runs.columns]
    if not disponibles:
        return []
    df_plot = df_runs.copy()
    df_plot['config'] = df_plot['algorithm'] + '-' + df_plot['init']
    artefactos = []

    # 1. Panel General (Barplots)
    ncols = 3
    nrows = int(np.ceil(len(disponibles) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for i, m in enumerate(disponibles):
        ax = axes[i]
        sns.barplot(data=df_plot, x='config', y=m, ax=ax, hue='config', palette='muted', legend=False, errorbar='sd', capsize=.2)
        ax.set_title(f'Media ± Std: {m}', fontsize=12, fontweight='bold')
        ax.tick_params(axis='x', rotation=35)
    for j in range(i + 1, len(axes)): axes[j].axis('off')
    fig.tight_layout()
    ruta_p = os.path.join(dir_salida, f'media_std_panel_metricas_finales_{etiqueta_modo}.png')
    fig.savefig(ruta_p, dpi=dpi)
    artefactos.append((ruta_p, "Panel de Tendencia Central (Media ± Std)"))
    if emitir_log:
        imprimir_grafico_guardado(ruta_p, "Panel de Tendencia Central (Media ± Std)")
    
    # 2. Individuales (Barplots)
    for m in disponibles:
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df_plot, x='config', y=m, hue='config', palette='muted', legend=False, errorbar='sd', capsize=.2)
        plt.title(f'Media ± Desviación Estándar de {m}')
        plt.xticks(rotation=35, ha='right')
        plt.tight_layout()
        ruta_i = os.path.join(dir_salida, f'media_std_metricas_{m}_{etiqueta_modo}.png')
        plt.savefig(ruta_i, dpi=dpi)
        artefactos.append((ruta_i, f"Media ± std {m}"))
        if emitir_log:
            imprimir_grafico_guardado(ruta_i, f"Media ± std {m}")
    plt.close('all')
    return artefactos

def graficar_boxplot_metricas(df_runs: pd.DataFrame, dir_salida: str, etiqueta_modo: str,
                              dpi=300, emitir_log: bool = True) -> List[Tuple[str, str]]:
    """Genera panel general y diagramas de caja individuales para las métricas finales."""
    metricas = ['Range', 'SumMin', 'MinSum', 'MaxToleranceRate', 'AvgToleranceRate', 'AvgHammingDistance', 'Hypervolume']
    disponibles = [m for m in metricas if m in df_runs.columns]
    if not disponibles:
        return []
    
    df_plot = df_runs.copy()
    df_plot['config'] = df_plot['algorithm'] + '-' + df_plot['init']
    artefactos = []
    
    # 1. Panel General
    ncols = 3
    nrows = int(np.ceil(len(disponibles) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for i, m in enumerate(disponibles):
        ax = axes[i]
        sns.boxplot(data=df_plot, x='config', y=m, ax=ax, palette='Set3', hue='config', legend=False)
        sns.stripplot(data=df_plot, x='config', y=m, ax=ax, color='black', alpha=0.3, size=4)
        ax.set_title(f'Distribución de {m}', fontsize=12, fontweight='bold')
        ax.tick_params(axis='x', rotation=35)
    for j in range(i + 1, len(axes)): axes[j].axis('off')
    fig.tight_layout()
    ruta_p = os.path.join(dir_salida, f'boxplots_metricas_finales_{etiqueta_modo}.png')
    fig.savefig(ruta_p, dpi=dpi, bbox_inches='tight')
    artefactos.append((ruta_p, "Panel de Boxplots comparativos"))
    if emitir_log:
        imprimir_grafico_guardado(ruta_p, "Panel de Boxplots comparativos")
    
    # 2. Individuales
    for m in disponibles:
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df_plot, x='config', y=m, palette='Set3', hue='config', legend=False)
        plt.title(f'Distribución de {m} (Boxplot)')
        plt.xticks(rotation=35, ha='right')
        plt.tight_layout()
        ruta_i = os.path.join(dir_salida, f'boxplot_metricas_{m}_{etiqueta_modo}.png')
        plt.savefig(ruta_i, dpi=dpi)
        artefactos.append((ruta_i, f"Distribución {m} (Boxplot)"))
        if emitir_log:
            imprimir_grafico_guardado(ruta_i, f"Distribución {m} (Boxplot)")
    plt.close('all')
    return artefactos

def graficar_ranking_global(df_runs: pd.DataFrame, dir_salida: str, etiqueta_modo: str, dpi=300):
    """
    Calcula y visualiza el ranking acumulado de los métodos basado en métricas clave.
    """
    if df_runs.empty: return
    
    df_plot = df_runs.copy()
    df_plot['method'] = df_plot['algorithm'] + '-' + df_plot['init']
    
    # Agrupar por media
    resumen = df_plot.groupby('method')[['Range', 'SumMin', 'MinSum']].mean().reset_index()
    
    for m in ['Range', 'SumMin', 'MinSum']:
        asce = True if m in ['SumMin', 'MinSum'] else False
        resumen[f'{m}_rank'] = resumen[m].rank(ascending=asce)
        
    resumen['Puntuación Total'] = resumen[[c for c in resumen.columns if '_rank' in c]].sum(axis=1)
    resumen = resumen.sort_values('Puntuación Total')
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=resumen, x='method', y='Puntuación Total', palette='viridis', hue='method', legend=False)
    plt.xticks(rotation=35, ha='right')
    plt.title('Ranking Global por Suma de Posiciones (Métricas de Convergencia)')
    plt.tight_layout()
    
    ruta = os.path.join(dir_salida, f'ranking_global_{etiqueta_modo}.png')
    plt.savefig(ruta, dpi=dpi)
    imprimir_grafico_guardado(ruta, "Gráfico de Ranking Global")
    plt.close()
