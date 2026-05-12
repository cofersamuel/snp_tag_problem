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
from matplotlib.colors import ListedColormap, BoundaryNorm
from snp_tag.utils.terminal import imprimir_grafico_guardado, imprimir_subseccion

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
    ax = sns.heatmap(heat_norm_better, annot=heat_df_plot, fmt='.3f', cmap='RdYlGn', linewidths=0.5)
    
    # Personalizar la barra de color (leyenda)
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['Peor', 'Mejor'])
    
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


def graficar_analisis_estadistico(df_runs, dir_salida, etiqueta_modo, col_group='config', dpi=300, indent=9):
    """
    Realiza el análisis estadístico de Friedman + Nemenyi para una dimensión específica.
    Retorna el ranking promedio.
    """
    if df_runs.empty: return None
    
    import scipy.stats as ss
    try:
        import scikit_posthocs as sp
    except ImportError:
        return None

    espacios = " " * indent
    df_plot = df_runs.copy()
    if 'config' not in df_plot.columns:
        df_plot['config'] = df_plot['algorithm'] + '-' + df_plot['init']
    
    # 1. Preparar datos
    metricas = ['Hypervolume', 'Range', 'MinSum', 'SumMin', 'MaxToleranceRate', 'AvgToleranceRate', 'AvgHammingDistance']
    higher_is_better = ['Hypervolume', 'Range', 'MaxToleranceRate', 'AvgToleranceRate', 'AvgHammingDistance']
    disponibles = [m for m in metricas if m in df_plot.columns]
    
    if not disponibles: return None
    
    titulos = {
        'config': 'Método (Algoritmo + Inicialización)',
        'algorithm': 'Algoritmo',
        'init': 'Inicialización'
    }
    titulo = titulos.get(col_group, col_group)
    
    # Media por grupo
    resumen = df_plot.groupby(col_group)[disponibles].mean().reset_index()
    
    rank_matrix = []
    for m in disponibles:
        asce = False if m in higher_is_better else True
        rank_matrix.append(resumen[m].rank(ascending=asce).values)
    rank_matrix = np.array(rank_matrix) # (metrics, groups)
    
    # 2. Test de Friedman
    stat, p_value = ss.friedmanchisquare(*rank_matrix.T)
    
    # Encabezado manual con sangría
    sub_line = "─" * (len(titulo) + 24)
    print(f"\n{espacios}📊  \033[1mTEST DE FRIEDMAN ({titulo})\033[0m")
    print(f"{espacios}{sub_line}")
    print(f"{espacios}    Estadístico: {stat:.4f}")
    print(f"{espacios}    P-valor: {p_value:.4e}")
    print(f"{espacios}    Significativo (p < 0.05): {'Sí' if p_value < 0.05 else 'No'}\n")
    
    # 3. Gráfico de Rangos Promedio (Siempre se genera)
    avg_ranks = np.mean(rank_matrix, axis=0)
    resumen['AvgRank'] = avg_ranks
    resumen_sort = resumen.sort_values('AvgRank')
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=resumen_sort, x=col_group, y='AvgRank', palette='viridis', hue=col_group, legend=False)
    plt.xticks(rotation=45, ha='right')
    plt.title(f'Ranking Promedio - {titulo} (Menor es Mejor)')
    plt.ylabel('Rango Promedio')
    plt.xlabel(titulo)
    plt.tight_layout()
    
    # Organizar directorios
    sufijo = f"_{col_group}" if col_group != 'config' else ""
    nombre_barras = f'rangos_promedio{sufijo}_{etiqueta_modo}.png'
    ruta_barras = os.path.join(dir_salida, nombre_barras)
    plt.savefig(ruta_barras, dpi=dpi)
    
    print(f"{espacios}    ", end="")
    imprimir_grafico_guardado(ruta_barras, f"Gráfico de Rangos Promedio ({titulo})")
    plt.close()

    # 4. Post-hoc de Nemenyi
    if p_value < 0.05:
        p_values_nemenyi = sp.posthoc_nemenyi_friedman(rank_matrix)
        p_values_nemenyi.columns = resumen[col_group]
        p_values_nemenyi.index = resumen[col_group]
        
        plt.figure(figsize=(14, 12))
        mask = np.triu(np.ones_like(p_values_nemenyi, dtype=bool))
        cmap = ListedColormap(['#228B22', '#90EE90', '#FFC1C1'])
        norm = BoundaryNorm([0, 0.01, 0.05, 1.0], cmap.N)
        
        ax = sns.heatmap(p_values_nemenyi, mask=mask, annot=False, cmap=cmap, norm=norm, 
                         linewidths=0.5, linecolor='white',
                         cbar_kws={"ticks": [0.005, 0.03, 0.5], "label": "Nivel de Significancia"})
        
        cbar = ax.collections[0].colorbar
        cbar.set_ticklabels(['p < 0.01', '0.01 ≤ p < 0.05', 'NS (p ≥ 0.05)'])
        
        plt.xticks(rotation=45, ha='right')
        plt.suptitle(f'Diagrama Nemenyi - {titulo}', fontsize=16, y=0.98)
        plt.tight_layout()
        
        nombre_heatmap = f'heatmap_nemenyi{sufijo}_{etiqueta_modo}.png'
        ruta_heatmap = os.path.join(dir_salida, nombre_heatmap)
        plt.savefig(ruta_heatmap, dpi=dpi, bbox_inches='tight')
        
        print(f"{espacios}    ", end="")
        imprimir_grafico_guardado(ruta_heatmap, f"Heatmap de Significancia Nemenyi ({titulo})")
        plt.close()
    else:
        print(f"{espacios}    • ⚠️  Test de Friedman no significativo para {titulo}. Omitiendo post-hoc.")

    return resumen[[col_group, 'AvgRank']]


def graficar_analisis_kruskal_dunn(df_runs, dir_salida, metrica_objetivo, etiqueta_modo, dpi=300, indent=9):
    """
    Realiza Kruskal-Wallis + Dunn para una métrica específica.
    """
    if df_runs.empty or metrica_objetivo not in df_runs.columns: return
    
    import scipy.stats as ss
    try:
        import scikit_posthocs as sp
    except ImportError:
        return

    espacios = " " * indent
    df_plot = df_runs.copy()
    if 'config' not in df_plot.columns:
        df_plot['config'] = df_plot['algorithm'] + '-' + df_plot['init']
    
    # 1. Agrupar datos
    grupos = []
    nombres = sorted(df_plot['config'].unique())
    for n in nombres:
        grupos.append(df_plot[df_plot['config'] == n][metrica_objetivo].values)
    
    # 2. Kruskal-Wallis
    stat, p_val = ss.kruskal(*grupos)
    
    sub_line = "─" * (len(metrica_objetivo) + 28)
    print(f"\n{espacios}📊  \033[1mVALIDACIÓN ESTADÍSTICA ({metrica_objetivo})\033[0m")
    print(f"{espacios}{sub_line}")
    print(f"{espacios}    Estadístico H (Kruskal-Wallis): {stat:.4f}")
    print(f"{espacios}    P-valor: {p_val:.4e}")
    print(f"{espacios}    Significativo (p < 0.05): {'Sí' if p_val < 0.05 else 'No'}\n")
    
    if p_val < 0.05:
        p_dunn = sp.posthoc_dunn(df_plot, val_col=metrica_objetivo, group_col='config', p_adjust='bonferroni')
        
        plt.figure(figsize=(14, 12))
        mask = np.triu(np.ones_like(p_dunn, dtype=bool))
        cmap = ListedColormap(['#228B22', '#90EE90', '#FFC1C1'])
        norm = BoundaryNorm([0, 0.01, 0.05, 1.0], cmap.N)
        
        sns.heatmap(p_dunn, mask=mask, annot=False, cmap=cmap, norm=norm, 
                    linewidths=0.5, linecolor='white',
                    cbar_kws={"ticks": [0.005, 0.03, 0.5], "label": "Nivel de Significancia"})
        
        plt.xticks(rotation=45, ha='right')
        plt.suptitle(f'Post-hoc de Dunn: {metrica_objetivo}', fontsize=16, y=0.98)
        plt.tight_layout()
        
        ruta_h = os.path.join(dir_salida, f"heatmap_dunn_{metrica_objetivo.lower()}_{etiqueta_modo}.png")
        plt.savefig(ruta_h, dpi=dpi, bbox_inches='tight')
        
        print(f"{espacios}    ", end="")
        imprimir_grafico_guardado(ruta_h, f"Heatmap de Comparaciones Dunn (Post-hoc)")
        plt.close()


def graficar_diagrama_diferencia_critica(df_runs, dir_salida, metrica_objetivo, etiqueta_modo, dpi=300, indent=9):
    """
    Genera un diagrama de Diferencia Crítica (CD) para una métrica específica.
    """
    if metrica_objetivo not in df_runs.columns: return
    
    import scipy.stats as ss
    espacios = " " * indent
    df_plot = df_runs.copy()
    if 'config' not in df_plot.columns:
        df_plot['config'] = df_plot['algorithm'] + '-' + df_plot['init']
    
    # 1. Calcular rangos por réplica
    configs = sorted(df_plot['config'].unique())
    replicas = sorted(df_plot['run'].unique())
    higher_is_better = ['Hypervolume', 'Range', 'MaxToleranceRate', 'AvgToleranceRate', 'AvgHammingDistance']
    asce = False if metrica_objetivo in higher_is_better else True
    
    ranks = []
    for r in replicas:
        slice_r = df_plot[df_plot['run'] == r].set_index('config')[metrica_objetivo]
        if len(slice_r) == len(configs):
            ranks.append(slice_r.rank(ascending=asce).values)
    
    if not ranks: return
    
    avg_ranks = np.mean(ranks, axis=0)
    n_configs = len(configs)
    n_replicas = len(replicas)
    
    q_alpha = 3.2 
    cd = q_alpha * np.sqrt((n_configs * (n_configs + 1)) / (6 * n_replicas))
    
    plt.figure(figsize=(12, n_configs * 0.4 + 2))
    order = np.argsort(avg_ranks)
    sorted_ranks = avg_ranks[order]
    sorted_labels = [configs[i] for i in order]
    
    plt.hlines(y=range(len(sorted_labels)), xmin=1, xmax=n_configs, colors='gray', linestyles='dotted', alpha=0.3)
    plt.plot(sorted_ranks, range(len(sorted_labels)), 'ro', markersize=8)
    
    for i, (r, label) in enumerate(zip(sorted_ranks, sorted_labels)):
        plt.text(r, i + 0.2, f"{r:.2f}", ha='center', fontsize=9)
        plt.text(0.8, i, label, ha='right', va='center', fontweight='bold')
    
    for i in range(len(sorted_ranks)):
        for j in range(i + 1, len(sorted_ranks)):
            if (sorted_ranks[j] - sorted_ranks[i]) <= cd:
                plt.plot([sorted_ranks[i], sorted_ranks[j]], [i - 0.1, i - 0.1], color='blue', linewidth=3, alpha=0.5)

    plt.title(f'Diagrama CD - {metrica_objetivo}', fontsize=14, pad=20)
    plt.xlabel('Rango Promedio')
    plt.xlim(0.5, n_configs + 0.5)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    ruta_cd = os.path.join(dir_salida, f"diagrama_cd_{metrica_objetivo.lower()}_{etiqueta_modo}.png")
    plt.savefig(ruta_cd, dpi=dpi, bbox_inches='tight')
    
    print(f"{espacios}    ", end="")
    imprimir_grafico_guardado(ruta_cd, f"Diagrama de Diferencia Crítica (CD)")
    plt.close()
