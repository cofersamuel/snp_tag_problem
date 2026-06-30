"""
Módulo de Reportes Sintéticos (reporting.py)
-------------------------------------------
Genera comparaciones estadísticas globales, rankings y resúmenes de rendimiento
entre los diferentes algoritmos e inicializaciones.
"""

# =============================================================================
# LIBRERÍAS ESTÁNDAR
# =============================================================================
import os
from typing import Any, List, Optional, Tuple

# =============================================================================
# LIBRERÍAS DE TERCEROS
# =============================================================================
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import BoundaryNorm, ListedColormap

# =============================================================================
# MÓDULOS LOCALES (snp_tag)
# =============================================================================
from snp_tag.engine.stats_logic import (compute_kruskal_dunn)
from snp_tag.utils.ai_exports import exportar_gemelo_ia_csv
from snp_tag.utils.terminal import (imprimir_grafico_guardado,
                                    imprimir_subseccion)


def graficar_rendimiento_tiempo(df_runs: pd.DataFrame, dir_salida: str, etiqueta_modo: str, dpi: int = 300) -> None:
    """
    Visualiza el tiempo de ejecución promedio por cada configuración algorítmica.

    Parámetros:
    -----------
    df_runs : pd.DataFrame
        Dataset con los tiempos de ejecución de cada iteración independiente.
    dir_salida : str
        Directorio destino para exportar la figura.
    etiqueta_modo : str
        Sufijo identificador del experimento.
    dpi : int
        Resolución de la imagen generada.
    """
    df_plot = df_runs.copy()
    df_plot['config'] = df_plot['algorithm'] + '-' + df_plot['init'] + '-' + df_plot['crossover']
    n_configs = df_plot['config'].nunique()
    ancho_dinamico = max(12, n_configs * 0.35)
    

    
    # 2. Media +- Std (Barplot)
    plt.figure(figsize=(ancho_dinamico, 6))
    sns.barplot(data=df_plot, x='config', y='time_seg', hue='config', palette='muted', legend=False, errorbar='sd', capsize=.2)
    plt.title('Media ± Desviación Estándar del Tiempo de Ejecución')
    plt.xticks(rotation=35, ha='right', rotation_mode='anchor')
    plt.tight_layout()
    ruta_std = os.path.join(dir_salida, f'media_std_tiempo_{etiqueta_modo}.png')
    plt.savefig(ruta_std, dpi=dpi, bbox_inches='tight')
    exportar_gemelo_ia_csv(ruta_std, df_datos=df_plot[['config', 'time_seg']])
    imprimir_grafico_guardado(ruta_std, "Media ± std tiempo ejecución")
    

        
    plt.close('all')

def graficar_comparativa_objetivos(df_runs: pd.DataFrame, dir_salida: str, etiqueta_modo: str, dpi: int = 300) -> None:
    """
    Genera un heatmap comparativo de rendimiento escalado entre todas las métricas.

    Parámetros:
    -----------
    df_runs : pd.DataFrame
        Dataset consolidado con las métricas finales.
    dir_salida : str
        Ruta de exportación.
    etiqueta_modo : str
        Sufijo identificador.
    dpi : int
        Calidad de imagen.
    """
    # 1. Agregación y preparación
    cols_met = ['Range', 'SumMin', 'MinSum', 'MaxToleranceRate', 'AvgToleranceRate', 'AvgHammingDistance', 'Hypervolume', 'IGD+', 'GD+']
    disponibles = [c for c in cols_met if c in df_runs.columns]
    
    resumen = df_runs.groupby(['algorithm', 'init', 'crossover'])[disponibles].mean().reset_index()
    resumen['method'] = resumen['algorithm'] + '-' + resumen['init'] + '-' + resumen['crossover']
    
    heat_df_plot = resumen.set_index('method')[disponibles].copy()
    heat_norm_better = heat_df_plot.copy()
    
    # 2. Lógica de Normalización de Calidad (Legacy)
    higher_is_better = ['Hypervolume', 'Range', 'MaxToleranceRate', 'AvgToleranceRate', 'AvgHammingDistance']
    lower_is_better = ['SumMin', 'MinSum', 'IGD+', 'GD+']
    
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
    
    n_configs = len(heat_df_plot)
    alto_dinamico = max(8.0, n_configs * 0.35)
    ancho_dinamico = max(14.0, len(disponibles) * 0.8)

    # 3. Graficado exacto
    plt.figure(figsize=(ancho_dinamico, alto_dinamico))
    ax = sns.heatmap(heat_norm_better, annot=heat_df_plot, fmt='.3f', cmap='RdYlGn', linewidths=0.5)
    
    # Personalizar la barra de color (leyenda)
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['Peor', 'Mejor'])
    
    plt.title('Comparativa de Algoritmos: Rojo (Peor) vs Verde (Mejor)', fontsize=15, pad=20)
    plt.tight_layout()
    
    ruta_h = os.path.join(dir_salida, f"heatmap_comparativa_{etiqueta_modo}.png")
    plt.savefig(ruta_h, dpi=dpi, bbox_inches='tight')
    exportar_gemelo_ia_csv(ruta_h, df_datos=heat_df_plot.reset_index())
    imprimir_grafico_guardado(ruta_h, "Mapa de calor comparativo (Benchmark)")
    plt.close()

def graficar_violin_metricas(df_runs: pd.DataFrame, dir_salida: str, etiqueta_modo: str,
                             dpi: int = 300, emitir_log: bool = True) -> List[Tuple[str, str]]:
    """
    Genera diagramas de violín para representar las distribuciones de rendimiento.

    Parámetros:
    -----------
    df_runs : pd.DataFrame
        Dataset con métricas finales.
    dir_salida : str
        Directorio base de guardado.
    etiqueta_modo : str
        Sufijo del archivo.
    dpi : int
        Resolución del gráfico.
    emitir_log : bool
        Indica si se reporta la generación en la consola.

    Retorna:
    --------
    List[Tuple[str, str]]
        Rutas y descripciones de las imágenes exportadas.
    """
    metricas = ['Range', 'SumMin', 'MinSum', 'MaxToleranceRate', 'AvgToleranceRate', 'AvgHammingDistance', 'Hypervolume', 'IGD+', 'GD+']
    disponibles = [m for m in metricas if m in df_runs.columns]
    if not disponibles:
        return []
    df_plot = df_runs.copy()
    df_plot['config'] = df_plot['algorithm'] + '-' + df_plot['init'] + '-' + df_plot['crossover']
    n_configs = df_plot['config'].nunique()
    ancho_dinamico = max(12.0, n_configs * 0.35)
    artefactos = []
    
    # 2. Individuales
    for m in disponibles:
        plt.figure(figsize=(ancho_dinamico, 6))
        sns.violinplot(data=df_plot, x='config', y=m, inner="quart", hue='config', palette="Pastel1", legend=False)
        sns.stripplot(data=df_plot, x='config', y=m, color="black", alpha=0.3, size=3)
        plt.title(f'Distribución Detallada: {m} (Violin Plot)')
        plt.xticks(rotation=35, ha='right', rotation_mode='anchor')
        plt.tight_layout()
        ruta_i = os.path.join(dir_salida, f'violin_metricas_{m}_{etiqueta_modo}.png')
        plt.savefig(ruta_i, dpi=dpi, bbox_inches='tight')
        exportar_gemelo_ia_csv(ruta_i, df_datos=df_plot[['config', m]])
        artefactos.append((ruta_i, f"Distribución {m} (Violin)"))
        if emitir_log:
            imprimir_grafico_guardado(ruta_i, f"Distribución {m} (Violin)")
    plt.close('all')
    return artefactos

def graficar_media_std_metricas(df_runs: pd.DataFrame, dir_salida: str, etiqueta_modo: str,
                                dpi: int = 300, emitir_log: bool = True) -> List[Tuple[str, str]]:
    """
    Genera diagramas de barras con intervalos de confianza de desviación estándar.

    Parámetros:
    -----------
    df_runs : pd.DataFrame
        Dataset con métricas finales.
    dir_salida : str
        Directorio destino.
    etiqueta_modo : str
        Sufijo de nomenclatura.
    dpi : int
        Resolución visual.
    emitir_log : bool
        Manejo de registros por consola.

    Retorna:
    --------
    List[Tuple[str, str]]
        Lista de recursos de imagen exportados.
    """
    metricas = ['Range', 'SumMin', 'MinSum', 'MaxToleranceRate', 'AvgToleranceRate', 'AvgHammingDistance', 'Hypervolume', 'IGD+', 'GD+']
    disponibles = [m for m in metricas if m in df_runs.columns]
    if not disponibles:
        return []
    df_plot = df_runs.copy()
    df_plot['config'] = df_plot['algorithm'] + '-' + df_plot['init'] + '-' + df_plot['crossover']
    n_configs = df_plot['config'].nunique()
    ancho_dinamico = max(12.0, n_configs * 0.35)
    artefactos = []

    # 2. Individuales (Barplots)
    for m in disponibles:
        plt.figure(figsize=(ancho_dinamico, 6))
        sns.barplot(data=df_plot, x='config', y=m, hue='config', palette='muted', legend=False, errorbar='sd', capsize=.2)
        plt.title(f'Media ± Desviación Estándar de {m}')
        plt.xticks(rotation=35, ha='right', rotation_mode='anchor')
        plt.tight_layout()
        ruta_i = os.path.join(dir_salida, f'media_std_metricas_{m}_{etiqueta_modo}.png')
        plt.savefig(ruta_i, dpi=dpi, bbox_inches='tight')
        exportar_gemelo_ia_csv(ruta_i, df_datos=df_plot[['config', m]])
        artefactos.append((ruta_i, f"Media ± std {m}"))
        if emitir_log:
            imprimir_grafico_guardado(ruta_i, f"Media ± std {m}")
    plt.close('all')
    return artefactos

def graficar_boxplot_metricas(df_runs: pd.DataFrame, dir_salida: str, etiqueta_modo: str,
                              dpi: int = 300, emitir_log: bool = True) -> List[Tuple[str, str]]:
    """
    Genera diagramas de caja para evaluar la dispersión intercuartil.

    Parámetros:
    -----------
    df_runs : pd.DataFrame
        Dataset estadístico.
    dir_salida : str
        Destino en disco.
    etiqueta_modo : str
        Clave identificadora.
    dpi : int
        Nitidez de salida.
    emitir_log : bool
        Logs por terminal.

    Retorna:
    --------
    List[Tuple[str, str]]
        Detalles de los gráficos exportados.
    """
    metricas = ['Range', 'SumMin', 'MinSum', 'MaxToleranceRate', 'AvgToleranceRate', 'AvgHammingDistance', 'Hypervolume', 'IGD+', 'GD+']
    disponibles = [m for m in metricas if m in df_runs.columns]
    if not disponibles:
        return []
    
    df_plot = df_runs.copy()
    df_plot['config'] = df_plot['algorithm'] + '-' + df_plot['init'] + '-' + df_plot['crossover']
    n_configs = df_plot['config'].nunique()
    ancho_dinamico = max(12.0, n_configs * 0.35)
    artefactos = []
    
    # 2. Individuales
    for m in disponibles:
        plt.figure(figsize=(max(10.0, n_configs * 0.35), 6))
        sns.boxplot(data=df_plot, x='config', y=m, palette='Set3', hue='config', legend=False)
        plt.title(f'Distribución de {m} (Boxplot)')
        plt.xticks(rotation=35, ha='right', rotation_mode='anchor')
        plt.tight_layout()
        ruta_i = os.path.join(dir_salida, f'boxplot_metricas_{m}_{etiqueta_modo}.png')
        plt.savefig(ruta_i, dpi=dpi, bbox_inches='tight')
        exportar_gemelo_ia_csv(ruta_i, df_datos=df_plot[['config', m]])
        artefactos.append((ruta_i, f"Distribución {m} (Boxplot)"))
        if emitir_log:
            imprimir_grafico_guardado(ruta_i, f"Distribución {m} (Boxplot)")
    plt.close('all')
    return artefactos


def graficar_analisis_kruskal_dunn(df_runs: pd.DataFrame, dir_salida: str, metrica_objetivo: str, etiqueta_modo: str, col_group: str = 'config', dpi: int = 300, indent: int = 9, graficar: bool = True) -> None:
    """
    Evalúa contrastes no paramétricos multivariables y exporta un heatmap de los p-values.

    Parámetros:
    -----------
    df_runs : pd.DataFrame
        Histórico de rendimiento por ejecución.
    dir_salida : str
        Carpeta destino.
    metrica_objetivo : str
        Nombre del indicador de rendimiento evaluado.
    etiqueta_modo : str
        Distintivo del modo.
    dpi : int
        Puntos por pulgada.
    indent : int
        Espacios de margen en terminal.
    """
    if df_runs.empty or metrica_objetivo not in df_runs.columns: return
    
    espacios = " " * indent
    df_plot = df_runs.copy()
    if 'config' not in df_plot.columns:
        df_plot['config'] = df_plot['algorithm'] + '-' + df_plot['init'] + '-' + df_plot['crossover']
    
    n_configs = df_plot[col_group].nunique()
    
    stat, p_val, p_dunn = compute_kruskal_dunn(df_plot, metrica_objetivo, col_group)
    
    titulos = {
        'config': 'Configuración (Alg + Init + Cruce)',
        'algorithm': 'Algoritmo',
        'init': 'Inicialización',
        'crossover': 'Cruce'
    }
    titulo = titulos.get(col_group, col_group)

    sub_line = "─" * (len(metrica_objetivo) + len(titulo) + 31)
    if stat is None:
        print(f"\n{espacios}📊  \033[1mVALIDACIÓN ESTADÍSTICA ({metrica_objetivo} | {titulo})\033[0m")
        print(f"{espacios}{sub_line}")
        print(f"{espacios}    ⚠️  Kruskal-Wallis omitido: se requieren al menos 2 grupos.\n")
        return

    print(f"\n{espacios}📊  \033[1mVALIDACIÓN ESTADÍSTICA ({metrica_objetivo} | {titulo})\033[0m")
    print(f"{espacios}{sub_line}")
    print(f"{espacios}    Estadístico H (Kruskal-Wallis): {stat:.4f}")
    print(f"{espacios}    P-valor: {p_val:.4e}")
    print(f"{espacios}    Significativo (p < 0.05): {'Sí' if p_val < 0.05 else 'No'}\n")
    
    if p_dunn is not None and graficar:
        
        tamano_hm = max(14.0, n_configs * 0.4)
        plt.figure(figsize=(tamano_hm, tamano_hm))
        mask = np.triu(np.ones_like(p_dunn, dtype=bool))
        cmap = ListedColormap(['#228B22', '#90EE90', '#FFC1C1'])
        norm = BoundaryNorm([0, 0.01, 0.05, 1.0], cmap.N)
        
        ax = sns.heatmap(p_dunn, mask=mask, annot=False, cmap=cmap, norm=norm, 
                    linewidths=0.5, linecolor='white',
                    cbar_kws={"ticks": [0.005, 0.03, 0.525]})
        cbar = ax.collections[0].colorbar
        cbar.set_ticklabels(['Alta Sig. (p < 0.01)', 'Sig. (p < 0.05)', 'No Sig. (p \u2265 0.05)'])
        cbar.set_label("Nivel de Significancia", size=12)
        
        plt.xticks(rotation=45, ha='right', rotation_mode='anchor')
        plt.suptitle(f'Post-hoc de Dunn: {metrica_objetivo}', fontsize=16, y=0.98)
        plt.tight_layout()
        
        ruta_h = os.path.join(dir_salida, f"heatmap_dunn_{metrica_objetivo.lower()}_{col_group}_{etiqueta_modo}.png")
        plt.savefig(ruta_h, dpi=dpi, bbox_inches='tight')
        exportar_gemelo_ia_csv(ruta_h, df_datos=p_dunn)
        
        print(f"{espacios}    ", end="")
        imprimir_grafico_guardado(ruta_h, f"Heatmap de Comparaciones Dunn (Post-hoc)")
        plt.close()



