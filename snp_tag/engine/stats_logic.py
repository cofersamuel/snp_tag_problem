"""
Módulo del Motor Estadístico (stats.py)
---------------------------------------
Realiza pruebas estadísticas no paramétricas sobre los resultados del experimento.
"""

import numpy as np
import pandas as pd
import scipy.stats as ss

try:
    import scikit_posthocs as sp
except ImportError:
    sp = None

def prepare_rank_matrix(df_plot: pd.DataFrame, col_group: str) -> tuple:
    """Prepara la matriz de rankings para test de Friedman."""
    metricas = ['Hypervolume', 'Range', 'MinSum', 'SumMin', 'MaxToleranceRate', 'AvgToleranceRate', 'AvgHammingDistance', 'IGD+', 'GD+']
    higher_is_better = ['Hypervolume', 'Range', 'MaxToleranceRate', 'AvgToleranceRate', 'AvgHammingDistance']
    disponibles = [m for m in metricas if m in df_plot.columns]
    
    if not disponibles:
        return None, None
        
    resumen = df_plot.groupby(col_group)[disponibles].mean().reset_index()
    
    rank_matrix = []
    for m in disponibles:
        asce = False if m in higher_is_better else True
        rank_matrix.append(resumen[m].rank(ascending=asce).values)
    
    rank_matrix = np.array(rank_matrix) # (metrics, groups)
    return rank_matrix, resumen

def compute_friedman_nemenyi(rank_matrix: np.ndarray, resumen: pd.DataFrame, col_group: str) -> tuple:
    """Ejecuta el test de Friedman y post-hoc Nemenyi."""
    n_groups = rank_matrix.shape[1]
    if n_groups < 3:
        return None, 1.0, None, np.mean(rank_matrix, axis=0)
        
    stat, p_value = ss.friedmanchisquare(*rank_matrix.T)
    avg_ranks = np.mean(rank_matrix, axis=0)
    
    p_values_nemenyi = None
    if p_value < 0.05 and sp is not None:
        p_values_nemenyi = sp.posthoc_nemenyi_friedman(rank_matrix)
        p_values_nemenyi.columns = resumen[col_group]
        p_values_nemenyi.index = resumen[col_group]
        
    return stat, p_value, p_values_nemenyi, avg_ranks

def compute_kruskal_dunn(df_plot: pd.DataFrame, metrica_objetivo: str, col_group: str = 'config') -> tuple:
    """Ejecuta test Kruskal-Wallis y post-hoc Dunn."""
    grupos = []
    nombres = sorted(df_plot[col_group].unique())
    for n in nombres:
        grupos.append(df_plot[df_plot[col_group] == n][metrica_objetivo].values)
        
    if len(grupos) < 2:
        return None, 1.0, None
        
    stat, p_val = ss.kruskal(*grupos)
    
    p_dunn = None
    if p_val < 0.05 and sp is not None:
        p_dunn = sp.posthoc_dunn(df_plot, val_col=metrica_objetivo, group_col=col_group, p_adjust='bonferroni')
        
    return stat, p_val, p_dunn
