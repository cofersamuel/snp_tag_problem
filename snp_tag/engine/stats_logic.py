"""
Módulo del Motor Estadístico (stats.py)
---------------------------------------
Realiza pruebas estadísticas no paramétricas sobre los resultados del experimento.
"""

# =============================================================================
# LIBRERÍAS DE TERCEROS
# =============================================================================
import numpy as np
import pandas as pd
import scipy.stats as ss

try:
    # =============================================================================
# LIBRERÍAS DE TERCEROS
# =============================================================================
    import scikit_posthocs as sp
except ImportError:
    sp = None


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
        p_dunn = sp.posthoc_dunn(df_plot, val_col=metrica_objetivo, group_col=col_group, p_adjust='holm')
        
    return stat, p_val, p_dunn
