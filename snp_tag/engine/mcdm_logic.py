"""
Módulo del Motor MCDM (mcdm.py)
--------------------------------
Procesa datos de frentes Pareto y ejecuta estrategias MCDM (Multi-Criteria Decision Making).
"""

import numpy as np
import pandas as pd
from typing import Tuple

from pymoo.decomposition.asf import ASF
from pymoo.mcdm.pseudo_weights import PseudoWeights
from pymoo.mcdm.high_tradeoff import HighTradeoffPoints

from snp_tag.engine.metrics_logic import decodificar_objetivos_reales

_HIGHER_IS_BETTER = {1, 2}  # Tolerancia, Hamming (índices de columna)

def decodificar_y_filtrar(df_fronts: pd.DataFrame,
                          modo_transformacion: str) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Decodifica objetivos a escala real y filtra soluciones factibles.

    Returns:
        (df_factible, F_real)  donde F_real tiene columnas
        [Compacidad, Tolerancia, Hamming, Balance].
    """
    cols_F = [
        'f1_compactness', 'f2_transformed_tolerance',
        'f3_transformed_hamming_avg', 'f4_balance_var',
    ]
    F_raw = df_fronts[cols_F].to_numpy(dtype=float)

    reales = decodificar_objetivos_reales(F_raw, modo_transformacion)
    F_real = np.column_stack([
        reales['compacidad'],
        reales['tolerancia_real'],
        reales['hamming_prom_real'],
        reales['balance_var'],
    ])

    mask = reales['min_cobertura'] >= 1.0 - 1e-9
    return df_fronts[mask].reset_index(drop=True), F_real[mask]


def normalizar_minimizacion(F_real: np.ndarray) -> np.ndarray:
    """
    Normaliza objetivos a [0, 1] orientados a minimización.

    Compacidad (col 0): menor es mejor  → normalización directa.
    Tolerancia (col 1): mayor es mejor  → se invierte.
    Hamming    (col 2): mayor es mejor  → se invierte.
    Balance    (col 3): menor es mejor  → normalización directa.
    """
    F_norm = F_real.copy()
    for j in range(F_norm.shape[1]):
        fmin, fmax = F_norm[:, j].min(), F_norm[:, j].max()
        rango = fmax - fmin
        if rango > 0:
            F_norm[:, j] = (F_norm[:, j] - fmin) / rango
        else:
            F_norm[:, j] = 0.0

    # Invertir objetivos de maximización
    for j in _HIGHER_IS_BETTER:
        F_norm[:, j] = 1.0 - F_norm[:, j]

    return F_norm


def detectar_knee_points(F_norm: np.ndarray) -> np.ndarray:
    """Detecta puntos de alto trade-off."""
    try:
        dm = HighTradeoffPoints()
        indices = dm(F_norm)
        if indices is None:
            return np.array([], dtype=int)
        return np.asarray(indices, dtype=int)
    except Exception:
        return np.array([], dtype=int)


def evaluar_estrategias_mcdm(F_norm: np.ndarray, pesos_usuario: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Aplica las estrategias MCDM sobre la matriz normalizada.
    
    Returns:
        (I_asf, I_pw, pw_mat, I_knee)
    """
    I_asf = np.array([ASF().do(F_norm, weights=pesos_usuario).argmin()])
    I_pw_val, pw_mat = PseudoWeights(pesos_usuario).do(F_norm, return_pseudo_weights=True)
    I_pw = np.array([I_pw_val])
    I_knee = detectar_knee_points(F_norm)
    
    return I_asf, I_pw, pw_mat, I_knee
