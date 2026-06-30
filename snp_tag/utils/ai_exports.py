"""
Módulo de Exportación de Gemelos de Datos (ai_exports.py)
---------------------------------------------------------
Proporciona utilidades para exportar los datos subyacentes de cada gráfica
generada a un archivo CSV contiguo, facilitando la auditoría automatizada
por parte de agentes de IA sin necesidad de interpretar píxeles.
"""

# =============================================================================
# LIBRERÍAS ESTÁNDAR
# =============================================================================
import os

# =============================================================================
# LIBRERÍAS DE TERCEROS
# =============================================================================
import numpy as np
import pandas as pd


def exportar_gemelo_ia_csv(
    ruta_imagen: str,
    df_datos: pd.DataFrame = None,
    array_datos: np.ndarray = None,
    columnas: list = None,
) -> None:
    """
    Exporta el dataset completo usado para generar una gráfica a un archivo CSV contiguo.

    El archivo se guarda con el mismo nombre base que la imagen, pero con sufijo ``_data.csv``.
    Acepta indistintamente un DataFrame de pandas o un ndarray de NumPy (con columnas opcionales).

    Parámetros
    ----------
    ruta_imagen : str
        Ruta absoluta de la imagen (.png) asociada.
    df_datos : pd.DataFrame, optional
        DataFrame con los datos tabulares a exportar.
    array_datos : np.ndarray, optional
        Matriz NumPy a exportar (se convierte internamente a DataFrame).
    columnas : list, optional
        Nombres de columnas cuando se proporciona ``array_datos``.
    """
    base, _ = os.path.splitext(ruta_imagen)
    ruta_csv = f"{base}_data.csv"

    if df_datos is not None:
        if hasattr(df_datos, 'empty') and df_datos.empty:
            return
        df_datos.to_csv(ruta_csv, index=False)
    elif array_datos is not None and array_datos.size > 0:
        df = pd.DataFrame(array_datos, columns=columnas)
        df.to_csv(ruta_csv, index=False)
