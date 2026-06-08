"""
Módulo de Utilidades de Terminal y Registro (terminal.py)
---------------------------------------------------------
Proporciona funciones auxiliares para la interfaz de usuario en consola,
incluyendo formateo de texto, enlaces clicables y un sistema de volcado (Tee)
para el registro de la sesión en disco.
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Optional
from snp_tag.utils.logger import logger

def obtener_bit_string_estilizado(bits: np.ndarray) -> str:
    """Retorna una cadena de bits con los ceros en gris para mejorar la legibilidad."""
    gris = "\033[90m"
    reset = "\033[0m"
    return "".join([f"{gris if b == 0 else ''}{str(int(b))}{reset if b == 0 else ''}" for b in bits])

def obtener_enlace_terminal(ruta: str, etiqueta: Optional[str] = None) -> str:
    """
    Genera una secuencia de escape OSC 8 para un enlace hipertextual en terminales compatibles.
    """
    if etiqueta is None:
        etiqueta = os.path.basename(ruta)
    ruta_abs = os.path.abspath(ruta)
    url = f"file://{ruta_abs}"
    return f"\033]8;;{url}\033\\ \033[36m{etiqueta}\033[0m\033]8;;\033\\"

def imprimir_encabezado(titulo: str, color: str = "\033[93m") -> None:
    """
    Imprime un encabezado estilizado con bordes dobles y color persistente.
    """
    reset = "\033[0m"
    bold = "\033[1m"
    width = 80
    border_top = f"{color}{bold}╔" + "═" * (width - 2) + "╗"
    content = f"║{titulo.center(width - 2)}║"
    border_bot = f"╚" + "═" * (width - 2) + f"╝{reset}"
    logger.info(f"\n{border_top}\n{color}{bold}{content}\n{color}{bold}{border_bot}")

def imprimir_subseccion(titulo: str, icono: str = "🔹") -> None:
    """
    Imprime un encabezado de subsección descriptivo y en negrita.
    """
    # Estandarización de iconos: eliminar VS16 existentes para evitar duplicados
    icono_base = icono.replace("\ufe0f", "")
    
    # Lista de emojis que requieren VS16 y espacio extra por renderizado terminal
    necesitan_vs16 = ["⚙", "⚖", "⏱", "⚠️", "↔"]
    if icono_base in necesitan_vs16:
        icono_final = icono_base + "\ufe0f"
        # Tres espacios: uno suele ser 'absorbido' por el renderizado del glifo ancho
        espaciado = "   "
    else:
        icono_final = icono_base
        espaciado = "  "

    # Imprimir con el espaciado calculado
    logger.info(f"\n  {icono_final}{espaciado}\033[1m{titulo}\033[0m")
    logger.info("  " + "─" * (len(titulo) + 6))

def imprimir_metadato(etiqueta: str, valor: str, sangria: int = 6) -> None:
    """
    Imprime un par etiqueta-valor con formato de viñeta (separador ':').
    """
    espacios = " " * sangria
    logger.info(f"{espacios}• \033[1m{etiqueta}\033[0m: {valor}")

def imprimir_paso(mensaje: str, icono: str = "🚀") -> None:
    """
    Imprime un hito del proceso evolutivo.
    """
    logger.info(f"    • {mensaje}") if icono == "" else logger.info(f"\n  {icono}  \033[1m{mensaje}\033[0m")

def imprimir_estado(mensaje: str, exito: bool = True) -> None:
    """
    Imprime el resultado de una operación con indicadores cromáticos.
    """
    icono = "✅" if exito else "❌"
    color = "\033[92m" if exito else "\033[91m"
    reset = "\033[0m"
    logger.info(f"       {color}{icono}  {mensaje}{reset}")

def imprimir_grafico_guardado(ruta: str, descripcion: str) -> None:
    """
    Notifica la generación de un archivo gráfico con su enlace correspondiente (OSC 8).
    """
    enlace = obtener_enlace_terminal(ruta)
    # Dos espacios tras el icono y uno tras el colon
    logger.info(f"      🖼️  {descripcion}: {enlace}")

def imprimir_tabla(df: pd.DataFrame, titulo: Optional[str] = None) -> None:
    """
    Muestra un subconjunto de un DataFrame de forma tabular en la consola.
    """
    if titulo:
        logger.info(f"\n--- 📊️  {titulo} " + "─" * (60 - len(titulo)))
    with pd.option_context('display.max_rows', 15, 'display.max_columns', None, 
                           'display.width', 1000, 'display.precision', 4):
        logger.info(df.to_string(index=False))

