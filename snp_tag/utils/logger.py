"""
Módulo de Registro Estructurado (logger.py)
-------------------------------------------
Proporciona un logger centralizado, configurado para salida en terminal
con formato ANSI y salida a disco limpia con timestamps.
"""

import logging
import sys
import re

class _ANSIEscapeStripper(logging.Formatter):
    """Filtra secuencias ANSI (colores, hipervínculos) para archivos de log limpios."""
    ansi_escape = re.compile(r'\x1B\]8;;.*?\x1B\\|\x1B\[[0-9;]*[a-zA-Z]')
    
    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        return self.ansi_escape.sub('', msg)

def setup_logger(name: str = "snp_tag") -> logging.Logger:
    """Configura el logger base con salida a terminal."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Evitar handlers duplicados
    if logger.hasHandlers():
        return logger

    # Handler de Consola (con color y sin formato extra para mantener estética)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch_formatter = logging.Formatter('%(message)s')
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)

    return logger

def add_file_handler(logger: logging.Logger, log_path: str) -> None:
    """Añade un FileHandler al logger que limpie los escapes ANSI."""
    # Verificar que no exista ya un FileHandler para no duplicar
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            return
            
    fh = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    fh.setLevel(logging.INFO)
    fh_formatter = _ANSIEscapeStripper('[%(asctime)s] [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)

# Instancia global del logger
logger = setup_logger()
