"""
Módulo de Registro Estructurado (logger.py)
-------------------------------------------
Proporciona un logger centralizado, configurado para salida en terminal
con formato ANSI y salida a disco limpia con timestamps.
"""

# =============================================================================
# LIBRERÍAS ESTÁNDAR
# =============================================================================
import logging  # Módulo estándar para el registro de eventos y logs.
import re  # Módulo estándar para soporte de expresiones regulares.
import sys  # Acceso a parámetros y funciones del sistema (como stdout).


class _ANSIEscapeStripper(logging.Formatter):
    """Filtra secuencias ANSI (colores, hipervínculos) para archivos de log limpios."""
    # Compilación de expresión regular para encontrar códigos de escape ANSI y secuencias de hipervínculos.
    ansi_escape = re.compile(r'\x1B\]8;;.*?\x1B\\|\x1B\[[0-9;]*[a-zA-Z]')
    
    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)  # Llama al formateador padre para generar el mensaje de texto original.
        return self.ansi_escape.sub('', msg)  # Elimina todas las secuencias ANSI encontradas reemplazándolas por nada.

def setup_logger(name: str = "snp_tag") -> logging.Logger:
    """Configura el logger base con salida a terminal."""
    logger = logging.getLogger(name)  # Obtiene o crea una instancia del logger con el nombre dado.
    logger.setLevel(logging.INFO)  # Establece el nivel mínimo de registro de logs a INFO.

    # Evitar handlers duplicados
    if logger.hasHandlers():  # Comprueba si el logger ya dispone de algún manejador asignado.
        return logger  # Retorna el logger existente inmediatamente para evitar redundancia.

    # Handler de Consola (con color y sin formato extra para mantener estética)
    ch = logging.StreamHandler(sys.stdout)  # Crea un StreamHandler para redirigir los logs a la salida estándar.
    ch.setLevel(logging.INFO)  # Fija el nivel de salida del StreamHandler a INFO.
    ch_formatter = logging.Formatter('%(message)s')  # Formatea para mostrar solo el mensaje crudo sin añadidos.
    ch.setFormatter(ch_formatter)  # Asocia este formateador plano al StreamHandler de la consola.
    logger.addHandler(ch)  # Registra el StreamHandler en el objeto del logger.

    return logger  # Devuelve la instancia del logger configurada.

def add_file_handler(logger: logging.Logger, log_path: str) -> None:
    """Añade un FileHandler al logger que limpie los escapes ANSI."""
    # Verificar que no exista ya un FileHandler para no duplicar
    for handler in logger.handlers:  # Itera sobre la lista de manejadores activos del logger.
        if isinstance(handler, logging.FileHandler):  # Verifica si el manejador es de tipo FileHandler.
            return  # Si ya existe un FileHandler, aborta el proceso para evitar duplicidades.
            
    fh = logging.FileHandler(log_path, mode='w', encoding='utf-8')  # Crea el FileHandler para escritura con codificación UTF-8.
    fh.setLevel(logging.INFO)  # Configura el nivel del FileHandler a INFO.
    # Instancia el limpiador ANSI con un patrón de mensaje detallado que incluye hora y nivel de log.
    fh_formatter = _ANSIEscapeStripper('[%(asctime)s] [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(fh_formatter)  # Aplica el formateador de limpieza al FileHandler.
    logger.addHandler(fh)  # Añade el FileHandler a la instancia del logger.

# Instancia global del logger
logger = setup_logger()  # Llama a setup_logger para inicializar la variable global.
