"""
Módulo de Interfaz de Línea de Comandos (__main__.py)
------------------------------------------------------
Punto de entrada para la ejecución del pipeline modular Tag SNP.
Gestiona el parseo de argumentos.
"""

# =============================================================================
# LIBRERÍAS ESTÁNDAR
# =============================================================================
import os  # Gestión de rutas y verificación de existencia de directorios.
import sys  # Acceso a parámetros y funciones del sistema (no utilizado actualmente).
import argparse  # Procesamiento y parseo de argumentos de línea de comandos.

# =============================================================================
# MÓDULOS LOCALES (snp_tag)
# =============================================================================
# Importación desde config.py de:
# - la lista de modos de ejecución disponibles (perfiles predefinidos de coste/búsqueda):
#       - fast: ejecución ultrarrápida de depuración (pob=10, gen=2, runs=2)
#       - medium: ejecución intermedia por defecto (pob=84, gen=50, runs=2)
#       - high: ejecución de coste y calidad altos (pob=120, gen=100, runs=3)
#       - full: ejecución exhaustiva completa (pob=220, gen=500, runs=5)
#       - full_20: ejecución exhaustiva con 20 ejecuciones independientes para validación robusta (pob=220, gen=500, runs=20)
#       - full_30: ejecución exhaustiva con 30 ejecuciones independientes para validación robusta (pob=220, gen=500, runs=30)
# - la lista de fuentes de datos permitidas:
#       - hinds2005: bloque histórico real del estudio de Hinds et al. (2005)
#       - synthetic: bloque sintético generado artificialmente con linkage disequilibrium configurable
from snp_tag.config import MODOS_DISPONIBLES, FUENTES_DATOS_DISPONIBLES
from snp_tag.utils.logger import logger  # Registro y salida de mensajes informativos y de error.
# Generación de hipervínculos interactivos en la terminal.
from snp_tag.utils.terminal import obtener_enlace_terminal
# Orquestador del pipeline modular para ejecutar la búsqueda evolutiva completa o generar reportes exclusivos (modo report-only-csv).
from snp_tag.orchestrator import ejecutar_pipeline, ejecutar_pipeline_report_only_csv

def main():
    """
    Punto de entrada principal para la ejecución del pipeline modular Tag SNP.
    Procesa los argumentos de la línea de comandos e inicia el orquestador correspondiente.
    """
    # Inicialización del analizador de argumentos de línea de comandos con una descripción y desactivando la ayuda por defecto para personalizarla.
    parser = argparse.ArgumentParser(description='Pipeline Modular para la Selección de Tag SNPs', #  Descripción del proyecto.
                                     add_help=False) # 'add_help=False' evita que ArgumentParser muestre su ayuda automática. Usamos un argumento personalizado para 'help'.
    
    # Adición de un argumento personalizado para mostrar el mensaje de ayuda de forma explícita.
    parser.add_argument('-h', '--help', #  Opción para mostrar la ayuda.
                        action='help', # 'action='help'' hace que el programa muestre la ayuda y salga. 
                        help='Mostrar este mensaje de ayuda y salir') # Mensaje que se muestra al usar la ayuda.
    
    # Adición del argumento opcional '--mode' (o '-m') para elegir el nivel de optimización (por defecto, 'medium').
    parser.add_argument('--mode', '-m', # Opción para elegir el nivel de optimización. Se almacena en args.mode.
                        choices=list(MODOS_DISPONIBLES), # 'choices=list(MODOS_DISPONIBLES)' hace que solo se puedan usar los valores de MODOS_DISPONIBLES.
                        default='medium', # 'default='medium'' hace que por defecto se use 'medium'.
                        help='Modo de ejecución que define el coste y la calidad de la búsqueda (por defecto: %(default)s)')
    
    # Adición del argumento opcional '--data-source' (o '-d') para especificar el conjunto de datos de entrada (por defecto, 'hinds2005').
    parser.add_argument('--data-source', '-d', # Opción para elegir el conjunto de datos de entrada. Se almacena en args.data_source.
                        choices=list(FUENTES_DATOS_DISPONIBLES), # 'choices=list(FUENTES_DATOS_DISPONIBLES)' hace que solo se puedan usar los valores de FUENTES_DATOS_DISPONIBLES.
                        default='hinds2005', # 'default='hinds2005'' hace que por defecto se use 'hinds2005'.
                        help='Fuente de datos o conjunto de genotipos de entrada para el pipeline (por defecto: %(default)s)')
    
    # Adición de una bandera (booleano) para activar el modo de generación exclusiva de reportes a partir de archivos CSV existentes.
    parser.add_argument(
        '--report-only-csv',  # Nombre del argumento en la línea de comandos. Se almacena en args.report_only_csv.
        action='store_true',  # Al indicarse la bandera, almacena el valor True.
        help=(  # Mensaje informativo que se muestra al invocar la ayuda del script.
            "Activa el modo de reporte exclusivo. El sistema selecciona automáticamente los CSV "
            "más recientes en el directorio snp_tag/input/ para generar las visualizaciones."
        ),
    )
    args = parser.parse_args()  # Procesa y valida los argumentos de la CLI, guardando los valores como atributos en el objeto 'args' (args.mode, args.data_source, args.report_only_csv).
    
    try:  # Bloque try-except para capturar y controlar posibles excepciones ocurridas durante la ejecución del pipeline.
        if args.report_only_csv:  # Condición: si el usuario ha activado la bandera '--report-only-csv'.
            ruta_base = ejecutar_pipeline_report_only_csv(args)  # Ejecuta el orquestador únicamente en modo "report-only-csv" y obtiene la ruta de resultados.
        else:  # En caso contrario (ejecución normal del pipeline evolutivo).
            ruta_base = ejecutar_pipeline(args)  # Ejecuta la optimización completa y obtiene la ruta del directorio de salida.
        
        if ruta_base and os.path.exists(ruta_base):  # Verifica que la ruta de resultados devuelta sea válida y exista físicamente en el disco.
            log_path = os.path.join(ruta_base, 'ejecucion.log')  # Construye la ruta al archivo donde se almacena el registro del proceso.
            enlace_carpeta = obtener_enlace_terminal(ruta_base, ruta_base)  # Genera el enlace hipervínculo clicable a la carpeta de salida.
            enlace_log = obtener_enlace_terminal(log_path, "ejecucion.log")  # Genera el enlace hipervínculo clicable directo al archivo de registro de ejecución.
            logger.info(f"\n\033[94m📁  Carpeta de los experimentos en: {enlace_carpeta}\033[0m")  # Muestra en la terminal la ruta a la carpeta con estilo azul.
            logger.info(f"\033[92m💾  Salida de la terminal guardada en {enlace_log}\033[0m\n")  # Muestra en la terminal la ruta al log con estilo verde.
            
    except Exception as e:  # Captura de cualquier excepción ocurrida dentro del bloque de ejecución principal.
        logger.error(f"\n❌  Error fatal durante la ejecución: {e}\n")  # Registra el error en la terminal con formato visual descriptivo.
        raise  # Propaga la excepción capturada para que el sistema detenga la ejecución indicando el fallo.

if __name__ == '__main__':  # Comprobación de que el archivo se está ejecutando directamente y no importándose como módulo.
    main()  # Llama a la función principal para iniciar la ejecución del proyecto.
