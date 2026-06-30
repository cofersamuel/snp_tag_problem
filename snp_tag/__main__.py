"""
Módulo de Interfaz de Línea de Comandos (__main__.py)
------------------------------------------------------
Punto de entrada para la ejecución del pipeline modular Tag SNP.
Gestiona el parseo de argumentos.
"""

# =============================================================================
# LIBRERÍAS ESTÁNDAR
# =============================================================================
import argparse  # Procesamiento y parseo de argumentos de línea de comandos.
import os  # Gestión de rutas y verificación de existencia de directorios.
import sys  # Acceso a parámetros y funciones del sistema (no utilizado actualmente).

# =============================================================================
# MÓDULOS LOCALES (snp_tag)
# =============================================================================
# Importación desde config.py de:
# - la lista de modos de ejecución disponibles (perfiles predefinidos de coste/búsqueda):
#       - fast: ejecución ultrarrápida de depuración (pob=10, gen=2, runs=3)
#       - medium: ejecución intermedia por defecto (pob=84, gen=50, runs=3)
#       - high: ejecución de coste y calidad altos (pob=120, gen=100, runs=3)
#       - full: ejecución exhaustiva completa (pob=220, gen=500, runs=5)
#       - full_21: ejecución exhaustiva con 21 ejecuciones independientes para validación robusta (pob=220, gen=500, runs=21)
#       - full_31: ejecución exhaustiva con 31 ejecuciones independientes para validación robusta (pob=220, gen=500, runs=31)
# - la lista de fuentes de datos permitidas:
#       - hinds2005: bloque histórico real del estudio de Hinds et al. (2005)
#       - synthetic: bloque sintético generado artificialmente con linkage disequilibrium configurable
from snp_tag.config import FUENTES_DATOS_DISPONIBLES, MODOS_DISPONIBLES
# Orquestador del pipeline modular para ejecutar la búsqueda evolutiva completa o generar reportes exclusivos (modo report-only-csv).
from snp_tag.orchestrator import (ejecutar_pipeline,
                                  ejecutar_pipeline_postprocessing)
from snp_tag.utils.logger import \
    logger  # Registro y salida de mensajes informativos y de error.
# Generación de hipervínculos interactivos en la terminal.
from snp_tag.utils.terminal import obtener_enlace_terminal


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
    
    # Adición del argumento opcional '--resume' (o '-r') para reanudar un experimento anterior.
    parser.add_argument('--resume', '-r',
                        type=str,
                        default=None,
                        help='Ruta al directorio de un experimento previo para reanudarlo (lee la configuración exportada)')
    
    # Adición de una bandera (booleano) para activar el modo de generación exclusiva de reportes a partir de archivos CSV existentes.
    parser.add_argument(
        '--post-processing',  # Nombre del argumento en la línea de comandos. Se almacena en args.post_processing.
        action='store_true',  # Al indicarse la bandera, almacena el valor True.
        help=(  # Mensaje informativo que se muestra al invocar la ayuda del script.
            "Activa el modo de postprocesamiento exclusivo. El sistema selecciona automáticamente los CSV "
            "más recientes en el directorio snp_tag/input/ para generar visualizaciones y estadísticas."
        ),
    )
    args = parser.parse_args()  # Procesa y valida los argumentos de la CLI, guardando los valores como atributos en el objeto 'args' (args.mode, args.data_source, args.post_processing).
    
    try:  # Bloque try-except para capturar y controlar posibles excepciones ocurridas durante la ejecución del pipeline.
        if args.post_processing:  # Condición: si el usuario ha activado la bandera '--post-processing'.
            import sys
            argumentos_invalidos = []
            if '--mode' in sys.argv or '-m' in sys.argv:
                argumentos_invalidos.append('--mode')
            if '--data-source' in sys.argv or '-d' in sys.argv:
                argumentos_invalidos.append('--data-source')
            if '--resume' in sys.argv or '-r' in sys.argv:
                argumentos_invalidos.append('--resume')
            if argumentos_invalidos:
                parser.error(f"Los argumentos {', '.join(argumentos_invalidos)} no están permitidos con --post-processing. Estos valores se leen de la sección [Argumentos] del archivo exportado.")
                
            ruta_base = ejecutar_pipeline_postprocessing(args) # TOREAD  # Ejecuta el orquestador únicamente en modo post-processing y obtiene la ruta de resultados.
        elif args.resume:  # Condición: reanudar experimento previo.
            import sys
            argumentos_invalidos = []
            if '--mode' in sys.argv or '-m' in sys.argv:
                argumentos_invalidos.append('--mode')
            if '--data-source' in sys.argv or '-d' in sys.argv:
                argumentos_invalidos.append('--data-source')
            if argumentos_invalidos:
                parser.error(f"Los argumentos {', '.join(argumentos_invalidos)} no están permitidos con --resume. Estos valores se infieren de la configuración exportada.")
            ruta_base = ejecutar_pipeline(args)
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
