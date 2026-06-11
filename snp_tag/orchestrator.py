# Módulo principal para la coordinación de la ejecución del pipeline modular Tag SNP.
# Proporciona el punto de entrada oficial para la orquestación del experimento.
"""
Módulo Principal (main.py)
--------------------------
Punto de entrada oficial para la ejecución del pipeline modular Tag SNP.
Coordina la configuración, carga de datos, diagnóstico, ejecución evolutiva
y síntesis de resultados.
"""

# =============================================================================
# LIBRERÍAS ESTÁNDAR
# =============================================================================
import argparse  # Para parsear argumentos de la línea de comandos
import concurrent.futures  # Para ejecutar tareas concurrentes utilizando hilos o procesos
import multiprocessing  # Para paralelismo a nivel de procesos
# Importación de bibliotecas estándar del sistema
import os  # Para manipulación de directorios y rutas del sistema operativo
import shutil  # Para realizar operaciones de copiado de archivos
import sys  # Para acceder a variables y funciones del intérprete de Python
import tempfile  # Para gestionar archivos y directorios temporales si fuesen necesarios
import time  # Para medir tiempos de ejecución en distintas fases
from itertools import \
    combinations  # Para generar todas las combinaciones posibles de elementos
# Importación de bibliotecas científicas y manipulación de datos
from pathlib import \
    Path  # Para manipulación de rutas del sistema de archivos orientada a objetos
# Tipado estático para anotaciones de tipos
from typing import Any, Dict, List, Optional, Tuple

# =============================================================================
# LIBRERÍAS DE TERCEROS
# =============================================================================
import numpy as np  # Para operaciones y manipulaciones numéricas eficientes
import pandas as pd  # Para manejo y estructuración de datos en DataFrames

# =============================================================================
# MÓDULOS LOCALES (snp_tag)
# =============================================================================
# Módulo de métricas para inyección dinámica de evaluación
import snp_tag.engine.metrics_logic
# Importaciones de módulos internos del paquete snp_tag
# Configuración del experimento e información relacionada
from snp_tag.config import (FUENTES_DATOS_DISPONIBLES, MODOS_DISPONIBLES,
                            RUTA_USER_CONFIG, ConfiguracionExperimento,
                            cargar_params_tunables_desde_ini,
                            construir_configuracion, informar_configuracion,
                            resolver_modo_evaluacion)
# Algoritmos y generación de direcciones de referencia en optimización
from snp_tag.core.algorithm import construir_direcciones_referencia
# Carga y exportación de conjuntos de datos genómicos
from snp_tag.data.loader import cargar_dataset_objetivo, exportar_dataset
# Funciones lógicas para diagnóstico y EDA de desequilibrio de ligamiento (LD)
from snp_tag.engine.diagnostics_logic import (analizar_similitud_genotipica,
                                              calcular_ld_completo,
                                              detectar_bloques_ld,
                                              ejecutar_diagnostico_ld)
# Funciones lógicas para la evaluación de métricas
from snp_tag.engine.metrics_logic import (
    construir_metricas_generacionales, decodificar_objetivos_reales,
    evaluar_metricas_finales, filtrar_soluciones_factibles,
    obtener_referencias_estaticas_dataset)
# Pipeline para la ejecución del diagnóstico del dataset
from snp_tag.pipelines.diagnostics_pipeline import \
    ejecutar_pipeline_diagnostico
# Pipeline principal de ejecución evolutiva
from snp_tag.pipelines.evolution_pipeline import ejecutar_suite_completa
# Pipeline para generar reportes y visualizaciones gráficas
from snp_tag.pipelines.reporting_pipeline import \
    ejecutar_reportes_visualizacion
# Creación de directorios para organizar resultados
from snp_tag.utils.filesystem import crear_arbol_directorios_dataset
# Registro de eventos (logger) y manejador de archivos
from snp_tag.utils.logger import add_file_handler, logger
# Cálculo adaptativo del número de trabajadores en paralelo
from snp_tag.utils.runtime import calcular_max_workers_paralelo
# Utilidades de impresión formateada en terminal
from snp_tag.utils.terminal import (imprimir_encabezado, imprimir_estado,
                                    imprimir_grafico_guardado,
                                    imprimir_metadato, imprimir_paso,
                                    imprimir_subseccion,
                                    obtener_bit_string_estilizado,
                                    obtener_enlace_terminal)
# Gráfico específico para el rendimiento en tiempo
from snp_tag.visualization.stats_plot import graficar_rendimiento_tiempo


def inicializar_configuracion(modo: str = 'medium', data_source: str = 'hinds2005', overrides: Optional[Dict[str, Any]] = None) -> ConfiguracionExperimento:
    """
    Configura los parámetros iniciales del experimento mediante inyección de dependencias.
    """
    # Llama a construir_configuracion con los parámetros y los overrides opcionales
    return construir_configuracion(modo=modo, data_source=data_source, overrides=overrides)


def _construir_df_fronts_desde_resultados(resultados: List[Any], modo_transformacion_objetivos: str = 'neg') -> pd.DataFrame:
    """Construye un DataFrame tabular de soluciones de frentes finales para CSV/plots."""
    # Nombres de las columnas que compondrán el DataFrame resultante
    columnas = [
        'algorithm', 'init', 'crossover', 'run', 'seed',
        'f1_compactness',
        'f2_transformed_tolerance',
        'f3_transformed_hamming_avg',
        'f4_balance_var',
    ]
    # Lista vacía donde se irán guardando las filas del frente
    filas = []
    # Iteración sobre los resultados de cada ejecución o réplica
    for rr in resultados:
        # Si no hay soluciones en el frente final o es None, se salta esta ejecución
        if rr.F_final is None or len(rr.F_final) == 0:
            continue  # Salta a la siguiente iteración
        # Se filtran únicamente las soluciones que sean factibles según la transformación
        F_factibles = filtrar_soluciones_factibles(
            rr.F_final,
            modo_transformacion_objetivos=modo_transformacion_objetivos,
        )
        # Iteración sobre cada punto/solución factible en el frente filtrado
        for f in F_factibles:
            # Añade un diccionario con la información y valores de objetivos a la lista de filas
            filas.append({
                'algorithm': rr.algoritmo,
                'init': rr.inicializacion,
                'crossover': rr.crossover,
                'run': rr.replica,
                'seed': rr.semilla,
                'f1_compactness': f[0],
                'f2_transformed_tolerance': f[1],
                'f3_transformed_hamming_avg': f[2],
                'f4_balance_var': f[3],
            })
    # Si la lista de filas quedó vacía, se devuelve un DataFrame vacío con las columnas definidas
    if not filas:
        return pd.DataFrame(columns=columnas)
    # Se genera el DataFrame a partir de la lista de diccionarios
    df = pd.DataFrame(filas, columns=columnas)
    # Lista de columnas de referencia para identificar y eliminar filas duplicadas
    subset = [
        'algorithm', 'init', 'crossover', 'run', 'seed',
        'f1_compactness',
        'f2_transformed_tolerance',
        'f3_transformed_hamming_avg',
        'f4_balance_var',
    ]
    # Elimina los registros duplicados conservando la primera aparición y resetea el índice
    return df.drop_duplicates(subset=subset, keep='first').reset_index(drop=True)


def _inferir_modo_desde_nombre_csv(ruta_csv: Path, prefijo: str) -> Optional[str]:
    """Extrae el modo desde un nombre tipo '<prefijo><modo>.csv'."""
    # Obtención del nombre del archivo
    nombre = ruta_csv.name
    # Comprobación de que el archivo empieza con el prefijo y termina en la extensión correcta
    if not (nombre.startswith(prefijo) and nombre.endswith('.csv')):
        return None  # Retorna None si no cumple el formato esperado
    # Extrae el fragmento correspondiente al modo restando el prefijo y la extensión
    return nombre[len(prefijo):-4]


def _normalizar_columnas_frentes_csv(df_fronts: pd.DataFrame) -> Tuple[pd.DataFrame, List[Tuple[str, str]]]:
        """
        Normaliza nombres de columnas de frentes para compatibilidad en report-only.

        Esquema canónico actual:
            - f2_transformed_tolerance
            - f3_transformed_hamming_avg

        Alias compatibles (reportes antiguos):
            - f2_neg_tolerance
            - f3_neg_hamming_avg
        """
        # Si el DataFrame no es válido o está vacío, se retorna tal cual sin realizar cambios
        if df_fronts is None or df_fronts.empty:
                return df_fronts, []  # Retorna DataFrame original y lista de mapeos vacía

        # Se realiza una copia del DataFrame para evitar modificar el original directamente
        df_norm = df_fronts.copy()
        # Inicialización de la lista que guardará los pares de columnas renombradas
        columnas_renombradas = []

        # Diccionario con el mapeo de nombres de columnas antiguos a los nuevos canónicos
        alias_a_canonico = {
                'f2_neg_tolerance': 'f2_transformed_tolerance',
                'f3_neg_hamming_avg': 'f3_transformed_hamming_avg',
        }

        # Iteración sobre los elementos del mapeo de alias
        for alias, canonica in alias_a_canonico.items():
                # Si la columna canónica no existe pero el alias sí está presente
                if canonica not in df_norm.columns and alias in df_norm.columns:
                        # Renombrado in-place de la columna alias por su equivalente canónico
                        df_norm.rename(columns={alias: canonica}, inplace=True)
                        # Guarda el registro del cambio de nombre efectuado
                        columnas_renombradas.append((alias, canonica))

        # Devuelve el DataFrame normalizado y la lista de columnas modificadas
        return df_norm, columnas_renombradas


def _seleccionar_mas_reciente(ruta_input: Path, patron: str) -> Optional[Path]:
    """Devuelve el CSV más reciente por fecha de modificación."""
    # Busca todos los archivos que coincidan con el patrón proporcionado en la ruta
    candidatos = [p for p in ruta_input.glob(patron) if p.is_file()]
    # Si no hay archivos candidatos que cumplan los requisitos, retorna None
    if not candidatos:
        return None  # Retorna None indicando ausencia de archivos
    # Obtiene el candidato con el mayor tiempo de última modificación (mtime)
    return max(candidatos, key=lambda p: p.stat().st_mtime)


def _cargar_dataframes_report_only_csv(ruta_input: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[str]]:
    """Carga CSVs desde snp_tag/input para ejecutar sólo reportes/visualización (modo report-only-csv)."""
    # Si la ruta especificada no existe o no es un directorio válido, lanza un error
    if not ruta_input.exists() or not ruta_input.is_dir():
        raise FileNotFoundError(f"No existe el directorio de entrada fijo: {ruta_input}")

    # Se busca el CSV de resultados detallados más reciente en la carpeta de entrada
    ruta_detallado = _seleccionar_mas_reciente(ruta_input, 'resultados_detallados_*.csv')
    # Si no se encuentra ningún archivo de este tipo, se detiene con excepción
    if ruta_detallado is None:
        raise FileNotFoundError(
            f"No se encontró ningún 'resultados_detallados_*.csv' en {ruta_input}"
        )

    # Imprime en logs informativos la ruta del CSV detallado seleccionado
    logger.info(f"      • CSV detallado seleccionado: {ruta_detallado}")
    # Lectura del archivo CSV para guardarlo en un DataFrame de pandas
    df_final = pd.read_csv(ruta_detallado)

    # Comprueba que el DataFrame contenga información y no esté vacío
    if df_final.empty:
        raise ValueError(f"El CSV detallado está vacío: {ruta_detallado}")

    # Deducción del modo del experimento a partir del nombre del archivo detallado
    modo_detectado = _inferir_modo_desde_nombre_csv(ruta_detallado, 'resultados_detallados_')

    # Inicialización de la variable que alojará la ruta del histórico generacional
    ruta_hist = None
    # Si se ha detectado un modo, se busca el archivo histórico correspondiente a ese modo
    if modo_detectado:
        candidato_modo = ruta_input / f"historico_generacional_{modo_detectado}.csv"
        # Si el archivo del modo específico existe, se asigna como objetivo de carga
        if candidato_modo.exists() and candidato_modo.is_file():
            ruta_hist = candidato_modo
    # En caso de no existir, se toma el histórico más reciente que coincida con el patrón general
    if ruta_hist is None:
        ruta_hist = _seleccionar_mas_reciente(ruta_input, 'historico_generacional_*.csv')

    # Si se ha encontrado un archivo histórico válido
    if ruta_hist is not None:
        # Registro informativo en el logger
        logger.info(f"      • CSV histórico seleccionado: {ruta_hist}")
        # Carga del CSV del histórico generacional a un DataFrame
        df_gen = pd.read_csv(ruta_hist)
    else:
        # Advertencia en logs de que se procederá omitiendo el bloque de convergencia generacional
        logger.warning("      • ⚠️  No se encontró CSV histórico; se omitirá el bloque de convergencia.")
        # Se define un DataFrame vacío para evitar errores posteriores de nulidad
        df_gen = pd.DataFrame()

    # Inicialización de la variable que contendrá la ruta de los frentes de Pareto
    ruta_fronts = None
    # Si se detectó el modo, se busca el archivo de frentes específico
    if modo_detectado:
        candidato_fronts = ruta_input / f"frentes_pareto_{modo_detectado}.csv"
        # Comprueba existencia y asigna
        if candidato_fronts.exists() and candidato_fronts.is_file():
            ruta_fronts = candidato_fronts
    # Si sigue siendo None, se busca el más reciente que coincida con el patrón
    if ruta_fronts is None:
        ruta_fronts = _seleccionar_mas_reciente(ruta_input, 'frentes_pareto_*.csv')

    # Si no se localizó ningún CSV de frentes
    if ruta_fronts is None:
        # Registro de advertencia avisando que no se generará el análisis de Pareto
        logger.warning("      • ⚠️  No hay CSV de frentes; se omitirá Pareto con aviso explícito.")
        # Creación de DataFrame de frentes vacío
        df_fronts = pd.DataFrame()
    else:
        # Registro en logger del archivo de frentes seleccionado
        logger.info(f"      • CSV frentes seleccionado: {ruta_fronts}")
        # Carga de los frentes de Pareto a un DataFrame de pandas
        df_fronts = pd.read_csv(ruta_fronts)
        # Normalización de los nombres de columnas para que coincidan con la estructura canónica
        df_fronts, columnas_renombradas = _normalizar_columnas_frentes_csv(df_fronts)
        # Si se realizó alguna renombración de columnas, se notifica en los logs
        if columnas_renombradas:
            # Creación de una cadena con los detalles del renombrado de columnas
            detalle = ', '.join([f"{src}→{dst}" for src, dst in columnas_renombradas])
            # Impresión de logs del proceso de compatibilidad
            logger.info(f"      • Compatibilidad report-only aplicada en frentes: {detalle}")

    # Devuelve todos los dataframes leídos y el modo detectado del experimento
    return df_final, df_gen, df_fronts, modo_detectado


def ejecutar_pipeline_report_only_csv(args: Any) -> str:
    """Ejecuta únicamente la fase de reportes usando CSVs preexistentes en snp_tag/input (modo report-only-csv)."""
    # Captura el instante de tiempo en el que da comienzo el proceso
    inicio_total = time.time()

    # Define el directorio input relativo al archivo del orquestador
    ruta_input = Path(__file__).parent / 'input'

    # Imprime en la consola la sección correspondiente a la carga de datos CSV
    imprimir_subseccion("Carga de CSVs para Report-Only", icono="📥")
    # Realiza la llamada interna para recuperar los DataFrames y el modo de ejecución
    df_final, df_gen, df_fronts_total, modo_detectado = _cargar_dataframes_report_only_csv(
        # ruta_input: directorio absoluto que almacena los ficheros CSV históricos y la configuración INI
        ruta_input 
    )

    # Inicialización de overrides (parámetros del archivo .ini que sobrescriben la configuración por defecto) y la ruta del INI
    overrides = None
    candidato_ini = None
    # Si se conoce el modo, se intenta localizar su respectivo archivo de configuración .ini
    if modo_detectado:
        candidato_ini_modo = ruta_input / f"user_config_{modo_detectado}.ini"
        # Si el .ini específico del modo existe, se selecciona
        if candidato_ini_modo.exists():
            candidato_ini = candidato_ini_modo
    
    # Si no se ha encontrado el archivo específico, se busca el archivo de configuración más reciente
    if candidato_ini is None:
        # Lista los archivos con extensión .ini en el directorio input
        candidatos = list(ruta_input.glob("user_config*.ini"))
        # Si se encuentran archivos .ini, se toma el que se haya modificado más recientemente
        if candidatos:
            candidato_ini = max(candidatos, key=lambda p: p.stat().st_mtime)

    # Si se localizó algún archivo de configuración candidato
    if candidato_ini:
        # Se registra que se cargará la configuración del experimento correspondiente
        logger.info(f"      • Configuración cargada desde input: {candidato_ini.name}")
        try:
            # Intento de carga de parámetros desde el archivo .ini a una estructura dictionary
            overrides = cargar_params_tunables_desde_ini(candidato_ini)
        except Exception as e:
            # Advertencia en caso de que ocurra algún error inesperado al leer el .ini
            logger.warning(f"      • ⚠️ Error cargando config de input, usando por defecto: {e}")
    else:
        # Registro en logs de que se recurrirá a la configuración por defecto del sistema
        logger.warning("      • ⚠️ No se encontró user_config en input, usando por defecto.")

    # Se inicializa el objeto de configuración del experimento combinando el modo, origen de datos y overrides
    cfg = inicializar_configuracion(modo=args.mode, data_source=args.data_source, overrides=overrides)

    # Sobrescribe el modo de evaluación global estático con el valor extraído en la configuración (inyección dinámica)
    snp_tag.engine.metrics_logic._MODO_EVALUACION_GLOBAL = resolver_modo_evaluacion(cfg.modo_evaluacion)

    # Informa y muestra las características cargadas de la configuración
    informar_configuracion(cfg)

    # Crea los directorios específicos para guardar los reportes del dataset en report-only
    ruta_base, carpetas = crear_arbol_directorios_dataset(cfg, cfg.origen_datos, is_report_only_csv=True)
    # Asigna la colección de rutas generadas al objeto de configuración
    cfg.carpetas = carpetas
    
    # Añade un manejador de logs tipo archivo para guardar las trazas generadas
    add_file_handler(logger, os.path.join(ruta_base, "ejecucion.log"))
    
    # Registro del directorio base de salida para report-only
    logger.info(f"      • Carpeta de salida report-only: {ruta_base}")

    # Llama a la suite de visualización y reportes con todos los DataFrames
    ejecutar_reportes_visualizacion(cfg, df_final, df_gen, df_fronts_total)

    # Calcula la duración total que ha tomado la generación exclusiva de reportes
    duracion = time.time() - inicio_total
    # Conversión de segundos a formato legible de horas, minutos y segundos
    hor, rem = divmod(duracion, 3600)
    minu, seg = divmod(rem, 60)
    # Impresión estructurada del resumen temporal en logs
    logger.info(f"\n{'='*40}")
    logger.info(f"⏱️  TIEMPO TOTAL DE EJECUCIÓN (REPORT-ONLY): {int(hor)}h {int(minu)}m {seg:.2f}s")
    logger.info(f"{'='*40}\n")

    # Retorna la ruta raíz donde se han almacenado las imágenes y reportes finales
    return ruta_base


def ejecutar_pipeline(args: Any) -> str:
    """
    Ejecuta el pipeline Tag SNP de principio a fin con el estilo visual original.
    """
    # Obtiene el instante de tiempo en el que arranca todo el pipeline
    inicio_total = time.time()
    
    # 1. Configuración del sistema
    # Inicializa los parámetros operacionales a partir de los argumentos recibidos
    cfg = inicializar_configuracion(modo=args.mode, data_source=args.data_source)
    # Imprime en la consola la tabla/información estructurada de la configuración
    informar_configuracion(cfg)
    
    # Crear árbol de directorios para la salida de datos del experimento
    ruta_base, carpetas = crear_arbol_directorios_dataset(cfg, cfg.origen_datos)
    # Se actualizan las carpetas de salida en la variable de configuración
    cfg.carpetas = carpetas
    
    # Vincula el archivo físico de logs para este experimento específico
    add_file_handler(logger, os.path.join(ruta_base, "ejecucion.log"))
    
    # 2. Carga de Datos & 3. Diagnóstico y Análisis Exploratorio de Datos (EDA)
    # Llama al pipeline de diagnóstico, cargando genotipos y precalculando distancias
    H, snp_ids, posiciones, hap_ids, dvals = ejecutar_pipeline_diagnostico(cfg)
    
    # 5. Fase de ejecución del algoritmo evolutivo
    # Encabezado para delimitar el bloque del motor evolutivo en consola
    imprimir_encabezado("MOTOR MULTIOBJETIVO")
    
    # Se recupera el tamaño de las listas de configuraciones del experimento
    n_algoritmos = len(cfg.algoritmos_activos)  # Cantidad de algoritmos evolutivos habilitados
    n_inits = len(cfg.opciones_init)  # Tipos de inicialización a evaluar
    n_cross = len(cfg.crossover_operadores_activos)  # Operadores de cruce habilitados
    # Cálculo aritmético de la cantidad de ejecuciones que se van a procesar en total
    n_ejec_total = n_algoritmos * n_inits * n_cross * cfg.n_ejecuciones
    
    # Sección informativa en pantalla del plan de ejecuciones evolutivas
    imprimir_subseccion("Planificación de Ejecución", icono="📅")
    # Muestra el desglose total de ejecuciones previstas
    print(
        f"      • Total: {n_algoritmos} alg. x {n_inits} inits. x {n_cross} cruces x {cfg.n_ejecuciones} runs = {n_ejec_total} ejecuciones"
    )
    # Informa detalles del tamaño de población, generaciones y modos matemáticos
    print(
        f"      • Modo de ejecución: {cfg.modo_ejecucion.upper()} "
        f"(Población: {cfg.tam_poblacion}, Generaciones: {cfg.n_generaciones}, "
        f"Evaluación: {cfg.modo_evaluacion}, Transformación: {cfg.modo_transformacion_objetivos})"
    )
    
    # Construcción de vectores de referencia para algoritmos basados en descomposición (como NSGA-III)
    dirs_ref, n_part = construir_direcciones_referencia(cfg.tam_poblacion)
    # Registro en log de los puntos de referencia calculados y el número de particiones
    logger.info(f"      • Referencias (Das and Dennis): {len(dirs_ref)} puntos | Particiones: {n_part}")
    
    # Genera los índices de pares de haplotipos necesarios para evaluar distancias Hamming en paralelo
    pares_idx = np.array(list(combinations(range(cfg.n_haplotipos), 2)), dtype=np.int32)
    # Ejecuta el suite multiobjetivo completo devolviendo la lista de objetos de resultados
    resultados = ejecutar_suite_completa(H, pares_idx, cfg)
    
    # 6. Fase de recopilación y síntesis de métricas finales
    # Encabezado visual para separar el procesamiento de resultados en la terminal
    imprimir_encabezado("SÍNTESIS DE RESULTADOS")
    # Imprime información sobre la resolución de las gráficas que se van a generar
    imprimir_metadato("Configuración de síntesis", f"DPI={cfg.report_plot_dpi}", sangria=2)
    
    # Análisis de Rendimiento y Tiempos de Cómputo
    imprimir_subseccion("Análisis de Rendimiento y Tiempo", icono="⏱️")
    # Captura la marca temporal para medir este subproceso
    t0_perf = time.time()
    # Construcción de un dataframe resumen con los tiempos e información de cada réplica realizada
    df_perf = pd.DataFrame([
        {'algorithm': rr.algoritmo, 'init': rr.inicializacion, 'crossover': rr.crossover, 'time_seg': rr.tiempo_seg, 'run': rr.replica, 'frente_size': len(rr.F_final) if rr.F_final is not None else 0}
        for rr in resultados
    ])
    # Comprueba si el dataframe temporal contiene filas de información
    if not df_perf.empty:
        # Si la gráfica de tiempo está activa dentro de las opciones deseadas
        if 'tiempo' in cfg.graficas_activas:
            # Llama a la función gráfica para plasmar los diagramas de caja y bigotes de tiempos
            graficar_rendimiento_tiempo(df_perf, cfg.carpetas['tiempo'], cfg.modo_ejecucion)
        else:
            # Avisa que el usuario ha desactivado esta representación específica
            logger.info("      • ⚠️  Gráfica de tiempo omitida (user_config.ini).")
    # Registro de logs del tiempo invertido en este análisis específico
    logger.info(f"      • Tiempo bloque 'Análisis de Rendimiento y Tiempo': {time.time() - t0_perf:.1f}s")

    
    # Procesamiento y cálculo de métricas agregadas finales
    imprimir_subseccion("Procesamiento de Métricas (con progreso)", icono="🧮")
    # Log indicando el número de resultados de réplica a procesar
    logger.info(f"    • Iniciando métricas finales: {len(resultados)} ejecuciones")
    # Captura del tiempo inicial de cálculo de métricas finales
    t0_fin = time.time()
    # Ejecuta el cálculo masivo de métricas finales (hipervolumen, spacing, etc.)
    df_final, ideal_g, nadir_g = evaluar_metricas_finales(
        resultados, n_snps_total=cfg.n_snps, 
        modo_normalizacion=cfg.modo_normalizacion,
        hamming_pares=dvals,
        modo_transformacion_objetivos=cfg.modo_transformacion_objetivos,
    )
    # Notifica de la finalización e imprime el tiempo de cómputo invertido
    logger.info(f"      • Métricas finales completadas: {len(resultados)} ejecuciones en {time.time() - t0_fin:.1f}s")
    
    # Comienzo del cálculo y agregación de métricas generacionales paso a paso
    logger.info(f"\n    • Iniciando métricas generacionales: {len(resultados)} ejecuciones con historial")
    # Captura de tiempo inicial para métricas generacionales
    t0_gen = time.time()
    
    # Si el paso generacional está fijado en 0, se deshabilita la extracción de esta métrica
    if getattr(cfg, 'paso_generacional_metricas', 10) == 0:
        # Registro de advertencia por consola
        logger.warning("      • ⚠️ Cálculo de métricas generacionales deshabilitado (paso = 0).")
        # Generación de dataframe vacío
        df_gen = pd.DataFrame()
    else:
        # Se genera el dataframe con la evolución a lo largo de las generaciones de los algoritmos
        df_gen = construir_metricas_generacionales(
            resultados, ideal_g, nadir_g, n_snps_total=cfg.n_snps,
            modo_normalizacion=cfg.modo_normalizacion,
            hamming_pares=dvals,
            modo_transformacion_objetivos=cfg.modo_transformacion_objetivos,
            paso_generacional_metricas=cfg.paso_generacional_metricas,
        )
    # Notifica de la finalización e imprime el tiempo empleado en las métricas generacionales
    logger.info(f"      • Métricas generacionales completadas: {len(resultados)} ejecuciones en {time.time() - t0_gen:.1f}s")

    # Extrae y construye el DataFrame de frentes de Pareto combinando los resultados de cada ejecución
    df_fronts_total = _construir_df_fronts_desde_resultados(
        resultados,
        modo_transformacion_objetivos=cfg.modo_transformacion_objetivos,
    )
    
    # Exportación física y persistencia de la información en formato de tablas CSV
    imprimir_subseccion("Trazabilidad y Exportación de Datos (CSV)", icono="📊️")
    # Define la ruta absoluta y nombre del archivo para los resultados finales estructurados
    ruta_csv = os.path.join(cfg.carpetas['ejecuciones'], f"resultados_detallados_{cfg.modo_ejecucion}.csv")
    # Escritura del DataFrame a un archivo físico CSV
    df_final.to_csv(ruta_csv, index=False)
    # Impresión en terminal del guardado del archivo
    imprimir_grafico_guardado(ruta_csv, "Resultados detallados por ejecución")
    
    # Define la ruta absoluta para el histórico de progreso evolutivo
    ruta_hist = os.path.join(cfg.carpetas['ejecuciones'], f"historico_generacional_{cfg.modo_ejecucion}.csv")
    # Escritura del DataFrame histórico generacional a CSV
    df_gen.to_csv(ruta_hist, index=False)
    # Impresión en terminal de la acción finalizada
    imprimir_grafico_guardado(ruta_hist, "Historial evolutivo generacional")

    # Define la ruta absoluta de las soluciones asociadas a los frentes finales
    ruta_fronts_csv = os.path.join(cfg.carpetas['ejecuciones'], f"frentes_pareto_{cfg.modo_ejecucion}.csv")
    # Escritura a CSV de la información agregada del frente de Pareto
    df_fronts_total.to_csv(ruta_fronts_csv, index=False)
    # Impresión del correspondiente aviso
    imprimir_grafico_guardado(ruta_fronts_csv, "Soluciones de frentes finales (CSV)")
    
    # Copia el archivo .ini de entrada original para mantener la trazabilidad de los parámetros configurados
    ruta_config_exp = os.path.join(cfg.carpetas['ejecuciones'], f"user_config_{cfg.modo_ejecucion}.ini")
    # Si existe el archivo de configuración de partida original utilizado en la sesión
    if RUTA_USER_CONFIG.exists():
        # Se copia el archivo al directorio de ejecuciones junto a los resultados
        shutil.copyfile(RUTA_USER_CONFIG, ruta_config_exp)
        # Notifica en consola la exportación exitosa de la configuración
        imprimir_grafico_guardado(ruta_config_exp, "Configuración del experimento (INI)")
    
    # Bloque de información acerca de las cotas teóricas del espacio de objetivos
    imprimir_subseccion("Puntos Críticos del Espacio de Objetivos", icono="📍")
    
    def format_arr(arr):
        """Formatea arrays de coordenadas reales a strings compactos."""
        # Genera el string formateando cada elemento numérico con cuatro decimales
        return "[" + ", ".join([f"{x:12.4f}" for x in arr]) + "]"
        
    # Impresión de los límites empíricos extremos localizados (Ideal y Nadir)
    imprimir_metadato("Punto Ideal Empírico (mejor)", format_arr(ideal_g))
    imprimir_metadato("Punto Nadir Empírico (peor)", format_arr(nadir_g))
    
    # En caso de que se use un modo de normalización estático predefinido
    if cfg.modo_normalizacion in ('static_dataset_limits', 'static_proportional_limits'):
        # Obtiene las cotas teóricas estáticas correspondientes al dataset utilizado
        ideal_teorico, denom_teorico = obtener_referencias_estaticas_dataset(
            n_snps_total=cfg.n_snps,
            hamming_pares=dvals,
            modo_transformacion_objetivos=cfg.modo_transformacion_objetivos,
            modo_normalizacion=cfg.modo_normalizacion,
        )
        # Deducción matemática del punto Nadir Teórico
        nadir_teorico = denom_teorico + ideal_teorico - 1e-9
        # Cálculo del punto de referencia estándar para el cómputo del Hipervolumen (nadir + 10%)
        ref_hv_teorico = ideal_teorico + 1.1 * denom_teorico
        
        # Muestra en consola las tres referencias teóricas estáticas calculadas
        imprimir_metadato("Punto Ideal Teórico (Ref)", format_arr(ideal_teorico))
        imprimir_metadato("Punto Nadir Teórico (Ref)", format_arr(nadir_teorico))
        imprimir_metadato("Punto Ref. Hipervolumen (1.1)", format_arr(ref_hv_teorico))
    
    # 7. Generación final de reportes de visualización y gráficos estáticos
    ejecutar_reportes_visualizacion(cfg, df_final, df_gen, df_fronts_total)
    
    # Cálculo de la duración temporal total del pipeline evolutivo
    duracion = time.time() - inicio_total
    # Conversión del tiempo a formato de horas, minutos y segundos
    hor, rem = divmod(duracion, 3600)
    minu, seg = divmod(rem, 60)
    # Impresión final en consola de conclusión exitosa del proceso completo
    logger.info(f"\n{'='*40}")
    logger.info(f"⏱️  TIEMPO TOTAL DE EJECUCIÓN: {int(hor)}h {int(minu)}m {seg:.2f}s")
    logger.info(f"{'='*40}\n")
    
    # Retorna la ruta base donde se almacenaron todos los directorios y archivos de la simulación
    return ruta_base
