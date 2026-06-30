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


def _construir_df_fronts_desde_resultados(resultados: List[Any], modo_transformacion_objetivos: str = 'neg', modo_evaluacion: str = 'absoluta') -> pd.DataFrame:
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
            modo_evaluacion=modo_evaluacion,
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


def _cargar_dataframes_postprocessing(ruta_input: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[str]]:
    """Carga CSVs desde snp_tag/input para ejecutar sólo el postprocesamiento estadístico y visual."""
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
            logger.info(f"      • Compatibilidad postprocessing aplicada en frentes: {detalle}")

    # Devuelve todos los dataframes leídos y el modo detectado del experimento
    return df_final, df_gen, df_fronts, modo_detectado


def ejecutar_pipeline_postprocessing(args: Any) -> str:
    """Ejecuta únicamente la fase de postprocesamiento usando CSVs preexistentes en snp_tag/input."""
    # Captura el instante de tiempo en el que da comienzo el proceso
    inicio_total = time.time()

    # Define el directorio input relativo al archivo del orquestador
    ruta_input = Path(__file__).parent / 'input'

    # Imprime en la consola la sección correspondiente a la carga de datos CSV
    imprimir_subseccion("Carga de CSVs para Post-procesamiento", icono="📥")
    # Realiza la llamada interna para recuperar los DataFrames y el modo de ejecución
    df_final, df_gen, df_fronts_total, modo_detectado = _cargar_dataframes_postprocessing(
        # ruta_input: directorio absoluto que almacena los ficheros CSV históricos y la configuración INI
        ruta_input 
    )

    import configparser
    
    # Inicialización de overrides (parámetros del archivo .ini que sobrescriben la configuración por defecto) y la ruta del INI
    overrides = None
    candidato_ini = None
    
    # Buscar estrictamente el archivo exportado
    candidatos = list(ruta_input.glob("exported_user_config*.ini"))
    if candidatos:
        # Se toma el que se haya modificado más recientemente
        candidato_ini = max(candidatos, key=lambda p: p.stat().st_mtime)

    # Si se localizó el archivo de configuración exportado
    modo_final = None
    data_source_final = None
    
    if candidato_ini:
        # Se registra que se cargará la configuración del experimento correspondiente
        logger.info(f"      • Configuración cargada desde input: {candidato_ini.name}")
        
        # Extraer modo y dataset manualmente de la sección [Argumentos]
        parser_args = configparser.ConfigParser()
        parser_args.read(candidato_ini, encoding='utf-8')
        if parser_args.has_section('Argumentos'):
            if parser_args.has_option('Argumentos', 'modo_ejecucion'):
                modo_final = parser_args.get('Argumentos', 'modo_ejecucion').strip()
            if parser_args.has_option('Argumentos', 'origen_datos'):
                data_source_final = parser_args.get('Argumentos', 'origen_datos').strip()
        else:
            raise ValueError(f"El archivo {candidato_ini.name} no tiene la sección [Argumentos] obligatoria.")
        
        try:
            # Intento de carga de parámetros tunables ignorando [Argumentos] internamente
            overrides = cargar_params_tunables_desde_ini(candidato_ini)
        except Exception as e:
            # Error fatal en caso de que ocurra algún error inesperado al leer el .ini
            raise RuntimeError(f"Error cargando config exportada desde input: {e}")
    else:
        # Error fatal si no hay archivo exportado
        raise FileNotFoundError("No se encontró ningún archivo 'exported_user_config*.ini' en la carpeta input/. Es obligatorio para --post-processing.")

    if not modo_final or not data_source_final:
        raise ValueError("El archivo exportado no contiene 'modo_ejecucion' u 'origen_datos' dentro de [Argumentos].")

    # Se inicializa el objeto de configuración del experimento usando los valores extraídos del archivo
    cfg = inicializar_configuracion(modo=modo_final, data_source=data_source_final, overrides=overrides)

    # Informa y muestra las características cargadas de la configuración
    informar_configuracion(cfg)

    # Crea los directorios específicos para guardar los reportes del dataset en postprocesamiento
    ruta_base, carpetas = crear_arbol_directorios_dataset(cfg, cfg.origen_datos, is_postprocessing=True)
    # Asigna la colección de rutas generadas al objeto de configuración
    cfg.carpetas = carpetas
    
    # Añade un manejador de logs tipo archivo para guardar las trazas generadas
    add_file_handler(logger, os.path.join(ruta_base, "ejecucion.log"))
    
    # Registro del directorio base de salida para postprocesamiento
    logger.info(f"      • Carpeta de salida postprocesamiento: {ruta_base}")

    # Llama a la suite de visualización y reportes con todos los DataFrames
    # === PARCHE RECALCULO DE METRICAS ===
    if "metricas_finales" in cfg.postprocesamiento_activo or "metricas_generacionales" in cfg.postprocesamiento_activo:
        logger.info("      • Recalculando métricas IGD+ y GD+ debido a flags explícitas en configuración...")
        from pymoo.indicators.igd_plus import IGDPlus
        from pymoo.indicators.gd_plus import GDPlus
        import numpy as np
        
        # 1. Obtener frente de referencia global corregido desde df_fronts_total
        cols_obj = [c for c in df_fronts_total.columns if c.startswith('f1_') or c.startswith('f2_') or c.startswith('f3_') or c.startswith('f4_')]
        all_points = df_fronts_total[cols_obj].values
        
        ideal_g = all_points.min(axis=0)
        nadir_g = all_points.max(axis=0)
        denom_g = nadir_g - ideal_g + 1e-9
        
        from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
        nds = NonDominatedSorting()
        F_actual = None
        for i in range(0, len(all_points), 20000):
            chunk_P = all_points[i:i+20000]
            if F_actual is not None:
                chunk_P = np.vstack((chunk_P, F_actual))
            if len(chunk_P) > 1:
                chunk_P = np.unique(chunk_P, axis=0)
            fronts = nds.do(chunk_P, only_non_dominated_front=True)
            F_actual = chunk_P[fronts]
            
        F_ref_norm = np.clip((F_actual - ideal_g) / denom_g, 0, 1)
        igd_plus_metric = IGDPlus(F_ref_norm)
        gd_plus_metric = GDPlus(F_ref_norm)
        
        # 2. Recálculo Métricas Finales (por combo, todas las runs agregadas)
        if "metricas_finales" in cfg.postprocesamiento_activo:
            logger.info("      • [metricas_finales] Recalculando IGD+ y GD+ en df_final (por combo)...")
            combos = df_final[['algorithm', 'init', 'crossover']].drop_duplicates()
            for _, combo in combos.iterrows():
                mask_fronts = (
                    (df_fronts_total['algorithm'] == combo['algorithm']) &
                    (df_fronts_total['init']      == combo['init'])      &
                    (df_fronts_total['crossover'] == combo['crossover'])
                )
                exec_points = df_fronts_total.loc[mask_fronts, cols_obj].values
                if len(exec_points) == 0:
                    continue
                exec_points = np.unique(exec_points, axis=0)
                F_crudo_norm = np.clip((exec_points - ideal_g) / denom_g, 0, 1)
                igd_val = float(igd_plus_metric.do(F_crudo_norm))
                gd_val  = float(gd_plus_metric.do(F_crudo_norm))
                # Asignar el mismo valor a todas las runs de este combo
                mask_df = (
                    (df_final['algorithm'] == combo['algorithm']) &
                    (df_final['init']      == combo['init'])      &
                    (df_final['crossover'] == combo['crossover'])
                )
                df_final.loc[mask_df, 'IGD+'] = igd_val
                df_final.loc[mask_df, 'GD+']  = gd_val
                logger.info(f"        – {combo['algorithm']}+{combo['init']}+{combo['crossover']}: IGD+={igd_val:.4f}  GD+={gd_val:.4f}")

        # 3. Recálculo Métricas Generacionales
        if "metricas_generacionales" in cfg.postprocesamiento_activo:
            logger.info("      • [metricas_generacionales] Leyendo checkpoints NPZ para recalcular df_gen...")
            dir_checkpoints = ruta_input / "checkpoints"
            if not dir_checkpoints.exists():
                logger.warning("      • NO EXISTE la carpeta checkpoints. Se omite el recálculo generacional.")
            else:
                for npz_file in dir_checkpoints.glob("*.npz"):
                    # Extraer metadata del nombre del archivo chk_{alg}_{ini}_{cro}_rep{run}.npz
                    stem = npz_file.stem
                    parts = stem.split('_')
                    if len(parts) >= 5:
                        alg = parts[1]
                        cro = parts[-2]
                        run_str = parts[-1].replace("rep", "")
                        run = int(run_str) if run_str.isdigit() else 1
                        ini = "_".join(parts[2:-2])
                        
                        mask_df = (df_gen['algorithm'] == alg) & \
                                  (df_gen['init'] == ini) & \
                                  (df_gen['crossover'] == cro) & \
                                  (df_gen['run'] == run)
                        
                        if not mask_df.any():
                            continue
                            
                        try:
                            with np.load(npz_file) as data:
                                indices = data['gen_indices']
                                for i, gen in enumerate(indices):
                                    F_crudo = data.get(f'hist_F_{i}')
                                    if F_crudo is None or len(F_crudo) == 0:
                                        continue
                                        
                                    # El F_crudo guardado en numpy array no está filtrado ni transformado
                                    # Por simplicidad, ya que el historial npz guarda crudo... espera.
                                    # Oh, en metrics_logic.py se aplica filtrar_soluciones_factibles
                                    from snp_tag.engine.metrics_logic import filtrar_soluciones_factibles
                                    F_crudo = np.array(F_crudo, dtype=float)
                                    if F_crudo.ndim == 1:
                                        F_crudo = F_crudo.reshape(1, -1)
                                    F_crudo = filtrar_soluciones_factibles(F_crudo, modo_transformacion_objetivos=cfg.modo_transformacion_objetivos, modo_evaluacion=cfg.modo_evaluacion)
                                    
                                    if len(F_crudo) > 0:
                                        F_crudo = np.unique(F_crudo, axis=0)
                                        F_crudo_norm = np.clip((F_crudo - ideal_g) / denom_g, 0, 1)
                                        val_igd = float(igd_plus_metric.do(F_crudo_norm))
                                        val_gd = float(gd_plus_metric.do(F_crudo_norm))
                                        
                                        # Actualizar dataframe
                                        idx_to_update = df_gen[mask_df & (df_gen['generation'] == gen)].index
                                        for i_update in idx_to_update:
                                            df_gen.at[i_update, 'IGD+'] = val_igd
                                            df_gen.at[i_update, 'GD+'] = val_gd
                        except Exception as e:
                            logger.error(f"Error procesando {npz_file.name}: {e}")

        # 4. Guardar CSVs recalculados en 1_ejecuciones/ del experimento de postprocesamiento
        from pathlib import Path as _Path
        _ruta_ejec = _Path(cfg.carpetas['ejecuciones'])
        _ruta_ejec.mkdir(parents=True, exist_ok=True)

        if "metricas_finales" in cfg.postprocesamiento_activo:
            _ruta_csv_final = _ruta_ejec / f"resultados_detallados_{modo_detectado}.csv"
            df_final.to_csv(_ruta_csv_final, index=False)
            logger.info(f"      • CSV final recalculado guardado: {_ruta_csv_final.name}")

        if "metricas_generacionales" in cfg.postprocesamiento_activo:
            _ruta_csv_gen = _ruta_ejec / f"historico_generacional_{modo_detectado}.csv"
            df_gen.to_csv(_ruta_csv_gen, index=False)
            logger.info(f"      • CSV generacional recalculado guardado: {_ruta_csv_gen.name}")
    # === FIN PARCHE RECALCULO ===

    ejecutar_reportes_visualizacion(cfg, df_final, df_gen, df_fronts_total, is_postprocessing=True)

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


def _anexar_columnas_lowess(df: pd.DataFrame, modo_transformacion_objetivos: str = 'neg', modo_evaluacion: str = 'absoluta') -> pd.DataFrame:
    """Calcula y anexa las curvas suavizadas LOWESS para los pares de métricas principales en escala física."""
    if df.empty:
        return df
        
    import statsmodels.api as sm
    import warnings
    from snp_tag.engine.metrics_logic import decodificar_objetivos_reales
    
    # Decodificar objetivos a escala física real
    requeridas = ['f1_compactness', 'f2_transformed_tolerance', 'f3_transformed_hamming_avg', 'f4_balance_var']
    faltantes = [c for c in requeridas if c not in df.columns]
    if faltantes:
        return df # Si faltan columnas, no hacer nada
        
    objetivos_reales = decodificar_objetivos_reales(
        df[requeridas].to_numpy(dtype=float),
        modo_transformacion_objetivos=modo_transformacion_objetivos,
        modo_evaluacion=modo_evaluacion,
    )
    
    # Asignar columnas temporales en escala física
    df['_temp_Comp'] = objetivos_reales['compacidad']
    df['_temp_Tol'] = objetivos_reales['tolerancia_real']
    df['_temp_Hamm'] = objetivos_reales['hamming_prom_real']
    df['_temp_Bal'] = objetivos_reales['balance_var']
    
    # Pares de métricas a suavizar (x_col, y_col) en la escala real
    pares_metricas = [
        ('_temp_Comp', '_temp_Tol'),
        ('_temp_Comp', '_temp_Hamm'),
        ('_temp_Comp', '_temp_Bal'),
        ('_temp_Tol', '_temp_Hamm'),
        ('_temp_Tol', '_temp_Bal'),
        ('_temp_Hamm', '_temp_Bal')
    ]
    
    # Nombres de las nuevas columnas para el CSV
    nombres_cols = {
        ('_temp_Comp', '_temp_Tol'): 'lowess_Comp_vs_Tol',
        ('_temp_Comp', '_temp_Hamm'): 'lowess_Comp_vs_Hamm',
        ('_temp_Comp', '_temp_Bal'): 'lowess_Comp_vs_Bal',
        ('_temp_Tol', '_temp_Hamm'): 'lowess_Tol_vs_Hamm',
        ('_temp_Tol', '_temp_Bal'): 'lowess_Tol_vs_Bal',
        ('_temp_Hamm', '_temp_Bal'): 'lowess_Hamm_vs_Bal',
    }
    
    for col in nombres_cols.values():
        df[col] = np.nan
        
    df['init_cross'] = df['init'].astype(str) + '+' + df['crossover'].astype(str)
    grupos = df.groupby(['algorithm', 'init_cross'])
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*divide.*")
        
        for name, group in grupos:
            for x_col, y_col in pares_metricas:
                col_name = nombres_cols[(x_col, y_col)]
                datos_ord = group[[x_col, y_col]].sort_values(by=x_col).dropna()
                
                if len(datos_ord) > 12 and datos_ord[x_col].std() > 1e-5:
                    try:
                        # Frac=0.66 y return_sorted=False como pidió el usuario
                        z_y = sm.nonparametric.lowess(datos_ord[y_col], datos_ord[x_col], frac=0.66, it=3, return_sorted=False)
                        df.loc[datos_ord.index, col_name] = z_y
                    except Exception:
                        pass
                        
    # Limpiar columnas temporales
    df = df.drop(columns=['init_cross', '_temp_Comp', '_temp_Tol', '_temp_Hamm', '_temp_Bal'])
    
    return df


def ejecutar_pipeline(args: Any) -> str:
    """
    Ejecuta el pipeline Tag SNP de principio a fin con el estilo visual original.
    """
    # Obtiene el instante de tiempo en el que arranca todo el pipeline
    inicio_total = time.time()
    
    is_resume = getattr(args, 'resume', None) is not None
    
    if is_resume:
        from pathlib import Path
        import configparser
        from snp_tag.config import cargar_params_tunables_desde_ini
        
        ruta_resume = Path(args.resume)
        if not ruta_resume.exists() or not ruta_resume.is_dir():
            raise FileNotFoundError(f"Directorio de reanudación no encontrado: {ruta_resume}")
        
        archivos_exportados = list((ruta_resume / '1_ejecuciones').glob('exported_user_config_*.ini'))
        if not archivos_exportados:
            raise FileNotFoundError(f"No se encontró archivo de configuración exportado en {ruta_resume / '1_ejecuciones'}")
        
        ruta_config_exp = archivos_exportados[0]
        parser_ini = configparser.ConfigParser(interpolation=None)
        parser_ini.read(ruta_config_exp, encoding='utf-8')
        try:
            modo_final = parser_ini.get('Argumentos', 'modo_ejecucion').strip()
            data_source_final = parser_ini.get('Argumentos', 'origen_datos').strip()
        except configparser.NoOptionError:
            raise ValueError("El archivo exportado no contiene 'modo_ejecucion' o 'origen_datos'.")
            
        overrides = cargar_params_tunables_desde_ini(ruta_config_exp)
        cfg = inicializar_configuracion(modo=modo_final, data_source=data_source_final, overrides=overrides)
        informar_configuracion(cfg)
        
        ruta_base = str(ruta_resume)
        
        comparativa_root = ruta_resume / '2_comparativa'
        frentes_root = comparativa_root / '1_frentes'
        sintesis_root = comparativa_root / '3_sintesis'

        carpetas = {
            'datos': ruta_resume / '0_datos_previos',
            'ejecuciones': ruta_resume / '1_ejecuciones',
            'checkpoints': ruta_resume / '1_ejecuciones' / 'checkpoints',
            'comparativa': comparativa_root,
            'tiempo': comparativa_root / '0_tiempo',
            'frentes': frentes_root,
            'frentes_pareto': frentes_root / 'frentes_pareto',
            'frentes_paralelas': frentes_root / 'coordenadas_paralelas',
            'frentes_otros': frentes_root / 'otros',
            'metricas_convergencia': comparativa_root / '2_metricas_convergencia',
            'sintesis': sintesis_root,
            'sintesis_boxplots': sintesis_root / '0_boxplots',
            'sintesis_violines': sintesis_root / '1_violines',
            'sintesis_barras': sintesis_root / '2_barras',
            'rankings': comparativa_root / '4_rankings',
            'decision_mcdm': comparativa_root / '5_decision_mcdm',
        }

        # Asegurar que todas existan físicamente para que matplotlib no falle
        for p in carpetas.values():
            p.mkdir(parents=True, exist_ok=True)

        carpetas_str = {k: str(v) for k, v in carpetas.items()}
        carpetas_str['comparativa_root'] = carpetas_str['comparativa']
        cfg.carpetas = carpetas_str
        
        logger.info(f"      • Reanudando experimento desde: {ruta_base}")
    else:
        # 1. Configuración del sistema
        # Inicializa los parámetros operacionales a partir de los argumentos recibidos
        cfg = inicializar_configuracion(modo=args.mode, data_source=args.data_source)
        # Imprime en la consola la tabla/información estructurada de la configuración
        informar_configuracion(cfg)
        
        # Crear árbol de directorios para la salida de datos del experimento
        ruta_base, carpetas = crear_arbol_directorios_dataset(cfg, cfg.origen_datos)
        # Se actualizan las carpetas de salida en la variable de configuración
        cfg.carpetas = carpetas
        
        # Copia el archivo .ini de entrada original al principio para capturar la configuración exacta
        ruta_config_exp = os.path.join(cfg.carpetas['ejecuciones'], f"exported_user_config_{cfg.modo_ejecucion}.ini")
        if RUTA_USER_CONFIG.exists():
            with open(RUTA_USER_CONFIG, 'r', encoding='utf-8') as f:
                contenido_ini = f.read()
                
            seccion_argumentos = f"\n\n[Argumentos]\nmodo_ejecucion = {cfg.modo_ejecucion}\norigen_datos = {cfg.origen_datos}\n"
            
            with open(ruta_config_exp, 'w', encoding='utf-8') as f:
                f.write(contenido_ini + seccion_argumentos)
    
    # Vincula el archivo físico de logs para este experimento específico
    add_file_handler(logger, os.path.join(ruta_base, "ejecucion.log"))
    
    # 2. Carga de Datos & 3. Diagnóstico y Análisis Exploratorio de Datos (EDA)
    # Llama al pipeline de diagnóstico, cargando genotipos y precalculando distancias
    H, snp_ids, posiciones, hap_ids, dvals = ejecutar_pipeline_diagnostico(cfg)
    
    # 5. Fase de ejecución del algoritmo evolutivo
    # Encabezado para delimitar el bloque del motor evolutivo en consola
    imprimir_encabezado("OPTIMIZACIÓN")
    
    # Se recupera el tamaño de las listas de configuraciones del experimento
    n_algoritmos = len(cfg.algoritmos_activos)  # Cantidad de algoritmos evolutivos habilitados
    n_inits = len(cfg.opciones_init)  # Tipos de inicialización a evaluar
    n_cross = len(cfg.crossover_operadores_activos)  # Operadores de cruce habilitados
    # Cálculo aritmético de la cantidad de ejecuciones que se van a procesar en total
    n_ejec_total = n_algoritmos * n_inits * n_cross * cfg.n_ejecuciones
    
    # Sección informativa en pantalla del plan de ejecuciones evolutivas
    imprimir_subseccion("Planificación", icono="📅")
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
    imprimir_encabezado("SÍNTESIS")
    # Imprime información sobre la resolución de las gráficas que se van a generar
    imprimir_metadato("Configuración de síntesis", f"DPI={cfg.report_plot_dpi}", sangria=2)
    
    # Análisis de Rendimiento y Tiempos de Cómputo
    imprimir_subseccion("Rendimiento y Tiempos", icono="⏱️")
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
        if 'tiempo' in cfg.postprocesamiento_activo:
            # Llama a la función gráfica para plasmar los diagramas de caja y bigotes de tiempos
            graficar_rendimiento_tiempo(df_perf, cfg.carpetas['tiempo'], cfg.modo_ejecucion)
        else:
            # Avisa que el usuario ha desactivado esta representación específica
            logger.info("      • ⚠️  Gráfica de tiempo omitida (user_config.ini).")
    # Registro de logs del tiempo invertido en este análisis específico
    logger.info(f"      • Tiempo bloque 'Análisis de Rendimiento y Tiempo': {time.time() - t0_perf:.1f}s")

    
    # Procesamiento y cálculo de métricas agregadas finales
    imprimir_subseccion("Cálculo de Métricas", icono="🧮")
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
        modo_evaluacion=cfg.modo_evaluacion,
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
            modo_evaluacion=cfg.modo_evaluacion,
        )
    # Notifica de la finalización e imprime el tiempo empleado en las métricas generacionales
    logger.info(f"      • Métricas generacionales completadas: {len(resultados)} ejecuciones en {time.time() - t0_gen:.1f}s")

    # Extrae y construye el DataFrame de frentes de Pareto combinando los resultados de cada ejecución
    df_fronts_total = _construir_df_fronts_desde_resultados(
        resultados,
        modo_transformacion_objetivos=cfg.modo_transformacion_objetivos,
        modo_evaluacion=cfg.modo_evaluacion,
    )
    
    # Calcula y anexa las columnas LOWESS para suavizado en el mismo CSV
    df_fronts_total = _anexar_columnas_lowess(
        df_fronts_total,
        modo_transformacion_objetivos=cfg.modo_transformacion_objetivos,
        modo_evaluacion=cfg.modo_evaluacion
    )
    
    # Exportación física y persistencia de la información en formato de tablas CSV
    imprimir_subseccion("Exportación de Datos", icono="📊️")
    
    # Reordenar las columnas de df_final para que las métricas coincidan con la consola
    cols_config = ['algorithm', 'init', 'crossover', 'run', 'seed']
    cols_metricas = [
        'Range', 'MinSum', 'SumMin', 'MaxToleranceRate', 'AvgToleranceRate', 
        'AvgHammingDistance', 'Hypervolume', 'IGD+', 'GD+'
    ]
    # Se obtienen las columnas extra asegurándose de no duplicar
    cols_extra = [c for c in df_final.columns if c not in cols_config and c not in cols_metricas]
    # Se concatenan en el orden correcto asegurando que existan en el DataFrame original
    orden_final = [c for c in cols_config + cols_metricas + cols_extra if c in df_final.columns]
    df_final = df_final[orden_final]

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
