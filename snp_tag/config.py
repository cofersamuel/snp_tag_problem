"""
Módulo de Configuración Experimental (config.py)
-----------------------------------------------
Define las estructuras de datos y parámetros globales requeridos para la
ejecución controlada de experimentos de búsqueda multiobjetivo de Tag SNPs.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from pathlib import Path
import re
import configparser


RUTA_USER_CONFIG = Path(__file__).resolve().parent / "user_config.ini"

CLAVES_TUNABLES_REQUERIDAS = (
    'num_bloques',
    'semilla_maestra',
    'n_snps',
    'dir_salida_base',
    'pc',
    'vecinos_moead',
    'prob_vecindad_moead',
    'theta_moead_pbi',
    'algoritmos_activos',
    'opciones_init',
    'modo_normalizacion',
    'modo_evaluacion',
    'modo_semillas',
    'modo_transformacion_objetivos',
    'prob_flip_sintetico',
    'dif_min_pares_sintetico',
    'intentos_max_sintetico',
    'report_plot_dpi',
    'prob_aleatoria_gi',
)


def _parsear_valor_atomico(valor_raw: str) -> Any:
    """Parsea un valor escalar desde texto plano (int/float/bool/str)."""
    valor = str(valor_raw).strip()
    if not valor:
        raise ValueError("Se encontró un valor vacío en la configuración.")

    if (valor.startswith("'") and valor.endswith("'")) or (
        valor.startswith('"') and valor.endswith('"')
    ):
        return valor[1:-1]

    valor_lower = valor.lower()
    if valor_lower == 'true':
        return True
    if valor_lower == 'false':
        return False

    if re.fullmatch(r"[+-]?\d+", valor):
        return int(valor)

    if re.fullmatch(r"[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?", valor):
        return float(valor)

    return valor


def _parsear_valor_tunable(valor_raw: str) -> Any:
    """Parsea un valor tunable, permitiendo listas estilo [a, b, c] o a, b, c."""
    valor = str(valor_raw).strip()

    if valor.startswith('[') and valor.endswith(']'):
        contenido = valor[1:-1].strip()
        if not contenido:
            return []
        tokens = [t.strip() for t in contenido.split(',')]
        if any(not t for t in tokens):
            raise ValueError(f"Lista mal formada: {valor}")
        return [_parsear_valor_atomico(t) for t in tokens]

    if ',' in valor:
        tokens = [t.strip() for t in valor.split(',')]
        if any(not t for t in tokens):
            raise ValueError(f"Lista mal formada: {valor}")
        return [_parsear_valor_atomico(t) for t in tokens]

    return _parsear_valor_atomico(valor)


def cargar_params_tunables_desde_ini(ruta_config: Path = RUTA_USER_CONFIG) -> Dict[str, Any]:
    """
    Carga parámetros tunables desde un fichero INI con secciones.

    Formato recomendado:
        [Seccion]
        clave = valor ; explicación
    """
    ruta = Path(ruta_config)
    if not ruta.exists() or not ruta.is_file():
        raise FileNotFoundError(
            f"No se encontró el archivo de configuración de usuario: {ruta}. "
            "Debes crear/editar 'user_config.ini' dentro del paquete 'snp_tag/' (por defecto: snp_tag/user_config.ini)."
        )

    parser = configparser.ConfigParser(
        interpolation=None,
        inline_comment_prefixes=('#', ';')
    )
    parser.optionxform = str
    parser.read(ruta, encoding='utf-8')

    params: Dict[str, Any] = {}

    for seccion in parser.sections():
        for clave, valor_txt in parser.items(seccion):
            if clave in params:
                raise ValueError(
                    f"Clave duplicada en {ruta}: '{clave}' aparece en más de una sección."
                )

            valor_limpio = str(valor_txt).strip()
            if not valor_limpio:
                raise ValueError(
                    f"Valor vacío para '{clave}' en la sección [{seccion}] de {ruta}."
                )

            try:
                params[clave] = _parsear_valor_tunable(valor_limpio)
            except ValueError as e:
                raise ValueError(
                    f"Error parseando '{clave}' en sección [{seccion}] de {ruta}: {e}"
                ) from e

    claves_desconocidas = sorted(set(params.keys()) - set(CLAVES_TUNABLES_REQUERIDAS))
    if claves_desconocidas:
        raise ValueError(
            "Se encontraron claves no soportadas en user_config.ini: "
            f"{claves_desconocidas}. "
            f"Claves válidas: {sorted(CLAVES_TUNABLES_REQUERIDAS)}"
        )

    faltantes = sorted(set(CLAVES_TUNABLES_REQUERIDAS) - set(params.keys()))
    if faltantes:
        raise ValueError(
            "Faltan parámetros obligatorios en user_config.ini: "
            f"{faltantes}"
        )

    return params


# -----------------------------------------------------------------------------
# BLOQUE ÚNICO DE PARÁMETROS (editar aquí)
#
# Guía rápida:
# - perfiles_modo: define presets de ejecución (fast/medium/high/full).
# - fuentes_datos_disponibles: datasets aceptados por el sistema.
# - modos_normalizacion_disponibles: modos válidos para normalizar métricas.
# - tunables: parámetros editables del experimento.
#
# Recomendación:
# - Si quieres cambiar comportamiento global, empieza por "tunables".
# - Si quieres ajustar coste/calidad de búsqueda, modifica "perfiles_modo".
# -----------------------------------------------------------------------------
PARAMETROS_CONFIGURACION = {
    # Perfiles predefinidos de ejecución.
    # Opciones disponibles: 'fast', 'medium', 'high', 'full'.
    # Cada perfil controla:
    # - tam_pob: tamaño de población.
    # - n_gen: número de generaciones.
    # - descendencia: número de descendientes por generación.
    # - n_runs: repeticiones por configuración algoritmo-init.
    'perfiles_modo': {
        'fast': {'tam_pob': 10, 'n_gen': 2, 'descendencia': 10, 'n_runs': 2},
        'medium': {'tam_pob': 100, 'n_gen': 50, 'descendencia': 100, 'n_runs': 2},
        'high': {'tam_pob': 120, 'n_gen': 100, 'descendencia': 120, 'n_runs': 3},
        'full': {'tam_pob': 200, 'n_gen': 500, 'descendencia': 200, 'n_runs': 5},
    },

    # Fuentes de datos permitidas al construir configuración.
    # Valores válidos:
    # - 'hinds2005': benchmark real (bloque histórico Hinds et al.)
    # - 'synthetic': bloque sintético generado por el sistema
    'fuentes_datos_disponibles': ('hinds2005', 'synthetic'),

    # Modos permitidos para normalización de métricas.
    # Valores válidos:
    # - 'per_algorithm': normaliza por algoritmo.
    # - 'global_all_pairs': normaliza con referencias globales.
    # - 'static_dataset_limits': normaliza con límites fijos del dataset.
    # - 'static_proportional_limits': normaliza asumiendo que los objetivos son proporciones (ej. Ting).
    'modos_normalizacion_disponibles': ('per_algorithm', 'global_all_pairs', 'static_dataset_limits', 'static_proportional_limits'),

    # Modos permitidos para la evaluación del fitness.
    # - 'absoluta': usa las distancias de Hamming totales.
    # - 'proportional': escala las distancias por la cantidad de SNPs (ej. métrica de Ting).
    'modos_evaluacion_disponibles': ('absoluta', 'proportional'),

    # Modos de semilla para el motor evolutivo.
    # - 'non_deterministic': semillas por tiempo/sistema (modo habitual)
    # - 'deterministic': semillas derivadas de semilla_maestra
    'modos_semilla_disponibles': ('non_deterministic', 'deterministic'),

    # Modos de transformación de objetivos de maximización a minimización.
    # - 'neg': usa -f (formulación actual)
    # - 'inverse': usa 1/f (estilo Ting para objetivos de maximización)
    'modos_transformacion_objetivos_disponibles': ('neg', 'inverse'),

    # Algoritmos disponibles para ejecución (selección tunable).
    # MOEA/D se expone en tres variantes independientes.
    'algoritmos_disponibles': (
        'NSGA2', 'NSGA3', 'SPEA2',
        'MOEAD_TCHE', 'MOEAD_PBI', 'MOEAD_WS'
    ),

    # Parámetros tunables editables (se cargan desde user_config.ini).
    'tunables': {},
}

PARAMETROS_CONFIGURACION['tunables'] = cargar_params_tunables_desde_ini()

# Alias de compatibilidad interna/externa
PERFILES_MODO = PARAMETROS_CONFIGURACION['perfiles_modo']
MODOS_DISPONIBLES = tuple(PERFILES_MODO.keys())
FUENTES_DATOS_DISPONIBLES = tuple(PARAMETROS_CONFIGURACION['fuentes_datos_disponibles'])
MODOS_NORMALIZACION_DISPONIBLES = tuple(PARAMETROS_CONFIGURACION['modos_normalizacion_disponibles'])
MODOS_EVALUACION_DISPONIBLES = tuple(PARAMETROS_CONFIGURACION['modos_evaluacion_disponibles'])
MODOS_SEMILLA_DISPONIBLES = tuple(PARAMETROS_CONFIGURACION['modos_semilla_disponibles'])
MODOS_TRANSFORMACION_OBJETIVOS_DISPONIBLES = tuple(PARAMETROS_CONFIGURACION['modos_transformacion_objetivos_disponibles'])
ALGORITMOS_DISPONIBLES = tuple(PARAMETROS_CONFIGURACION['algoritmos_disponibles'])
PARAMS_TUNABLES_DEFECTO = PARAMETROS_CONFIGURACION['tunables']
CLAVES_TUNABLES_PERMITIDAS = frozenset(PARAMS_TUNABLES_DEFECTO.keys())

@dataclass
class ConfiguracionExperimento:
    """
    Encapsula la totalidad de hiperparámetros y metadatos del sistema.
    """
    
    modo_ejecucion: str
    num_bloques: int
    semilla_maestra: int
    origen_datos: str
    n_snps: int
    tam_bloque_sintetico: int
    n_haplotipos: int
    dir_salida_base: str
    carpetas: Dict[str, str]
    tam_poblacion: int
    n_generaciones: int
    n_descendencia: int
    pc: float
    pm: float
    vecinos_moead: int
    prob_vecindad_moead: float
    theta_moead_pbi: float
    n_ejecuciones: int
    
    # Parámetros con valores por defecto (deben ir al final)
    INIT_PERMITIDAS: List[str] = field(default_factory=lambda: [
        'random_sparse', 'random_dense',
        'greedy_hybrid', 'greedy_pure'
    ])

    ALGORITMOS_PERMITIDOS: List[str] = field(default_factory=lambda: list(ALGORITMOS_DISPONIBLES))
    
    # Parámetros para generación sintética
    prob_flip_sintetico: float = PARAMS_TUNABLES_DEFECTO['prob_flip_sintetico']
    dif_min_pares_sintetico: int = PARAMS_TUNABLES_DEFECTO['dif_min_pares_sintetico']
    intentos_max_sintetico: int = PARAMS_TUNABLES_DEFECTO['intentos_max_sintetico']
    
    # DPI de los reportes gráficos
    report_plot_dpi: int = PARAMS_TUNABLES_DEFECTO['report_plot_dpi']
    
    # Ruta al dataset histórico de Hinds et al. (2005) - Localizado dentro del paquete (snp_tag/data/datasets/)
    ruta_hinds2005: str = str(Path(__file__).parent / "data" / "datasets" / "hinds2005_1032.txt")
    
    # Probabilidad de bit=1 en inicialización aleatoria de GI 50/50
    prob_aleatoria_gi: float = PARAMS_TUNABLES_DEFECTO['prob_aleatoria_gi']
    
    algoritmos_activos: List[str] = field(default_factory=list)
    opciones_init: List[str] = field(default_factory=list)
    modo_normalizacion: str = PARAMS_TUNABLES_DEFECTO.get('modo_normalizacion', 'static_dataset_limits')
    modo_evaluacion: str = PARAMS_TUNABLES_DEFECTO.get('modo_evaluacion', 'absoluta')
    modo_semillas: str = PARAMS_TUNABLES_DEFECTO.get('modo_semillas', 'non_deterministic')
    modo_transformacion_objetivos: str = PARAMS_TUNABLES_DEFECTO.get('modo_transformacion_objetivos', 'neg')


def es_opcion_init_valida(nombre_init: str) -> bool:
    """
    Verifica si una estrategia de inicialización es compatible con el sistema.
    """
    if nombre_init in ['random_sparse', 'random_dense', 'greedy_hybrid', 'greedy_pure']:
        return True
    return False


def es_algoritmo_valido(nombre_algoritmo: str) -> bool:
    """
    Verifica si un algoritmo solicitado está soportado por el sistema.
    """
    return str(nombre_algoritmo) in ALGORITMOS_DISPONIBLES


def resolver_modo_normalizacion(modo: Optional[str]) -> str:
    """
    Normaliza y valida el esquema de escalado de objetivos.
    """
    modo_normalizado = str(modo or 'global_all_pairs').strip().lower()
    permitidos = set(MODOS_NORMALIZACION_DISPONIBLES)
    if modo_normalizado not in permitidos:
        raise ValueError(f"Esquema de normalización no soportado: {modo}. Opciones válidas: {sorted(permitidos)}")
    return modo_normalizado


def resolver_modo_evaluacion(modo: Optional[str]) -> str:
    """
    Normaliza y valida el esquema de evaluación (fitness).
    """
    modo_normalizado = str(modo or 'absoluta').strip().lower()
    permitidos = set(MODOS_EVALUACION_DISPONIBLES)
    if modo_normalizado not in permitidos:
        raise ValueError(f"Esquema de evaluación no soportado: {modo}. Opciones válidas: {sorted(permitidos)}")
    return modo_normalizado


def resolver_modo_semillas(modo: Optional[str]) -> str:
    """
    Normaliza y valida el esquema de semillas.
    """
    modo_normalizado = str(modo or 'non_deterministic').strip().lower()
    permitidos = set(MODOS_SEMILLA_DISPONIBLES)
    if modo_normalizado not in permitidos:
        raise ValueError(f"Modo de semillas no soportado: {modo}. Opciones válidas: {sorted(permitidos)}")
    return modo_normalizado


def resolver_modo_transformacion_objetivos(modo: Optional[str]) -> str:
    """
    Normaliza y valida la transformación de objetivos de maximización.
    """
    modo_normalizado = str(modo or 'neg').strip().lower()
    permitidos = set(MODOS_TRANSFORMACION_OBJETIVOS_DISPONIBLES)
    if modo_normalizado not in permitidos:
        raise ValueError(
            f"Modo de transformación de objetivos no soportado: {modo}. "
            f"Opciones válidas: {sorted(permitidos)}"
        )
    return modo_normalizado


def _validar_overrides(overrides: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Valida y normaliza overrides de parámetros tunables."""
    if overrides is None:
        return {}

    if not isinstance(overrides, dict):
        raise TypeError("'overrides' debe ser un diccionario con pares clave-valor.")

    claves_desconocidas = sorted(set(overrides.keys()) - CLAVES_TUNABLES_PERMITIDAS)
    if claves_desconocidas:
        raise ValueError(
            "Se recibieron parámetros tunables no soportados: "
            f"{claves_desconocidas}. "
            f"Claves válidas: {sorted(CLAVES_TUNABLES_PERMITIDAS)}"
        )

    return dict(overrides)


def construir_configuracion(modo: str = 'medium', data_source: str = 'hinds2005',
                           overrides: Optional[Dict[str, Any]] = None) -> ConfiguracionExperimento:
    """Construye la configuración completa a partir de parámetros tunables centralizados."""
    modo = str(modo or 'medium').strip().lower()
    if modo not in PERFILES_MODO:
        raise ValueError(f"Modo no soportado: {modo}. Opciones válidas: {list(MODOS_DISPONIBLES)}")

    data_source = str(data_source or 'hinds2005').strip().lower()
    if data_source not in FUENTES_DATOS_DISPONIBLES:
        raise ValueError(
            f"Fuente de datos no soportada: {data_source}. Opciones válidas: {list(FUENTES_DATOS_DISPONIBLES)}"
        )

    params = dict(PARAMS_TUNABLES_DEFECTO)
    params.update(_validar_overrides(overrides))

    opciones_init = list(params.get('opciones_init', []))
    for nombre_init in opciones_init:
        if not es_opcion_init_valida(str(nombre_init)):
            raise ValueError(f"Inicialización no válida: {nombre_init}")

    algoritmos_activos = list(params.get('algoritmos_activos', []))
    if not algoritmos_activos:
        raise ValueError("Debe definirse al menos un algoritmo en 'algoritmos_activos'.")
    for nombre_algoritmo in algoritmos_activos:
        if not es_algoritmo_valido(str(nombre_algoritmo)):
            raise ValueError(f"Algoritmo no válido: {nombre_algoritmo}")

    theta_pbi = float(params['theta_moead_pbi'])
    if theta_pbi <= 0:
        raise ValueError("'theta_moead_pbi' debe ser > 0.")

    prob_vec = float(params['prob_vecindad_moead'])
    if not (0.0 <= prob_vec <= 1.0):
        raise ValueError("'prob_vecindad_moead' debe estar en [0, 1].")

    n_snps = int(params['n_snps'])
    num_bloques = int(params['num_bloques'])
    # Calcular el tamaño de cada bloque LD a partir del número de bloques solicitado.
    # Se garantiza un mínimo de 1 SNP por bloque para evitar divisiones degeneradas.
    tam_bloque_sintetico = max(1, n_snps // num_bloques)
    modo_cfg = PERFILES_MODO[modo]

    return ConfiguracionExperimento(
        modo_ejecucion=modo,
        num_bloques=num_bloques,
        semilla_maestra=int(params['semilla_maestra']),
        origen_datos=data_source,
        n_snps=n_snps,
        tam_bloque_sintetico=tam_bloque_sintetico,
        n_haplotipos=48 if data_source == 'hinds2005' else 40,
        dir_salida_base=str(params['dir_salida_base']),
        carpetas={},
        tam_poblacion=int(modo_cfg['tam_pob']),
        n_generaciones=int(modo_cfg['n_gen']),
        n_descendencia=int(modo_cfg['descendencia']),
        pc=float(params['pc']),
        pm=1.0 / max(1, n_snps),
        vecinos_moead=int(params['vecinos_moead']),
        prob_vecindad_moead=prob_vec,
        theta_moead_pbi=theta_pbi,
        n_ejecuciones=int(modo_cfg['n_runs']),
        prob_flip_sintetico=float(params['prob_flip_sintetico']),
        dif_min_pares_sintetico=int(params['dif_min_pares_sintetico']),
        intentos_max_sintetico=int(params['intentos_max_sintetico']),
        report_plot_dpi=int(params['report_plot_dpi']),
        prob_aleatoria_gi=float(params['prob_aleatoria_gi']),
        algoritmos_activos=algoritmos_activos,
        opciones_init=opciones_init,
        modo_normalizacion=resolver_modo_normalizacion(params.get('modo_normalizacion')),
        modo_evaluacion=resolver_modo_evaluacion(params.get('modo_evaluacion')),
        modo_semillas=resolver_modo_semillas(params.get('modo_semillas')),
        modo_transformacion_objetivos=resolver_modo_transformacion_objetivos(
            params.get('modo_transformacion_objetivos')
        ),
    )
