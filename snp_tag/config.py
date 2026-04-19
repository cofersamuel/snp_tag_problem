"""
Módulo de Configuración Experimental (config.py)
-----------------------------------------------
Define las estructuras de datos y parámetros globales requeridos para la
ejecución controlada de experimentos de búsqueda multiobjetivo de Tag SNPs.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from pathlib import Path


# -----------------------------------------------------------------------------
# BLOQUE ÚNICO DE PARÁMETROS (editar aquí)
#
# Guía rápida:
# - perfiles_modo: define presets de ejecución (fast/medium/full).
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
    # Opciones disponibles: 'fast', 'medium', 'full'.
    # Cada perfil controla:
    # - tam_pob: tamaño de población.
    # - n_gen: número de generaciones.
    # - descendencia: número de descendientes por generación.
    # - n_runs: repeticiones por configuración algoritmo-init.
    'perfiles_modo': {
        'fast': {'tam_pob': 10, 'n_gen': 2, 'descendencia': 10, 'n_runs': 2},
        'medium': {'tam_pob': 100, 'n_gen': 50, 'descendencia': 100, 'n_runs': 2},
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
    'modos_normalizacion_disponibles': ('per_algorithm', 'global_all_pairs', 'static_dataset_limits'),

    # Parámetros tunables editables (punto principal de ajuste).
    'tunables': {
        # Número de bloques lógicos del experimento (actualmente informativo).
        'num_bloques': 10,

        # Semilla maestra para reproducibilidad global.
        'semilla_maestra': 42,

        # Número total de SNPs del bloque objetivo.
        'n_snps': 1032,

        # Directorio raíz de salida de resultados.
        'dir_salida_base': 'results',

        # Probabilidad de cruce (crossover probability).
        'pc': 0.7,

        # Vecinos de MOEA/D (solo aplica a ese algoritmo).
        'vecinos_moead': 15,

        # Inicializaciones activas en los experimentos.
        # Opciones: random_sparse, random_dense, greedy_pure,
        # greedy_hybrid.
        'opciones_init': ['random_sparse', 'random_dense', 'greedy_pure', 'greedy_hybrid'],

        # Modo de normalización usado en la fase de métricas/reporting.
        # Debe estar en "modos_normalizacion_disponibles".
        'modo_normalizacion': 'static_dataset_limits',

        # Probabilidad de flip por SNP al generar datos sintéticos.
        'prob_flip_sintetico': 0.03,

        # Distancia Hamming mínima objetivo entre pares sintéticos.
        'dif_min_pares_sintetico': 100,

        # Intentos máximos para forzar la distancia mínima sintética.
        'intentos_max_sintetico': 1000,

        # Resolución (DPI) para figuras y reportes.
        'report_plot_dpi': 300,

        # Cobertura máxima usada por estrategias greedy.
        'cobertura_max_greedy': 50,

        # Probabilidad de bit=1 para parte aleatoria en GI 50/50.
        'prob_aleatoria_gi': 0.5,
    },
}

# Alias de compatibilidad interna/externa
PERFILES_MODO = PARAMETROS_CONFIGURACION['perfiles_modo']
MODOS_DISPONIBLES = tuple(PERFILES_MODO.keys())
FUENTES_DATOS_DISPONIBLES = tuple(PARAMETROS_CONFIGURACION['fuentes_datos_disponibles'])
MODOS_NORMALIZACION_DISPONIBLES = tuple(PARAMETROS_CONFIGURACION['modos_normalizacion_disponibles'])
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
    n_ejecuciones: int
    
    # Parámetros con valores por defecto (deben ir al final)
    INIT_PERMITIDAS: List[str] = field(default_factory=lambda: [
        'random', 'random_sparse', 'random_dense', 
        'greedy_hybrid', 'greedy_pure'
    ])
    
    # Parámetros para generación sintética
    prob_flip_sintetico: float = PARAMS_TUNABLES_DEFECTO['prob_flip_sintetico']
    dif_min_pares_sintetico: int = PARAMS_TUNABLES_DEFECTO['dif_min_pares_sintetico']
    intentos_max_sintetico: int = PARAMS_TUNABLES_DEFECTO['intentos_max_sintetico']
    
    # DPI de los reportes gráficos
    report_plot_dpi: int = PARAMS_TUNABLES_DEFECTO['report_plot_dpi']
    
    # Ruta al dataset histórico de Hinds et al. (2005) - Localizado dentro del paquete (snp_tag/data/datasets/)
    ruta_hinds2005: str = str(Path(__file__).parent / "data" / "datasets" / "hinds2005_1032.txt")
    
    # Cobertura máxima para estrategias greedy
    cobertura_max_greedy: int = PARAMS_TUNABLES_DEFECTO['cobertura_max_greedy']
    
    # Probabilidad de bit=1 en inicialización aleatoria de GI 50/50
    prob_aleatoria_gi: float = PARAMS_TUNABLES_DEFECTO['prob_aleatoria_gi']
    
    opciones_init: List[str] = field(default_factory=list)
    modo_normalizacion: str = PARAMS_TUNABLES_DEFECTO['modo_normalizacion']


def es_opcion_init_valida(nombre_init: str) -> bool:
    """
    Verifica si una estrategia de inicialización es compatible con el sistema.
    """
    if nombre_init in ['random', 'random_sparse', 'random_dense', 'greedy_hybrid', 'greedy_pure']:
        return True
    return False


def resolver_modo_normalizacion(modo: Optional[str]) -> str:
    """
    Normaliza y valida el esquema de escalado de objetivos.
    """
    modo_normalizado = str(modo or 'global_all_pairs').strip().lower()
    permitidos = set(MODOS_NORMALIZACION_DISPONIBLES)
    if modo_normalizado not in permitidos:
        raise ValueError(f"Esquema de normalización no soportado: {modo}. Opciones válidas: {sorted(permitidos)}")
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

    n_snps = int(params['n_snps'])
    modo_cfg = PERFILES_MODO[modo]

    return ConfiguracionExperimento(
        modo_ejecucion=modo,
        num_bloques=int(params['num_bloques']),
        semilla_maestra=int(params['semilla_maestra']),
        origen_datos=data_source,
        n_snps=n_snps,
        tam_bloque_sintetico=n_snps,
        n_haplotipos=48 if data_source == 'hinds2005' else 40,
        dir_salida_base=str(params['dir_salida_base']),
        carpetas={},
        tam_poblacion=int(modo_cfg['tam_pob']),
        n_generaciones=int(modo_cfg['n_gen']),
        n_descendencia=int(modo_cfg['descendencia']),
        pc=float(params['pc']),
        pm=1.0 / max(1, n_snps),
        vecinos_moead=int(params['vecinos_moead']),
        n_ejecuciones=int(modo_cfg['n_runs']),
        prob_flip_sintetico=float(params['prob_flip_sintetico']),
        dif_min_pares_sintetico=int(params['dif_min_pares_sintetico']),
        intentos_max_sintetico=int(params['intentos_max_sintetico']),
        report_plot_dpi=int(params['report_plot_dpi']),
        cobertura_max_greedy=int(params['cobertura_max_greedy']),
        prob_aleatoria_gi=float(params['prob_aleatoria_gi']),
        opciones_init=opciones_init,
        modo_normalizacion=resolver_modo_normalizacion(params.get('modo_normalizacion')),
    )
