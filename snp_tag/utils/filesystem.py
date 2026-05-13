"""
Módulo de Gestión de Sistema de Archivos (filesystem.py)
--------------------------------------------------------
Administra la jerarquía de directorios para la organización de resultados,
utilizando rutas robustas basadas en pathlib.
"""

from pathlib import Path
from datetime import datetime
from snp_tag.config import ConfiguracionExperimento

def crear_arbol_directorios_dataset(cfg: ConfiguracionExperimento, tipo_dataset: str):
    """
    Crea un árbol de directorios específico para un dataset y devuelve la ruta base y el diccionario de carpetas.
    Prioriza snp_tag/results como raíz de resultados.
    """
    ts = datetime.now().strftime('%Y%m%dT%H%M%S')
    
    # Etiqueta de inicializaciones
    opciones_init = getattr(cfg, 'opciones_init', [])
    if isinstance(opciones_init, list) and len(opciones_init) > 0:
        etiqueta_inits = "-".join(sorted([str(x) for x in opciones_init]))
    else:
        etiqueta_inits = "init_por_defecto"
        
    # Localizar la carpeta results dentro del paquete
    ruta_base_paquete = Path(__file__).parent.parent / "results"
    
    ruta_base = ruta_base_paquete / cfg.modo_ejecucion / tipo_dataset / etiqueta_inits / ts
    
    comparativa_root = ruta_base / '2_comparativa'
    frentes_root = comparativa_root / '1_frentes'
    sintesis_root = comparativa_root / '3_sintesis'

    carpetas = {
        'datos': ruta_base / '0_datos_previos',
        'ejecuciones': ruta_base / '1_ejecuciones',
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
        'estadistica_hv': comparativa_root / '4_rankings' / 'estadistica_hv',
        'decision_mcdm': comparativa_root / '5_decision_mcdm',
    }

    for p in carpetas.values():
        p.mkdir(parents=True, exist_ok=True)

    # Convertir a cadenas para compatibilidad con el resto del código
    carpetas_str = {k: str(v) for k, v in carpetas.items()}
    carpetas_str['comparativa_root'] = carpetas_str['comparativa']
    
    return str(ruta_base), carpetas_str
