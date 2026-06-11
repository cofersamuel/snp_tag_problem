"""
Módulo de Orquestación de Diagnósticos (diagnostics_pipeline.py)
----------------------------------------------------------------
Gestiona la carga de datos, el análisis exploratorio (EDA) y la 
visualización de desequilibrio de ligamiento (LD).
"""

# =============================================================================
# LIBRERÍAS ESTÁNDAR
# =============================================================================
import os
from typing import List, Tuple

# =============================================================================
# LIBRERÍAS DE TERCEROS
# =============================================================================
import numpy as np

# =============================================================================
# MÓDULOS LOCALES (snp_tag)
# =============================================================================
from snp_tag.config import ConfiguracionExperimento
from snp_tag.data.loader import cargar_dataset_objetivo
from snp_tag.engine.diagnostics_logic import (analizar_similitud_genotipica,
                                              calcular_ld_completo,
                                              detectar_bloques_ld,
                                              ejecutar_diagnostico_ld)
from snp_tag.utils.logger import logger
from snp_tag.utils.terminal import imprimir_encabezado, imprimir_subseccion
from snp_tag.visualization.diagnostics_plot import (
    graficar_bloques_ld, graficar_conteo_alelos, graficar_histograma_alelico,
    graficar_histograma_hamming, graficar_ld_detallado,
    graficar_mapa_calor_haplotipos, graficar_variabilidad_snps)


def ejecutar_pipeline_diagnostico(cfg: ConfiguracionExperimento) -> Tuple[np.ndarray, List[str], np.ndarray, List[str], np.ndarray]:
    """
    Ejecuta el análisis exploratorio de datos (EDA) y diagnóstico LD.
    
    Parámetros:
    -----------
    cfg : ConfiguracionExperimento
        Contexto del experimento.
        
    Retorna:
    --------
    Tuple[np.ndarray, List[str], np.ndarray, List[str], np.ndarray]
        Matriz haplotípica H, IDs de SNPs, posiciones, IDs de haplotipos y matriz de distancias de Hamming (dvals).
    """
    imprimir_encabezado("DIAGNÓSTICO DE DATOS Y DESEQUILIBRIO (LD)")
    
    imprimir_subseccion("Metadatos y Dimensiones del Dataset", icono="📊️")
    H, snp_ids, posiciones, hap_ids = cargar_dataset_objetivo(cfg)
    
    fichero_rel = os.path.relpath(cfg.ruta_hinds2005) if cfg.origen_datos == 'hinds2005' else "N/A"
    logger.info(f"      • N_SNPS={len(snp_ids)} | N_PATRONES={len(hap_ids)} | FICHERO={fichero_rel}")
    
    imprimir_subseccion("Visualización de la Estructura de Haplotipos", icono="🧬")
    if 'diagnostico_datos' in cfg.graficas_activas:
        graficar_mapa_calor_haplotipos(H, cfg.carpetas, cfg.modo_ejecucion)
        graficar_bloques_ld(H, cfg.carpetas, cfg.modo_ejecucion, cfg=cfg)
    else:
        logger.info("      • ⚠️  Gráficas de estructura de haplotipos omitidas (user_config.ini).")
    
    imprimir_subseccion("Análisis de Variabilidad y Frecuencia Alélica", icono="📈")
    if 'diagnostico_datos' in cfg.graficas_activas:
        graficar_histograma_alelico(H, cfg.carpetas, cfg.modo_ejecucion)
        graficar_variabilidad_snps(H, cfg.carpetas, cfg.modo_ejecucion)
        graficar_conteo_alelos(H, cfg.carpetas, cfg.modo_ejecucion)
    else:
        logger.info("      • ⚠️  Gráficas de variabilidad omitidas (user_config.ini).")
    
    dvals = analizar_similitud_genotipica(H)
    if 'diagnostico_datos' in cfg.graficas_activas:
        graficar_histograma_hamming(dvals, cfg.carpetas, cfg.modo_ejecucion)
    
    # LD y Veredicto
    media_ld, corrs, corr_full = calcular_ld_completo(H)
    segmentos = detectar_bloques_ld(H) 
    
    # Generar gráficos LD y capturar rutas
    rutas_ld = []
    if 'diagnostico_datos' in cfg.graficas_activas:
        rutas_ld = graficar_ld_detallado(corr_full, corrs, cfg.carpetas, cfg.modo_ejecucion)
    
    # Ejecutar reporte de diagnóstico con enlaces integrados
    ejecutar_diagnostico_ld(H, cfg, rutas_ld=rutas_ld)
    
    return H, snp_ids, posiciones, hap_ids, dvals
