# --- Funciones auxiliares para paralelización de métricas (deben estar a nivel superior para ser picklables) ---
# NOTA: Las métricas finales ya no se paralelizan individualmente porque la
# normalización requiere conocer todas las ejecuciones para construir
# referencias consistentes de ideal/nadir.

def _eval_gen_metric_single(args):
    """Wrapper para métricas generacionales con referencias pre-calculadas.

    Las referencias de Range/SumMin/MinSum dependen del modo de normalización
    elegido en configuración (per_algorithm | global_all_pairs | static_dataset_limits).
    """
    run_result, algo_ideal, algo_safe_denom, ideal_global, safe_denom_global = args
    from __main__ import _build_generation_rows_for_run
    rows, gen_count = _build_generation_rows_for_run(
        run_result,
        algo_ideal, algo_safe_denom,
        ideal_global, safe_denom_global
    )
    import pandas as _pd
    return _pd.DataFrame(rows)
# --- Función auxiliar para graficado paralelo (debe estar a nivel superior para ser picklable) ---
def tarea_plot(tipo, df_front_obj, FOLDERS, mode_tag_local, REPORT_DPI, master_seed):
    if tipo == 'all_fronts':
        _plot_all_pareto_fronts(df_front_obj, FOLDERS, mode_tag_local, dpi=REPORT_DPI)
    elif tipo == 'correlation':
        _plot_objective_correlation(df_front_obj, FOLDERS, mode_tag_local, dpi=REPORT_DPI)
    elif tipo == 'parallel':
        _plot_parallel_coordinates(df_front_obj, master_seed, FOLDERS, mode_tag_local, dpi=REPORT_DPI)
#!/usr/bin/env python3
import os
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.spea2 import SPEA2
from pymoo.core.problem import Problem
from pymoo.core.sampling import Sampling
from pymoo.operators.crossover.ux import UX
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.util.ref_dirs import get_reference_directions
import matplotlib
matplotlib.use('Agg') # Forzar modo no interactivo para evitar bloqueos
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import seaborn as sns
import subprocess
import time
import json
from datetime import datetime
import sys
import shutil


@dataclass
class ExperimentConfig:
    # Opciones de inicialización permitidas a nivel de clase
    INIT_OPTIONS_PERMITIDAS = ['random', 'random_sparse', 'random_dense', 'greedy_hybrid', 'greedy_hybrid_50-50', 'greedy_pure']
    """Almacena la configuración global del experimento."""
    # Opciones de inicialización permitidas
    execution_mode: str
    num_blocks: int
    master_seed: int
    data_source: str
    n_snps: int
    syn_block_size: int
    n_haplotypes: int
    base_out_dir: str
    folders: dict
    pop_size: int
    n_gen: int
    offspring: int
    pc: float
    pm: float
    moead_neighbors: int
    n_runs: int
    thresh_opt_pct: float
    thresh_opt_mean: float
    thresh_opt_abs_corr: float
    thresh_accept_mean: float
    syn_flip_prob: float
    report_plot_dpi: int
    # Diversificación de dataset sintético (distancia de Hamming mínima entre pares)
    synthetic_min_pairwise_diff: int
    synthetic_max_attempts: int
    # Ruta al fichero del dataset Hinds et al. (2005) en formato texto binario
    hinds2005_path: str = ""
    # Cobertura máxima para inicialización greedy
    greedy_max_coverage: int = 5
    # Probabilidad de bit=1 en la mitad aleatoria de la GI 50/50
    gi_random_p: float = 0.5
    # Selección de inicializaciones a usar (modificable por el usuario)
    init_options: list = None

    # Modo de normalización para Range/SumMin/MinSum: per_algorithm | global_all_pairs | static_dataset_limits
    normalization_mode: str = 'static_dataset_limits'


def _is_valid_init_option(init_name: str) -> bool:
    """Valida opciones de inicialización estáticas y variantes con cobertura."""
    if init_name in ExperimentConfig.INIT_OPTIONS_PERMITIDAS:
        return True
    if isinstance(init_name, str) and init_name.startswith('greedy_pure_'):
        cov = init_name[len('greedy_pure_'):]
        return cov.isdigit() and int(cov) > 0
    return False


def _resolve_normalization_mode(normalization_mode: str | None) -> str:
    """Normaliza y valida el modo de normalización configurado en código."""
    mode = str(normalization_mode or 'global_all_pairs').strip().lower()
    allowed = {'per_algorithm', 'global_all_pairs', 'static_dataset_limits'}
    if mode not in allowed:
        raise ValueError(f"normalization_mode no soportado: {normalization_mode}. Opciones permitidas: {sorted(allowed)}")
    return mode


def setup_configuration(execution_mode: str | None = None, num_blocks: int | None = None, data_source: str | None = None) -> ExperimentConfig:
    """Configura los parámetros globales y crea los directorios.

    Parameters
    ----------
    execution_mode : str | None
        Optional execution mode override ('fast', 'medium', 'full'). If None, defaults to 'medium'.
    num_blocks : int | None
        Optional number of blocks override. If None, defaults to 1.
    data_source : str | None
        Optional data source override. If None, defaults to 'hinds2005'.
    """
    import time
    sns.set_theme(style='whitegrid')
    pd.set_option('display.max_columns', 200)
    
    EXECUTION_MODE = execution_mode if execution_mode is not None else 'medium' # fast / medium / full | Selector de modo
    DATA_SOURCE = data_source if data_source is not None else "hinds2005"  # hinds2005 / synthetic
    
    # --- Selección de inicializaciones (unificada e independiente del dataset) ---
    init_options_usuario = ['random_sparse', 'random_dense', 'greedy_pure']
    # Todas las inicializaciones base permitidas: ['random', 'greedy_hybrid', 'greedy_pure']

    # Validación de inicializaciones
    for init in init_options_usuario:
        if not _is_valid_init_option(init):
            raise ValueError(f"Inicialización no soportada: {init}. Opciones permitidas: {ExperimentConfig.INIT_OPTIONS_PERMITIDAS}")

    NUM_BLOCKS = int(num_blocks) if num_blocks is not None else 1
    MASTER_SEED = 42
    np.random.seed(MASTER_SEED)
    random.seed(MASTER_SEED)
    N_SNPS = 1032
    SYN_BLOCK_SIZE = int(N_SNPS / max(1, NUM_BLOCKS))
    # Para hinds2005 el número real de patrones alelicos se actualiza al cargar los datos
    N_HAPLOTYPES = 48 if DATA_SOURCE == "hinds2005" else 40
    BASE_OUT_DIR = "resultados"
    BASE_OUT_MODE_DIR = os.path.join(BASE_OUT_DIR, EXECUTION_MODE)
    # Solo crear la carpeta base del modo, no los subdirectorios
    FOLDERS = {
        "datos": os.path.join(BASE_OUT_MODE_DIR, "0_datos_previos"),
        "ejecuciones": os.path.join(BASE_OUT_MODE_DIR, "1_ejecuciones"),
        "csv": os.path.join(BASE_OUT_MODE_DIR, "1_ejecuciones", "csv"),
        "comparativa_root": os.path.join(BASE_OUT_MODE_DIR, "2_comparativa"),
        "comparativa": os.path.join(BASE_OUT_MODE_DIR, "2_comparativa"),
        "heatmaps": os.path.join(BASE_OUT_MODE_DIR, "2_comparativa", "heatmaps"),
        "resumenes": os.path.join(BASE_OUT_MODE_DIR, "2_comparativa", "resumenes"),
        "rankings": os.path.join(BASE_OUT_MODE_DIR, "2_comparativa", "rankings"),
        "metricas": os.path.join(BASE_OUT_MODE_DIR, "2_comparativa", "metricas"),
        "frentes": os.path.join(BASE_OUT_MODE_DIR, "2_comparativa", "frentes"),
        "boxplots": os.path.join(BASE_OUT_MODE_DIR, "2_comparativa", "boxplots"),
    }
    REPORT_PLOT_DPI = 300
    # Solo crear la carpeta base del modo
    os.makedirs(BASE_OUT_MODE_DIR, exist_ok=True)
        
    MODE_CONFIG = {
        'fast': {'POP_SIZE': 10, 'N_GEN': 2, 'OFFSPRING': 10, 'PC': 0.7, 'MOEAD_NEIGHBORS': 5, 'N_RUNS': 2},
        'medium': {'POP_SIZE': 100, 'N_GEN': 50, 'OFFSPRING': 100, 'PC': 0.7, 'MOEAD_NEIGHBORS': 15, 'N_RUNS': 2},
        'full': {'POP_SIZE': 200, 'N_GEN': 500, 'OFFSPRING': 200, 'PC': 0.7, 'MOEAD_NEIGHBORS': 15, 'N_RUNS': 5},
    }
    
    cfg = MODE_CONFIG[EXECUTION_MODE]
    # Ruta al fichero del dataset Hinds et al. (2005) — bloque de 48 patrones × 1032 SNPs
    HINDS2005_PATH = os.path.join("data", "hinds2005_1032.txt")
    # Cobertura máxima greedy (respaldo): estandarizada a 50 para todos los datasets.
    GREEDY_MAX_COVERAGE = 50
    # Normalización basada en límites estáticos del dataset (reproduce los resultados del paper).
    NORMALIZATION_MODE = _resolve_normalization_mode('static_dataset_limits')

    print_step("CONFIGURACIÓN", icon="⚙")
    print(f"      • Modo={EXECUTION_MODE} | POP_SIZE={cfg['POP_SIZE']} | N_GEN={cfg['N_GEN']} | OFFSPRING={cfg['OFFSPRING']} | PC={cfg['PC']} | PM={1.0/N_SNPS:.6f} | N_RUNS={cfg['N_RUNS']}")
    print_subsection("Metadatos y Dimensiones del Dataset", icon="📊")
    if DATA_SOURCE == "hinds2005":
        print(f"      • ORIGEN_DATOS={DATA_SOURCE} (Hinds et al. 2005 / Perlegen) | N_SNPS={N_SNPS} | N_PATRONES={N_HAPLOTYPES} | PM={1.0/N_SNPS:.6f}")
        print(f"      • FICHERO={HINDS2005_PATH} | GREEDY_MAX_COVERAGE={GREEDY_MAX_COVERAGE}")
    else:
        print(f"      • ORIGEN_DATOS={DATA_SOURCE} | NUM_BLOQUES={NUM_BLOCKS} | N_SNPS={N_SNPS} | PM={1.0/N_SNPS:.6f}")
    print(f"      • REPORT_DPI={REPORT_PLOT_DPI} | NORMALIZATION_MODE={NORMALIZATION_MODE}")
    
    return ExperimentConfig(
        execution_mode=EXECUTION_MODE,
        num_blocks=NUM_BLOCKS,
        master_seed=MASTER_SEED,
        data_source=DATA_SOURCE,
        n_snps=N_SNPS,
        syn_block_size=SYN_BLOCK_SIZE,
        n_haplotypes=N_HAPLOTYPES,
        base_out_dir=BASE_OUT_DIR,
        folders=FOLDERS,
        pop_size=cfg['POP_SIZE'],
        n_gen=cfg['N_GEN'],
        offspring=cfg['OFFSPRING'],
        pc=cfg['PC'],
        pm=1.0 / N_SNPS,
        syn_flip_prob=0.03,
        moead_neighbors=cfg['MOEAD_NEIGHBORS'],
        n_runs=cfg['N_RUNS'],
        thresh_opt_pct=0.95,
        thresh_opt_mean=0.18,
        thresh_opt_abs_corr=0.45,
        thresh_accept_mean=0.12,
        report_plot_dpi=REPORT_PLOT_DPI,
        synthetic_min_pairwise_diff=100,
        synthetic_max_attempts=1000,
        hinds2005_path=HINDS2005_PATH,
        greedy_max_coverage=GREEDY_MAX_COVERAGE,
        init_options=init_options_usuario,
        normalization_mode=NORMALIZATION_MODE,
    )


def create_dataset_tree(cfg: ExperimentConfig, dataset_type: str):
    """Create a per-dataset directory tree and return (base_dataset_dir, folders_dict).

    Naming convention: resultados/<mode>/<dataset_type>/<num_blocks>_bloques/
    """
    # Siempre añadir timestamp al nombre del bloque
    ts = datetime.now().strftime('%Y%m%dT%H%M%S')
    
    # Crear etiqueta de inicializaciones ordenadas alfabéticamente (ej: greedy_pure-random_dense)
    opciones_init = getattr(cfg, 'init_options', [])
    if isinstance(opciones_init, list) and len(opciones_init) > 0:
        etiqueta_inits = "-".join(sorted([str(x) for x in opciones_init]))
    else:
        etiqueta_inits = "default_init"
        
    # Construir la ruta base: resultados/<modo>/<tipo_dataset>/<inicializaciones>/<timestamp>/
    base_dataset_dir = os.path.join(
        cfg.base_out_dir, 
        cfg.execution_mode, 
        dataset_type, 
        etiqueta_inits,
        ts
    )

    data_dir = os.path.join(base_dataset_dir, '0_datos_previos')
    ejec_dir = os.path.join(base_dataset_dir, '1_ejecuciones')
    ejec_csv_dir = os.path.join(ejec_dir, 'csv')
    comp_dir = os.path.join(base_dataset_dir, '2_comparativa')
    comp_heatmaps = os.path.join(comp_dir, 'heatmaps')
    comp_frentes = os.path.join(comp_dir, 'frentes')
    comp_metricas = os.path.join(comp_dir, 'metricas')
    comp_rankings = os.path.join(comp_dir, 'rankings')
    comp_resumenes = os.path.join(comp_dir, 'resumenes')
    boxplots_dir = os.path.join(comp_dir, 'boxplots')

    for p in [data_dir, ejec_dir, ejec_csv_dir, comp_dir, comp_heatmaps, comp_frentes, comp_metricas, comp_rankings, comp_resumenes, boxplots_dir]:
        os.makedirs(p, exist_ok=True)

    folders = {
        'datos': data_dir,
        'ejecuciones': ejec_dir,
        'csv': ejec_csv_dir,
        'comparativa_root': comp_dir,
        'comparativa': comp_dir,
        'heatmaps': comp_heatmaps,
        'resumenes': comp_resumenes,
        'rankings': comp_rankings,
        'metricas': comp_metricas,
        'frentes': comp_frentes,
        'boxplots': boxplots_dir,
    }

    return base_dataset_dir, folders

def load_target_dataset(cfg: ExperimentConfig):
    """Carga o genera los haplotipos basados en la configuración."""
    if cfg.data_source == "synthetic":
        # Generar datos sintéticos con los parámetros de configuración
        H, attempts, achieved_min = generate_ld_haplotype_block(
            n_haplotypes=cfg.n_haplotypes, n_snps=cfg.n_snps,
            block_size=cfg.syn_block_size, flip_prob=cfg.syn_flip_prob, seed=cfg.master_seed,
            min_pairwise_diff=cfg.synthetic_min_pairwise_diff,
            max_attempts=cfg.synthetic_max_attempts)

        snp_ids = [f"snp_{i}" for i in range(cfg.n_snps)]
        snp_positions = np.arange(cfg.n_snps, dtype=int)
        haplotype_ids = [f"hap_{i}" for i in range(cfg.n_haplotypes)]

        # Calcular estadísticas de distancias
        try:
            n = H.shape[0]
            dists = []
            for a in range(n):
                for b in range(a+1, n):
                    dists.append(int((H[a] != H[b]).sum()))
            dists = np.array(dists) if dists else np.array([0])
            min_d, max_d, mean_d = int(dists.min()), int(dists.max()), float(dists.mean())
            theoretical_tol = max(0, min_d - 1)

            # Preparar rutas de salida usando helper centralizado
            dataset_type = 'synthetic'
            # If cfg.folders already points to a per-dataset tree created upstream, reuse it
            if cfg.folders and cfg.folders.get('datos') and f"{cfg.num_blocks}_bloques" in cfg.folders.get('datos'):
                base_dataset_dir = os.path.dirname(cfg.folders['datos'])
                local_folders = cfg.folders
            else:
                base_dataset_dir, local_folders = create_dataset_tree(cfg, dataset_type)
            # Override cfg.folders so subsequent pipeline stages write into the per-dataset tree
            cfg.folders = local_folders
            data_dir = local_folders['datos']
            ejec_csv_dir = local_folders['csv']
            comp_dir = local_folders['comparativa']

            # Exportar usando la función ya existente para nombres con mode tag
            _exportar_dataset_seleccionado(H, snp_ids, snp_positions, haplotype_ids, cfg.folders, cfg.execution_mode)

            # También guardar metadata adicional en JSON
            metadata = {
                'min_dist': int(min_d),
                'max_dist': int(max_d),
                'mean_dist': float(mean_d),
                'theoretical_tolerance': int(theoretical_tol),
                'n_haplotypes': int(n),
                'n_snps': int(H.shape[1]),
                'generator_params': {
                    'min_pairwise_diff': int(cfg.synthetic_min_pairwise_diff),
                    'max_attempts': int(cfg.synthetic_max_attempts),
                },
                'attempts': int(attempts),
                'achieved_min': int(achieved_min)
            }
            metadata_path = os.path.join(data_dir, 'metadata.json')
            with open(metadata_path, 'w') as jf:
                json.dump(metadata, jf, indent=2)

            # Construir un log de terminal más completo con la salida en español esperada
            def osc8_link(path, label=None):
                if label is None:
                    label = os.path.basename(path)
                url = f'file://{os.path.abspath(path)}'
                return f"\033]8;;{url}\033\\{label}\033]8;;\033\\"

            # Rutas de plots esperadas (serán creadas por etapas posteriores)
            mode = cfg.execution_mode
            expected_plots = {
                'heatmap_matriz': os.path.join(cfg.folders['heatmaps'], f'heatmap_matriz_haplotipos_{mode}.png'),
                'bloques_ld': os.path.join(cfg.folders['heatmaps'], f'bloques_ld_haplotipos_{mode}.png'),
                'zoom_10x10': os.path.join(cfg.folders['heatmaps'], f'zoom_matriz_haplotipos_10x10_{mode}.png'),
                'hist_freq': os.path.join(cfg.folders['heatmaps'], f'histograma_frecuencia_alelica_{mode}.png'),
                'std_snps': os.path.join(cfg.folders['heatmaps'], f'desviacion_estandar_snps_{mode}.png'),
                'count_alleles': os.path.join(cfg.folders['heatmaps'], f'conteo_alelos_por_haplotipo_{mode}.png'),
                'hist_hamming': os.path.join(cfg.folders['heatmaps'], f'histograma_distancia_hamming_{mode}.png'),
                'heatmap_corr': os.path.join(cfg.folders['heatmaps'], f'heatmap_correlacion_completa_{mode}.png'),
                'hist_corr': os.path.join(cfg.folders['heatmaps'], f'histograma_correlaciones_ld_{mode}.png'),
                'cdf_corr': os.path.join(cfg.folders['heatmaps'], f'cdf_correlacion_absoluta_ld_{mode}.png'),
            }

            report_lines = []
            report_lines.append('')
            report_lines.append('  ⚙ CONFIGURACIÓN')
            report_lines.append(f"      • Modo={cfg.execution_mode} | POP_SIZE={cfg.pop_size} | N_GEN={cfg.n_gen} | OFFSPRING={cfg.offspring} | PC={cfg.pc} | PM={cfg.pm:.6f} | N_RUNS={cfg.n_runs}")
            report_lines.append('')
            report_lines.append('  📊 Metadatos y Dimensiones del Dataset')
            report_lines.append('  ───────────────────────────────────────')
            report_lines.append(f"      • ORIGEN_DATOS={cfg.data_source} | NUM_BLOQUES={cfg.num_blocks} | N_SNPS={cfg.n_snps} | PM={cfg.pm:.6f}")
            report_lines.append(f"      • REPORT_DPI={cfg.report_plot_dpi}")
            report_lines.append(f"      ✅ Generador sintético alcanzó min_pairwise_diff={int(metadata['achieved_min'])} tras {int(metadata['attempts'])} intentos.")
            report_lines.append(f"      ✅ Sintético H: min_dist={metadata['min_dist']}, max_dist={metadata['max_dist']}, mean_dist={metadata['mean_dist']:.1f}. Tolerancia teórica={metadata['theoretical_tolerance']}")
            report_lines.append(f"      ✅ Dataset exportado: {metadata['n_snps']} SNPs x {metadata['n_haplotypes']} haplotipos")
            report_lines.append(f"      🖼  Matriz de haplotipos (CSV): {os.path.basename(os.path.join(data_dir, f'matriz_haplotipos_seleccionada_{mode}.csv'))}")
            report_lines.append(f"      🖼  Metadatos de SNPs (CSV): {os.path.basename(os.path.join(data_dir, f'metadatos_snps_seleccionados_{mode}.csv'))}")
            report_lines.append('')
            report_lines.append('\n'.join(['╔' + '═'*78 + '╗','║                  DIAGNÓSTICO DE DATOS Y DESEQUILIBRIO (LD)                   ║','╚' + '═'*78 + '╝']))

            # Añadir links a plots esperados
            report_lines.append('')
            report_lines.append('  🧬 Visualización de la Estructura de Haplotipos')
            report_lines.append('  ───────────────────────────────────────────────')
            report_lines.append(f"      🖼  Mapa de calor de haplotipos: {os.path.basename(expected_plots['heatmap_matriz'])}")
            report_lines.append(f"      🖼  Estructura de bloques LD (1 bloques reales): {os.path.basename(expected_plots['bloques_ld'])}")
            report_lines.append(f"      🖼  Zoom matriz H (10x10): {os.path.basename(expected_plots['zoom_10x10'])}")

            report_lines.append('')
            report_lines.append('  📈 Análisis de Variabilidad y Frecuencia Alélica')
            report_lines.append('  ─────────────────────────────────────────────────')
            report_lines.append(f"      🖼  Distribución de frecuencia alélica: {os.path.basename(expected_plots['hist_freq'])}")
            report_lines.append(f"      🖼  Variabilidad por SNP: {os.path.basename(expected_plots['std_snps'])}")
            report_lines.append(f"      🖼  Alelos dominantes por haplotipo: {os.path.basename(expected_plots['count_alleles'])}")
            report_lines.append(f"      🖼  Distribución distancias de Hamming: {os.path.basename(expected_plots['hist_hamming'])}")

            report_lines.append('')
            report_lines.append('  🔗 Coeficientes Globales de Correlación LD')
            report_lines.append('  ──────────────────────────────────────────')
            # compute LD summary now so we can show some numbers
            ld_mean, corrs, corr_full = full_ld_check(H)
            total_pairs = corrs.size
            report_lines.append(f"      • Correlación media absoluta (global): {ld_mean:.4f}")
            report_lines.append(f"      • Total de pares evaluados: {total_pairs}")
            report_lines.append(f"      🖼  Mapa de calor de correlación (LD): {os.path.basename(expected_plots['heatmap_corr'])}")
            report_lines.append(f"      🖼  Distribución de correlaciones (LD): {os.path.basename(expected_plots['hist_corr'])}")
            report_lines.append(f"      🖼  CDF de correlación LD: {os.path.basename(expected_plots['cdf_corr'])}")

            report_lines.append('')
            report_lines.append('  ⚖ Veredicto de Desequilibrio de Ligamento')
            report_lines.append('  ───────────────────────────────────────────')
            pct_over = 0.0
            try:
                pct_over = 100.0 * (np.abs(corrs) >= cfg.thresh_opt_abs_corr).sum() / max(1, corrs.size)
            except Exception:
                pct_over = 0.0
            is_opt = (np.abs(ld_mean) >= cfg.thresh_opt_mean) or (pct_over >= (cfg.thresh_opt_pct * 100.0))
            report_lines.append(f"      • Correlación media absoluta (global): {ld_mean:.4f}")
            report_lines.append(f"      • % pares con |corr| >= {cfg.thresh_opt_abs_corr}: {pct_over:.2f}%")
            report_lines.append(f"      • Total de pares evaluados: {total_pairs}")
            report_lines.append(f"      ✅ Verificación previa: {'ÓPTIMO' if is_opt else 'NO ÓPTIMO'}")

            # Similitud genotípica (pares)
            # calcular algunos percentiles y pares extremos
            dlist = []
            for a in range(H.shape[0]):
                for b in range(a+1, H.shape[0]):
                    dlist.append(((a,b), int((H[a]!=H[b]).sum())))
            dvals = np.array([v for (_,v) in dlist]) if dlist else np.array([0])
            p33 = float(np.percentile(dvals,33))
            p66 = float(np.percentile(dvals,66))
            report_lines.append('')
            report_lines.append('  📐 Análisis de Similitud Genotípica (Pares de Haplotipos)')
            report_lines.append('  ──────────────────────────────────────────────────────────')
            report_lines.append(f"      • Número de pares de haplotipos: {len(dvals)}")
            report_lines.append(f"      • Pares mostrados: 3 similares / 3 distintos")
            report_lines.append(f"      • Vista parcial: primeros 32 SNPs")
            report_lines.append(f"      • Percentiles (Hamming): P33={p33:.2f}, P66={p66:.2f}")

            # Encontrar 3 pares más similares y 3 más distintos
            sorted_pairs = sorted(dlist, key=lambda x: x[1])
            top_sim = sorted_pairs[:3]
            top_dist = sorted_pairs[-3:][::-1]
            report_lines.append('')
            report_lines.append('    🤝 Pares de mayor similitud genética')
            for (ia, ib), val in top_sim:
                report_lines.append(f"      • Par ({ia+1}, {ib+1}) | Hamming={val} | muy similar")
                report_lines.append(f"        h{ia+1:03d}: {(_as_bits(H[ia,:32]))}...")
                report_lines.append(f"        h{ib+1:03d}: {(_as_bits(H[ib,:32]))}...")
            report_lines.append('')
            report_lines.append('    ↔ Pares de mayor divergencia genética')
            for (ia, ib), val in top_dist:
                report_lines.append(f"      • Par ({ia+1}, {ib+1}) | Hamming={val} | muy distinto")
                report_lines.append(f"        h{ia+1:03d}: {(_as_bits(H[ia,:32]))}...")
                report_lines.append(f"        h{ib+1:03d}: {(_as_bits(H[ib,:32]))}...")

            # Sumar cierre
            # NOTE: do not write a partial terminal log here; the full stdout/stderr
            # will be captured at runtime and copied into the dataset folder at the end
            # of the pipeline. This avoids truncated logs.
            print_status("Terminal log will be saved at pipeline end.")

        except Exception as e:
            print_status(f"Error en post-procesado sintético: {e}", success=False)
            raise
    elif cfg.data_source == "hinds2005":
        # Carga el bloque exacto de 48 patrones alelicos × 1032 SNPs utilizado
        # en Hinds et al. (2005) y reproducido por Ting et al. (2010) y Moqa et al. (2022).
        H, snp_ids, snp_positions, haplotype_ids = load_hinds2005_block(cfg.hinds2005_path)
        cfg.n_haplotypes = int(H.shape[0])
        # No crear árbol de directorios de nuevo; reutilizar cfg.folders establecido en main
    else:
        raise ValueError(f"DATA_SOURCE no soportado: {cfg.data_source}")

    return H, snp_ids, snp_positions, haplotype_ids

def _exportar_dataset_seleccionado(H, snp_ids, snp_positions, haplotype_ids, folders, mode_tag):
    """
    Exporta la matriz de haplotipos y metadatos de SNPs a un archivo CSV para análisis externo.
    """
    try:
        # ── 1. Construir el DataFrame con las etiquetas adecuadas ──────────────────
        df = pd.DataFrame(H, index=haplotype_ids, columns=snp_ids)
        
        # ── 2. Crear una fila adicional para las posiciones genómicas ──────────────
        # Insertamos la posición como primera fila (opcionalmente) o en un archivo adjunto.
        # Aquí la incluiremos como metadatos en un archivo separado para mayor limpieza
        # o como cabecera extra si el usuario lo prefiere. Por ahora, CSV estándar.
        
        ruta_csv = os.path.join(folders['datos'], f'matriz_haplotipos_seleccionada_{mode_tag}.csv')
        df.to_csv(ruta_csv)
        
        # ── 3. Guardar metadatos de SNPs (IDs y posiciones) por separado ───────────
        df_meta = pd.DataFrame({
            'snp_id': snp_ids,
            'posicion': snp_positions
        })
        ruta_meta = os.path.join(folders['datos'], f'metadatos_snps_seleccionados_{mode_tag}.csv')
        df_meta.to_csv(ruta_meta, index=False)
        
        print_status(f"Dataset exportado: {len(snp_ids)} SNPs x {len(haplotype_ids)} haplotipos", success=True)
        print_saved_plot(ruta_csv, "Matriz de haplotipos (CSV)")
        print_saved_plot(ruta_meta, "Metadatos de SNPs (CSV)")
        
    except Exception as e:
        print_status(f"Error al exportar dataset: {e}", success=False)

# --- UI HELPERS ---
def get_terminal_link(path, label=None):
    """Genera una secuencia de escape OSC 8 para un enlace clickable en la terminal."""
    if label is None:
        label = os.path.basename(path)
    abs_path = os.path.abspath(path)
    # Algunos terminales necesitan el protocolo file://
    url = f"file://{abs_path}"
    return f"\033]8;;{url}\033\\\033[36m{label}\033[0m\033]8;;\033\\"

def print_header(title, color="\033[93m"):
    reset = "\033[0m"
    bold = "\033[1m"
    width = 80
    border_top = f"{color}{bold}╔" + "═" * (width - 2) + "╗"
    content = f"║ {title.center(width - 4)} ║"
    border_bot = f"╚" + "═" * (width - 2) + f"╝{reset}"
    print(f"\n{border_top}\n{content}\n{border_bot}")

def print_subsection(title, icon="🔹"):
    """Imprime un encabezado de subsección con un formato discreto pero claro."""
    print(f"\n  {icon} \033[1m{title}\033[0m")
    print("  " + "─" * (len(title) + 4))

def print_step(message, icon="🚀"):
    print(f"\n  {icon} \033[1m{message}\033[0m")

def print_status(message, success=True):
    icon = "✅" if success else "❌"
    color = "\033[92m" if success else "\033[91m"
    reset = "\033[0m"
    print(f"      {color}{icon} {message}{reset}")

def print_saved_plot(path, description="Gráfico guardado"):
    link = get_terminal_link(path)
    print(f"      🖼️  {description}: {link}")

def save_report_figure(path, dpi=300, tight=False, fig=None):
    if fig is None:
        fig = plt.gcf()
    save_kwargs = {'dpi': dpi}
    if tight:
        save_kwargs['bbox_inches'] = 'tight'
    fig.savefig(path, **save_kwargs)

def print_table(df, title=None):
    if title:
        print(f"\n--- 📊 {title} " + "─" * (60 - len(title)))

    with pd.option_context('display.max_rows', 15, 'display.max_columns', None, 'display.width', 1000, 'display.precision', 4):
        print(df.to_string(index=False))


def generate_ld_haplotype_block(n_haplotypes=40, n_snps=1200, block_size=50, flip_prob=0.02, seed=42,
                                min_pairwise_diff: int = 0, max_attempts: int = 1000):
    """Genera una matriz haplotipo-SNP binaria con estructura de bloques de LD.

    Optionally enforces a minimum pairwise Hamming distance between haplotypes
    (best-effort, bounded by max_attempts).
    """
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n_snps / block_size))
    base_patterns = rng.integers(0, 2, size=(n_haplotypes, n_blocks), dtype=np.int8)
    X = np.zeros((n_haplotypes, n_snps), dtype=np.int8)
    snp_idx = 0
    for b in range(n_blocks):
        remaining = n_snps - snp_idx
        width = min(block_size, remaining)
        leader = base_patterns[:, b].copy()
        for _ in range(width):
            col = leader.copy()
            flip_mask = rng.random(n_haplotypes) < flip_prob
            col[flip_mask] = 1 - col[flip_mask]
            X[:, snp_idx] = col
            snp_idx += 1

    # If no enforcement requested, return immediately
    if not min_pairwise_diff or min_pairwise_diff <= 0:
        return X

    # Helper: compute pairwise Hamming distances (flattened for i<j)
    def _pairwise_dists(H):
        n = H.shape[0]
        idx_i, idx_j = np.triu_indices(n, k=1)
        # XOR and sum across SNP axis
        diffs = (H[idx_i] != H[idx_j]).sum(axis=1)
        return diffs, idx_i, idx_j

    attempts = 0
    H = X.copy()
    diffs, idx_i, idx_j = _pairwise_dists(H)
    cur_min = int(diffs.min()) if diffs.size > 0 else 0
    while cur_min < min_pairwise_diff and attempts < max_attempts:
        # select the pair with smallest distance
        arg = int(np.argmin(diffs))
        i = int(idx_i[arg]); j = int(idx_j[arg])

        # choose which haplotype to alter (pick the one with smaller mean distance)
        mean_i = ((H[i] != H).sum(axis=1)).mean()
        mean_j = ((H[j] != H).sum(axis=1)).mean()
        hap_to_mut = i if mean_i <= mean_j else j

        # positions where they are equal are candidates to flip
        eq_pos = np.where(H[i] == H[j])[0]
        if eq_pos.size == 0:
            # fallback: flip a random position
            pos = rng.integers(0, n_snps, size=1)
        else:
            needed = min_pairwise_diff - cur_min
            flip_count = min(eq_pos.size, max(1, int(np.ceil(needed))))
            pos = rng.choice(eq_pos, size=flip_count, replace=False)

        # apply flips
        H[hap_to_mut, pos] = 1 - H[hap_to_mut, pos]

        # recompute distances
        diffs, idx_i, idx_j = _pairwise_dists(H)
        cur_min = int(diffs.min()) if diffs.size > 0 else 0
        attempts += 1

    if cur_min < min_pairwise_diff:
        print(f"      ⚠ No se pudo alcanzar min_pairwise_diff={min_pairwise_diff} tras {attempts} intentos. Logrado {cur_min}.")
    else:
        print(f"      ✅ Generador sintético alcanzó min_pairwise_diff={cur_min} tras {attempts} intentos.")

    # Return also attempts and achieved minimum distance for diagnostics
    return H, attempts, cur_min

def load_hinds2005_block(filepath: str):
    """
    Carga el bloque de haplotipos de Hinds et al. (2005) desde un fichero
    de texto binario (formato Ting 2010: una fila por clase alélica,
    caracteres '0'/'1' sin separadores).

    El fichero corresponde al bloque utilizado en:
      - Ting et al. (2010) Bioinformatics 26(11):1446-1452
      - Moqa et al. (2022) PLOS ONE 17(12):e0278560

    Dimensiones confirmadas: 48 patrones alelicos × 1032 SNPs.

    Parámetros
    ----------
    filepath : str
        Ruta al fichero hinds2005_1032.txt.

    Devuelve
    --------
    H : np.ndarray (n_patrones, n_snps), dtype int8
    snp_ids : list[str]
    snp_positions : np.ndarray (n_snps,), dtype int
    haplotype_ids : list[str]
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Fichero Hinds 2005 no encontrado: '{filepath}'. "
            f"Asegúrese de que data/hinds2005_1032.txt existe en el directorio del proyecto."
        )
    with open(filepath) as f:
        filas = [l.strip() for l in f if l.strip()]
    if not filas:
        raise ValueError(f"El fichero '{filepath}' está vacío o no contiene datos válidos.")
    H = np.array([[int(c) for c in fila] for fila in filas], dtype=np.int8)
    n_patrones, n_snps = H.shape
    snp_ids = [f"snp_{i}" for i in range(n_snps)]
    snp_positions = np.arange(n_snps, dtype=int)
    haplotype_ids = [f"patron_{i}" for i in range(n_patrones)]
    print_status(
        f"Hinds 2005 cargado: {n_patrones} patrones alelicos × {n_snps} SNPs "
        f"desde '{filepath}'",
        success=True
    )
    return H, snp_ids, snp_positions, haplotype_ids



def full_ld_check(X): # Definir función para calcular el Desequilibrio de Ligamiento (LD) completo
    # Calcular la matriz de correlación de Pearson entre las columnas (SNPs).
    # Se utiliza np.errstate para suprimir advertencias (RuntimeWarning) causadas por SNPs
    # monomórficos (desviación estándar cero), que son comunes en datasets reales.
    with np.errstate(divide='ignore', invalid='ignore'):
        corr_full = np.corrcoef(X.T)
    
    # Sustituir NaNs e infinitos por 0.0 (los NaNs ocurren por la división por cero mencionada)
    corr_full = np.nan_to_num(corr_full, nan=0.0, posinf=0.0, neginf=0.0)
    tri_i, tri_j = np.triu_indices(corr_full.shape[0], k=1) # Obtener los índices de la parte superior de la matriz (sin la diagonal)
    corrs = corr_full[tri_i, tri_j] # Extraer todos los coeficientes de correlación del triángulo superior
    ld_mean = float(np.mean(np.abs(corrs))) if corrs.size > 0 else 0.0 # Calcular la media de las correlaciones absolutas
    return ld_mean, corrs, corr_full # Devolver la media, el array de correlaciones y la matriz completa

def etiqueta_distancia_por_percentil(dist, p33, p66):
    """Devuelve una etiqueta basada en los percentiles precalculados."""
    if dist <= p33:
        return 'muy similar'
    elif dist <= p66:
        return 'intermedio'
    return 'muy distinto'

def _as_bits(arr):
    return ''.join(arr.astype(str).tolist())

def evaluate_candidate(mask, H, pair_idx=None):
    if pair_idx is None:
        # Intentar obtener del ámbito global
        pair_idx = globals().get('PAIR_IDX')
    
    # Convertir la máscara a tipo booleano
    mask = mask.astype(bool)
    # Contar cuántos SNPs han sido seleccionados (SNPs "tag")
    k = int(mask.sum())
    # Si no hay ningún SNP seleccionado, elegir uno al azar para evitar errores
    if k == 0:
        ridx = np.random.randint(0, H.shape[1])
        mask = mask.copy()
        mask[ridx] = True
        k = 1
    # Filtrar la matriz de haplotipos para quedarse solo con los SNPs seleccionados
    Hs = H[:, mask]
    # Obtener los alelos de los pares de individuos que queremos comparar
    a = Hs[pair_idx[:, 0], :]
    b = Hs[pair_idx[:, 1], :]
    # Calcular la distancia de Hamming (suma de diferencias absolutas) para cada par
    d = np.abs(a - b).sum(axis=1).astype(float)
    # Tolerancia a datos faltantes (definición del paper): min(D_ij) - 1
    # NOTA: No se acota inferiormente (puede ser negativa si hay pares sin cubrir).
    tolerance_real = float(d.min() - 1.0)
    # El objetivo es maximizar la separación promedio entre pares
    hamming_avg_real = float(d.mean())
    # El objetivo es minimizar la varianza de las distancias (uniformidad)
    balance_var = float(d.var())
    # El objetivo es minimizar el número de SNPs seleccionados
    compactness = float(k)
    # Definir los objetivos del modelo (f2 y f3 en negativo para maximizar)
    f1 = compactness        # Minimizar tamaño del conjunto
    f2 = -tolerance_real    # Maximizar resolución mínima
    f3 = -hamming_avg_real  # Maximizar distancia promedio
    f4 = balance_var        # Minimizar varianza del balance
    # Devolver el vector de objetivos y un diccionario con los valores reales
    return np.array([f1, f2, f3, f4], dtype=float), {
        'compactness_real': compactness,
        'tolerance_real': tolerance_real,
        'hamming_avg_real': hamming_avg_real,
        'balance_var_real': balance_var
    }

def evaluate_population(Xpop, H, pair_idx=None):
    """Evaluación secuencial por compatibilidad con código externo."""
    if pair_idx is None:
        pair_idx = globals().get('PAIR_IDX')
    F = np.zeros((Xpop.shape[0], 4), dtype=float)
    aux = []
    for i in range(Xpop.shape[0]):
        f, a = evaluate_candidate(Xpop[i], H, pair_idx=pair_idx)
        F[i] = f
        aux.append(a)
    return F, aux


def _evaluar_poblacion_vectorizado(X_bool: np.ndarray,
                                   diff_matrix: np.ndarray) -> np.ndarray:
    """
    Evaluación completamente vectorizada de toda la población sobre los 4 objetivos.

    La clave del rendimiento es que la distancia de Hamming de cada par bajo
    la selección del individuo i se puede expresar como un producto matricial:

        D = (diff_matrix @ X_bool.T).T      shape: (pop_size, n_pairs)

    donde diff_matrix[j, s] = 1 si el par j difiere en el SNP s.
    Con D calculada de una sola vez para toda la población:
        f1 = numero de SNPs seleccionados      (minimizar)
        f2 = -min(D)                           (maximizar resolucion)
        f3 = -mean(D)                          (maximizar distancia media)
        f4 = var(D)                            (minimizar varianza)
    """
    pop_size = X_bool.shape[0]

    # Numero de SNPs seleccionados por cada individuo
    k = X_bool.sum(axis=1).astype(float)              # (pop_size,)

    # Individuo sin ningun SNP: seleccionar el primero para evitar division por 0
    sin_seleccion = (k == 0)
    if sin_seleccion.any():
        X_bool = X_bool.copy()
        X_bool[sin_seleccion, 0] = True
        k[sin_seleccion] = 1.0

    # Distancias de Hamming por par para toda la poblacion en un solo producto matricial
    # diff_matrix: (n_pairs, n_snps)  — int16
    # X_bool.T:    (n_snps, pop_size) — bool cast a int
    D = (diff_matrix.astype(np.int32) @ X_bool.T.astype(np.int32)).T.astype(float)
    # D shape: (pop_size, n_pairs)

    # Tolerancia (paper): min(D_ij) - 1 (sin clamp inferior)
    tolerancia = (D.min(axis=1) - 1.0)
    hamming_med = D.mean(axis=1)      # (pop_size,)
    varianza = D.var(axis=1)          # (pop_size,)

    F = np.column_stack([
        k,               # f1: compacidad
        -tolerancia,     # f2: resolución (negativo para maximizar)
        -hamming_med,    # f3: distancia media (negativo para maximizar)
        varianza,        # f4: varianza
    ])
    return F.astype(float)


class TagSNPProblem(Problem):
    """Problema de selección de Tag SNPs para búsqueda multiobjetivo."""

    def __init__(self, H: np.ndarray, pair_idx: np.ndarray, normalize_for_search: bool = False):
        """
        Parameters
        ----------
        H        : matriz de haplotipos (n_hap, n_snps), dtype int8
        pair_idx : pares de haplotipos a comparar, shape (n_pairs, 2)
        """
        self.H = H
        self.pair_idx = pair_idx
        self.normalize_for_search = bool(normalize_for_search)
        # Precomputa la matriz de diferencias una sola vez — evita recalculo en cada generacion
        self.diff_matrix = (
            H[pair_idx[:, 0], :] != H[pair_idx[:, 1], :]
        ).astype(np.int16)            # (n_pairs, n_snps)

        n_var = H.shape[1]
        # Rango máximo demostrable: matriz de distancias completa con todos los SNPs activos
        D_full = self.diff_matrix.sum(axis=1).astype(float)
        
        self._scale_f1 = max(1.0, float(n_var))
        self._scale_f2 = max(1.0, float(D_full.min() - 1.0))
        self._scale_f3 = max(1.0, float(D_full.mean()))
        self._scale_f4 = max(1.0, float(D_full.var()))

        super().__init__(n_var=n_var, n_obj=4, n_ieq_constr=0, xl=0, xu=1, vtype=bool)

    def _evaluate(self, X, out, *args, **kwargs):
        """Evalua toda la poblacion de forma vectorizada en un solo pase NumPy."""
        X_bool = X.astype(bool)
        F_raw = _evaluar_poblacion_vectorizado(X_bool, self.diff_matrix)
        if self.normalize_for_search:
            F_scaled = F_raw.copy()
            F_scaled[:, 0] = F_scaled[:, 0] / self._scale_f1
            F_scaled[:, 1] = F_scaled[:, 1] / self._scale_f2
            F_scaled[:, 2] = F_scaled[:, 2] / self._scale_f3
            F_scaled[:, 3] = F_scaled[:, 3] / self._scale_f4
            out['F'] = F_scaled
        else:
            out['F'] = F_raw

def snp_distinguishability(H, pair_idx=None):
    if pair_idx is None:
        pair_idx = globals().get('PAIR_IDX')
    # Obtiene los haplotipos para el primer elemento de cada par de comparación
    a = H[pair_idx[:, 0], :]
    # Obtiene los haplotipos para el segundo elemento de cada par de comparación
    b = H[pair_idx[:, 1], :]
    # Calcula la diferencia bit a bit entre los pares y la convierte a entero
    diff = (a != b).astype(np.int8)
    # Suma las diferencias por cada SNP y devuelve el resultado como float
    return diff.sum(axis=0).astype(float)


def _build_distinguishability_groups(scores_desc_order: np.ndarray, dscore: np.ndarray):
    """Agrupa índices SNP contiguos con la misma distinguibilidad (orden descendente)."""
    groups = []
    if len(scores_desc_order) == 0:
        return groups
    current = [int(scores_desc_order[0])]
    current_score = float(dscore[scores_desc_order[0]])
    for idx in scores_desc_order[1:]:
        score = float(dscore[idx])
        if score == current_score:
            current.append(int(idx))
        else:
            groups.append(np.array(current, dtype=int))
            current = [int(idx)]
            current_score = score
    groups.append(np.array(current, dtype=int))
    return groups


def _build_order_with_random_ties(score_groups: list, rng: np.random.Generator) -> np.ndarray:
    """
    Mantiene el orden greedy por score, pero rompe empates de forma aleatoria.
    """
    if not score_groups:
        return np.array([], dtype=int)
    parts = []
    for group in score_groups:
        if len(group) <= 1:
            parts.append(group)
        else:
            parts.append(rng.permutation(group))
    return np.concatenate(parts).astype(int, copy=False)

def greedy_construct(H, coverage_target=1, sorted_idx=None, pair_idx=None):
    if pair_idx is None:
        pair_idx = globals().get('PAIR_IDX')
    # Obtiene el número de SNPs de la matriz de haplotipos
    n_snps = H.shape[1]
    # Obtiene el número total de pares a distinguir
    n_pairs = pair_idx.shape[0]
    # Si no se proporciona un orden de SNPs, se calcula según su capacidad de distinción
    if sorted_idx is None:
        dscore = snp_distinguishability(H)
        sorted_idx = np.argsort(-dscore)
    # Inicializa el array de SNPs seleccionados (booleano)
    selected = np.zeros(n_snps, dtype=bool)
    # Inicializa el contador de cuántas veces ha sido cubierto cada par
    covered = np.zeros(n_pairs, dtype=np.int32)
    # Precalcula las diferencias para todos los pares en todos los SNPs
    a = H[pair_idx[:, 0], :]
    b = H[pair_idx[:, 1], :]
    diff_all = (a != b)
    # Itera sobre los SNPs siguiendo el orden de importancia (Greedy)
    for s in sorted_idx:
        # Si todos los pares ya han alcanzado el objetivo de cobertura, termina
        if np.all(covered >= coverage_target):
            break
        # Obtiene la contribución del SNP actual para distinguir pares
        contrib = diff_all[:, s].astype(np.int32)
        # Si el SNP ayuda a cubrir algún par que aún no tiene la cobertura objetivo
        if np.any((covered < coverage_target) & (contrib > 0)):
            # Selecciona el SNP y actualiza la cobertura de los pares
            selected[s] = True
            covered += contrib
    # Si no se ha seleccionado ningún SNP, selecciona al menos el mejor puntuado
    if not selected.any():
        selected[sorted_idx[0]] = True
    # Devuelve el array booleano con los SNPs seleccionados
    return selected

class GreedyHybridTagSNPSampling(Sampling):  # Inicialización híbrida (Greedy + Aleatorio)
    def __init__(self, H, max_coverage=5, random_fill_ratio=0.2, seed=42):
        super().__init__()
        self.H = H
        self.max_coverage = max_coverage
        self.random_fill_ratio = random_fill_ratio
        self.rng = np.random.default_rng(seed)
        dscore = snp_distinguishability(H)
        self.sorted_idx = np.argsort(-dscore)
        self.score_groups = _build_distinguishability_groups(self.sorted_idx, dscore)

    def _do(self, problem, n_samples, **kwargs):
        X = np.zeros((n_samples, problem.n_var), dtype=bool)
        n_random = int(n_samples * self.random_fill_ratio)
        n_greedy = n_samples - n_random
        coverages = np.linspace(1, self.max_coverage, n_greedy).astype(int) if n_greedy > 0 else []

        for i in range(n_greedy):
            c = int(coverages[i]) if len(coverages) > 0 else 1
            sorted_idx_i = _build_order_with_random_ties(self.score_groups, self.rng)
            X[i] = greedy_construct(self.H, coverage_target=max(1, c), sorted_idx=sorted_idx_i)

        for i in range(n_greedy, n_samples):
            row = self.rng.random(problem.n_var) < 0.05
            if not row.any():
                row[self.rng.integers(0, problem.n_var)] = True
            X[i] = row
        return X


class GreedyHybrid5050TagSNPSampling(Sampling):  # Inicialización GI 50/50 (Greedy + Aleatorio)
    """Inicialización híbrida 50/50 al estilo Ting.

    - 50% de individuos: greedy con coverage_target=1
    - 50% de individuos: aleatorio con p(bit=1)=cfg.gi_random_p (por defecto 0.5)
    """

    def __init__(self, H, random_bit_p: float = 0.5, seed: int = 42):
        super().__init__()
        self.H = H
        self.random_bit_p = float(random_bit_p)
        self.rng = np.random.default_rng(seed)
        dscore = snp_distinguishability(H)
        self.sorted_idx = np.argsort(-dscore)
        self.score_groups = _build_distinguishability_groups(self.sorted_idx, dscore)

    def _do(self, problem, n_samples, **kwargs):
        X = np.zeros((n_samples, problem.n_var), dtype=bool)
        n_greedy = int(n_samples // 2)
        n_random = int(n_samples - n_greedy)

        for i in range(n_greedy):
            sorted_idx_i = _build_order_with_random_ties(self.score_groups, self.rng)
            row = greedy_construct(self.H, coverage_target=1, sorted_idx=sorted_idx_i)
            if not row.any():
                row[self.rng.integers(0, problem.n_var)] = True
            X[i] = row

        for j in range(n_random):
            i = n_greedy + j
            row = (self.rng.random(problem.n_var) < self.random_bit_p)
            if not row.any():
                row[self.rng.integers(0, problem.n_var)] = True
            X[i] = row

        return X

class GreedyPureTagSNPSampling(Sampling):  # Inicialización 100% Greedy
    def __init__(self, H, max_coverage=5, seed=42):
        super().__init__()
        self.H = H
        self.max_coverage = max_coverage
        self.rng = np.random.default_rng(seed)
        dscore = snp_distinguishability(H)
        self.sorted_idx = np.argsort(-dscore)
        self.score_groups = _build_distinguishability_groups(self.sorted_idx, dscore)

    def _do(self, problem, n_samples, **kwargs):
        X = np.zeros((n_samples, problem.n_var), dtype=bool)
        coverages = np.linspace(1, self.max_coverage, n_samples).astype(int) if n_samples > 0 else []

        for i in range(n_samples):
            c = int(coverages[i]) if len(coverages) > 0 else 1
            sorted_idx_i = _build_order_with_random_ties(self.score_groups, self.rng)
            row = greedy_construct(self.H, coverage_target=max(1, c), sorted_idx=sorted_idx_i)
            if not row.any():
                row[self.rng.integers(0, problem.n_var)] = True
            X[i] = row
        return X

def build_ref_dirs_for_pop(pop_size, n_obj=4):
    best_ref_dirs = get_reference_directions('das-dennis', n_obj, n_partitions=1)
    best_partitions = 1
    p = 1
    while True:
        cand = get_reference_directions('das-dennis', n_obj, n_partitions=p + 1)
        if len(cand) > pop_size:
            break
        best_ref_dirs = cand
        best_partitions = p + 1
        p += 1
    return best_ref_dirs, best_partitions

class SparseBinaryRandomSampling(Sampling):
    def __init__(self, prob: float = 0.05, seed=42):
        super().__init__()
        self.prob = prob
        self.rng = np.random.default_rng(seed)

    def _do(self, problem, n_samples, **kwargs):
        X = self.rng.random((n_samples, problem.n_var)) < self.prob
        # Garantizar que ningún cromosoma quede completamente vacío
        empty = ~X.any(axis=1)
        if empty.any():
            for i in np.where(empty)[0]:
                X[i, self.rng.integers(0, problem.n_var)] = True
        return X

def build_algorithm(problem, H, algo_name, init_name, cfg, seed=42, ref_dirs=None):
    # Alias de compatibilidad hacia atrás: 'greedy' -> 'greedy_hybrid'
    if init_name == 'greedy':
        init_name = 'greedy_hybrid'

    init_name = str(init_name)
    base_init_name = init_name
    coverage_override = None
    if init_name.startswith('greedy_pure_'):
        cov_txt = init_name[len('greedy_pure_'):]
        if not cov_txt.isdigit() or int(cov_txt) <= 0:
            raise ValueError(f'Inicialización greedy_pure inválida: {init_name}')
        base_init_name = 'greedy_pure'
        coverage_override = int(cov_txt)

    # Selección del método de inicialización
    if base_init_name in ['random', 'random_sparse']:
        # Instanciamos la clase con inicialización dispersa (Sparse)
        # Esto penaliza la inflación artificial del Average Hamming Distance en Random.
        probabilidad_esperada = max(0.01, min(0.5, 70.0 / problem.n_var)) 
        sampling = SparseBinaryRandomSampling(prob=probabilidad_esperada, seed=seed)
    elif base_init_name == 'random_dense':
        # Instanciamos la clase con el muestreo aleatorio denso por defecto de PyMoo (prob=0.5)
        # Útil como control para demostrar la inflación de métricas de distancia.
        from pymoo.operators.sampling.rnd import BinaryRandomSampling
        sampling = BinaryRandomSampling()
    elif base_init_name == 'greedy_hybrid':
        cobertura_max = int(coverage_override) if coverage_override is not None else int(getattr(cfg, 'greedy_max_coverage', 5))
        sampling = GreedyHybridTagSNPSampling(H=H, max_coverage=cobertura_max, random_fill_ratio=0.2, seed=seed)
    elif base_init_name == 'greedy_hybrid_50-50':
        p_bit = float(getattr(cfg, 'gi_random_p', 0.5))
        sampling = GreedyHybrid5050TagSNPSampling(H=H, random_bit_p=p_bit, seed=seed)
    elif base_init_name == 'greedy_pure':
        cobertura_max = int(coverage_override) if coverage_override is not None else int(getattr(cfg, 'greedy_max_coverage', 5))
        sampling = GreedyPureTagSNPSampling(H=H, max_coverage=cobertura_max, seed=seed)
    else:
        raise ValueError(f'Inicialización no soportada: {init_name}')

    crossover = UX(prob=cfg.pc)
    mutation = BitflipMutation(prob=cfg.pm)

    if algo_name == 'NSGA2':
        return NSGA2(
            pop_size=cfg.pop_size,
            sampling=sampling,
            crossover=crossover,
            mutation=mutation,
            eliminate_duplicates=True,
            n_offsprings=cfg.offspring
        )

    if algo_name == 'SPEA2':
        return SPEA2(
            pop_size=cfg.pop_size,
            sampling=sampling,
            crossover=crossover,
            mutation=mutation,
            eliminate_duplicates=True
        )

    if ref_dirs is None:
        ref_dirs, _ = build_ref_dirs_for_pop(cfg.pop_size, n_obj=4)

    if algo_name == 'NSGA3':
        return NSGA3(
            pop_size=cfg.pop_size,
            ref_dirs=ref_dirs,
            sampling=sampling,
            crossover=crossover,
            mutation=mutation,
            eliminate_duplicates=True,
            n_offsprings=cfg.offspring
        )

    if algo_name == 'MOEAD':
        return MOEAD(
            ref_dirs=ref_dirs,
            n_neighbors=cfg.moead_neighbors,
            sampling=sampling,
            crossover=crossover,
            mutation=mutation
        )

    raise ValueError(f'Algoritmo no soportada: {algo_name}')

def build_algorithms(problem, H, cfg, seed=42):
    ref_dirs, n_part = build_ref_dirs_for_pop(cfg.pop_size, n_obj=4)
    configs = {}
    init_options = cfg.init_options
    for algo_name in ['NSGA2', 'NSGA3', 'SPEA2', 'MOEAD']:
        for init_name in init_options:
            configs[(algo_name, init_name)] = build_algorithm(
                problem=problem,
                H=H,
                algo_name=algo_name,
                init_name=init_name,
                cfg=cfg,
                seed=seed,
                ref_dirs=ref_dirs
            )
    return configs, ref_dirs, n_part

@dataclass
class RunResult:
    algorithm: str
    init: str
    run: int
    seed: int
    elapsed_sec: float
    X_final: np.ndarray
    F_final: np.ndarray
    F_history: list



def _ejecutar_un_experimento(args: dict) -> 'RunResult':
    """
    Función de ejecución a nivel de módulo (picklable).
    Ejecuta una única llamada minimize() y devuelve un RunResult.
    """
    algo_name  = args['algo_name']
    init_name  = args['init_name']
    run_idx    = args['run_idx']
    seed       = args['seed']
    H          = args['H']
    pair_idx   = args['pair_idx']
    cfg        = args['cfg']
    ref_dirs   = args['ref_dirs']

    # Reconstruir el problema en el subproceso (no se puede compartir entre procesos)
    problema = TagSNPProblem(H, pair_idx, normalize_for_search=(algo_name == 'MOEAD'))
    algo = build_algorithm(
        problem=problema, H=H, algo_name=algo_name, init_name=init_name,
        cfg=cfg, seed=seed, ref_dirs=ref_dirs
    )
    termination = get_termination('n_gen', cfg.n_gen)

    # Utilizar callback ligero en lugar de save_history=True para evitar OOM.
    # save_history=True almacena una copia profunda del algoritmo completo por
    # generación (~1.5 GB/worker en modo full).  El callback solo guarda F.
    from pymoo.core.callback import Callback

    class _LightweightCallback(Callback):
        def __init__(self, H_local, pair_idx_local):
            super().__init__()
            self._H = H_local
            self._pair_idx = pair_idx_local
            self.F_history = []

        def notify(self, algorithm):
            pop = algorithm.pop
            if pop is None:
                return
            try:
                X_gen = np.array(pop.get('X'), dtype=bool)
            except Exception:
                try:
                    X_gen = np.array(getattr(pop, 'X', None), dtype=bool)
                except Exception:
                    return
            if X_gen is None or X_gen.size == 0:
                return
            F_gen, _ = evaluate_population(X_gen, self._H, pair_idx=self._pair_idx)
            if F_gen.ndim == 1:
                F_gen = F_gen.reshape(1, -1)
            if F_gen.shape[1] >= 4:
                self.F_history.append(F_gen[:, :4].copy())

    cb = _LightweightCallback(H, pair_idx)

    t0 = time.time()
    res = minimize(problema, algo, termination, seed=seed, verbose=False,
                   save_history=False, callback=cb)
    elapsed = time.time() - t0

    n_var = problema.n_var
    Xf = np.array(res.X, dtype=bool) if getattr(res, 'X', None) is not None else np.empty((0, n_var), dtype=bool)
    # Para garantizar métricas comparables, siempre reconstruimos F en escala cruda
    if Xf.shape[0] > 0:
        Ff, _ = evaluate_population(Xf, H, pair_idx=pair_idx)
    else:
        Ff = np.empty((0, 4), dtype=float)

    # El historial F ya fue capturado por el callback — no se necesita iterar
    # sobre res.history (que ahora está vacío gracias a save_history=False)
    F_history = cb.F_history

    return RunResult(
        algorithm=algo_name, init=init_name, run=run_idx,
        seed=seed, elapsed_sec=float(elapsed),
        X_final=Xf, F_final=Ff, F_history=F_history
    )


def run_all_experiments(problem, H, cfg, n_runs=None, master_seed=None):
    """
    Ejecuta todos los experimentos multiobjetivo de forma secuencial.
    """
    if n_runs is None:
        n_runs = cfg.n_runs
    if master_seed is None:
        master_seed = cfg.master_seed

    n_runs = int(n_runs)
    base_seeds = np.random.default_rng(master_seed).integers(0, 10_000_000, size=n_runs)
    ref_dirs, _ = build_ref_dirs_for_pop(cfg.pop_size, n_obj=4)
    pair_idx = problem.pair_idx  # Extraer del problema vectorizado

    # Construir lista plana de todos los trabajos independientes
    init_options = cfg.init_options
    algoritmos   = ['NSGA2', 'NSGA3', 'SPEA2', 'MOEAD']
    trabajos = []
    for algo_name in algoritmos:
        for init_name in init_options:
            for r in range(n_runs):
                seed = int(base_seeds[r] + (hash((algo_name, init_name)) % 10000))
                trabajos.append({
                    'algo_name': algo_name,
                    'init_name': init_name,
                    'run_idx':   r + 1,
                    'seed':      seed,
                    'H':         H,
                    'pair_idx':  pair_idx,
                    'cfg':       cfg,
                    'ref_dirs':  ref_dirs,
                })

    total_runs = len(trabajos)
    print_subsection("Fase Evolutiva", icon="🧬")
    print(f"    • Iniciando {total_runs} experimentos en modo paralelo seguro")

    # --- Paralelización segura usando concurrent.futures ---
    # Implementación en paralelo de los experimentos evolutivos independientes.
    # Se utiliza ProcessPoolExecutor para aprovechar múltiples núcleos de CPU,
    # pero se limita el número de procesos a un máximo seguro (8 por defecto)
    # para evitar sobrecargar el sistema y asegurar estabilidad en cualquier equipo.
    import concurrent.futures
    import os

    # Detectar número seguro de workers de forma adaptativa.
    # 1. Reservar 2 núcleos para el SO.
    # 2. Limitar por RAM disponible: estimar ~300 MB por worker (modo full) y
    #    reservar 2 GB para el SO + proceso principal.
    cpu_limit = max(1, (os.cpu_count() or 2) - 2)
    try:
        with open('/proc/meminfo', 'r') as _memf:
            for _line in _memf:
                if _line.startswith('MemAvailable:'):
                    available_mb = int(_line.split()[1]) / 1024  # kB -> MB
                    break
            else:
                available_mb = 8000  # fallback conservador
        ram_per_worker_mb = 350  # estimación conservadora por worker
        ram_reserved_mb = 2048  # reservar 2 GB para SO + proceso principal
        ram_limit = max(1, int((available_mb - ram_reserved_mb) / ram_per_worker_mb))
    except Exception:
        ram_limit = 4  # fallback muy conservador si no puede leerse /proc/meminfo
    max_workers = min(cpu_limit, ram_limit)
    print(f"      • Paralelizando con hasta {max_workers} procesos en paralelo "
          f"(CPUs libres: {cpu_limit}, límite RAM: {ram_limit})")

    resultados_desordenados = {}
    done_runs = 0
    w = len(str(total_runs))

    # Ejecutar en paralelo, recogiendo resultados y errores
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_tarea = {executor.submit(_ejecutar_un_experimento, tarea): tarea for tarea in trabajos}
        for future in concurrent.futures.as_completed(future_to_tarea):
            tarea = future_to_tarea[future]
            try:
                rr = future.result()
            except Exception as e:
                print(
                    f"      • ⚠ Error en [{tarea['algo_name']}-{tarea['init_name']}] "
                    f"ejecución {tarea['run_idx']}: {e}"
                )
                continue

            done_runs += 1
            print(
                f"      • [Progreso: {done_runs:>{w}}/{total_runs}] | "
                f"[{rr.algorithm}-{rr.init}] ejecución {rr.run}/{n_runs} "
                f"({rr.elapsed_sec:.1f}s)"
            )
            key = (rr.algorithm, rr.init, rr.run)
            resultados_desordenados[key] = rr

    # Reordenar resultados para mantener consistencia con la version secuencial
    resultados = []
    for algo_name in algoritmos:
        for init_name in init_options:
            for r in range(1, n_runs + 1):
                rr = resultados_desordenados.get((algo_name, init_name, r))
                if rr is not None:
                    resultados.append(rr)

    return resultados


def global_ideal_nadir(run_results):
    # Obtener todos los frentes finales de los resultados de ejecución que no estén vacíos
    all_F = [rr.F_final for rr in run_results if rr.F_final is not None and len(rr.F_final) > 0]
    # Si no hay frentes, lanzar un error
    if len(all_F) == 0:
        raise ValueError('No hay frentes finales para normalizar.')
    # Apilar todos los frentes en una sola matriz
    F_stack = np.vstack(all_F)
    # Calcular el punto ideal (mínimos de cada objetivo)
    ideal = F_stack.min(axis=0)
    # Calcular el punto nadir (máximos de cada objetivo)
    nadir = F_stack.max(axis=0)
    # Retornar los puntos ideal y nadir
    return ideal, nadir

def minmax_normalize(F, ideal, nadir):
    # Calcular el denominador para la normalización (rango de cada objetivo)
    denom = nadir - ideal
    # Evitar la división por cero reemplazando ceros por 1.0 (el resultado será 0 en esos casos)
    safe = np.where(denom == 0, 1.0, denom)
    # Aplicar la fórmula de normalización Min-Max
    F_norm = (F - ideal) / safe
    # Identificar dónde el denominador era originalmente cero
    deg = denom == 0
    # Si hay casos de división por cero (rango nulo), fijar la normalización a 0.0
    if np.any(deg):
        F_norm[:, deg] = 0.0
    # Retornar el frente normalizado
    return F_norm


def static_dataset_limits_references(n_snps_total: int = 1032, pair_hamming=None):
    """Devuelve (ideal, safe_denom) con límites fijos basados en la estructura del dataset.

    Se define un rango teórico/empírico por objetivo, consistente con los objetivos crudos
    usados en este proyecto (alineado con la escala de resultados de Moqa et al. 2022):

    - f1 = k (min):              [1, L]
    - f2 = -tolerancia (min):    [-(min_H - 1), 1]  
    - f3 = -hamming_avg (min):   [-mean_H, 0]      
    - f4 = var(D) (min):         [0, var_H or L^2/4]
    """
    L = float(max(1, int(n_snps_total)))
    
    if pair_hamming is not None and len(pair_hamming) > 0:
        max_tol = float(pair_hamming.min() - 1.0)
        max_ham = float(pair_hamming.mean())
        # Variación aproximada empírica
        max_var = float(np.var(pair_hamming)) if len(pair_hamming) > 1 else (L * L) / 4.0
    else:
        max_tol = L - 1.0
        max_ham = L
        max_var = (L * L) / 4.0

    ideal = np.array([
        1.0,           # k mínimo
        -max_tol,      # mejor (tolerancia máxima demostrable)
        -max_ham,      # mejor (distancia media máxima demostrable)
        0.0            # varianza mínima
    ], dtype=float)
    nadir = np.array([
        L,             # k máximo
        1.0,           # peor (tolerancia = -1)
        0.0,           # peor (distancia media = 0)
        max_var        # peor (cota empírica)
    ], dtype=float)
    safe = (nadir - ideal + 1e-9)
    return ideal, safe

def compute_range_summin_minsum(F_norm):
    # Calcular el mínimo de cada columna en el frente normalizado
    col_min = F_norm.min(axis=0)
    # Calcular el máximo de cada columna en el frente normalizado
    col_max = F_norm.max(axis=0)
    # Métrica de 'range': suma de los rangos de todos los objetivos
    range_metric = float((col_max - col_min).sum())
    # Métrica de 'summin': suma de los mínimos alcanzados en cada objetivo
    summin_metric = float(col_min.sum())
    # Métrica de 'minsum': el valor mínimo de la suma de objetivos entre todas las soluciones
    minsum_metric = float(F_norm.sum(axis=1).min())
    # Retornar las tres métricas calculadas
    return range_metric, summin_metric, minsum_metric

def compute_raw_aux_metrics(F_raw, n_snps_total: int = 1032):
    """
    Calcula métricas auxiliares crudas (independientes de la normalización del frente).

    Parámetros
    ----------
    F_raw : np.ndarray (n_soluciones, 4)
        Frente en escala cruda: [compacidad, -tolerancia, -Hamming_avg, varianza].
    n_snps_total : int
        Número total de SNPs del bloque (1032 para Hinds 2005 y hapmap_phase2).
        Se utiliza para calcular la distancia de Hamming normalizada comparable
        con la Tabla 5 de Moqa et al. (2022).
    """
    # Extraer la compacidad (primer objetivo)
    compactness = F_raw[:, 0]
    # Extraer la tolerancia real (segundo objetivo, negado porque PyMoo minimiza)
    tolerance_real = -F_raw[:, 1]
    # Extraer el Hamming promedio real (tercer objetivo, negado)
    hamming_avg_real = -F_raw[:, 2]
    # Evitar división por cero en compacidad (nº de SNPs seleccionados)
    safe_comp = np.where(compactness <= 0, np.nan, compactness)
    # Tasa de tolerancia: tolerancia / compacidad (ratio adimensional)
    tr = tolerance_real / safe_comp
    # Máximo de la tasa de tolerancia
    max_tr = float(np.nanmax(tr))
    # Promedio de la tasa de tolerancia
    avg_tr = float(np.nanmean(tr))
    # Promedio de distancia de Hamming en escala cruda (conteo absoluto)
    avg_hamming = float(np.nanmean(hamming_avg_real))
    # Distancia de Hamming normalizada: comparable con Tabla 5 del paper.
    # El paper reporta valores en ~[0.07, 0.15] dividiendo el conteo entre n_snps_total.
    avg_hamming_norm = avg_hamming / max(1, n_snps_total)
    return max_tr, avg_tr, avg_hamming, avg_hamming_norm

def compute_hypervolume(F_norm):
    # Si no hay puntos, no se puede calcular HV
    if F_norm is None or len(F_norm) == 0:
        return np.nan

    hv = _get_hv_indicator(int(F_norm.shape[1]))
    return float(hv(F_norm))


@lru_cache(maxsize=8)
def _get_hv_indicator(n_obj: int):
    from pymoo.indicators.hv import HV
    ref_point = np.ones(int(n_obj), dtype=float)
    return HV(ref_point=ref_point)

def _calcular_referencias_por_algoritmo(run_results):
    """Calcula ideal/nadir por algoritmo para normalización de doble ámbito.

    Agrupa todas las ejecuciones por nombre de algoritmo y computa el punto
    ideal y nadir usando exclusivamente los frentes finales de ese algoritmo.
    Esto produce métricas Range/SumMin/MinSum no triviales (el paper emplea
    esta granularidad en sus Tablas 4 y 5).

    Returns
    -------
    dict : {nombre_algoritmo: (ideal_algo, safe_denom_algo)}
    """
    from collections import defaultdict
    grupos = defaultdict(list)
    for rr in run_results:
        grupos[rr.algorithm].append(rr)

    refs = {}
    for algo, rrs in grupos.items():
        ideal_a, nadir_a = global_ideal_nadir(rrs)
        refs[algo] = (ideal_a, nadir_a - ideal_a + 1e-9)
    return refs


def _calcular_referencias_por_modo(run_results, normalization_mode='global_all_pairs', pair_hamming=None):
    """Devuelve referencias para Range/SumMin/MinSum según modo configurado."""
    mode = _resolve_normalization_mode(normalization_mode)
    if mode == 'per_algorithm':
        return _calcular_referencias_por_algoritmo(run_results)

    if mode == 'static_dataset_limits':
        ideal_p, safe_p = static_dataset_limits_references(n_snps_total=1032, pair_hamming=pair_hamming)
        algos = sorted({str(rr.algorithm) for rr in run_results})
        return {algo: (ideal_p, safe_p) for algo in algos}

    ideal_g, nadir_g = global_ideal_nadir(run_results)
    safe_g = nadir_g - ideal_g + 1e-9
    algos = sorted({str(rr.algorithm) for rr in run_results})
    return {algo: (ideal_g, safe_g) for algo in algos}


def evaluate_final_metrics(run_results, show_progress=True, normalization_mode='global_all_pairs', pair_hamming=None):
    """Calcula las métricas finales con normalización configurable en código.

    - Range/SumMin/MinSum: usan referencias según normalization_mode.
    - Hypervolume: mantiene normalización global para comparabilidad cruzada.
    """
    if not run_results:
        return pd.DataFrame(), None, None

    normalization_mode = _resolve_normalization_mode(normalization_mode)

    # --- Normalización GLOBAL (para Hypervolume) ---
    ideal_global, nadir_global = global_ideal_nadir(run_results)
    safe_denom_global = (nadir_global - ideal_global + 1e-9)

    # --- Normalización para Range / SumMin / MinSum según modo ---
    algo_ref = _calcular_referencias_por_modo(run_results, normalization_mode=normalization_mode, pair_hamming=pair_hamming)

    # Progreso detallado por ejecución
    rows = []
    t_start = time.time()
    total_runs = len(run_results)
    w = len(str(total_runs))
    idx = 0
    for rr in run_results:
        F_raw = rr.F_final
        if F_raw is None or len(F_raw) == 0:
            continue

        # Normalización por algoritmo → Range, SumMin, MinSum
        ideal_a, safe_denom_a = algo_ref[rr.algorithm]
        F_norm_algo = np.clip((F_raw - ideal_a) / safe_denom_a, 0, 1)
        if np.any(np.isnan(F_norm_algo)) or np.any(np.isinf(F_norm_algo)):
            raise ValueError('Se detecta NaN/Inf en normalización por algoritmo.')
        rng, smn, msn = compute_range_summin_minsum(F_norm_algo)

        # Normalización global → Hypervolume
        F_norm_global = np.clip((F_raw - ideal_global) / safe_denom_global, 0, 1)
        hv_val = compute_hypervolume(F_norm_global)

        # Métricas crudas (ratios independientes de escala)
        max_tr, avg_tr, avg_h, avg_h_norm = compute_raw_aux_metrics(F_raw, n_snps_total=1032)

        rows.append({
            'algorithm': rr.algorithm,
            'init': rr.init,
            'run': rr.run,
            'seed': rr.seed,
            'elapsed_sec': rr.elapsed_sec,
            'n_solutions_final_front': len(F_raw),
            'Range': rng,
            'SumMin': smn,
            'MinSum': msn,
            'MaxToleranceRate': max_tr,
            'AvgToleranceRate': avg_tr,
            'AvgHammingDistance': avg_h,
            'AvgHammingDistance_norm': avg_h_norm,  # Normalizada: comparable con Tabla 5 del paper
            'Hypervolume': hv_val
        })
        idx += 1
        if show_progress:
            print(f"      • [Progreso: {idx:>{w}}/{total_runs}] | [{rr.algorithm}-{rr.init}] ejecución {rr.run}")
    # Retornar el DataFrame construido junto con los puntos de referencia globales
    return pd.DataFrame(rows), ideal_global, nadir_global

def _build_generation_rows_for_run(rr, ideal_algo, safe_denom_algo,
                                    ideal_global, safe_denom_global):
    """Construye filas de métricas por generación.

    Parámetros
    ----------
    rr : RunResult
        Resultado de una ejecución individual.
    ideal_algo, safe_denom_algo : np.ndarray
        Referencias de normalización para Range/SumMin/MinSum.
    ideal_global, safe_denom_global : np.ndarray
        Punto ideal y denominador seguro GLOBAL (para Hypervolume).
    """
    rows = []
    processed_generations = 0

    F_history = getattr(rr, 'F_history', None)
    if F_history is None or len(F_history) == 0:
        return rows, processed_generations

    for gen_idx, F_raw in enumerate(F_history):
        if F_raw is None or len(F_raw) == 0:
            continue

        F_raw = np.array(F_raw, dtype=float)
        if F_raw.ndim == 1:
            F_raw = F_raw.reshape(1, -1)

        # Normalización por algoritmo → Range, SumMin, MinSum
        F_norm_algo = np.clip((F_raw - ideal_algo) / safe_denom_algo, 0, 1)
        rng, smn, msn = compute_range_summin_minsum(F_norm_algo)

        # Normalización global → Hypervolume
        F_norm_global = np.clip((F_raw - ideal_global) / safe_denom_global, 0, 1)
        hv_val = compute_hypervolume(F_norm_global)

        # Métricas crudas (ratios independientes de escala)
        max_tr, avg_tr, avg_h, avg_h_norm = compute_raw_aux_metrics(F_raw, n_snps_total=1032)

        rows.append({
            'generation': gen_idx + 1,
            'algorithm': rr.algorithm,
            'init': rr.init,
            'run': rr.run,
            'seed': rr.seed,
            'n_solutions': len(F_raw),
            'Range': rng,
            'SumMin': smn,
            'MinSum': msn,
            'MaxToleranceRate': max_tr,
            'AvgToleranceRate': avg_tr,
            'AvgHammingDistance': avg_h,
            'AvgHammingDistance_norm': avg_h_norm,
            'Hypervolume': hv_val
        })
        processed_generations += 1

    return rows, processed_generations


def build_generation_metrics(run_results, ideal_global=None, nadir_global=None,
                             show_progress=True, show_summary=True,
                             normalization_mode='global_all_pairs'):
    """Calcula métricas por generación con normalización configurable.

    - Range/SumMin/MinSum: referencias según normalization_mode.
    - Hypervolume: referencias globales para comparabilidad cruzada.
    """
    if not run_results:
        return pd.DataFrame()

    normalization_mode = _resolve_normalization_mode(normalization_mode)

    # --- Normalización global (Hypervolume) ---
    if ideal_global is None or nadir_global is None:
        ideal_global, nadir_global = global_ideal_nadir(run_results)
    safe_denom_global = (nadir_global - ideal_global + 1e-9)

    # --- Normalización para Range / SumMin / MinSum según modo ---
    algo_ref = _calcular_referencias_por_modo(run_results, normalization_mode=normalization_mode)

    valid_runs = [rr for rr in run_results if getattr(rr, 'F_history', None) is not None and len(getattr(rr, 'F_history', [])) > 0]
    if not valid_runs:
        return pd.DataFrame()

    done_runs = 0
    done_gens = 0
    rows_by_key = {}

    # Progreso detallado por ejecución generacional
    t_start = time.time()
    total_runs = len(valid_runs)
    w = len(str(total_runs))
    idx = 0
    for rr in valid_runs:
        ideal_a, safe_denom_a = algo_ref[rr.algorithm]
        run_rows, gen_count = _build_generation_rows_for_run(
            rr, ideal_a, safe_denom_a, ideal_global, safe_denom_global
        )
        rows_by_key[(rr.algorithm, rr.init, rr.run)] = run_rows
        done_runs += 1
        done_gens += gen_count
        idx += 1
        elapsed_run = getattr(rr, 'elapsed_sec', 0.0)
        n_gens = len(getattr(rr, 'F_history', []) or [])
        if show_progress:
            print(f"      • [Progreso: {idx:>{w}}/{total_runs}] | [{rr.algorithm}-{rr.init}] ejecución {rr.run}")
    elapsed = time.time() - t_start
    rows = []
    for rr in valid_runs:
        key = (rr.algorithm, rr.init, rr.run)
        rows.extend(rows_by_key.get(key, []))
    # Mostrar resumen solo si se llama desde el proceso principal
    if show_summary and done_runs == len(valid_runs):
        print(f"      • Métricas generacionales completadas: {done_runs}/{total_runs} ejecuciones, {done_gens} generaciones en {elapsed:.1f}s")
    return pd.DataFrame(rows)

def plot_pareto_fronts(df_input, algorithm_name, init_name=None, folders=None, mode_tag=None, dpi=300):
    """Visualiza el frente de Pareto en 4 subgráficos 2D para (algoritmo, init).
    Si init_name es None, genera una figura independiente por cada init disponible del algoritmo.
    Estandariza ejes con límites globales calculados sobre todo df_input.
    """
    df_algo = df_input[df_input['algorithm'].str.upper() == str(algorithm_name).upper()].copy()
    if df_algo.empty:
        print(f"No hay datos para el algoritmo {algorithm_name}.")
        return

    df_algo['Compactness'] = df_algo['f1_compactness']
    df_algo['Tolerance'] = -df_algo['f2_neg_tolerance']
    df_algo['Hamming'] = -df_algo['f3_neg_hamming_avg']
    df_algo['Balance'] = df_algo['f4_balance_var']

    init_col = 'init_type' if 'init_type' in df_algo.columns else 'init'
    init_values = sorted(df_algo[init_col].dropna().astype(str).unique().tolist())

    if init_name is not None:
        init_values = [str(init_name)]

    default_palette = {
        'random': '#ff7f0e',
        'greedy_hybrid': '#1f77b4',
        'greedy_pure': '#2ca02c',
        'greedy_pure_15': '#2ca02c',
        'greedy_pure_30': '#1f9e89',
        'greedy_pure_50': '#31688e',
        'greedy': '#1f77b4',
    }

    axis_labels = {
        'Compactness': 'Compacidad (Nº Tag SNPs)',
        'Tolerance': 'Tolerancia',
        'Hamming': 'Distancia de Hamming Promedio',
        'Balance': 'Varianza / Balance',
    }

    df_global = df_input.copy()
    df_global['Compactness'] = df_global['f1_compactness']
    df_global['Tolerance'] = -df_global['f2_neg_tolerance']
    df_global['Hamming'] = -df_global['f3_neg_hamming_avg']
    df_global['Balance'] = df_global['f4_balance_var']

    pairs = [
        ('Compactness', 'Tolerance'),
        ('Tolerance', 'Hamming'),
        ('Hamming', 'Balance'),
        ('Compactness', 'Balance'),
    ]

    axis_limits = {}
    for x_col, y_col in pairs:
        x_min, x_max = df_global[x_col].min(), df_global[x_col].max()
        y_min, y_max = df_global[y_col].min(), df_global[y_col].max()

        x_pad = (x_max - x_min) * 0.05 if x_max > x_min else 1.0
        y_pad = (y_max - y_min) * 0.05 if y_max > y_min else 1.0

        axis_limits[(x_col, y_col)] = (
            (x_min - x_pad, x_max + x_pad),
            (y_min - y_pad, y_max + y_pad),
        )

    for init_val in init_values:
        df = df_algo[df_algo[init_col].astype(str) == str(init_val)]
        if df.empty:
            print(f"No hay datos para {algorithm_name} con init={init_val}.")
            continue

        sns.set_theme(style='whitegrid')
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle(f"Frente de Pareto: {algorithm_name} | {init_val}", fontsize=16, weight='bold')

        color = default_palette.get(str(init_val), '#1f77b4')
        subplots = [
            (axes[0, 0], 'Compactness', 'Tolerance', '(a) Compacidad vs. Tolerancia'),
            (axes[0, 1], 'Tolerance', 'Hamming', '(b) Tolerancia vs. Hamming'),
            (axes[1, 0], 'Hamming', 'Balance', '(c) Hamming vs. Balance/Varianza'),
            (axes[1, 1], 'Compactness', 'Balance', '(d) Compacidad vs. Balance/Varianza'),
        ]

        for ax, x_col, y_col, title in subplots:
            sns.scatterplot(
                data=df,
                x=x_col,
                y=y_col,
                ax=ax,
                s=60,
                alpha=0.8,
                color=color,
                edgecolor='w',
                linewidth=0.5,
                label=str(init_val),
            )

            # --- Línea de tendencia roja (rolling median) ---
            datos_ordenados = df[[x_col, y_col]].sort_values(by=x_col).dropna()
            if len(datos_ordenados) > 4:
                ventana = max(5, int(0.1 * len(datos_ordenados)))
                y_mediana = datos_ordenados[y_col].rolling(window=ventana, center=True, min_periods=1).median()
                ax.plot(
                    datos_ordenados[x_col],
                    y_mediana,
                    color='red',
                    linewidth=2.2,
                    alpha=0.95,
                    label='Tendencia Pareto'
                )

            x_lim, y_lim = axis_limits[(x_col, y_col)]
            ax.set_xlim(x_lim)
            ax.set_ylim(y_lim)
            ax.set_title(title, fontsize=13)
            ax.set_xlabel(axis_labels[x_col], fontsize=11)
            ax.set_ylabel(axis_labels[y_col], fontsize=11)
            ax.legend(title='Inicialización', loc='best', frameon=True)

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        if folders is not None and mode_tag is not None:
            os.makedirs(folders['frentes'], exist_ok=True)
            algo_tag = str(algorithm_name).lower()
            init_tag = str(init_val).lower()
            save_path = os.path.join(folders['frentes'], f'frentes_pareto_{algo_tag}_{init_tag}_{mode_tag}.png')
            save_report_figure(save_path, dpi=dpi, tight=True, fig=fig)
            print_saved_plot(save_path, f"Frente Pareto {algorithm_name} ({init_val})")
        plt.close(fig)


def _seleccionar_runs_representativos_por_hv(run_results, df_runs):
    """Selecciona 1 run por (algoritmo, init) usando mediana de Hypervolume.

    Criterio:
    1) Para cada grupo (algorithm, init), tomar HV mediano.
    2) Elegir el run con |HV - mediana(HV)| mínimo.
    3) Desempate determinista por run ascendente y luego seed ascendente.
    4) Si el grupo no tiene HV finito, usar run/seed mínimos.
    """
    if df_runs is None or df_runs.empty:
        return []

    required = {'algorithm', 'init', 'run', 'seed', 'Hypervolume'}
    if not required.issubset(df_runs.columns):
        return []

    selected_keys = []
    for (algo, init_name), g in df_runs.groupby(['algorithm', 'init'], dropna=False):
        g = g.copy()
        g['run'] = pd.to_numeric(g['run'], errors='coerce')
        g['seed'] = pd.to_numeric(g['seed'], errors='coerce')
        hv_num = pd.to_numeric(g['Hypervolume'], errors='coerce')
        g_valid = g[hv_num.notna()].copy()

        if not g_valid.empty:
            hv_vals = pd.to_numeric(g_valid['Hypervolume'], errors='coerce').to_numpy(dtype=float)
            hv_med = float(np.median(hv_vals))
            g_valid['hv_dist'] = np.abs(pd.to_numeric(g_valid['Hypervolume'], errors='coerce') - hv_med)
            chosen = g_valid.sort_values(['hv_dist', 'run', 'seed'], ascending=[True, True, True]).iloc[0]
        else:
            chosen = g.sort_values(['run', 'seed'], ascending=[True, True]).iloc[0]

        selected_keys.append((str(algo), str(init_name), int(chosen['run']), int(chosen['seed'])))

    rr_lookup = {(str(rr.algorithm), str(rr.init), int(rr.run), int(rr.seed)): rr for rr in run_results}
    selected_runs = [rr_lookup[k] for k in selected_keys if k in rr_lookup]
    return selected_runs

def aggregate_results(df_runs):
    # Verifica si el DataFrame de ejecuciones está vacío
    if df_runs.empty:
        return pd.DataFrame() # Retorna un DataFrame vacío si no hay datos
    # Lista de métricas a promediar y analizar
    metrics = [
        'Range', 'SumMin', 'MinSum',
        'MaxToleranceRate', 'AvgToleranceRate', 'AvgHammingDistance', 'Hypervolume',
        'elapsed_sec', 'n_solutions_final_front'
    ]
    # Agrupa por algoritmo e inicialización y calcula media y desviación estándar para cada métrica
    agg = df_runs.groupby(['algorithm', 'init'])[metrics].agg(['mean', 'std']).reset_index()
    # Aplana los nombres de las columnas resultantes uniéndolos con guiones bajos (ej. Metric_mean)
    agg.columns = ['_'.join(c).rstrip('_') for c in agg.columns.to_flat_index()]
    return agg # Retorna el DataFrame agregado

def plot_metricas_generacionales(df_gen_runs, out_dir=None, mode_tag=None, figsize=(18, 10), dpi=300, show=True):
    """
    Genera una figura por combinación (algoritmo, inicialización),
    con subplots de evolución generacional de métricas clave.

    Nota: el eje Y se estandariza de forma GLOBAL por métrica (todos los algoritmos + todas las inits),
    para que las figuras sean comparables entre sí.
    """
    required_cols = {'generation', 'algorithm', 'init'}
    missing = required_cols - set(df_gen_runs.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas en df_gen_runs: {sorted(missing)}")

    metricas_map = [
        ('Range', 'Rango (Range): Diversidad geométrica', 'Valor de Rango'),
        ('SumMin', 'SumMin: Convergencia marginal', 'Valor de SumMin'),
        ('MinSum', 'MinSum: Convergencia central', 'Valor de MinSum'),
        ('MaxToleranceRate', 'Tasa de Tolerancia Máxima', 'Tasa de Tolerancia Máxima'),
        ('AvgToleranceRate', 'Tasa de Tolerancia Promedio', 'Tasa de Tolerancia Promedio'),
        ('AvgHammingDistance', 'Distancia Hamming Promedio', 'Distancia Hamming Promedio'),
    ]

    metricas_disponibles = [m for m in metricas_map if m[0] in df_gen_runs.columns]
    if not metricas_disponibles:
        raise ValueError(
            "No se encontraron métricas válidas en df_gen_runs. "
            "Se esperaba al menos una de: " + ', '.join([m[0] for m in metricas_map])
        )

    metric_color_map = {
        'Range': '#1f77b4',
        'SumMin': '#ff7f0e',
        'MinSum': '#2ca02c',
        'MaxToleranceRate': '#d62728',
        'AvgToleranceRate': '#9467bd',
        'AvgHammingDistance': '#8c564b',
    }

    # --- Límites Y globales por métrica (GLOBAL por métrica) ---
    metric_ylim = {}
    for metric_col, _, _ in metricas_disponibles:
        series = df_gen_runs[metric_col].dropna()
        if series.empty:
            continue
        y_min = float(series.min())
        y_max = float(series.max())
        if y_max > y_min:
            pad = (y_max - y_min) * 0.05
        else:
            pad = abs(y_max) * 0.05 if y_max != 0 else 1.0
        metric_ylim[metric_col] = (y_min - pad, y_max + pad)

    sns.set_theme(style='whitegrid')
    n_metrics = len(metricas_disponibles)
    ncols = 3
    nrows = int(np.ceil(n_metrics / ncols))

    combos = (
        df_gen_runs[['algorithm', 'init']]
        .dropna()
        .drop_duplicates()
        .sort_values(['algorithm', 'init'])
        .to_records(index=False)
    )

    figures = []
    for algorithm_name, init_name in combos:
        sub = df_gen_runs[
            (df_gen_runs['algorithm'] == algorithm_name) &
            (df_gen_runs['init'] == init_name)
        ].copy()
        if sub.empty:
            continue

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True)
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

        for i, (metric_col, title, y_label) in enumerate(metricas_disponibles):
            ax = axes[i]
            sns.lineplot(
                data=sub,
                x='generation',
                y=metric_col,
                estimator='mean',
                errorbar=None,
                linewidth=2,
                color=metric_color_map.get(metric_col, '#1f77b4'),
                ax=ax,
            )
            if metric_col in metric_ylim:
                ax.set_ylim(metric_ylim[metric_col])
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.set_xlabel('Generación')
            ax.set_ylabel(y_label)

        for j in range(n_metrics, len(axes)):
            axes[j].axis('off')

        fig.suptitle(
            f"Evolución generacional | {algorithm_name} | {init_name}",
            fontsize=14,
            fontweight='bold'
        )
        fig.tight_layout(rect=(0, 0, 1, 0.95))

        if out_dir is not None and mode_tag is not None:
            os.makedirs(out_dir, exist_ok=True)
            algo_tag = str(algorithm_name).lower()
            init_tag = str(init_name).lower()
            save_path = os.path.join(out_dir, f'metricas_generacionales_{algo_tag}_{init_tag}_{mode_tag}.png')
            save_report_figure(save_path, dpi=dpi, tight=True, fig=fig)
            print_saved_plot(save_path, f"Métricas generacionales {algorithm_name} ({init_name})")

        plt.close(fig)
        figures.append((algorithm_name, init_name, fig, axes))

    return figures


def _plot_haplotype_heatmap(H, folders, mode_tag, dpi=200):
    sns.set_theme(style='whitegrid')
    plt.figure(figsize=(16, 5))
    sns.heatmap(H, cmap='gray', vmin=0, vmax=1, cbar_kws={'label': 'Alelo (0/1)'})
    plt.title('Matriz de haplotipos H (filas=haplotipos, columnas=SNPs)')
    plt.xlabel('SNP')
    plt.ylabel('Haplotipo')
    plt.tight_layout()
    save_path = os.path.join(folders['datos'], f'heatmap_haplotipos_{mode_tag}.png')
    plt.savefig(save_path, dpi=dpi)
    print_saved_plot(save_path, "Mapa de calor de haplotipos")
    plt.close()

def _detectar_bloques_ld(H: np.ndarray,
                         ventana_suavizado: int = 11,
                         umbral: float = 0.10,
                         min_snps_bloque: int = 10):
    """
    Detecta fronteras de bloques LD con máximo realismo biológico.

    Algoritmo:
      1. Calcula r² adyacente entre cada par de SNPs consecutivos (i, i+1)
         directamente sobre los vectores de haplotipos — resolución máxima.
      2. Suaviza la señal con una media móvil de `ventana_suavizado` posiciones
         para eliminar fluctuaciones puntuales de origen técnico (ruido).
      3. Declara una frontera de bloque en cada posición donde el r² suavizado
         cae por debajo de `umbral`, indicando un hotspot de recombinación real.
      4. Fusiona cualquier bloque más estrecho que `min_snps_bloque` SNPs
         con su vecino para evitar micro-bloques biológicamente implausibles.

    Devuelve una lista de tuplas (inicio_snp, fin_snp) para cada bloque.
    """
    n_snps  = H.shape[1]
    n_hap   = H.shape[0]
    H_f     = H.astype(float)

    # ── Paso 1: r² adyacente a nivel de SNP individual ────────────────────────
    media  = H_f.mean(axis=0)
    desvio = H_f.std(axis=0) + 1e-9
    H_norm = (H_f - media) / desvio          # (n_hap, n_snps)

    # Producto punto entre columnas adyacentes normalizado → correlación de Pearson
    r_adj  = np.einsum('hi,hi->i', H_norm[:, :-1], H_norm[:, 1:]) / n_hap
    r2_adj = r_adj ** 2                      # longitud: n_snps - 1

    # ── Paso 2: Suavizado con media móvil ─────────────────────────────────────
    nucleo      = np.ones(ventana_suavizado) / ventana_suavizado
    r2_suavizado = np.convolve(r2_adj, nucleo, mode='same')

    # ── Paso 3: Detectar posiciones de frontera ────────────────────────────────
    # r2_suavizado[i] representa la correlación entre SNP i y SNP i+1;
    # si cae bajo el umbral, el bloque nuevo comienza en el SNP i+1.
    pos_fronteras = np.where(r2_suavizado < umbral)[0] + 1

    # ── Paso 4: Aplicar anchura mínima (eliminar micro-bloques) ───────────────
    todos_cortes = np.concatenate([[0], pos_fronteras, [n_snps]])
    cortes_validos = [0]
    acumulado = 0
    for idx, (corte, anchura) in enumerate(zip(todos_cortes[1:], np.diff(todos_cortes))):
        acumulado += int(anchura)
        es_ultimo = (idx == len(todos_cortes) - 2)
        if acumulado >= min_snps_bloque or es_ultimo:
            cortes_validos.append(int(corte))
            acumulado = 0
    # Garantizar que el último corte llega hasta el final
    if cortes_validos[-1] != n_snps:
        cortes_validos[-1] = n_snps

    return [(cortes_validos[i], cortes_validos[i + 1])
            for i in range(len(cortes_validos) - 1)]


def _plot_ld_blocks(H, folders, mode_tag, dpi=200):
    """Visualiza los bloques LD reales detectados sobre la ventana de SNPs cargada."""
    VENTANA_SUAVIZADO = 11    # SNPs de media móvil para suprimir ruido genético
    UMBRAL_FRONTERA   = 0.10  # r² suavizado bajo el cual se declara hotspot
    MIN_SNPS_BLOQUE   = 10    # Anchura mínima para descartar micro-bloques

    # ── 1. Detectar bloques reales ─────────────────────────────────────────────
    segmentos = _detectar_bloques_ld(H, VENTANA_SUAVIZADO, UMBRAL_FRONTERA, MIN_SNPS_BLOQUE)
    n_bloques_reales = len(segmentos)

    # ── 2. Calcular medias por bloque y ordenar haplotipos ─────────────────────
    H_f = H.astype(float)
    n_hap = H_f.shape[0]
    H_media_bloque = np.zeros((n_hap, n_bloques_reales), dtype=float)
    for b, (s, e) in enumerate(segmentos):
        H_media_bloque[:, b] = H_f[:, s:e].mean(axis=1)

    pesos     = np.linspace(1.0, 2.0, n_bloques_reales)
    orden_hap = np.argsort(H_media_bloque @ pesos)
    H_ord     = H_media_bloque[orden_hap, :]   # (n_hap, n_bloques) — valores medios

    # ── 3. Construir malla pcolormesh con anchura proporcional al nº de SNPs ────
    # Los bordes del eje X son las posiciones reales de inicio/fin de cada bloque
    bordes_x = np.array([s for s, _ in segmentos] + [segmentos[-1][1]], dtype=float)
    bordes_y = np.arange(n_hap + 1, dtype=float)

    fig, ax = plt.subplots(figsize=(16, 6))
    malla = ax.pcolormesh(
        bordes_x, bordes_y, H_ord,
        cmap='viridis', vmin=0, vmax=1, shading='flat'
    )
    fig.colorbar(malla, ax=ax, label='Fracción de alelo 1 (media del bloque)')

    # ── 4. Líneas de frontera y etiquetas de bloque ────────────────────────────
    for b, (s, e) in enumerate(segmentos):
        if s > 0:
            ax.axvline(x=s, color='red', linewidth=1.2, linestyle='--', alpha=0.85)
        centro  = (s + e) / 2
        anchura = e - s
        ax.text(
            centro, n_hap + 1.5,
            f'B{b + 1}  {anchura} SNPs',
            ha='center', va='top', fontsize=8, color='black',
            bbox=dict(boxstyle='round,pad=0.2', fc='lightyellow', ec='grey', alpha=0.8)
        )

    texto_bloques = "bloque detectado" if n_bloques_reales == 1 else "bloques detectados"
    ax.set_title(
        f'Bloques LD reales por haplotipo | {n_bloques_reales} {texto_bloques}\n'
        f'(suavizado={VENTANA_SUAVIZADO} SNPs, umbral r²={UMBRAL_FRONTERA}, '
        f'mín. bloque={MIN_SNPS_BLOQUE} SNPs) — anchura proporcional al nº de SNPs'
    )
    ax.set_xlabel('Posición SNP (escala real, proporcional)')
    ax.set_ylabel('Haplotipo (ordenado para visualizar patrones)')
    ax.set_xlim(bordes_x[0], bordes_x[-1])
    ax.set_ylim(0, n_hap)
    ax.invert_yaxis()
    plt.tight_layout()

    save_path = os.path.join(folders['datos'], f'bloques_ld_haplotipos_{mode_tag}.png')
    plt.savefig(save_path, dpi=dpi)
    print_saved_plot(save_path, f"Estructura de bloques LD ({n_bloques_reales} bloques reales)")
    plt.close()

def _plot_haplotype_zoom(H, folders, mode_tag, dpi=200):
    N_HAP_VIEW = 10
    N_SNP_VIEW = 10
    h_view = min(N_HAP_VIEW, H.shape[0])
    s_view = min(N_SNP_VIEW, H.shape[1])
    H_view = H[:h_view, :s_view]
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        H_view, cmap='gray', vmin=0, vmax=1, annot=True, fmt='.0f',
        linewidths=0.5, cbar_kws={'label': 'Alelo (0/1)'}
    )
    plt.title(f'Zoom matriz H ({h_view} haplotipos × {s_view} SNPs)')
    plt.xlabel('SNP (subconjunto inicial)')
    plt.ylabel('Haplotipo (subconjunto inicial)')
    plt.tight_layout()
    save_path = os.path.join(folders['datos'], f'zoom_matriz_haplotipos_{h_view}x{s_view}_{mode_tag}.png')
    plt.savefig(save_path, dpi=dpi)
    print_saved_plot(save_path, f"Zoom matriz H ({h_view}x{s_view})")
    plt.close()

def _plot_allele_frequency(H, folders, mode_tag, dpi=200):
    allele_freq = H.mean(axis=0)
    plt.figure(figsize=(12, 5))
    sns.histplot(allele_freq, bins=30, kde=True, color='steelblue')
    plt.title('Distribución de frecuencia alélica por SNP')
    plt.xlabel('Frecuencia de alelo 1')
    plt.ylabel('Número de SNPs')
    plt.tight_layout()
    save_path = os.path.join(folders['datos'], f'histograma_frecuencia_alelica_{mode_tag}.png')
    plt.savefig(save_path, dpi=dpi)
    print_saved_plot(save_path, "Distribución de frecuencia alélica")
    plt.close()

def _plot_snp_variability(H, folders, mode_tag, dpi=200):
    snp_std = H.std(axis=0)
    mean_std = float(snp_std.mean())
    plt.figure(figsize=(14, 4))
    plt.plot(np.arange(len(snp_std)), snp_std, linewidth=1.2, color='darkorange', label='Desviación típica por SNP')
    plt.axhline(mean_std, color='crimson', linestyle='--', linewidth=1.5, label=f'Media = {mean_std:.4f}')
    plt.title('Variabilidad por SNP (desviación típica)')
    plt.xlabel('Índice SNP')
    plt.ylabel('Desviación típica')
    plt.legend(loc='upper right')
    plt.tight_layout()
    save_path = os.path.join(folders['datos'], f'desviacion_estandar_snps_{mode_tag}.png')
    plt.savefig(save_path, dpi=dpi)
    print_saved_plot(save_path, "Variabilidad por SNP")
    plt.close()

def _plot_dominant_alleles(H, folders, mode_tag, dpi=200):
    hap_ones = H.sum(axis=1)
    plt.figure(figsize=(10, 4))
    sns.barplot(x=np.arange(len(hap_ones)), y=hap_ones, color='teal')
    plt.title('Número de alelos 1 por haplotipo')
    plt.xlabel('Haplotipo')
    plt.ylabel('Conteo de alelos 1')
    plt.tight_layout()
    save_path = os.path.join(folders['datos'], f'conteo_alelos_por_haplotipo_{mode_tag}.png')
    plt.savefig(save_path, dpi=dpi)
    print_saved_plot(save_path, "Alelos dominantes por haplotipo")
    plt.close()

def _plot_hamming_distribution(H, folders, mode_tag, dpi=200):
    pair_dists = []
    for i in range(H.shape[0]):
        for j in range(i + 1, H.shape[0]):
            pair_dists.append(np.abs(H[i] - H[j]).sum())
    pair_dists = np.array(pair_dists, dtype=float)
    plt.figure(figsize=(10, 5))
    sns.histplot(pair_dists, bins=25, kde=True, color='purple')
    plt.title('Distribución de distancia de Hamming entre haplotipos')
    plt.xlabel('Distancia de Hamming')
    plt.ylabel('Número de pares')
    plt.tight_layout()
    save_path = os.path.join(folders['datos'], f'histograma_distancia_hamming_{mode_tag}.png')
    plt.savefig(save_path, dpi=dpi) 
    print_saved_plot(save_path, "Distribución distancias de Hamming")
    plt.close()

def run_exploratory_data_analysis(H, snp_ids, snp_positions, cfg):
    """Genera las visualizaciones exploratorias (Heatmaps, Histogramas) mediante subrutinas."""
    EXECUTION_MODE = cfg.execution_mode
    FOLDERS = cfg.folders
    mode_tag_local = EXECUTION_MODE
    REPORT_DPI = cfg.report_plot_dpi

    print_subsection("Visualización de la Estructura de Haplotipos", icon="🧬")
    _plot_haplotype_heatmap(H, FOLDERS, mode_tag_local, dpi=REPORT_DPI)
    _plot_ld_blocks(H, FOLDERS, mode_tag_local, dpi=REPORT_DPI)
    _plot_haplotype_zoom(H, FOLDERS, mode_tag_local, dpi=REPORT_DPI)
    
    print_subsection("Análisis de Variabilidad y Frecuencia Alélica", icon="📈")
    _plot_allele_frequency(H, FOLDERS, mode_tag_local, dpi=REPORT_DPI)
    _plot_snp_variability(H, FOLDERS, mode_tag_local, dpi=REPORT_DPI)
    _plot_dominant_alleles(H, FOLDERS, mode_tag_local, dpi=REPORT_DPI)
    _plot_hamming_distribution(H, FOLDERS, mode_tag_local, dpi=REPORT_DPI)

    

def _plot_ld_correlation_heatmap(corr_full, folders, mode_tag, dpi=200):
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_full, cmap='vlag', center=0, cbar_kws={'label': 'Correlación'})
    plt.title('Mapa de correlación completo SNP×SNP')
    plt.xlabel('SNP')
    plt.ylabel('SNP')
    plt.tight_layout()
    save_path = os.path.join(folders['datos'], f'heatmap_correlacion_completa_{mode_tag}.png')
    plt.savefig(save_path, dpi=dpi)
    print_saved_plot(save_path, "Mapa de calor de correlación (LD)")
    plt.close()

def _plot_ld_correlation_hist(ld_corrs, folders, mode_tag, dpi=200):
    plt.figure(figsize=(10, 5))
    sns.histplot(ld_corrs, bins=40, kde=True, color='slateblue')
    plt.title('Correlaciones entre todos los pares de SNPs')
    plt.xlabel('Correlación de Pearson')
    plt.ylabel('Frecuencia')
    plt.tight_layout()
    save_path = os.path.join(folders['datos'], f'histograma_correlaciones_ld_{mode_tag}.png')
    plt.savefig(save_path, dpi=dpi)
    print_saved_plot(save_path, "Distribución de correlaciones (LD)")
    plt.close()

def _plot_ld_cdf(abs_sorted, cdf, folders, mode_tag, dpi=200):
    plt.figure(figsize=(10, 5))
    if len(abs_sorted) > 0:
        plt.plot(abs_sorted, cdf, color='darkgreen', linewidth=2)
    else:
        plt.text(0.5, 0.5, 'Sin pares disponibles', ha='center', va='center')
    plt.title('CDF empírica de |correlación| entre SNPs (todos los pares)')
    plt.xlabel('|Correlación|')
    plt.ylabel('Probabilidad acumulada')
    plt.tight_layout()
    save_path = os.path.join(folders['datos'], f'cdf_correlacion_absoluta_ld_{mode_tag}.png')
    plt.savefig(save_path, dpi=dpi)
    print_saved_plot(save_path, "CDF de correlación LD")
    plt.close()

def perform_ld_diagnostic(H, cfg):
    """
    Analiza la matriz H y caracteriza la estructura de Desequilibrio de Ligamiento (LD).
    Proporciona una descripción estadística sin emitir juicios de optimalidad.
    """
    ld_mean, ld_corrs, corr_full = full_ld_check(H)
    abs_corr = np.abs(ld_corrs)
    
    # Detección de bloques para la caracterización descriptiva
    VENTANA_SUAVIZADO = 11
    UMBRAL_FRONTERA   = 0.10
    MIN_SNPS_BLOQUE   = 10
    segmentos = _detectar_bloques_ld(H, VENTANA_SUAVIZADO, UMBRAL_FRONTERA, MIN_SNPS_BLOQUE)
    n_bloques_reales = len(segmentos)

    if len(abs_corr) > 0:
        abs_sorted = np.sort(abs_corr)
        cdf = np.arange(1, len(abs_sorted) + 1) / len(abs_sorted)
    else:
        abs_sorted = np.array([])
        cdf = np.array([])

    print_subsection("Caracterización Global de Correlación (LD)", icon="🔗")
    print(f'      • Correlación media absoluta (global |r|): {ld_mean:.4f}')
    print(f'      • Total de pares evaluados: {len(ld_corrs)}')

    REPORT_DPI = cfg.report_plot_dpi
    _plot_ld_correlation_heatmap(corr_full, cfg.folders, cfg.execution_mode, dpi=REPORT_DPI)
    _plot_ld_correlation_hist(ld_corrs, cfg.folders, cfg.execution_mode, dpi=REPORT_DPI)
    _plot_ld_cdf(abs_sorted, cdf, cfg.folders, cfg.execution_mode, dpi=REPORT_DPI)

    THRESH_OPT_ABS_CORR = cfg.thresh_opt_abs_corr
    pct_abs_corr_ge_thresh = float((abs_corr >= THRESH_OPT_ABS_CORR).mean() * 100.0) if len(abs_corr) > 0 else 0.0

    print_subsection("Resumen Estructural del Dataset", icon="⚖️")
    print(f'      • Estructura detectada: {n_bloques_reales} bloques de ligamiento')
    print(f'      • Correlación media absoluta (|r|): {ld_mean:.4f}')
    print(f'      • Proporción de pares con |r| >= {THRESH_OPT_ABS_CORR:.2f}: {pct_abs_corr_ge_thresh:.2f}%')
    print(f'      • Naturaleza del dato: {"Benchmark biológico (Hinds)" if cfg.data_source == "hinds2005" else "Simulación sintética controlada"}')

def setup_evolutionary_environment(H, cfg):
    """Calcula los pares de combinaciones de haplotipos."""
    N_HAPLOTYPES = cfg.n_haplotypes
    PAIR_IDX = np.array(list(combinations(range(N_HAPLOTYPES), 2)), dtype=np.int32)
    N_PAIRS = len(PAIR_IDX)

    
    pair_hamming = np.abs(H[PAIR_IDX[:, 0]] - H[PAIR_IDX[:, 1]]).sum(axis=1).astype(int)
    p33, p66 = np.percentile(pair_hamming, [33.33, 66.67])
    n_examples = min(3, N_PAIRS)
    sorted_idx = np.argsort(pair_hamming)
    most_similar_idx = sorted_idx[:n_examples]
    most_different_idx = sorted_idx[-n_examples:][::-1]
    preview_snps = min(32, H.shape[1])
    
    print_subsection("Análisis de Similitud Genotípica (Pares de Haplotipos)", icon="📐")
    print(f'      • Número de pares de haplotipos: {N_PAIRS}')
    print(f'      • Pares mostrados: {n_examples} similares / {n_examples} distintos')
    print(f'      • Vista parcial: primeros {preview_snps} SNPs')
    print(f'      • Percentiles (Hamming): P33={p33:.2f}, P66={p66:.2f}')
    print('      • [Etiquetas: <=P33 -> muy similar | (P33,P66] -> intermedio | >P66 -> muy distinto]')

    def _print_pair_diagnostic(indices, header, icon):
        print(f'\n    {icon} \033[1m{header}\033[0m')
        for idx in indices:
            i, j = PAIR_IDX[idx]
            d = pair_hamming[idx]
            etiqueta = etiqueta_distancia_por_percentil(d, p33, p66)
            s_i = _as_bits(H[i, :preview_snps])
            s_j = _as_bits(H[j, :preview_snps])
            
            print(f'      • \033[1mPar ({i}, {j})\033[0m | Hamming=\033[1m{d}\033[0m | {etiqueta}')
            print(f'        h{i:03d}: \033[90m{s_i}\033[0m...')
            print(f'        h{j:03d}: \033[90m{s_j}\033[0m...')

    _print_pair_diagnostic(most_similar_idx, "Pares de mayor similitud genética", "🤝")
    _print_pair_diagnostic(most_different_idx, "Pares de mayor divergencia genética", "↔️")
    
    return PAIR_IDX, pair_hamming, p33, p66


def execute_algorithms(H, PAIR_IDX, pair_hamming, p33, p66, cfg):
    """Ejecuta todos los algoritmos multiobjetivo configurados."""
    EXECUTION_MODE = cfg.execution_mode
    NUM_BLOCKS = cfg.num_blocks
    DATA_SOURCE = cfg.data_source
    N_SNPS = cfg.n_snps
    N_HAPLOTYPES = cfg.n_haplotypes
    FOLDERS = cfg.folders
    POP_SIZE = cfg.pop_size
    N_GEN = cfg.n_gen
    OFFSPRING = cfg.offspring
    PC = cfg.pc
    PM = cfg.pm
    N_RUNS = cfg.n_runs
    MASTER_SEED = cfg.master_seed
    MOEAD_NEIGHBORS = cfg.moead_neighbors

    problem = TagSNPProblem(H, PAIR_IDX)
    algo_configs, ref_dirs_used, ref_partitions = build_algorithms(problem, H, cfg=cfg, seed=MASTER_SEED)
    num_algos = 4  # NSGA2, NSGA3, SPEA2, MOEAD
    num_inits = len(cfg.init_options)
    n_runs = cfg.n_runs
    total_execs = len(algo_configs) * n_runs

    print_subsection("Configuración del Motor Evolutivo", icon="⚙️")
    print(f"      • Modo={EXECUTION_MODE} | POP_SIZE={POP_SIZE} | N_GEN={N_GEN} | OFFSPRING={OFFSPRING} | PC={PC} | PM={1.0/N_SNPS:.6f} | N_RUNS={N_RUNS}")
    print(f'      • Desglose: {num_algos} algoritmos x {num_inits} inicializaciones x {n_runs} runs = {total_execs} ejecuciones')
    print(f'      • Configuraciones únicas (algoritmo-init): {len(algo_configs)}')
    print(f'      • Puntos de referencia (ref_dirs): {len(ref_dirs_used)} | Particiones: {ref_partitions}')
    print(f'      • Tamaño de población (pop_size): {POP_SIZE}')
    RUN_EXPERIMENTS = True
    if RUN_EXPERIMENTS:
        run_results = run_all_experiments(problem, H, cfg=cfg, n_runs=cfg.n_runs, master_seed=cfg.master_seed)
    else:
        run_results = []
        print('RUN_EXPERIMENTS=False: se omite la ejecución por ahora.')
    
    return run_results


def _plot_execution_time_boxplot(df_exec, order, folders, mode_tag, dpi=300):
    plt.figure(figsize=(max(10, 0.6 * len(order) + 4), 6))
    sns.boxplot(data=df_exec, x='config', y='elapsed_sec', order=order)
    sns.stripplot(data=df_exec, x='config', y='elapsed_sec', order=order, color='black', alpha=0.45, size=3, jitter=0.25)
    plt.title('Tiempo por run (s) - Comparativa por configuración')
    plt.xlabel('Configuración (algoritmo-init)')
    plt.ylabel('Tiempo (s)')
    plt.xticks(rotation=35, ha='right')
    plt.tight_layout()
    save_path = os.path.join(folders['boxplots'], f'boxplot_tiempo_ejecucion_{mode_tag}.png')
    save_report_figure(save_path, dpi=dpi)
    print_saved_plot(save_path, "Boxplot tiempo de ejecución")
    plt.close()

def _plot_execution_time_mean_std(df_exec, order, folders, mode_tag, dpi=300):
    df_time = (df_exec.groupby('config', as_index=False)['elapsed_sec'].agg(mean='mean', std='std'))
    df_time['std'] = df_time['std'].fillna(0.0)
    df_time = df_time.set_index('config').reindex(order).reset_index()

    plt.figure(figsize=(max(10, 0.6 * len(order) + 4), 6))
    ax = sns.barplot(data=df_time, x='config', y='mean', order=order, errorbar=None)
    ax.errorbar(x=np.arange(len(df_time)), y=df_time['mean'].to_numpy(), yerr=df_time['std'].to_numpy(), fmt='none', c='black', capsize=4, linewidth=1.2)
    plt.title('Tiempo medio (± std) - Comparativa por configuración')
    plt.xlabel('Configuración (algoritmo-init)')
    plt.ylabel('Tiempo (s)')
    plt.xticks(rotation=35, ha='right')
    plt.tight_layout()
    save_path = os.path.join(folders['boxplots'], f'media_std_tiempo_ejecucion_{mode_tag}.png')
    save_report_figure(save_path, dpi=dpi)
    print_saved_plot(save_path, "Media ± std tiempo ejecución")
    plt.close()

def _plot_front_size_boxplot(df_exec, order, folders, mode_tag, dpi=300):
    plt.figure(figsize=(max(10, 0.6 * len(order) + 4), 6))
    sns.boxplot(data=df_exec, x='config', y='n_solutions_final_front', order=order)
    sns.stripplot(data=df_exec, x='config', y='n_solutions_final_front', order=order, color='black', alpha=0.45, size=3, jitter=0.25)
    plt.title('Tamaño frente final por run - Comparativa por configuración')
    plt.xlabel('Configuración (algoritmo-init)')
    plt.ylabel('Número de soluciones en frente final')
    plt.xticks(rotation=35, ha='right')
    plt.tight_layout()
    save_path = os.path.join(folders['boxplots'], f'boxplot_tamano_frente_{mode_tag}.png')
    save_report_figure(save_path, dpi=dpi)
    print_saved_plot(save_path, "Boxplot tamaño de frente Pareto")
    plt.close()

def _plot_time_vs_frontsize_scatter(df_exec, folders, mode_tag, dpi=300):
    plt.figure(figsize=(9, 7))
    sns.scatterplot(data=df_exec, x='elapsed_sec', y='n_solutions_final_front', hue='algorithm', style='init', s=90, alpha=0.85, edgecolor='w', linewidth=0.4)
    plt.title('Tiempo vs tamaño de frente final - Comparativa')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Número de soluciones en frente final')
    plt.tight_layout()
    save_path = os.path.join(folders['boxplots'], f'tiempo_vs_tamano_frente_{mode_tag}.png')
    save_report_figure(save_path, dpi=dpi)
    print_saved_plot(save_path, "Tiempo vs Tamaño de frente")
    plt.close()


def _plot_all_pareto_fronts(df_front_obj, folders, mode_tag, dpi=300):
    if not df_front_obj.empty:
        plot_pareto_fronts(df_front_obj, 'NSGA3', folders=folders, mode_tag=mode_tag, dpi=dpi)
        plot_pareto_fronts(df_front_obj, 'MOEAD', folders=folders, mode_tag=mode_tag, dpi=dpi)
        plot_pareto_fronts(df_front_obj, 'NSGA2', folders=folders, mode_tag=mode_tag, dpi=dpi)
        plot_pareto_fronts(df_front_obj, 'SPEA2', folders=folders, mode_tag=mode_tag, dpi=dpi)
    else:
        print('DataFrame de frentes no disponible o vacío.')

def _plot_objective_correlation(df_front_obj, folders, mode_tag, dpi=300):
    if df_front_obj.empty:
        print('No hay datos suficientes para graficar.')
        return
    # Calcular correlación entre objetivos. Se suprimen advertencias en caso de objetivos constantes.
    with np.errstate(divide='ignore', invalid='ignore'):
        corr_obj = df_front_obj[['f1_compactness', 'f2_neg_tolerance', 'f3_neg_hamming_avg', 'f4_balance_var']].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_obj, annot=True, fmt='.2f', cmap='vlag', center=0)
    plt.title('Correlación entre objetivos en frentes finales')
    plt.tight_layout()
    save_path = os.path.join(folders['heatmaps'], f'correlacion_objetivos_pareto_{mode_tag}.png')
    save_report_figure(save_path, dpi=dpi)
    print_saved_plot(save_path, "Correlación objetivos Pareto")
    plt.close()

def _plot_parallel_coordinates(df_front_obj, seed, folders, mode_tag, dpi=300):
    # --- Implementación en español: graficar cada tipo de inicialización por separado para cada algoritmo ---
    if df_front_obj.empty:
        return
    import warnings
    warnings.filterwarnings('ignore')
    df_par = df_front_obj.copy()
    init_col = 'init_type' if 'init_type' in df_par.columns else 'init'
    algo_col = 'algorithm'

    df_par['Compacidad ($f_1$ min)'] = df_par['f1_compactness']
    df_par['Tolerancia Real ($f_2$ max)'] = -df_par['f2_neg_tolerance']
    df_par['Hamming Real ($f_3$ max)'] = -df_par['f3_neg_hamming_avg']
    df_par['Varianza ($f_4$ min)'] = df_par['f4_balance_var']

    display_cols = [
        'Compacidad ($f_1$ min)',
        'Tolerancia Real ($f_2$ max)',
        'Hamming Real ($f_3$ max)',
        'Varianza ($f_4$ min)',
    ]

    # Normalización global para todos los frentes (para mantener los ejes comparables)
    df_norm = df_par[[algo_col, init_col] + display_cols].copy()
    for c in display_cols:
        cmin, cmax = df_norm[c].min(), df_norm[c].max()
        if cmax > cmin:
            df_norm[c] = (df_norm[c] - cmin) / (cmax - cmin)
        else:
            df_norm[c] = 0.0

    # Mapas de color y estilo (opcional, pero mantenemos para consistencia visual)
    style_map = {
        'random': '--',
        'greedy_hybrid': '-',
        'greedy_pure': '-.',
        'greedy_pure_15': '-.',
        'greedy_pure_30': '-.',
        'greedy_pure_50': '-.',
        'greedy': '-'
    }
    color_map = {
        'random': '#ff7f0e',
        'greedy_hybrid': '#1f77b4',
        'greedy_pure': '#2ca02c',
        'greedy_pure_15': '#2ca02c',
        'greedy_pure_30': '#1f9e89',
        'greedy_pure_50': '#31688e',
        'greedy': '#1f77b4'
    }

    algorithms = sorted(df_norm[algo_col].dropna().astype(str).unique().tolist())
    init_values = sorted(df_norm[init_col].dropna().astype(str).unique().tolist())

    # Para cada algoritmo y cada tipo de inicialización, graficar por separado
    for algorithm_name in algorithms:
        for init_name in init_values:
            df_sub = df_norm[(df_norm[algo_col].astype(str) == str(algorithm_name)) & (df_norm[init_col].astype(str) == str(init_name))].copy()
            if df_sub.empty:
                continue

            sample_n = min(60, len(df_sub))
            df_plot = df_sub.sample(sample_n, random_state=seed) if len(df_sub) > sample_n else df_sub

            fig, ax = plt.subplots(figsize=(8, 5))
            c = color_map.get(str(init_name), '#1f77b4')
            s = style_map.get(str(init_name), '-')
            for i in range(len(df_plot)):
                row = df_plot.iloc[i]
                ax.plot(display_cols, row[display_cols].values, alpha=0.18, color=c, linestyle=s)

            ax.set_title(f'{algorithm_name} - {init_name}', fontsize=13, fontweight='bold')
            ax.set_ylim(-0.05, 1.05)
            ax.set_ylabel('Valor normalizado (0-1)')
            ax.grid(True, axis='y', linestyle='--', alpha=0.35)
            ax.tick_params(axis='x', rotation=25)

            # Leyenda opcional (solo un tipo de inicialización por plot)
            from matplotlib.lines import Line2D
            handles = [Line2D([0], [0], color=c, linestyle=s, linewidth=2, label=str(init_name))]
            ax.legend(handles=handles, title='Inicialización', loc='upper right', frameon=True)

            fig.suptitle('Coordenadas Paralelas (Objetivos reales normalizados)', fontsize=15, fontweight='bold')
            fig.tight_layout(rect=(0, 0, 1, 0.95))
            os.makedirs(folders['frentes'], exist_ok=True)
            nombre_archivo = f'coordenadas_paralelas_{algorithm_name}_{init_name}_{mode_tag}.png'
            save_report_figure(
                os.path.join(folders['frentes'], nombre_archivo),
                dpi=dpi,
                tight=True,
                fig=fig,
            )
            plt.close(fig)

def generate_final_synthesis(run_results, PAIR_IDX, pair_hamming, cfg):
    """Agrupa los resultados crudos, calcula métricas y exporta a CSV."""
    EXECUTION_MODE = cfg.execution_mode
    FOLDERS = cfg.folders
    REPORT_DPI = cfg.report_plot_dpi
    NORMALIZATION_MODE = _resolve_normalization_mode(getattr(cfg, 'normalization_mode', 'global_all_pairs'))
    
    print(f"  ⚙  Configuración de síntesis: DPI={REPORT_DPI}")

    df_runs = pd.DataFrame()
    df_gen_runs = pd.DataFrame()
    df_front_obj = pd.DataFrame()
    ideal_global, nadir_global = None, None

    if len(run_results) == 0:
        print("No hay resultados para procesar.")
        return df_runs, df_gen_runs, df_front_obj, ideal_global, nadir_global

    mode_tag_local = EXECUTION_MODE
    df_exec = pd.DataFrame([
        {
            'algorithm': rr.algorithm,
            'init': rr.init,
            'run': rr.run,
            'elapsed_sec': rr.elapsed_sec,
            'n_solutions_final_front': len(rr.F_final) if rr.F_final is not None else 0,
        }
        for rr in run_results
    ])
    df_exec['config'] = df_exec['algorithm'].astype(str) + '-' + df_exec['init'].astype(str)
    order = df_exec[['algorithm', 'init', 'config']].drop_duplicates().sort_values(['algorithm', 'init'])['config'].tolist()
    
    sns.set_theme(style='whitegrid')

    print_subsection("Análisis de Rendimiento y Tiempo", icon="⏱️")
    t_block = time.time()
    _plot_execution_time_boxplot(df_exec, order, FOLDERS, mode_tag_local, dpi=REPORT_DPI)
    _plot_execution_time_mean_std(df_exec, order, FOLDERS, mode_tag_local, dpi=REPORT_DPI)
    _plot_front_size_boxplot(df_exec, order, FOLDERS, mode_tag_local, dpi=REPORT_DPI)
    _plot_time_vs_frontsize_scatter(df_exec, FOLDERS, mode_tag_local, dpi=REPORT_DPI)
    print(f"      • Tiempo bloque 'Análisis de Rendimiento y Tiempo': {time.time() - t_block:.1f}s")

    print_subsection("Procesamiento de Métricas (con progreso)", icon="🧮")
    t_block_met = time.time()
    print(f"    • Iniciando métricas finales: {len(run_results)} ejecuciones")
    df_runs, ideal_global, nadir_global = evaluate_final_metrics(
        run_results, show_progress=True, normalization_mode=NORMALIZATION_MODE, pair_hamming=pair_hamming
    )

    import concurrent.futures
    max_workers = max(1, (os.cpu_count() or 2) - 2)
    algo_ref = _calcular_referencias_por_modo(run_results, normalization_mode=NORMALIZATION_MODE, pair_hamming=pair_hamming)
    safe_denom_global = (nadir_global - ideal_global + 1e-9)

    print()
    print(f"    • Iniciando métricas generacionales: {len(run_results)} ejecuciones con historial")
    df_gen_runs_list = []
    total_gen_runs = len(run_results)
    w_gen_runs = len(str(total_gen_runs))
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_rr = {
            executor.submit(
                _eval_gen_metric_single,
                (rr, algo_ref[rr.algorithm][0], algo_ref[rr.algorithm][1], ideal_global, safe_denom_global)
            ): rr for rr in run_results
        }
        done_gen_runs = 0
        for future in concurrent.futures.as_completed(future_to_rr):
            run_df = future.result()
            df_gen_runs_list.append(run_df)
            done_gen_runs += 1
            rr_local = future_to_rr[future]
            print(f"      • [Progreso: {done_gen_runs:>{w_gen_runs}}/{total_gen_runs}] | [{rr_local.algorithm}-{rr_local.init}] ejecución {rr_local.run}")
    
    df_gen_runs = pd.concat(df_gen_runs_list, ignore_index=True)
    print(f"      • Métricas generacionales completadas: {len(run_results)} ejecuciones en {time.time() - t_block_met:.1f}s")

    print_subsection("Trazabilidad y Exportación de Datos (CSV)", icon="📊")
    csv_dir = FOLDERS.get('csv', os.path.join(FOLDERS['ejecuciones'], 'csv'))
    os.makedirs(csv_dir, exist_ok=True)
    runs_csv_path = os.path.join(csv_dir, f"resultados_detallados_{EXECUTION_MODE}.csv")
    df_runs.to_csv(runs_csv_path, index=False)
    print_saved_plot(runs_csv_path, "Resultados detallados por ejecución")

    if not df_gen_runs.empty:
        gen_csv_path = os.path.join(csv_dir, f"historico_generacional_{EXECUTION_MODE}.csv")
        df_gen_runs.to_csv(gen_csv_path, index=False)
        print_saved_plot(gen_csv_path, "Historial evolutivo generacional")
        print(f"      • Registros en historial: {df_gen_runs.shape[0]}")

    print_subsection("Puntos Críticos del Espacio de Objetivos", icon="📍")
    print(f'      • Punto Ideal Global (mejor): {ideal_global}')
    print(f'      • Punto Nadir Global (peor):  {nadir_global}')

    runs_para_frentes = _seleccionar_runs_representativos_por_hv(run_results, df_runs)
    if not runs_para_frentes:
        runs_para_frentes = list(run_results)

    rows_obj = []
    for rr in runs_para_frentes:
        if rr.F_final is not None and len(rr.F_final) > 0:
            for row in rr.F_final:
                rows_obj.append({
                    'algorithm': rr.algorithm, 'init': rr.init,
                    'f1_compactness': float(row[0]), 'f2_neg_tolerance': float(row[1]),
                    'f3_neg_hamming_avg': float(row[2]), 'f4_balance_var': float(row[3])
                })
    df_front_obj = pd.DataFrame(rows_obj)

    return df_runs, df_gen_runs, df_front_obj, ideal_global, nadir_global

def generate_visual_reports(df_runs, df_gen_runs, df_front_obj, ideal_global, nadir_global, cfg):
    """Genera todos los reportes visuales y comparativas gráficas."""
    EXECUTION_MODE = cfg.execution_mode
    FOLDERS = cfg.folders
    REPORT_DPI = cfg.report_plot_dpi
    
    print(f"  ⚙  Configuración de visualización: DPI={REPORT_DPI}")
    max_workers_visual = max(1, (os.cpu_count() or 2) - 2)
    print(f"  ⚙  Paralelización: Ejecución concurrente activa ({max_workers_visual} hilos activos)")

    if df_front_obj is not None and not df_front_obj.empty:
        print_subsection("Distribución y Correlación de Frentes de Pareto", icon="🔍")
        t_block = time.time()
        import concurrent.futures
        max_workers = max(1, (os.cpu_count() or 2) - 2)
        plot_tasks = ['all_fronts', 'correlation', 'parallel']
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(tarea_plot, t, df_front_obj, FOLDERS, EXECUTION_MODE, REPORT_DPI, cfg.master_seed) for t in plot_tasks]
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"      • ⚠️ Error en graficado paralelo: {e}")
        print(f"      • Tiempo bloque 'Frentes de Pareto': {time.time() - t_block:.1f}s")

    if df_gen_runs is not None and not df_gen_runs.empty:
        print_subsection("Análisis de Convergencia Progresiva", icon="🔄")
        t_block = time.time()
        _ = plot_metricas_generacionales(df_gen_runs, out_dir=FOLDERS['metricas'], mode_tag=EXECUTION_MODE, dpi=REPORT_DPI)
        print(f"      • Tiempo bloque 'Convergencia Progresiva': {time.time() - t_block:.1f}s")

    df_summary = aggregate_results(df_runs) if (df_runs is not None and not df_runs.empty) else pd.DataFrame()
    if not df_summary.empty:
        mean_cols = [c for c in df_summary.columns if c.endswith('_mean')]
        heat_df = df_summary[['algorithm', 'init'] + mean_cols].copy()
        heat_df['method'] = heat_df['algorithm'] + '-' + heat_df['init']
        
        print_subsection("Resumen Estadístico de Métricas", icon="📊")
        exclude = ['elapsed_sec_mean', 'n_solutions_final_front_mean', 'n_solutions_mean']
        metrics_to_plot = [m for m in mean_cols if m not in exclude]
        if metrics_to_plot:
            heat_df_plot = heat_df.set_index('method')[metrics_to_plot].copy()
            clean_cols = [c.replace('_mean', '') for c in metrics_to_plot]
            heat_df_plot.columns = clean_cols
            heat_norm_better = heat_df_plot.copy()
            higher_is_better = ['Hypervolume', 'Range', 'MaxToleranceRate', 'AvgToleranceRate']
            lower_is_better = ['SumMin', 'MinSum', 'AvgHammingDistance']
            for col in clean_cols:
                c_min, c_max = heat_df_plot[col].min(), heat_df_plot[col].max()
                c_range = c_max - c_min
                if c_range == 0:
                    heat_norm_better[col] = 1.0
                elif any(k in col for k in higher_is_better):
                    heat_norm_better[col] = (heat_df_plot[col] - c_min) / c_range
                else:
                    heat_norm_better[col] = (c_max - heat_df_plot[col]) / c_range
            
            plt.figure(figsize=(14, 8))
            sns.heatmap(heat_norm_better, annot=heat_df_plot, fmt='.3f', cmap='RdYlGn', linewidths=0.5)
            plt.title('Comparativa de Algoritmos: Rojo (Peor) vs Verde (Mejor)', fontsize=15, pad=20)
            plt.tight_layout()
            save_path = os.path.join(FOLDERS['heatmaps'], f"heatmap_comparativa_{EXECUTION_MODE}.png")
            save_report_figure(save_path, dpi=REPORT_DPI, tight=True)
            print_saved_plot(save_path, "Mapa de calor comparativo (Benchmark)")
            plt.close()

        print_subsection("Ranking Global por Suma de Posiciones", icon="🏆")
        metricas_para_ranking = ['Range_mean', 'SumMin_mean', 'MinSum_mean']
        rank_df = heat_df[['method'] + [m for m in metricas_para_ranking if m in heat_df.columns]].copy()
        for m in metricas_para_ranking:
            if m in rank_df.columns:
                rank_df[m + '_pos'] = rank_df[m].rank(method='average', ascending=True)
        pos_cols = [c for c in rank_df.columns if c.endswith('_pos')]
        rank_df['posicion_total'] = rank_df[pos_cols].sum(axis=1)
        rank_df = rank_df.sort_values('posicion_total')
        plt.figure(figsize=(10, 5))
        sns.barplot(data=rank_df, x='method', y='posicion_total', color='royalblue')
        plt.xticks(rotation=25, ha='right')
        plt.tight_layout()
        save_path = os.path.join(FOLDERS['rankings'], f'ranking_global_total_{EXECUTION_MODE}.png')
        save_report_figure(save_path, dpi=REPORT_DPI)
        print_saved_plot(save_path, "Gráfico de Ranking Global")
        plt.close()

    if df_runs is not None and not df_runs.empty:
        print_subsection("Síntesis Estadística Comparativa", icon="📊")
        plot_metrics = ['Range', 'SumMin', 'MinSum', 'MaxToleranceRate', 'AvgToleranceRate', 'AvgHammingDistance', 'Hypervolume']
        available_metrics = [m for m in plot_metrics if m in df_runs.columns]
        
        if available_metrics:
            df_plot = df_runs.copy()
            df_plot['config'] = df_plot['algorithm'].astype(str) + ' - ' + df_plot['init'].astype(str)
            
            # --- 1. BOXPLOTS (PANEL + INDIVIDUALES) ---
            print(f"\n    📦 \033[1mResumen Global (Boxplots)\033[0m")
            
            # 1.a Panel consolidado
            ncols = 3
            nrows = int(np.ceil(len(available_metrics) / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows))
            axes = np.atleast_1d(axes).ravel()
            for ax, metric in zip(axes, available_metrics):
                sns.boxplot(data=df_plot, x='config', y=metric, ax=ax, color='#9ecae1')
                sns.stripplot(data=df_plot, x='config', y=metric, ax=ax, color='black', alpha=0.45, size=4)
                ax.set_title(f'Boxplot: {metric}', fontsize=12, fontweight='bold')
                ax.tick_params(axis='x', rotation=30)
            for ax in axes[len(available_metrics):]: ax.axis('off')
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            save_path = os.path.join(FOLDERS['boxplots'], f'boxplots_metricas_finales_{EXECUTION_MODE}.png')
            save_report_figure(save_path, dpi=REPORT_DPI)
            print_saved_plot(save_path, "Panel de Boxplots comparativos")
            plt.close()

            # 1.b Figuras individuales
            for metric in available_metrics:
                plt.figure(figsize=(10, 5))
                sns.boxplot(data=df_plot, x='config', y=metric, color='#9ecae1')
                sns.stripplot(data=df_plot, x='config', y=metric, color='black', alpha=0.45, size=5)
                plt.title(f'Distribución de {metric} (Boxplot)', fontsize=13, fontweight='bold')
                plt.xticks(rotation=30)
                plt.tight_layout()
                save_path = os.path.join(FOLDERS['boxplots'], f'boxplot_metricas_{metric}_{EXECUTION_MODE}.png')
                save_report_figure(save_path, dpi=REPORT_DPI)
                print_saved_plot(save_path, f"Distribución {metric} (Boxplot)")
                plt.close()

            # --- 2. VIOLIN PLOTS (PANEL + INDIVIDUALES) ---
            print(f"\n    🎻 \033[1mDistribuciones Detalladas (Violin Plots)\033[0m")
            
            # 2.a Panel consolidado
            fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows))
            axes = np.atleast_1d(axes).ravel()
            for ax, metric in zip(axes, available_metrics):
                sns.violinplot(data=df_plot, x='config', y=metric, ax=ax, inner='quartile', cut=0, color='#9ecae1')
                ax.set_title(f'Violin: {metric}', fontsize=12, fontweight='bold')
                ax.tick_params(axis='x', rotation=30)
            for ax in axes[len(available_metrics):]: ax.axis('off')
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            save_path = os.path.join(FOLDERS['boxplots'], f'violin_panel_metricas_finales_{EXECUTION_MODE}.png')
            save_report_figure(save_path, dpi=REPORT_DPI)
            print_saved_plot(save_path, "Panel de Distribución Violin")
            plt.close()

            # 2.b Figuras individuales
            for metric in available_metrics:
                plt.figure(figsize=(10, 5))
                sns.violinplot(data=df_plot, x='config', y=metric, inner='quartile', cut=0, color='#9ecae1')
                plt.title(f'Distribución de {metric} (Violin)', fontsize=13, fontweight='bold')
                plt.xticks(rotation=30)
                plt.tight_layout()
                save_path = os.path.join(FOLDERS['boxplots'], f'violin_metricas_{metric}_{EXECUTION_MODE}.png')
                save_report_figure(save_path, dpi=REPORT_DPI)
                print_saved_plot(save_path, f"Distribución {metric} (Violin)")
                plt.close()

            # --- 3. MEDIA ± STD (PANEL + INDIVIDUALES) ---
            print(f"\n    📉 \033[1mAnálisis de Tendencia Central (Media ± Std)\033[0m")
            summary_all = df_plot.groupby('config')[available_metrics].agg(['mean', 'std'])
            
            # 3.a Panel consolidado
            fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows))
            axes = np.atleast_1d(axes).ravel()
            for ax, metric in zip(axes, available_metrics):
                summary = pd.DataFrame({
                    'config': summary_all.index,
                    'm': summary_all[(metric, 'mean')],
                    's': summary_all[(metric, 'std')].fillna(0.0),
                })
                sns.barplot(data=summary, x='config', y='m', ax=ax, color='cadetblue', errorbar=None)
                ax.errorbar(x=np.arange(len(summary)), y=summary['m'].to_numpy(),
                            yerr=summary['s'].to_numpy(), fmt='none', c='black', capsize=4)
                ax.set_title(f'Media ± std: {metric}', fontsize=12, fontweight='bold')
                ax.set_ylabel('')
                ax.tick_params(axis='x', rotation=30)
            for ax in axes[len(available_metrics):]: ax.axis('off')
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            save_path = os.path.join(FOLDERS['boxplots'], f'media_std_panel_metricas_finales_{EXECUTION_MODE}.png')
            save_report_figure(save_path, dpi=REPORT_DPI)
            print_saved_plot(save_path, "Panel de Tendencia Central (Media ± Std)")
            plt.close()

            # 3.b Figuras individuales
            for metric in available_metrics:
                summary = pd.DataFrame({
                    'config': summary_all.index,
                    'metric_mean': summary_all[(metric, 'mean')],
                    'metric_std': summary_all[(metric, 'std')].fillna(0.0),
                })
                plt.figure(figsize=(10, 5))
                ax = sns.barplot(data=summary, x='config', y='metric_mean', color='cadetblue', errorbar=None)
                ax.errorbar(x=np.arange(len(summary)), y=summary['metric_mean'].to_numpy(),
                            yerr=summary['metric_std'].to_numpy(), fmt='none', c='black', capsize=4)
                plt.title(f'Estadística Descriptiva - {metric} (Media ± Std)', fontsize=13, fontweight='bold')
                plt.xlabel('Configuración')
                plt.ylabel(metric)
                plt.xticks(rotation=30)
                plt.tight_layout()
                save_path = os.path.join(FOLDERS['boxplots'], f'media_std_metricas_{metric}_{EXECUTION_MODE}.png')
                save_report_figure(save_path, dpi=REPORT_DPI)
                print_saved_plot(save_path, f"Media ± std {metric}")
                plt.close()
    # --- FINALIZACIÓN ---
    if 'START_TIME' in globals() or 'START_TIME' in locals():
        total_duration = time.time() - START_TIME
        hours, rem = divmod(total_duration, 3600)
        minutes, seconds = divmod(rem, 60)
        print(f"\n{'='*40}")
        print(f"⏱️  TIEMPO TOTAL DE EJECUCIÓN: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
        print(f"{'='*40}\n")
    

if __name__ == '__main__':
    # CLI: allow quick override of mode, data source and num blocks
    import argparse
    parser = argparse.ArgumentParser(description='Run TAG-SNP pipeline')
    parser.add_argument('--mode', '-m', choices=['fast', 'medium', 'full'], default='medium', help='Modo de ejecución (fast/medium/full)')
    parser.add_argument(
        '--data-source', '-d',
        choices=['hinds2005', 'synthetic'],
        default=None,
        help='Fuente de datos. Por defecto: hinds2005 (dataset exacto de Moqa 2022 y Ting 2010)'
    )
    args = parser.parse_args()

    START_TIME = time.time()
    print_header('PIPELINE TAG SNP')

    # --- LOG ÚNICO Y UNIFICADO ---
    # Se mantiene la captura de stdout/stderr en un único log temporal, que se copia al final.
    # Esto garantiza que toda la salida, incluyendo la de procesos paralelos, quede registrada en un solo archivo.
    import tempfile
    temp_log_fh = tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.log')
    temp_log = temp_log_fh.name
    log_fh = temp_log_fh

    class Tee:
        def __init__(self, *streams):
            self.streams = streams
        def write(self, data):
            for s in self.streams:
                try:
                    s.write(data)
                except Exception:
                    pass
        def flush(self):
            for s in self.streams:
                try:
                    s.flush()
                except Exception:
                    pass

    # Redirigir stdout/stderr a Tee para capturar todo en un solo log
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    sys.stdout = Tee(orig_stdout, log_fh)
    sys.stderr = Tee(orig_stderr, log_fh)

    try:
        # 1. Configuración y carga (allow overriding mode)
        cfg = setup_configuration(execution_mode=args.mode, data_source=args.data_source)

        # Create per-dataset tree early so we can later copy the full log into it
        base_dataset_dir, local_folders = create_dataset_tree(cfg, cfg.data_source)
        cfg.folders = local_folders

        H, snp_ids, snp_positions, haplotype_ids = load_target_dataset(cfg)
    except Exception as e:
        # restore stdout/stderr temporarily to ensure further exceptions are visible
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
        log_fh.flush()
        log_fh.close()
        raise
    _exportar_dataset_seleccionado(H, snp_ids, snp_positions, haplotype_ids, cfg.folders, cfg.execution_mode)
    
    # 2. Exploración y Diagnóstico
    print_header('DIAGNÓSTICO DE DATOS Y DESEQUILIBRIO (LD)')
    run_exploratory_data_analysis(H, snp_ids, snp_positions, cfg)
    perform_ld_diagnostic(H, cfg)
    # El análisis de similitud genética también es diagnóstico
    PAIR_IDX, pair_hamming, p33, p66 = setup_evolutionary_environment(H, cfg)
    
    # 3. Búsqueda Multiobjetivo
    print_header('MOTOR MULTIOBJETIVO')
    
    # 4. Ejecutar Algoritmos
    resultados = execute_algorithms(H, PAIR_IDX, pair_hamming, p33, p66, cfg)
    
    # 5. Generar Síntesis de Resultados
    print_header('SÍNTESIS DE RESULTADOS')
    df_runs, df_gen_runs, df_front_obj, ideal_global, nadir_global = \
        generate_final_synthesis(resultados, PAIR_IDX, pair_hamming, cfg)
    
    # 6. Generar Reportes y Visualización
    print_header('REPORTES Y VISUALIZACIÓN')
    generate_visual_reports(df_runs, df_gen_runs, df_front_obj, ideal_global, nadir_global, cfg)

    # FINALIZACIÓN: restaurar stdout/stderr, cerrar log temporal y copiarlo al dataset
    try:
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
    except Exception:
        pass
    try:
        log_fh.flush()
        log_fh.close()
    except Exception:
        pass
    try:
        final_log_path = os.path.join(base_dataset_dir, 'terminal_output.log')
        shutil.copyfile(temp_log, final_log_path)
        print_status(f"Salida de la terminal guardada en: {final_log_path}")
    except Exception as e:
        print_status(f"No se pudo guardar la salida de la terminal en: {e}", success=False)

