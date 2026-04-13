#!/usr/bin/env python3
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
    """Almacena la configuración global del experimento."""
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
    auto_download: bool
    block_len: int
    block_start: int
    syn_flip_prob: float
    hapmap_chr: int
    hapmap_pop: str
    hapmap_release: str
    hapmap_dir: str
    report_plot_dpi: int
    report_heartbeat_sec: int
    # Synthetic dataset diversification (min pairwise Hamming distance target)
    synthetic_min_pairwise_diff: int
    synthetic_max_attempts: int


def setup_configuration(execution_mode: str | None = None, num_blocks: int | None = None) -> ExperimentConfig:
    """Configura los parámetros globales y crea los directorios.

    Parameters
    ----------
    execution_mode : str | None
        Optional execution mode override ('fast', 'medium', 'full'). If None, defaults to 'medium'.
    num_blocks : int | None
        Optional number of blocks override. If None, defaults to 1.
    """
    import time
    sns.set_theme(style='whitegrid')
    pd.set_option('display.max_columns', 200)
    EXECUTION_MODE = execution_mode if execution_mode is not None else 'medium' # fast / medium / full | Selector de modo
    NUM_BLOCKS = int(num_blocks) if num_blocks is not None else 1
    MASTER_SEED = 42
    np.random.seed(MASTER_SEED)
    random.seed(MASTER_SEED)
    DATA_SOURCE = "hapmap_phase2" # synthetic / hapmap_phase2
    N_SNPS = 1032
    SYN_BLOCK_SIZE = int(N_SNPS / max(1, NUM_BLOCKS))
    N_HAPLOTYPES = 40
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
    REPORT_HEARTBEAT_SEC = 60
    # Solo crear la carpeta base del modo
    os.makedirs(BASE_OUT_MODE_DIR, exist_ok=True)
        
    MODE_CONFIG = {
        'fast': {'POP_SIZE': 10, 'N_GEN': 2, 'OFFSPRING': 10, 'PC': 0.7, 'MOEAD_NEIGHBORS': 5, 'N_RUNS': 2},
        'medium': {'POP_SIZE': 40, 'N_GEN': 10, 'OFFSPRING': 40, 'PC': 0.7, 'MOEAD_NEIGHBORS': 15, 'N_RUNS': 2},
        'full': {'POP_SIZE': 200, 'N_GEN': 500, 'OFFSPRING': 200, 'PC': 0.7, 'MOEAD_NEIGHBORS': 15, 'N_RUNS': 5},
    }
    
    cfg = MODE_CONFIG[EXECUTION_MODE]
    HAPMAP_DIR = os.path.join("data", "hapmap_phase2")
    BLOCK_START = 0
    NUM_BLOQUES_EFECTIVO = NUM_BLOCKS
    if DATA_SOURCE == "hapmap_phase2":
        try:
            import json
            map_path = os.path.join(HAPMAP_DIR, "block_map_1032.json")
            if os.path.exists(map_path):
                with open(map_path, "r") as f:
                    block_map = json.load(f)
                valid_keys = sorted([int(k) for k in block_map.keys()])
                rango_min, rango_max = min(valid_keys), max(valid_keys)
                # Clamping con aviso si el valor solicitado está fuera del rango
                if NUM_BLOCKS < rango_min:
                    print(f"      ⚠ NUM_BLOQUES={NUM_BLOCKS} fuera del rango válido [{rango_min}, {rango_max}]. "
                          f"Ajustando a {rango_min}.")
                    NUM_BLOQUES_EFECTIVO = rango_min
                elif NUM_BLOCKS > rango_max:
                    print(f"      ⚠ NUM_BLOQUES={NUM_BLOCKS} fuera del rango válido [{rango_min}, {rango_max}]. "
                          f"Ajustando a {rango_max}.")
                    NUM_BLOQUES_EFECTIVO = rango_max
                else:
                    NUM_BLOQUES_EFECTIVO = NUM_BLOCKS
                # Buscar clave exacta o la más cercana
                if NUM_BLOQUES_EFECTIVO not in valid_keys:
                    closest_k = min(valid_keys, key=lambda k: abs(k - NUM_BLOQUES_EFECTIVO))
                    print(f"      ⚠ Sin ventanas exactas para {NUM_BLOQUES_EFECTIVO} bloques. "
                          f"Usando la clave más cercana: {closest_k}.")
                    NUM_BLOQUES_EFECTIVO = closest_k
                BLOCK_START = int(np.random.choice(block_map[str(NUM_BLOQUES_EFECTIVO)]))
        except Exception as e:
            print(f"      ⚠ Error al leer block_map: {e}. Usando BLOCK_START=0.")

    print_step("CONFIGURACIÓN", icon="⚙")
    print(f"      • Modo={EXECUTION_MODE} | POP_SIZE={cfg['POP_SIZE']} | N_GEN={cfg['N_GEN']} | OFFSPRING={cfg['OFFSPRING']} | PC={cfg['PC']} | PM={1.0/N_SNPS:.6f} | N_RUNS={cfg['N_RUNS']}")
    print_subsection("Metadatos y Dimensiones del Dataset", icon="📊")
    if DATA_SOURCE == "hapmap_phase2":
        print(f"      • ORIGEN_DATOS={DATA_SOURCE} | NUM_BLOQUES={NUM_BLOQUES_EFECTIVO} (solicitados: {NUM_BLOCKS}) | BLOCK_START={BLOCK_START} | N_SNPS={N_SNPS} | PM={1.0/N_SNPS:.6f}")
    else:
        print(f"      • ORIGEN_DATOS={DATA_SOURCE} | NUM_BLOQUES={NUM_BLOCKS} | N_SNPS={N_SNPS} | PM={1.0/N_SNPS:.6f}")
    print(f"      • REPORT_DPI={REPORT_PLOT_DPI} | HEARTBEAT={REPORT_HEARTBEAT_SEC}s")
    
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
        moead_neighbors=cfg['MOEAD_NEIGHBORS'],
        n_runs=cfg['N_RUNS'],
        thresh_opt_pct=0.95,
        thresh_opt_mean=0.18,
        thresh_opt_abs_corr=0.45,
        thresh_accept_mean=0.12,
        auto_download=False,
        block_len=N_SNPS,
        block_start=BLOCK_START,
        syn_flip_prob=0.03,
        hapmap_chr=21,
        hapmap_pop="CEU",
        hapmap_release="r21",
        hapmap_dir=HAPMAP_DIR,
        report_plot_dpi=REPORT_PLOT_DPI,
        report_heartbeat_sec=REPORT_HEARTBEAT_SEC,
        # Synthetic generator settings: target min pairwise Hamming distance
        synthetic_min_pairwise_diff=100,
        synthetic_max_attempts=1000,
    )


def create_dataset_tree(cfg: ExperimentConfig, dataset_type: str):
    """Create a per-dataset directory tree and return (base_dataset_dir, folders_dict).

    Naming convention: resultados/<mode>/<dataset_type>/<num_blocks>_bloques/
    """
    # Siempre añadir timestamp al nombre del bloque
    ts = datetime.now().strftime('%Y%m%dT%H%M%S')
    base_dataset_dir = os.path.join(cfg.base_out_dir, cfg.execution_mode, dataset_type, f"{cfg.num_blocks}_bloques_{ts}")

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
            report_lines.append(f"      • REPORT_DPI={cfg.report_plot_dpi} | HEARTBEAT={cfg.report_heartbeat_sec}s")
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
            report_lines.append(f"      • es_optimo: {str(bool(is_opt))}")
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
    else:
        H, snp_ids, snp_positions, haplotype_ids = load_hapmap_phase2_block(
            chr_num=cfg.hapmap_chr, pop=cfg.hapmap_pop, release=cfg.hapmap_release,
            hapmap_dir=cfg.hapmap_dir, block_start=cfg.block_start, block_len=cfg.block_len,
            auto_download=cfg.auto_download)
        cfg.n_haplotypes = int(H.shape[0])
        # Do NOT create per-dataset folders again; use cfg.folders as set in main
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
    import pandas as pd
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

def _hapmap_phase2_prefix(chr_num: int, pop: str, release: str) -> str:
    return f"genotypes_chr{chr_num}_{pop}_{release}_nr_fwd"

def _ensure_hapmap_phase2_files(hapmap_dir: str, chr_num: int, pop: str, release: str, auto_download: bool):
    hapmap_dir_path = Path(hapmap_dir)
    hapmap_dir_path.mkdir(parents=True, exist_ok=True)

    prefix = _hapmap_phase2_prefix(chr_num=chr_num, pop=pop, release=release)
    base_url = "https://ftp.ncbi.nlm.nih.gov/hapmap/phasing/2006-07_phaseII/phased"
    files = {
        "phased": (hapmap_dir_path / f"{prefix}_phased.gz", f"{base_url}/{prefix}_phased.gz"),
        "legend": (hapmap_dir_path / f"{prefix}_legend.txt.gz", f"{base_url}/{prefix}_legend.txt.gz"),
        "sample": (hapmap_dir_path / f"{prefix}_sample.txt.gz", f"{base_url}/{prefix}_sample.txt.gz"),
    }

    missing = [key for key, (path, _) in files.items() if not path.exists()]
    if missing and not auto_download:
        missing_paths = ", ".join(str(files[k][0]) for k in missing)
        raise FileNotFoundError(
            f"Faltan ficheros HapMap en {hapmap_dir_path}. Faltan: {missing_paths}. "
            "Descárgalos manualmente o activa AUTO_DOWNLOAD=True."
        )

    if missing and auto_download:
        for key in missing:
            path, url = files[key]
            print(f"      • [HAPMAP] Descargando {key}: {url}")
            subprocess.run(["wget", "-c", url, "-O", str(path)], check=True)

    return {k: str(v[0]) for k, v in files.items()}

def load_hapmap_phase2_block(
    *,
    chr_num: int,
    pop: str,
    release: str,
    hapmap_dir: str,
    block_start: int,
    block_len: int,
    auto_download: bool = False,
    max_haplotypes: int | None = None,
    seed: int | None = None,
    shuffle_haplotypes: bool = False,
):
    """
    Carga HapMap Phase II (r21) faseado y devuelve un bloque contiguo de SNPs.

    - phased.gz: matriz (n_snps_total, n_haplotypes) con 0/1
    - legend.txt.gz: rs + posición (mismo orden que phased)
    - sample.txt.gz: (sample_id, hap) para cada columna (haplotipo)
    """
    paths = _ensure_hapmap_phase2_files(
        hapmap_dir=hapmap_dir, chr_num=chr_num, pop=pop, release=release, auto_download=auto_download
    )

    legend = pd.read_csv(paths["legend"], sep=r"\s+", compression="gzip")
    if not {"rs", "position"}.issubset(set(legend.columns)):
        raise ValueError(f"Legend inesperada. Columnas={list(legend.columns)}")

    phased_raw = pd.read_csv(
        paths["phased"],
        sep=r"\s+",
        header=None,
        compression="gzip",
        dtype=np.int8,
        engine="python",
    ).to_numpy(dtype=np.int8, copy=False)

    # Nota: en HapMap Phase II, algunos ficheros vienen como (n_haplotypes, n_snps_total)
    # en vez de (n_snps_total, n_haplotypes). Detectamos y corregimos automáticamente.
    if phased_raw.shape[0] == legend.shape[0]:
        phased = phased_raw
    elif phased_raw.shape[1] == legend.shape[0]:
        phased = phased_raw.T
        print(
            f"      • [HAPMAP] phased venía como (n_haplotypes, n_snps)={tuple(phased_raw.shape)}; transponiendo a {tuple(phased.shape)}"
        )
    else:
        raise ValueError(
            f"Inconsistencia legend/phased: legend={legend.shape[0]} SNPs; phased_raw shape={tuple(phased_raw.shape)}"
        )

    sample_df = pd.read_csv(
        paths["sample"],
        sep=r"\s+",
        header=None,
        names=["sample_id", "hap"],
        compression="gzip",
    )
    sample_ids = [f"{sid}_{hap}" for sid, hap in zip(sample_df["sample_id"].astype(str), sample_df["hap"].astype(int))]

    if phased.shape[1] == len(sample_ids):
        haplotype_ids = sample_ids
    else:
        # Algunos dumps no traen una correspondencia 1:1 clara en sample vs columnas de phased.
        # Para el procesamiento solo necesitamos H; usamos IDs genéricos si no cuadra.
        print(
            f"      • [HAPMAP] Aviso: sample trae {len(sample_ids)} IDs pero phased tiene {phased.shape[1]} haplotipos. Usando haplotype_ids genéricos."
        )
        haplotype_ids = [f"hap_{i}" for i in range(phased.shape[1])]

    start = int(block_start)
    end = int(block_start) + int(block_len)
    if start < 0 or end > phased.shape[0]:
        raise ValueError(f"Bloque fuera de rango: start={start}, end={end}, total_snps={phased.shape[0]}")

    phased_block = phased[start:end, :]
    snp_ids = legend.loc[start:end - 1, "rs"].astype(str).tolist()
    snp_positions = legend.loc[start:end - 1, "position"].astype(int).to_numpy()

    # Convertir a (n_haplotypes, n_snps) para compatibilidad con el resto del notebook
    H_block = phased_block.T.astype(np.int8, copy=False)

    if shuffle_haplotypes:
        if seed is None:
            raise ValueError("Si shuffle_haplotypes=True, proporciona seed=")
        rng = np.random.default_rng(seed)
        perm = rng.permutation(H_block.shape[0])
        H_block = H_block[perm, :]
        haplotype_ids = [haplotype_ids[i] for i in perm]

    if max_haplotypes is not None and max_haplotypes < H_block.shape[0]:
        H_block = H_block[:max_haplotypes, :]
        haplotype_ids = haplotype_ids[:max_haplotypes]

    return H_block, snp_ids, snp_positions, haplotype_ids

def full_ld_check(X): # Definir función para calcular el Desequilibrio de Ligamiento (LD) completo
    corr_full = np.corrcoef(X.T) # Calcular la matriz de correlación de Pearson entre las columnas (SNPs)
    corr_full = np.nan_to_num(corr_full, nan=0.0, posinf=0.0, neginf=0.0) # Sustituir NaNs e infinitos por 0.0
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
    # Se acota inferiormente a 0 para evitar valores negativos en casos límite.
    tolerance_real = float(max(0.0, d.min() - 1.0))
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

    tolerancia = np.maximum(0.0, D.min(axis=1) - 1.0)  # tolerancia de faltantes (paper)
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
        # Escalas teóricas para normalización interna en búsqueda (MOEA/D)
        self._scale_f1 = max(1.0, float(n_var))
        self._scale_f2 = max(1.0, float(n_var))
        self._scale_f3 = max(1.0, float(n_var))
        # varianza máxima aproximada para variable acotada en [0, n_var]
        self._scale_f4 = max(1.0, (float(n_var) ** 2) / 4.0)

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

def build_algorithm(problem, H, algo_name, init_name, cfg, seed=42, ref_dirs=None):
    # Fetch global config if not provided? Actually these are mostly passed or fixed.
    # But POP_SIZE, PC, PM, OFFSPRING, MOEAD_NEIGHBORS are used inside.
    # In Python, using global variables inside is fine IF they exist at runtime.
    
    # Alias de compatibilidad hacia atrás: 'greedy' -> 'greedy_hybrid'
    if init_name == 'greedy':
        init_name = 'greedy_hybrid'

    # Selección del método de inicialización
    if init_name == 'random':
        sampling = BinaryRandomSampling()
    elif init_name == 'greedy_hybrid':
        sampling = GreedyHybridTagSNPSampling(H=H, max_coverage=5, random_fill_ratio=0.2, seed=seed)
    elif init_name == 'greedy_pure':
        sampling = GreedyPureTagSNPSampling(H=H, max_coverage=5, seed=seed)
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
    init_options = ['random', 'greedy_hybrid', 'greedy_pure']
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

    t0 = time.time()
    res = minimize(problema, algo, termination, seed=seed, verbose=False, save_history=True)
    elapsed = time.time() - t0

    n_var = problema.n_var
    Xf = np.array(res.X, dtype=bool) if getattr(res, 'X', None) is not None else np.empty((0, n_var), dtype=bool)
    # Para garantizar métricas comparables, siempre reconstruimos F en escala cruda
    if Xf.shape[0] > 0:
        Ff, _ = evaluate_population(Xf, H, pair_idx=pair_idx)
    else:
        Ff = np.empty((0, 4), dtype=float)

    # Extraer historial generacional
    F_history = []
    for hist in getattr(res, 'history', []) or []:
        F_gen = None
        X_gen = None
        opt = getattr(hist, 'opt', None)
        pop = getattr(hist, 'pop', None)

        # 1) Intento preferente: reconstruir F crudo desde X de la población de la generación
        if pop is not None:
            try:
                X_gen = np.array(pop.get('X'), dtype=bool)
            except Exception:
                try:
                    X_gen = np.array(getattr(pop, 'X', None), dtype=bool)
                except Exception:
                    X_gen = None
            if X_gen is not None and getattr(X_gen, 'size', 0) > 0:
                F_gen, _ = evaluate_population(X_gen, H, pair_idx=pair_idx)

        # 2) Fallback: usar F reportada por pymoo
        if F_gen is None and opt is not None:
            try:
                F_gen = np.array(opt.get('F'), dtype=float)
            except Exception:
                try:
                    F_gen = np.array(getattr(opt, 'F', None), dtype=float)
                except Exception:
                    F_gen = None
        if F_gen is None or getattr(F_gen, 'size', 0) == 0:
            if pop is not None:
                try:
                    F_gen = np.array(pop.get('F'), dtype=float)
                except Exception:
                    try:
                        F_gen = np.array(getattr(pop, 'F', None), dtype=float)
                    except Exception:
                        F_gen = None
        if F_gen is None or getattr(F_gen, 'size', 0) == 0:
            continue
        if F_gen.ndim == 1:
            F_gen = F_gen.reshape(1, -1)
        if F_gen.shape[1] >= 4:
            F_history.append(F_gen[:, :4])

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
    init_options = ['random', 'greedy_hybrid', 'greedy_pure']
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
    print(f"    • Iniciando {total_runs} experimentos en modo secuencial")

    resultados_desordenados = {}
    done_runs = 0
    w = len(str(total_runs))

    for tarea in trabajos:
        try:
            rr = _ejecutar_un_experimento(tarea)
        except Exception as e:
            print(
                f"      • ⚠ Error en [{tarea['algo_name']}-{tarea['init_name']}] "
                f"ejecucion {tarea['run_idx']}: {e}"
            )
            continue

        done_runs += 1
        print(
            f"      • [Progreso: {done_runs:>{w}}/{total_runs}] | "
            f"[{rr.algorithm}-{rr.init}] ejecucion {rr.run}/{n_runs} "
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

def compute_raw_aux_metrics(F_raw):
    # Extraer la compacidad (primer objetivo)
    compactness = F_raw[:, 0]
    # Extraer la tolerancia real (segundo objetivo, negredo porque PyMoo minimiza)
    tolerance_real = -F_raw[:, 1]
    # Extraer el Hamming promedio real (tercer objetivo, negredo)
    hamming_avg_real = -F_raw[:, 2]
    # Evitar división por cero en compacidad (nº de SNPs seleccionados)
    safe_comp = np.where(compactness <= 0, np.nan, compactness)
    # Calcular la tasa de tolerancia por SNP
    tr = tolerance_real / safe_comp
    # Obtener el máximo de esta tasa
    max_tr = float(np.nanmax(tr))
    # Obtener el promedio de esta tasa
    avg_tr = float(np.nanmean(tr))
    # Obtener el promedio de Hamming real
    avg_hamming = float(np.nanmean(hamming_avg_real))
    # Retornar las métricas de rendimiento real no normalizadas
    return max_tr, avg_tr, avg_hamming

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

def evaluate_final_metrics(run_results, heartbeat_sec=30):
    # Si la lista de resultados está vacía, retorna un DataFrame vacío y puntos de referencia nulos
    if not run_results:
        return pd.DataFrame(), None, None

    # Llama a la función auxiliar para obtener el mejor (ideal) y peor (nadir) punto a través de todos los frentes
    ideal, nadir = global_ideal_nadir(run_results)

    safe_denom = (nadir - ideal + 1e-9)

    # Lista para acumular diccionarios con las métricas de cada ejecución
    rows = []
    total_runs = len(run_results)
    t0_metrics = time.time()
    next_heartbeat = t0_metrics + max(5, int(heartbeat_sec))
    print(f"      • Iniciando métricas finales: {total_runs} ejecuciones")
    # Itera sobre cada ejecución registrada
    for idx, rr in enumerate(run_results, start=1):
        # Extraer el frente de soluciones original (valores crudos de los objetivos)
        F_raw = rr.F_final
        # Si la ejecución falló o devolvió un frente vacío, se omite
        if F_raw is None or len(F_raw) == 0:
            continue

        # Realizar la normalización Min-Max del frente utilizando los puntos globales calculados
        # Se añade un pequeño epsilon (1e-9) para evitar división por cero en casos donde max == min
        F_norm = (F_raw - ideal) / safe_denom
        # Asegurar que ningún valor normalizado exceda los límites por errores de redondeo (0.0 a 1.0)
        F_norm = np.clip(F_norm, 0, 1)

        # Validación estricta: si tras la normalización persisten valores NaN o infinitos, lanza un error fatal
        if np.any(np.isnan(F_norm)) or np.any(np.isinf(F_norm)):
            raise ValueError('Se detecta NaN/Inf en normalizacion.')
        # Calcular métricas escalares (Range, SumMin, MinSum) sobre el frente normalizado
        rng, smn, msn = compute_range_summin_minsum(F_norm)
        # Calcular métricas biológicas y técnicas sobre los valores originales (raw)
        max_tr, avg_tr, avg_h = compute_raw_aux_metrics(F_raw)

        # Calcular el valor de hipervolumen para el frente normalizado actual
        hv_val = compute_hypervolume(F_norm)

        # Construir el registro detallado de la ejecución con metadatos y métricas calculadas
        rows.append({
            'algorithm': rr.algorithm,           # Algoritmo utilizado (ej. NSGA-III)
            'init': rr.init,                     # Método de inicialización (ej. Greedy)
            'run': rr.run,                       # Índice de la ejecución dentro del experimento
            'seed': rr.seed,                     # Semilla aleatoria empleada
            'elapsed_sec': rr.elapsed_sec,       # Tiempo total de ejecución en segundos
            'n_solutions_final_front': len(F_raw), # Número de soluciones encontradas en el frente de Pareto
            'Range': rng,                        # Diferencia entre el peor y mejor valor (diversidad)
            'SumMin': smn,                       # Métrica de proximidad al origen
            'MinSum': msn,                       # Métrica de cobertura del espacio
            'MaxToleranceRate': max_tr,          # Tasa de tolerancia máxima alcanzada
            'AvgToleranceRate': avg_tr,          # Tasa de tolerancia promedio en el frente
            'AvgHammingDistance': avg_h,         # Distancia de Hamming promedio entre Tag SNPs
            'Hypervolume': hv_val                # Valor de hipervolumen (mayor es mejor)
        })
        now = time.time()
        if now >= next_heartbeat or idx == total_runs:
            elapsed = now - t0_metrics
            rate = idx / elapsed if elapsed > 0 else 0.0
            remaining = max(0, total_runs - idx)
            eta = remaining / rate if rate > 0 else float('inf')
            eta_txt = f"{eta:.1f}s" if np.isfinite(eta) else "--"
            print(f"      • [Métricas finales] {idx}/{total_runs} | transcurrido={elapsed:.1f}s | ETA={eta_txt}")
            next_heartbeat = now + max(5, int(heartbeat_sec))
    # Retornar el DataFrame construido junto con los puntos de referencia utilizados
    return pd.DataFrame(rows), ideal, nadir

def _build_generation_rows_for_run(rr, ideal, safe_denom):
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

        F_norm = (F_raw - ideal) / safe_denom
        F_norm = np.clip(F_norm, 0, 1)

        rng, smn, msn = compute_range_summin_minsum(F_norm)
        max_tr, avg_tr, avg_h = compute_raw_aux_metrics(F_raw)
        hv_val = compute_hypervolume(F_norm)

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
            'Hypervolume': hv_val
        })
        processed_generations += 1

    return rows, processed_generations


def build_generation_metrics(run_results, ideal=None, nadir=None, heartbeat_sec=30):
    # Si no hay resultados, retornar un DataFrame vacío
    if not run_results:
        return pd.DataFrame()

    # Si no se proporcionan ideal/nadir, calcularlos de forma global
    if ideal is None or nadir is None:
        ideal, nadir = global_ideal_nadir(run_results)
    safe_denom = (nadir - ideal + 1e-9)

    valid_runs = [rr for rr in run_results if getattr(rr, 'F_history', None) is not None and len(getattr(rr, 'F_history', [])) > 0]
    if not valid_runs:
        return pd.DataFrame()

    total_runs = len(valid_runs)
    total_gens = sum(len(getattr(rr, 'F_history', []) or []) for rr in valid_runs)
    heartbeat = max(5, int(heartbeat_sec))
    started = time.time()
    next_heartbeat = started + heartbeat
    done_runs = 0
    done_gens = 0
    rows_by_key = {}

    print(f"      • Iniciando métricas generacionales: {total_runs} ejecuciones con historial ({total_gens} generaciones estimadas)")

    for rr in valid_runs:
        run_rows, gen_count = _build_generation_rows_for_run(rr, ideal, safe_denom)
        rows_by_key[(rr.algorithm, rr.init, rr.run)] = run_rows
        done_runs += 1
        done_gens += gen_count
        now = time.time()
        if now >= next_heartbeat or done_runs == total_runs:
            elapsed = now - started
            rate = done_runs / elapsed if elapsed > 0 else 0.0
            eta = (total_runs - done_runs) / rate if rate > 0 else float('inf')
            eta_txt = f"{eta:.1f}s" if np.isfinite(eta) else "--"
            print(f"      • [Métricas generacionales] runs={done_runs}/{total_runs} | gens={done_gens}/{total_gens} | transcurrido={elapsed:.1f}s | ETA={eta_txt}")
            next_heartbeat = now + heartbeat

    rows = []
    for rr in valid_runs:
        key = (rr.algorithm, rr.init, rr.run)
        rows.extend(rows_by_key.get(key, []))

    print(f"      • Métricas generacionales completadas en {time.time() - started:.1f}s")
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


def _plot_haplotype_heatmap(H, folders, mode_tag):
    sns.set_theme(style='whitegrid')
    plt.figure(figsize=(16, 5))
    sns.heatmap(H, cmap='gray', vmin=0, vmax=1, cbar_kws={'label': 'Alelo (0/1)'})
    plt.title('Matriz de haplotipos H (filas=haplotipos, columnas=SNPs)')
    plt.xlabel('SNP')
    plt.ylabel('Haplotipo')
    plt.tight_layout()
    save_path = os.path.join(folders['datos'], f'heatmap_matriz_haplotipos_{mode_tag}.png')
    plt.savefig(save_path, dpi=200)
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


def _plot_ld_blocks(H, folders, mode_tag):
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
    plt.savefig(save_path, dpi=200)
    print_saved_plot(save_path, f"Estructura de bloques LD ({n_bloques_reales} bloques reales)")
    plt.close()

def _plot_haplotype_zoom(H, folders, mode_tag):
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
    plt.savefig(save_path, dpi=200)
    print_saved_plot(save_path, f"Zoom matriz H ({h_view}x{s_view})")
    plt.close()

def _plot_allele_frequency(H, folders, mode_tag):
    allele_freq = H.mean(axis=0)
    plt.figure(figsize=(12, 5))
    sns.histplot(allele_freq, bins=30, kde=True, color='steelblue')
    plt.title('Distribución de frecuencia alélica por SNP')
    plt.xlabel('Frecuencia de alelo 1')
    plt.ylabel('Número de SNPs')
    plt.tight_layout()
    save_path = os.path.join(folders['datos'], f'histograma_frecuencia_alelica_{mode_tag}.png')
    plt.savefig(save_path, dpi=200)
    print_saved_plot(save_path, "Distribución de frecuencia alélica")
    plt.close()

def _plot_snp_variability(H, folders, mode_tag):
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
    plt.savefig(save_path, dpi=200)
    print_saved_plot(save_path, "Variabilidad por SNP")
    plt.close()

def _plot_dominant_alleles(H, folders, mode_tag):
    hap_ones = H.sum(axis=1)
    plt.figure(figsize=(10, 4))
    sns.barplot(x=np.arange(len(hap_ones)), y=hap_ones, color='teal')
    plt.title('Número de alelos 1 por haplotipo')
    plt.xlabel('Haplotipo')
    plt.ylabel('Conteo de alelos 1')
    plt.tight_layout()
    save_path = os.path.join(folders['datos'], f'conteo_alelos_por_haplotipo_{mode_tag}.png')
    plt.savefig(save_path, dpi=200)
    print_saved_plot(save_path, "Alelos dominantes por haplotipo")
    plt.close()

def _plot_hamming_distribution(H, folders, mode_tag):
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
    plt.savefig(save_path, dpi=200) 
    print_saved_plot(save_path, "Distribución distancias de Hamming")
    plt.close()

def run_exploratory_data_analysis(H, snp_ids, snp_positions, cfg):
    """Genera las visualizaciones exploratorias (Heatmaps, Histogramas) mediante subrutinas."""
    EXECUTION_MODE = cfg.execution_mode
    FOLDERS = cfg.folders
    mode_tag_local = EXECUTION_MODE

    print_subsection("Visualización de la Estructura de Haplotipos", icon="🧬")
    _plot_haplotype_heatmap(H, FOLDERS, mode_tag_local)
    _plot_ld_blocks(H, FOLDERS, mode_tag_local)
    _plot_haplotype_zoom(H, FOLDERS, mode_tag_local)
    
    print_subsection("Análisis de Variabilidad y Frecuencia Alélica", icon="📈")
    _plot_allele_frequency(H, FOLDERS, mode_tag_local)
    _plot_snp_variability(H, FOLDERS, mode_tag_local)
    _plot_dominant_alleles(H, FOLDERS, mode_tag_local)
    _plot_hamming_distribution(H, FOLDERS, mode_tag_local)

    

def _plot_ld_correlation_heatmap(corr_full, folders, mode_tag):
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_full, cmap='vlag', center=0, cbar_kws={'label': 'Correlación'})
    plt.title('Mapa de correlación completo SNP×SNP')
    plt.xlabel('SNP')
    plt.ylabel('SNP')
    plt.tight_layout()
    save_path = os.path.join(folders['datos'], f'heatmap_correlacion_completa_{mode_tag}.png')
    plt.savefig(save_path, dpi=200)
    print_saved_plot(save_path, "Mapa de calor de correlación (LD)")
    plt.close()

def _plot_ld_correlation_hist(ld_corrs, folders, mode_tag):
    plt.figure(figsize=(10, 5))
    sns.histplot(ld_corrs, bins=40, kde=True, color='slateblue')
    plt.title('Correlaciones entre todos los pares de SNPs')
    plt.xlabel('Correlación de Pearson')
    plt.ylabel('Frecuencia')
    plt.tight_layout()
    save_path = os.path.join(folders['datos'], f'histograma_correlaciones_ld_{mode_tag}.png')
    plt.savefig(save_path, dpi=200)
    print_saved_plot(save_path, "Distribución de correlaciones (LD)")
    plt.close()

def _plot_ld_cdf(abs_sorted, cdf, folders, mode_tag):
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
    plt.savefig(save_path, dpi=200)
    print_saved_plot(save_path, "CDF de correlación LD")
    plt.close()

def perform_ld_diagnostic(H, cfg):
    """Analiza la matriz H y evalúa los coeficientes de LD estructurales."""
    EXECUTION_MODE = cfg.execution_mode
    FOLDERS = cfg.folders
    mode_tag_local = EXECUTION_MODE

    ld_mean, ld_corrs, corr_full = full_ld_check(H)
    abs_corr = np.abs(ld_corrs)
    if len(abs_corr) > 0:
        abs_sorted = np.sort(abs_corr)
        cdf = np.arange(1, len(abs_sorted) + 1) / len(abs_sorted)
    else:
        abs_sorted = np.array([])
        cdf = np.array([])

    print_subsection("Coeficientes Globales de Correlación LD", icon="🔗")
    print(f'      • Correlación media absoluta (global): {ld_mean:.4f}')
    print(f'      • Total de pares evaluados: {len(ld_corrs)}')

    _plot_ld_correlation_heatmap(corr_full, FOLDERS, mode_tag_local)
    _plot_ld_correlation_hist(ld_corrs, FOLDERS, mode_tag_local)
    _plot_ld_cdf(abs_sorted, cdf, FOLDERS, mode_tag_local)

    THRESH_OPT_MEAN = cfg.thresh_opt_mean
    THRESH_OPT_ABS_CORR = cfg.thresh_opt_abs_corr
    THRESH_OPT_PCT = cfg.thresh_opt_pct
    THRESH_ACCEPT_MEAN = cfg.thresh_accept_mean

    pct_abs_corr_ge_thresh = float((abs_corr >= THRESH_OPT_ABS_CORR).mean() * 100.0) if len(abs_corr) > 0 else 0.0
    es_optimo = (ld_mean >= THRESH_OPT_MEAN) and (pct_abs_corr_ge_thresh >= THRESH_OPT_PCT)

    if es_optimo:
        estado = 'ÓPTIMO'
        icono = '✅'
    elif ld_mean >= THRESH_ACCEPT_MEAN:
        estado = 'ACEPTABLE'
        icono = '⚠️'
    else:
        estado = 'NO ÓPTIMO'
        icono = '❌'

    print_subsection("Veredicto de Desequilibrio de Ligamento", icon="⚖️")
    print(f'      • Correlación media absoluta (global): {ld_mean:.4f}')
    print(f'      • % pares con |corr| >= {THRESH_OPT_ABS_CORR:.2f}: {pct_abs_corr_ge_thresh:.2f}%')
    print(f'      • Total de pares evaluados: {len(ld_corrs)}')
    print(f'      • es_optimo: {es_optimo}')
    print_status(f"Verificación previa: {estado}", success=(estado != 'NO ÓPTIMO'))

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
    print_subsection("Configuración del Motor Evolutivo", icon="⚙️")
    print(f'      • Configuraciones a ejecutar: {len(algo_configs)}')
    print(f'      • Puntos de referencia (ref_dirs): {len(ref_dirs_used)} | Particiones: {ref_partitions}')
    print(f'      • Tamaño de población (pop_size): {POP_SIZE}')
    RUN_EXPERIMENTS = True
    if RUN_EXPERIMENTS:
        run_results = run_all_experiments(problem, H, cfg=cfg, n_runs=cfg.n_runs, master_seed=cfg.master_seed)
    else:
        run_results = []
        print('RUN_EXPERIMENTS=False: se omite la ejecucion por ahora.')
    
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
    if df_front_obj.empty:
        return
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

    df_norm = df_par[[algo_col, init_col] + display_cols].copy()
    for c in display_cols:
        cmin, cmax = df_norm[c].min(), df_norm[c].max()
        if cmax > cmin:
            df_norm[c] = (df_norm[c] - cmin) / (cmax - cmin)
        else:
            df_norm[c] = 0.0

    style_map = {'random': '--', 'greedy_hybrid': '-', 'greedy_pure': '-.', 'greedy': '-'}
    color_map = {'random': '#ff7f0e', 'greedy_hybrid': '#1f77b4', 'greedy_pure': '#2ca02c', 'greedy': '#1f77b4'}

    algorithms = sorted(df_norm[algo_col].dropna().astype(str).unique().tolist())
    init_values = sorted(df_norm[init_col].dropna().astype(str).unique().tolist())

    ncols = 2
    nrows = int(np.ceil(len(algorithms) / ncols)) if len(algorithms) > 0 else 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4.8 * nrows), sharey=True)
    axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    for ax in axes[len(algorithms):]:
        ax.axis('off')

    for idx, algorithm_name in enumerate(algorithms):
        ax = axes[idx]
        df_algo = df_norm[df_norm[algo_col].astype(str) == str(algorithm_name)].copy()
        if df_algo.empty:
            ax.axis('off')
            continue

        for init_name in init_values:
            df_sub = df_algo[df_algo[init_col].astype(str) == str(init_name)].copy()
            if df_sub.empty:
                continue

            sample_n = min(60, len(df_sub))
            df_plot = df_sub.sample(sample_n, random_state=seed) if len(df_sub) > sample_n else df_sub

            c = color_map.get(str(init_name), '#1f77b4')
            s = style_map.get(str(init_name), '-')
            for i in range(len(df_plot)):
                row = df_plot.iloc[i]
                ax.plot(display_cols, row[display_cols].values, alpha=0.18, color=c, linestyle=s)

        ax.set_title(str(algorithm_name), fontsize=12, fontweight='bold')
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel('Valor normalizado (0-1)')
        ax.grid(True, axis='y', linestyle='--', alpha=0.35)
        ax.tick_params(axis='x', rotation=25)

    from matplotlib.lines import Line2D
    handles = []
    for init_name in init_values:
        handles.append(Line2D(
            [0], [0],
            color=color_map.get(str(init_name), '#1f77b4'),
            linestyle=style_map.get(str(init_name), '-'),
            linewidth=2,
            label=str(init_name),
        ))

    fig.suptitle('Coordenadas Paralelas (Objetivos reales normalizados) - Comparativa global', fontsize=16, fontweight='bold')
    fig.legend(handles=handles, title='Inicialización', loc='upper center', ncol=min(len(handles), 4), frameon=True)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    os.makedirs(folders['frentes'], exist_ok=True)
    save_report_figure(
        os.path.join(folders['frentes'], f'coordenadas_paralelas_pareto_{mode_tag}.png'),
        dpi=dpi,
        tight=True,
        fig=fig,
    )
    plt.close()

def generate_final_reports(run_results, PAIR_IDX, pair_hamming, cfg):
    """Agrupa los resultados crudos y calcula las métricas finales."""
    EXECUTION_MODE = cfg.execution_mode
    FOLDERS = cfg.folders
    N_SNPS = cfg.n_snps
    REPORT_DPI = cfg.report_plot_dpi
    REPORT_HEARTBEAT_SEC = cfg.report_heartbeat_sec
    # START_TIME movido al bloque __main__ para capturar todo el pipeline
    print(f"  ⚙️  Configuración de reportes: DPI={REPORT_DPI} | modo=secuencial | heartbeat={REPORT_HEARTBEAT_SEC}s")

    if len(run_results) > 0:
        mode_tag_local = EXECUTION_MODE
    
        df_exec = pd.DataFrame(
            [
                {
                    'algorithm': rr.algorithm,
                    'init': rr.init,
                    'run': rr.run,
                    'elapsed_sec': rr.elapsed_sec,
                    'n_solutions_final_front': len(rr.F_final) if rr.F_final is not None else 0,
                }
                for rr in run_results
            ]
        )
    
        df_exec['config'] = df_exec['algorithm'].astype(str) + '-' + df_exec['init'].astype(str)
        order = (
            df_exec[['algorithm', 'init', 'config']]
            .drop_duplicates()
            .sort_values(['algorithm', 'init'])
            ['config']
            .tolist()
        )
    
        sns.set_theme(style='whitegrid')
    
        print_subsection("Análisis de Rendimiento y Tiempo", icon="⏱️")
        t_block = time.time()
        _plot_execution_time_boxplot(df_exec, order, FOLDERS, mode_tag_local, dpi=REPORT_DPI)
        _plot_execution_time_mean_std(df_exec, order, FOLDERS, mode_tag_local, dpi=REPORT_DPI)
        _plot_front_size_boxplot(df_exec, order, FOLDERS, mode_tag_local, dpi=REPORT_DPI)
        _plot_time_vs_frontsize_scatter(df_exec, FOLDERS, mode_tag_local, dpi=REPORT_DPI)
        print(f"      • Tiempo bloque 'Análisis de Rendimiento y Tiempo': {time.time() - t_block:.1f}s")
    else:
        print('No hay resultados (run_results vacío). Activa RUN_EXPERIMENTS o carga resultados previos.')
    
    if len(run_results) > 0:
        t_block = time.time()
        print_subsection("Procesamiento de Métricas (con progreso)", icon="🧮")
        df_runs, ideal_global, nadir_global = evaluate_final_metrics(run_results, heartbeat_sec=REPORT_HEARTBEAT_SEC)
        df_gen_runs = build_generation_metrics(
            run_results,
            ideal_global,
            nadir_global,
            heartbeat_sec=REPORT_HEARTBEAT_SEC
        )
        print(f"      • Tiempo bloque 'Procesamiento de Métricas': {time.time() - t_block:.1f}s")
    
        print_subsection("Trazabilidad y Exportación de Datos (CSV)", icon="📊")
        csv_dir = FOLDERS.get('csv', os.path.join(FOLDERS['ejecuciones'], 'csv'))
        os.makedirs(csv_dir, exist_ok=True)
        
        runs_csv_path = os.path.join(csv_dir, f"resultados_detallados_{EXECUTION_MODE}.csv")
        df_runs.to_csv(runs_csv_path, index=False)
        print_saved_plot(runs_csv_path, "Resultados detallados por ejecución")

        if df_gen_runs is not None and not df_gen_runs.empty:
            gen_csv_path = os.path.join(csv_dir, f"historico_generacional_{EXECUTION_MODE}.csv")
            df_gen_runs.to_csv(gen_csv_path, index=False)
            print_saved_plot(gen_csv_path, "Historial evolutivo generacional")
            print(f"      • Registros en historial: {df_gen_runs.shape[0]}")

        print_subsection("Puntos Críticos del Espacio de Objetivos", icon="📍")
        print(f'      • Punto Ideal Global (mejor): {ideal_global}')
        print(f'      • Punto Nadir Global (peor):  {nadir_global}')
        
        if df_gen_runs is None:
            print('      • ⚠️ Histórico generacional no disponible.')
    
        mode_tag_local = EXECUTION_MODE

        runs_para_frentes = _seleccionar_runs_representativos_por_hv(run_results, df_runs)
        if runs_para_frentes:
            print(f"      • Frentes estilo paper: {len(runs_para_frentes)} runs representativos (mediana HV por algoritmo/init).")
        else:
            runs_para_frentes = list(run_results)
            print("      • ⚠️ No fue posible seleccionar runs por mediana HV; se usan todos los runs para frentes.")

        rows_obj = []
        for rr in runs_para_frentes:
            if rr.F_final is None or len(rr.F_final) == 0:
                continue
            for row in rr.F_final:
                rows_obj.append({
                    'algorithm': rr.algorithm,
                    'init': rr.init,
                    'f1_compactness': float(row[0]),
                    'f2_neg_tolerance': float(row[1]),
                    'f3_neg_hamming_avg': float(row[2]),
                    'f4_balance_var': float(row[3])
                })
        df_front_obj = pd.DataFrame(rows_obj)
    else:
        df_runs = pd.DataFrame()
        df_gen_runs = pd.DataFrame()
        df_front_obj = pd.DataFrame()
        print("No hay resultados para procesar.")

    if 'df_front_obj' in locals() and not df_front_obj.empty:
        print_subsection("Distribución y Correlación de Frentes de Pareto", icon="🔍")
        t_block = time.time()
        _plot_all_pareto_fronts(df_front_obj, FOLDERS, mode_tag_local, dpi=REPORT_DPI)
        _plot_objective_correlation(df_front_obj, FOLDERS, mode_tag_local, dpi=REPORT_DPI)
        _plot_parallel_coordinates(df_front_obj, cfg.master_seed, FOLDERS, mode_tag_local, dpi=REPORT_DPI)
        print(f"      • Tiempo bloque 'Frentes de Pareto': {time.time() - t_block:.1f}s")
    else:
        print('      • No hay datos suficientes para generar coordenadas paralelas (df_front_obj vacío).')
    df_summary = aggregate_results(df_runs) if not df_runs.empty else pd.DataFrame()

    if not df_summary.empty:
        mode_tag_local = EXECUTION_MODE
    
        mean_cols = [c for c in df_summary.columns if c.endswith('_mean')]
        heat_df = df_summary[['algorithm', 'init'] + mean_cols].copy()
        heat_df['method'] = heat_df['algorithm'] + '-' + heat_df['init']
    if 'df_gen_runs' in locals() and df_gen_runs is not None and not df_gen_runs.empty:
        print_subsection("Análisis de Convergencia Progresiva", icon="🔄")
        mode_tag_local = EXECUTION_MODE
        t_block = time.time()
        _ = plot_metricas_generacionales(df_gen_runs, out_dir=FOLDERS['metricas'], mode_tag=mode_tag_local, dpi=REPORT_DPI)
        print(f"      • Tiempo bloque 'Convergencia Progresiva': {time.time() - t_block:.1f}s")
    else:
        print('      • ⚠️ Histórico generacional no disponible o vacío.')
    if 'heat_df' in locals() and not heat_df.empty:
        print_subsection("Resumen Estadístico de Métricas", icon="📊")
        metrics_to_plot = [c for c in heat_df.columns if c.endswith('_mean')]
        exclude = ['elapsed_sec_mean', 'n_solutions_final_front_mean', 'n_solutions_mean']
        metrics_to_plot = [m for m in metrics_to_plot if m not in exclude]
    
        if len(metrics_to_plot) == 0:
            print('      • ⚠️ No hay columnas *_mean adecuadas para el heatmap.')
        else:
            heat_df_plot = heat_df.set_index('method')[metrics_to_plot].copy()
    
            clean_cols = [c.replace('_mean', '') for c in metrics_to_plot]
            heat_df_plot.columns = clean_cols
    
            heat_norm_better = heat_df_plot.copy()
            higher_is_better = ['Hypervolume', 'Range', 'MaxToleranceRate', 'AvgToleranceRate']
            lower_is_better = ['SumMin', 'MinSum', 'AvgHammingDistance', 'AvgHamming']
    
            for col in clean_cols:
                c_min = heat_df_plot[col].min()
                c_max = heat_df_plot[col].max()
                c_range = c_max - c_min
                if c_range == 0:
                    heat_norm_better[col] = 1.0
                    continue
    
                if any(k in col for k in higher_is_better):
                    heat_norm_better[col] = (heat_df_plot[col] - c_min) / c_range
                elif any(k in col for k in lower_is_better):
                    heat_norm_better[col] = (c_max - heat_df_plot[col]) / c_range
                else:
                    heat_norm_better[col] = (heat_df_plot[col] - c_min) / c_range
    
            plt.figure(figsize=(14, 8))
            sns.heatmap(
                heat_norm_better,
                annot=heat_df_plot,
                fmt='.3f',
                cmap='RdYlGn',
                linewidths=0.5,
                cbar_kws={'label': 'Puntuación relativa (0=Peor, 1=Mejor)'}
            )
    
            plt.title('Comparativa de Algoritmos: Rojo (Peor) vs Verde (Mejor)', fontsize=15, pad=20)
            plt.ylabel('Configuración (Algoritmo - Inicialización)', fontsize=12)
            plt.xlabel('Métrica de Evaluación', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
    
            tag = mode_tag_local if 'mode_tag_local' in locals() else EXECUTION_MODE
            save_path = os.path.join(FOLDERS['heatmaps'], f"heatmap_comparativa_{tag}.png")
            save_report_figure(save_path, dpi=REPORT_DPI, tight=True)
            print_saved_plot(save_path, f"Mapa de calor comparativo (Benchmark)")
            plt.close()
    else:
        print('      • ⚠️ No hay datos suficientes (heat_df) para generar el heatmap.')
    if 'heat_df' in locals() and not heat_df.empty:
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
        plt.title('Ranking global por suma de posiciones (métricas principales)')
        plt.xlabel('Método')
        plt.ylabel('Suma de posiciones (menor mejor)')
        plt.xticks(rotation=25, ha='right')
        plt.tight_layout()
        save_path = os.path.join(FOLDERS['rankings'], f'ranking_global_total_{mode_tag_local}.png')
        save_report_figure(save_path, dpi=REPORT_DPI)
        print_saved_plot(save_path, "Gráfico de Ranking Global")
        plt.close()
    else:
        print('      • ⚠️ No hay datos suficientes para generar el ranking global.')
    if 'df_runs' not in locals() or df_runs.empty:
        print('      • ⚠️ No hay resultados en df_runs para generar comparativas estadísticas.')
    else:
        print_subsection("Síntesis Estadística Comparativa", icon="🎻")
        mode_tag_local = EXECUTION_MODE
        plot_metrics = ['Range', 'SumMin', 'MinSum', 'MaxToleranceRate', 'AvgToleranceRate', 'AvgHammingDistance', 'Hypervolume']
        available_metrics = [m for m in plot_metrics if m in df_runs.columns]
    
        if not available_metrics:
            print('      • ⚠️ No hay métricas disponibles para graficar.')
        else:
            df_plot = df_runs.copy()
            df_plot['config'] = df_plot['algorithm'].astype(str) + ' - ' + df_plot['init'].astype(str)
    
            print(f"\n    📦 \033[1mResumen Global (Boxplots)\033[0m")
            ncols = 3
            nrows = int(np.ceil(len(available_metrics) / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows))
            axes = np.atleast_1d(axes).ravel()
    
            for ax, metric in zip(axes, available_metrics):
                sns.boxplot(data=df_plot, x='config', y=metric, ax=ax, color='#9ecae1')
                sns.stripplot(data=df_plot, x='config', y=metric, ax=ax, color='black', alpha=0.45, size=4)
                ax.set_title(f'Boxplot - {metric}')
                ax.set_xlabel('Configuración')
                ax.tick_params(axis='x', rotation=30)
    
            for ax in axes[len(available_metrics):]:
                ax.axis('off')
    
            fig.suptitle('Métricas finales (boxplots) por configuración', fontsize=14, fontweight='bold')
            plt.tight_layout(rect=[0, 0, 1, 0.95])
    
            save_path = os.path.join(FOLDERS['boxplots'], f'boxplots_metricas_finales_{mode_tag_local}.png')
            save_report_figure(save_path, dpi=REPORT_DPI)
            print_saved_plot(save_path, "Panel de Boxplots comparativos")
            plt.close()

            # 2. Violin plots individuales
            print(f"\n    🎻 \033[1mDistribuciones Detalladas (Violin Plots)\033[0m")
            for metric in available_metrics:
                plt.figure(figsize=(10, 5))
                sns.violinplot(data=df_plot, x='config', y=metric, inner='quartile', cut=0, color='#9ecae1')
                sns.stripplot(data=df_plot, x='config', y=metric, alpha=0.45, size=4, color='black')
                plt.title(f'Análisis de Distribución - {metric} (Violin)')
                plt.xlabel('Configuración')
                plt.ylabel(metric)
                plt.xticks(rotation=30)
                plt.tight_layout()
    
                save_path = os.path.join(FOLDERS['boxplots'], f'violin_metricas_{metric}_{mode_tag_local}.png')
                save_report_figure(save_path, dpi=REPORT_DPI)
                print_saved_plot(save_path, f"Distribución {metric} (Violin)")
                plt.close()

            # 3. Media +- Std
            print(f"\n    📉 \033[1mAnálisis de Tendencia Central (Media ± Std)\033[0m")
            summary_all = df_plot.groupby('config')[available_metrics].agg(['mean', 'std'])
            for metric in available_metrics:
                summary = pd.DataFrame({
                    'config': summary_all.index,
                    'metric_mean': summary_all[(metric, 'mean')],
                    'metric_std': summary_all[(metric, 'std')],
                })
                summary['metric_std'] = summary['metric_std'].fillna(0.0)
    
                plt.figure(figsize=(10, 5))
                ax = sns.barplot(data=summary, x='config', y='metric_mean', color='cadetblue', errorbar=None)
                ax.errorbar(x=np.arange(len(summary)), y=summary['metric_mean'].to_numpy(),
                            yerr=summary['metric_std'].to_numpy(), fmt='none', c='black', capsize=4)
                plt.title(f'Estadística Descriptiva - {metric} (Media ± Std)')
                plt.xlabel('Configuración')
                plt.ylabel(metric)
                plt.xticks(rotation=30)
                plt.tight_layout()
    
                save_path = os.path.join(FOLDERS['boxplots'], f'media_std_metricas_{metric}_{mode_tag_local}.png')
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
    START_TIME = time.time()
    print_header('PIPELINE TAG SNP')

    # CLI: allow quick override of mode, data source and num blocks
    import argparse
    parser = argparse.ArgumentParser(description='Run TAG-SNP pipeline')
    parser.add_argument('--mode', '-m', choices=['fast', 'medium', 'full'], default='medium', help='Execution mode (fast/medium/full)')
    parser.add_argument('--data-source', '-d', choices=['synthetic', 'hapmap_phase2'], default=None, help='Data source to use')
    args = parser.parse_args()

    # Prepare a temp log to capture full stdout/stderr for the entire pipeline
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

    # swap stdout/stderr to tee
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    sys.stdout = Tee(orig_stdout, log_fh)
    sys.stderr = Tee(orig_stderr, log_fh)

    try:
        # 1. Configuración y carga (allow overriding mode)
        cfg = setup_configuration(execution_mode=args.mode)
        if args.data_source is not None:
            cfg.data_source = args.data_source

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
    
    # 5. Generar Reportes Finales
    print_header('SÍNTESIS Y REPORTES EXPERIMENTALES')
    generate_final_reports(resultados, PAIR_IDX, pair_hamming, cfg)

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
        print_status(f"Terminal log saved: {final_log_path}")
    except Exception as e:
        print_status(f"Could not copy pipeline log to dataset folder: {e}", success=False)

