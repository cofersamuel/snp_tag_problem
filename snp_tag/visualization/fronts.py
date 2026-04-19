"""
Módulo de Visualización de Frentes de Pareto (fronts.py)
-------------------------------------------------------
Proporciona herramientas para la representación gráfica de las soluciones
en el espacio de objetivos, incluyendo frentes de Pareto 2D y coordenadas paralelas.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import pandas as pd
from typing import Optional, Dict, List, Tuple
from pandas.plotting import parallel_coordinates
from matplotlib.lines import Line2D
import warnings

def graficar_frentes_pareto(df_datos: pd.DataFrame, nombre_algoritmo: str,
                            nombre_init: Optional[str] = None,
                            carpetas: Optional[Dict] = None,
                            etiqueta_modo: Optional[str] = None,
                            dpi: int = 300,
                            emitir_log: bool = True) -> List[Tuple[str, str]]:
    """
    Representa el frente de Pareto mediante cuatro proyecciones 2D entre objetivos.
    """
    df_algo = df_datos[df_datos['algorithm'].str.upper() == nombre_algoritmo.upper()].copy()
    if df_algo.empty:
        return []

    artefactos = []

    # Renombrar para claridad interna
    df_algo['Compacidad'] = df_algo['f1_compactness']
    df_algo['Tolerancia'] = -df_algo['f2_neg_tolerance']
    df_algo['Hamming'] = -df_algo['f3_neg_hamming_avg']
    df_algo['Balance'] = df_algo['f4_balance_var']

    init_col = 'init' if 'init' in df_algo.columns else 'init_type'
    unidades_init = [nombre_init] if nombre_init else sorted(df_algo[init_col].unique())

    # Paleta de colores consistente
    colores_init = {
        'random': '#ff7f0e', 'greedy_hybrid': '#1f77b4', 'greedy_pure': '#2ca02c'
    }

    etiquetas_ejes = {
        'Compacidad': 'Compacidad (Nº Tag SNPs)',
        'Tolerancia': 'Tolerancia',
        'Hamming': 'Distancia Hamming Promedio',
        'Balance': 'Varianza (Balance)'
    }

    for i_val in unidades_init:
        df = df_algo[df_algo[init_col] == i_val]
        if df.empty: continue
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle(f"Frente de Pareto: {nombre_algoritmo} | {i_val}", fontsize=16, weight='bold')
        
        color = colores_init.get(str(i_val), '#1f77b4')
        proyecciones = [
            (axes[0, 0], 'Compacidad', 'Tolerancia', '(a) Compacidad vs. Tolerancia'),
            (axes[0, 1], 'Tolerancia', 'Hamming', '(b) Tolerancia vs. Hamming'),
            (axes[1, 0], 'Hamming', 'Balance', '(c) Hamming vs. Balance'),
            (axes[1, 1], 'Compacidad', 'Balance', '(d) Compacidad vs. Balance')
        ]
        
        for ax, x_col, y_col, titulo in proyecciones:
            sns.scatterplot(data=df, x=x_col, y=y_col, ax=ax, s=60, alpha=0.8, color=color, edgecolor='w')
            
            # Línea de tendencia (Mediana móvil)
            datos_ord = df[[x_col, y_col]].sort_values(by=x_col).dropna()
            if len(datos_ord) > 4:
                ventana = max(5, int(0.1 * len(datos_ord)))
                y_mediana = datos_ord[y_col].rolling(window=ventana, center=True, min_periods=1).median()
                ax.plot(datos_ord[x_col], y_mediana, color='red', linewidth=2, alpha=0.9)

            ax.set_title(titulo, fontsize=13)
            ax.set_xlabel(etiquetas_ejes[x_col])
            ax.set_ylabel(etiquetas_ejes[y_col])
            
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        if carpetas and etiqueta_modo:
            dir_frentes = carpetas.get('frentes_pareto', carpetas.get('frentes'))
            ruta = os.path.join(dir_frentes, f'frentes_pareto_{nombre_algoritmo.lower()}_{str(i_val).lower()}_{etiqueta_modo}.png')
            fig.savefig(ruta, dpi=dpi, bbox_inches='tight')
            artefactos.append((ruta, f"Frente Pareto {nombre_algoritmo} ({i_val})"))
            if emitir_log:
                from snp_tag.utils.terminal import imprimir_grafico_guardado
                imprimir_grafico_guardado(ruta, f"Frente Pareto {nombre_algoritmo} ({i_val})")
        plt.close(fig)

    return artefactos

def graficar_correlacion_objetivos_pareto(df_total: pd.DataFrame, carpetas: Dict, etiqueta_modo: str,
                                          dpi: int = 300, emitir_log: bool = True) -> List[Tuple[str, str]]:
    """Genera un heatmap de la correlación de objetivos en los frentes de Pareto (Réplica Legacy)."""
    if df_total.empty:
        return []
    
    # Calcular correlación entre objetivos. Se suprimen advertencias en caso de objetivos constantes.
    with np.errstate(divide='ignore', invalid='ignore'):
        corr_obj = df_total[['f1_compactness', 'f2_neg_tolerance', 'f3_neg_hamming_avg', 'f4_balance_var']].corr()
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_obj, annot=True, fmt='.2f', cmap='vlag', center=0)
    plt.title('Correlación entre objetivos en frentes finales')
    plt.tight_layout()
    
    dir_otros = carpetas.get('frentes_others', carpetas.get('frentes'))
    ruta = os.path.join(dir_otros, f'correlacion_objetivos_pareto_{etiqueta_modo}.png')
    plt.savefig(ruta, dpi=dpi, bbox_inches='tight')
    if emitir_log:
        from snp_tag.utils.terminal import imprimir_grafico_guardado
        imprimir_grafico_guardado(ruta, "Correlación objetivos Pareto")
    plt.close()
    return [(ruta, "Correlación objetivos Pareto")]

def graficar_coordenadas_paralelas_pareto(df_total: pd.DataFrame, seed: int, carpetas: Dict, etiqueta_modo: str,
                                          dpi: int = 300, emitir_log: bool = True) -> List[Tuple[str, str]]:
    """Representa frentes en coordenadas paralelas normalizadas, por cada (algo, init) (Réplica Legacy)."""
    if df_total.empty:
        return []
    
    warnings.filterwarnings('ignore')
    df_par = df_total.copy()
    init_col = 'init'
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

    # Normalización global para todos los frentes
    df_norm = df_par[[algo_col, init_col] + display_cols].copy()
    for c in display_cols:
        cmin, cmax = df_norm[c].min(), df_norm[c].max()
        if cmax > cmin:
            df_norm[c] = (df_norm[c] - cmin) / (cmax - cmin)
        else:
            df_norm[c] = 0.0

    style_map = {'random': '--', 'greedy_hybrid': '-', 'greedy_pure': '-.'}
    color_map = {'random': '#ff7f0e', 'greedy_hybrid': '#1f77b4', 'greedy_pure': '#2ca02c'}

    algorithms = sorted(df_norm[algo_col].dropna().astype(str).unique().tolist())
    init_values = sorted(df_norm[init_col].dropna().astype(str).unique().tolist())
    artefactos = []

    for algorithm_name in algorithms:
        for init_name in init_values:
            df_sub = df_norm[(df_norm[algo_col].astype(str) == str(algorithm_name)) & 
                             (df_norm[init_col].astype(str) == str(init_name))].copy()
            if df_sub.empty: continue

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

            handles = [Line2D([0], [0], color=c, linestyle=s, linewidth=2, label=str(init_name))]
            ax.legend(handles=handles, title='Inicialización', loc='upper right', frameon=True)

            fig.suptitle('Coordenadas Paralelas (Objetivos reales normalizados)', fontsize=15, fontweight='bold')
            fig.tight_layout(rect=(0, 0, 1, 0.95))
            
            nombre_archivo = f'coordenadas_paralelas_{algorithm_name}_{init_name}_{etiqueta_modo}.png'
            dir_paralelas = carpetas.get('frentes_paralelas', carpetas.get('frentes'))
            ruta = os.path.join(dir_paralelas, nombre_archivo)
            fig.savefig(ruta, dpi=dpi, bbox_inches='tight')
            artefactos.append((ruta, f"Coordenadas paralelas {algorithm_name} ({init_name})"))
            if emitir_log:
                from snp_tag.utils.terminal import imprimir_grafico_guardado
                imprimir_grafico_guardado(ruta, f"Coordenadas paralelas {algorithm_name} ({init_name})")
            
            plt.close(fig)

    return artefactos
