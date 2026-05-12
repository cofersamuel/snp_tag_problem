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
import statsmodels.api as sm
import warnings

from snp_tag.engine.metrics import decodificar_objetivos_reales


def _normalizar_columnas_frentes(df_base: pd.DataFrame) -> pd.DataFrame:
    """Normaliza alias de columnas de frentes al esquema canónico actual."""
    if df_base is None or df_base.empty:
        return df_base

    df = df_base.copy()
    alias_a_canonico = {
        'f2_neg_tolerance': 'f2_transformed_tolerance',
        'f3_neg_hamming_avg': 'f3_transformed_hamming_avg',
    }
    for alias, canonica in alias_a_canonico.items():
        if canonica not in df.columns and alias in df.columns:
            df.rename(columns={alias: canonica}, inplace=True)
    return df


def _anexar_objetivos_reales(df_base: pd.DataFrame, modo_transformacion_objetivos: str = 'neg') -> pd.DataFrame:
    """Añade columnas en escala física para visualización consistente entre modos."""
    df_base = _normalizar_columnas_frentes(df_base)
    requeridas = ['f1_compactness', 'f2_transformed_tolerance', 'f3_transformed_hamming_avg', 'f4_balance_var']
    faltantes = [c for c in requeridas if c not in df_base.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas transformadas para graficar: {faltantes}")

    objetivos_reales = decodificar_objetivos_reales(
        df_base[requeridas].to_numpy(dtype=float),
        modo_transformacion_objetivos=modo_transformacion_objetivos,
    )

    df = df_base.copy()
    df['Compacidad'] = objetivos_reales['compacidad']
    df['Tolerancia'] = objetivos_reales['tolerancia_real']
    df['Hamming'] = objetivos_reales['hamming_prom_real']
    df['Balance'] = objetivos_reales['balance_var']
    df['min_cobertura'] = objetivos_reales['min_cobertura']
    return df[df['min_cobertura'] >= 1.0].copy()

def graficar_frentes_pareto(df_datos: pd.DataFrame, nombre_algoritmo: str,
                            nombre_init: Optional[str] = None,
                            carpetas: Optional[Dict] = None,
                            etiqueta_modo: Optional[str] = None,
                            dpi: int = 300,
                            emitir_log: bool = True,
                            limites_ejes: Optional[Dict[str, Tuple[float, float]]] = None,
                            modo_transformacion_objetivos: str = 'neg') -> List[Tuple[str, str]]:
    """
    Representa el frente de Pareto mediante cuatro proyecciones 2D entre objetivos.
    Permite estandarizar los límites de los ejes si se proporciona 'limites_ejes'.
    """
    df_algo = df_datos[df_datos['algorithm'].str.upper() == nombre_algoritmo.upper()].copy()
    if df_algo.empty:
        return []

    artefactos = []

    # Decodificar objetivos a escala real
    df_algo = _anexar_objetivos_reales(
        df_algo,
        modo_transformacion_objetivos=modo_transformacion_objetivos,
    )

    init_col = 'init' if 'init' in df_algo.columns else 'init_type'
    unidades_init = [nombre_init] if nombre_init else sorted(df_algo[init_col].unique())

    colores_init = {
        'random_sparse': '#ff7f0e',
        'random_dense': '#9467bd',
        'greedy_multi': '#2ca02c',
        'greedy_holistic': '#1f77b4',
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
            
            # Línea de tendencia suavizada (LOWESS - Locally Weighted Scatterplot Smoothing)
            datos_ord = df[[x_col, y_col]].sort_values(by=x_col).dropna()
            # Umbral mínimo de puntos y varianza para evitar errores numéricos
            if len(datos_ord) > 12 and datos_ord[x_col].std() > 1e-5:
                try:
                    with warnings.catch_warnings():
                        # Silenciamos advertencias internas de statsmodels por divisiones por cero en datasets pequeños
                        warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*divide.*")
                        z = sm.nonparametric.lowess(datos_ord[y_col], datos_ord[x_col], frac=0.3)
                    ax.plot(z[:, 0], z[:, 1], color='#e41a1c', linewidth=2.5, alpha=0.8, zorder=5)
                except Exception:
                    # Si falla el suavizado estadístico, simplemente no dibujamos la línea
                    pass

            # Aplicar límites estandarizados si están disponibles
            if limites_ejes:
                if x_col in limites_ejes:
                    ax.set_xlim(limites_ejes[x_col])
                if y_col in limites_ejes:
                    ax.set_ylim(limites_ejes[y_col])

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
                                          dpi: int = 300, emitir_log: bool = True,
                                          modo_transformacion_objetivos: str = 'neg') -> List[Tuple[str, str]]:
    """Genera un heatmap de la correlación de objetivos en los frentes de Pareto (Réplica Legacy)."""
    if df_total.empty:
        return []
    
    # Preparar datos en escala real para legibilidad consistente
    df_corr = _anexar_objetivos_reales(
        df_total,
        modo_transformacion_objetivos=modo_transformacion_objetivos,
    )[['Compacidad', 'Tolerancia', 'Hamming', 'Balance']].copy()
    
    # Calcular correlación entre objetivos. Se suprimen advertencias en caso de objetivos constantes.
    with np.errstate(divide='ignore', invalid='ignore'):
        corr_obj = df_corr.corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_obj, annot=True, fmt='.2f', cmap='vlag', center=0, square=True,
                vmin=-1, vmax=1,
                linewidths=.5, cbar_kws={"shrink": .8})
    plt.title('Correlación entre objetivos (Valores Reales)', fontsize=15, pad=20)
    plt.tight_layout()
    
    dir_otros = carpetas.get('frentes_otros', carpetas.get('frentes'))
    ruta = os.path.join(dir_otros, f'correlacion_objetivos_pareto_{etiqueta_modo}.png')
    plt.savefig(ruta, dpi=dpi, bbox_inches='tight')
    if emitir_log:
        from snp_tag.utils.terminal import imprimir_grafico_guardado
        imprimir_grafico_guardado(ruta, "Correlación objetivos Pareto (Valores Reales)")
    plt.close()
    return [(ruta, "Correlación objetivos Pareto (Valores Reales)")]

def graficar_coordenadas_paralelas_pareto(df_total: pd.DataFrame, seed: int, carpetas: Dict, etiqueta_modo: str,
                                          dpi: int = 300, emitir_log: bool = True,
                                          modo_transformacion_objetivos: str = 'neg') -> List[Tuple[str, str]]:
    """Representa frentes en coordenadas paralelas normalizadas, por cada (algo, init) (Réplica Legacy)."""
    if df_total.empty:
        return []
    
    warnings.filterwarnings('ignore')
    df_par = _anexar_objetivos_reales(
        df_total,
        modo_transformacion_objetivos=modo_transformacion_objetivos,
    )
    init_col = 'init'
    algo_col = 'algorithm'

    df_par['Compacidad ($f_1$ min)'] = df_par['Compacidad']
    df_par['Tolerancia Real ($f_2$ max)'] = df_par['Tolerancia']
    df_par['Hamming Real ($f_3$ max)'] = df_par['Hamming']
    df_par['Varianza ($f_4$ min)'] = df_par['Balance']

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

    style_map = {
        'random_sparse': '--',
        'random_dense': ':',
        'greedy_multi': '-.',
        'greedy_holistic': '-',
    }
    color_map = {
        'random_sparse': '#ff7f0e',
        'random_dense': '#9467bd',
        'greedy_multi': '#2ca02c',
        'greedy_holistic': '#1f77b4',
    }

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

def graficar_frentes_pareto_agregados(df_datos: pd.DataFrame, titulo_gen: str, 
                                     nombre_archivo: str,
                                     hue_col: str = 'init',
                                     carpetas: Optional[Dict] = None,
                                     dpi: int = 300,
                                     emitir_log: bool = True,
                                     limites_ejes: Optional[Dict[str, Tuple[float, float]]] = None,
                                     modo_transformacion_objetivos: str = 'neg') -> List[Tuple[str, str]]:
    """
    Genera un único gráfico de Pareto agregado (2x2) combinando múltiples configuraciones.
    Diferencia las categorías mediante colores (hue).
    """
    if df_datos.empty:
        return []

    # Preparar datos para el gráfico
    df = _anexar_objetivos_reales(
        df_datos,
        modo_transformacion_objetivos=modo_transformacion_objetivos,
    )

    etiquetas_ejes = {
        'Compacidad': 'Compacidad (Nº Tag SNPs)',
        'Tolerancia': 'Tolerancia',
        'Hamming': 'Distancia Hamming Promedio',
        'Balance': 'Varianza (Balance)'
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle(titulo_gen, fontsize=18, weight='bold')

    proyecciones = [
        (axes[0, 0], 'Compacidad', 'Tolerancia', '(a) Compacidad vs. Tolerancia'),
        (axes[0, 1], 'Tolerancia', 'Hamming', '(b) Tolerancia vs. Hamming'),
        (axes[1, 0], 'Hamming', 'Balance', '(c) Hamming vs. Balance'),
        (axes[1, 1], 'Compacidad', 'Balance', '(d) Compacidad vs. Balance')
    ]

    # Determinar paleta según el número de categorías
    n_colors = df[hue_col].nunique()
    palette = 'tab10' if n_colors <= 10 else 'husl'

    for ax, x_col, y_col, titulo in proyecciones:
        sns.scatterplot(data=df, x=x_col, y=y_col, ax=ax, hue=hue_col, 
                        s=50, alpha=0.7, palette=palette, edgecolor='w')
        
        # Aplicar límites estandarizados si están disponibles
        if limites_ejes:
            if x_col in limites_ejes: ax.set_xlim(limites_ejes[x_col])
            if y_col in limites_ejes: ax.set_ylim(limites_ejes[y_col])

        ax.set_title(titulo, fontsize=14, weight='semibold')
        ax.set_xlabel(etiquetas_ejes[x_col], fontsize=12)
        ax.set_ylabel(etiquetas_ejes[y_col], fontsize=12)
        ax.get_legend().remove() # Quitamos leyenda individual para poner una global

    # Añadir leyenda única fuera de los subplots
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', title=hue_col.capitalize(), 
               bbox_to_anchor=(1.12, 0.5), fontsize=11, title_fontsize=13)

    plt.tight_layout(rect=[0, 0, 0.98, 0.95])
    
    artefactos = []
    if carpetas:
        dir_frentes = carpetas.get('frentes_pareto', carpetas.get('frentes'))
        ruta = os.path.join(dir_frentes, nombre_archivo)
        fig.savefig(ruta, dpi=dpi, bbox_inches='tight')
        artefactos.append((ruta, f"Frente Agregado: {titulo_gen}"))
        
        if emitir_log:
            from snp_tag.utils.terminal import imprimir_grafico_guardado
            imprimir_grafico_guardado(ruta, f"Frente Agregado: {titulo_gen}")
            
    plt.close(fig)
    return artefactos
