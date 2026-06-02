"""
Módulo de Decisión Multi-Criterio (decision.py)
------------------------------------------------
Post-procesamiento MCDM sobre frentes de Pareto convergidos.
Implementa tres estrategias de selección nativas de pymoo:
1. Compromise Programming (ASF)
2. Pseudo-Weights
3. High Trade-off Points (Knee Points)

Flujo:
    1. Decodifica objetivos a escala real.
    2. Filtra soluciones factibles (min_cobertura >= 1).
    3. Normaliza al rango [0, 1] con orientación de minimización.
    4. Aplica las estrategias MCDM.
    5. Genera visualizaciones (Scatter con destacados, Radar) y CSV.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
from typing import List, Tuple, Optional, Dict

from pymoo.decomposition.asf import ASF
from pymoo.mcdm.pseudo_weights import PseudoWeights
from pymoo.mcdm.high_tradeoff import HighTradeoffPoints

from snp_tag.engine.metrics import decodificar_objetivos_reales
from snp_tag.utils.terminal import imprimir_grafico_guardado


# ---------------------------------------------------------------------------
# Constantes del módulo
# ---------------------------------------------------------------------------

_NOMBRES_OBJETIVOS = ['Compacidad', 'Tolerancia', 'Hamming', 'Balance']

_ETIQUETAS_EJES = {
    'Compacidad': 'Compacidad (Nº Tag SNPs)',
    'Tolerancia': 'Tolerancia',
    'Hamming': 'Distancia Hamming Promedio',
    'Balance': 'Varianza (Balance)',
}

# Objetivos donde mayor es mejor (se invierten para normalización a minimización)
_HIGHER_IS_BETTER = {1, 2}  # Tolerancia, Hamming

# Proyecciones 2D canónicas (consistentes con fronts.py)
_PROYECCIONES = [
    ('Compacidad', 'Tolerancia', '(a) Compacidad vs. Tolerancia'),
    ('Tolerancia', 'Hamming', '(b) Tolerancia vs. Hamming'),
    ('Hamming', 'Balance', '(c) Hamming vs. Balance'),
    ('Compacidad', 'Balance', '(d) Compacidad vs. Balance'),
]

# Umbral mínimo de soluciones para que el análisis MCDM sea significativo
_MIN_SOLUCIONES = 5


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _decodificar_y_filtrar(df_fronts: pd.DataFrame,
                           modo_transformacion: str) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Decodifica objetivos a escala real y filtra soluciones factibles.

    Returns:
        (df_factible, F_real)  donde F_real tiene columnas
        [Compacidad, Tolerancia, Hamming, Balance].
    """
    cols_F = [
        'f1_compactness', 'f2_transformed_tolerance',
        'f3_transformed_hamming_avg', 'f4_balance_var',
    ]
    F_raw = df_fronts[cols_F].to_numpy(dtype=float)

    reales = decodificar_objetivos_reales(F_raw, modo_transformacion)
    F_real = np.column_stack([
        reales['compacidad'],
        reales['tolerancia_real'],
        reales['hamming_prom_real'],
        reales['balance_var'],
    ])

    mask = reales['min_cobertura'] >= 1.0 - 1e-9
    return df_fronts[mask].reset_index(drop=True), F_real[mask]


def _normalizar_minimizacion(F_real: np.ndarray) -> np.ndarray:
    """
    Normaliza objetivos a [0, 1] orientados a minimización.

    Compacidad (col 0): menor es mejor  → normalización directa.
    Tolerancia (col 1): mayor es mejor  → se invierte.
    Hamming    (col 2): mayor es mejor  → se invierte.
    Balance    (col 3): menor es mejor  → normalización directa.
    """
    F_norm = F_real.copy()
    for j in range(F_norm.shape[1]):
        fmin, fmax = F_norm[:, j].min(), F_norm[:, j].max()
        rango = fmax - fmin
        if rango > 0:
            F_norm[:, j] = (F_norm[:, j] - fmin) / rango
        else:
            F_norm[:, j] = 0.0

    # Invertir objetivos de maximización
    for j in _HIGHER_IS_BETTER:
        F_norm[:, j] = 1.0 - F_norm[:, j]

    return F_norm


def _detectar_knee_points(F_norm: np.ndarray) -> np.ndarray:
    """Detecta puntos de alto trade-off."""
    try:
        dm = HighTradeoffPoints()
        indices = dm(F_norm)
        if indices is None:
            return np.array([], dtype=int)
        return np.asarray(indices, dtype=int)
    except Exception:
        return np.array([], dtype=int)


# ---------------------------------------------------------------------------
# Visualización y Terminal
# ---------------------------------------------------------------------------

def _imprimir_caja_info(titulo: str, lineas: List[str], espacios: str = "      ") -> None:
    """Imprime una caja ASCII con información formateada para la terminal."""
    # Ancho total fijo para asegurar alineación perfecta (caracteres totales)
    ancho_total = 72
    
    # Línea superior: ┌─ Titulo ───┐
    # ┌─ (2) + ' ' (1) + len(titulo) + ' ' (1) + ┐ (1) = 5 fijos
    num_guiones = ancho_total - len(titulo) - 5
    superior = f"┌─ {titulo} " + "─" * max(0, num_guiones) + "┐"
    print(f"{espacios}{superior}")
    
    # Líneas de contenido: │  Texto  │
    # │  (3) + '  │' (3) = 6 fijos
    ancho_texto = ancho_total - 6
    for linea in lineas:
        if len(linea) > ancho_texto:
            linea = linea[:ancho_texto - 3] + "..."
        print(f"{espacios}│  {linea.ljust(ancho_texto)}  │")
        
    # Línea inferior: └────────────┘
    # └ (1) + ┘ (1) = 2 fijos
    inferior = "└" + "─" * (ancho_total - 2) + "┘"
    print(f"{espacios}{inferior}")


def _graficar_scatter_mcdm(
    F_real: np.ndarray,
    destacados: Dict[str, Tuple[np.ndarray, str, str, str]], 
    titulo_global: str,
    ruta_salida: str,
    dpi: int = 300,
) -> None:
    """
    Genera un panel 2×2 con las selecciones MCDM destacadas.
    destacados: { 'Nombre_Criterio': (indices, color, marker, label) }
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle(titulo_global, fontsize=18, weight='bold')

    for ax, (x_col, y_col, subtitulo) in zip(axes.ravel(), _PROYECCIONES):
        idx_x = _NOMBRES_OBJETIVOS.index(x_col)
        idx_y = _NOMBRES_OBJETIVOS.index(y_col)

        # Fondo: todas las soluciones
        ax.scatter(
            F_real[:, idx_x], F_real[:, idx_y],
            s=30, alpha=0.25, color='#6c757d', edgecolor='none',
            zorder=2,
        )
        
        # Puntos MCDM destacados
        for crit, (indices, color, marker, label) in destacados.items():
            if len(indices) > 0:
                ax.scatter(
                    F_real[indices, idx_x], F_real[indices, idx_y],
                    s=140, alpha=0.9, color=color, edgecolor='white',
                    linewidth=1.2, marker=marker, zorder=5,
                )

        ax.set_title(subtitulo, fontsize=14, weight='semibold')
        ax.set_xlabel(_ETIQUETAS_EJES[x_col], fontsize=12)
        ax.set_ylabel(_ETIQUETAS_EJES[y_col], fontsize=12)
        ax.grid(True, alpha=0.15)

    # Leyenda global
    handles = [mpatches.Patch(color='#6c757d', alpha=0.4, label='Frente Pareto')]
    for crit, (indices, color, marker, label) in destacados.items():
        if len(indices) > 0:
            handles.append(
                plt.Line2D([0], [0], marker=marker, color='w', markerfacecolor=color,
                           markersize=11, label=label)
            )
            
    fig.legend(handles=handles, loc='lower center', ncol=len(handles), fontsize=13,
               frameon=True, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.savefig(ruta_salida, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def _graficar_radar_pseudo_pesos(
    valores_pw: np.ndarray,
    pesos_obj: np.ndarray,
    titulo: str,
    ruta_salida: str,
    dpi: int = 300
) -> None:
    """Radar chart comparando los pesos implícitos con las preferencias del usuario."""
    etiquetas = _NOMBRES_OBJETIVOS
    angles = np.linspace(0, 2 * np.pi, len(etiquetas), endpoint=False).tolist()
    
    val = np.concatenate((valores_pw, [valores_pw[0]]))
    ref = np.concatenate((pesos_obj, [pesos_obj[0]]))
    ang = angles + [angles[0]]
    
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    
    # Preferencia del usuario
    ax.plot(ang, ref, color='#e41a1c', linewidth=2, linestyle='--', label='Preferencias Usuario')
    ax.fill(ang, ref, color='#e41a1c', alpha=0.1)
    
    # Pesos implícitos de la solución elegida
    ax.plot(ang, val, color='#4daf4a', linewidth=2.5, label='Pesos Implícitos (Mejor Solución)')
    ax.fill(ang, val, color='#4daf4a', alpha=0.25)
    
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles), etiquetas, fontsize=12)
    
    max_val = max(np.max(val), np.max(ref))
    ax.set_ylim(0, max_val + 0.05)
    
    ax.set_title(titulo, weight='bold', size=15, pad=25)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=11)
    
    plt.tight_layout()
    fig.savefig(ruta_salida, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def _graficar_radar_mcdm(
    F_norm: np.ndarray,
    destacados: Dict[str, Tuple[np.ndarray, str, str, str]],
    titulo: str,
    ruta_salida: str,
    dpi: int = 300
) -> None:
    """Genera un Radar plot superponiendo las soluciones MCDM destacadas."""
    etiquetas = _NOMBRES_OBJETIVOS
    angles = np.linspace(0, 2 * np.pi, len(etiquetas), endpoint=False).tolist()
    ang = angles + [angles[0]]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for crit, (indices, color, marker, label) in destacados.items():
        if len(indices) > 0:
            idx = indices[0]
            # Invertir para que mayor área = mejor (1.0 = ideal, 0.0 = nadir)
            val = 1.0 - F_norm[idx].copy()
            val = np.concatenate((val, [val[0]]))
            ax.plot(ang, val, color=color, linewidth=2.5, marker=marker, markersize=8, label=label)
            ax.fill(ang, val, color=color, alpha=0.15)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles), etiquetas, fontsize=12)
    ax.set_ylim(0, 1.05)
    
    ax.set_title(f"{titulo}\n(Mayor área = Mejor rendimiento)", weight='bold', size=15, pad=25)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=11)

    plt.tight_layout()
    fig.savefig(ruta_salida, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def _graficar_petal_mcdm(
    F_norm: np.ndarray,
    destacados: Dict[str, Tuple[np.ndarray, str, str, str]],
    titulo: str,
    ruta_salida: str,
    dpi: int = 300
) -> None:
    """Genera un diagrama de pétalos (Petal diagram) para las soluciones MCDM."""
    valid_destacados = {k: v for k, v in destacados.items() if len(v[0]) > 0}
    n_plots = len(valid_destacados)
    if n_plots == 0:
        return

    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 6), subplot_kw=dict(polar=True))
    if n_plots == 1:
        axes = [axes]
    
    fig.suptitle(f"{titulo}\n(Pétalos más grandes = Mejor rendimiento)", fontsize=16, weight='bold', y=1.02)
    etiquetas = _NOMBRES_OBJETIVOS
    angles = np.linspace(0, 2 * np.pi, len(etiquetas), endpoint=False).tolist()

    # Colores base para cada métrica (pétalo)
    petal_colors = ['#3498db', '#e67e22', '#2ecc71', '#9b59b6']
    width = 2 * np.pi / len(etiquetas) * 0.85 

    for ax, (crit, (indices, color, marker, label)) in zip(axes, valid_destacados.items()):
        idx = indices[0]
        # Invertir para que pétalo grande = mejor
        val = 1.0 - F_norm[idx].copy()
        
        bars = ax.bar(angles, val, width=width, bottom=0.0, alpha=0.85, edgecolor='black', linewidth=1.2)
        for bar, c in zip(bars, petal_colors):
            bar.set_facecolor(c)
            
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_thetagrids(np.degrees(angles), etiquetas, fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.set_title(label, weight='bold', size=13, pad=15, color=color)

    plt.tight_layout()
    fig.savefig(ruta_salida, dpi=dpi, bbox_inches='tight')
    plt.close(fig)





# ---------------------------------------------------------------------------
# Función principal del módulo
# ---------------------------------------------------------------------------

def analizar_decision_mcdm(
    df_fronts: pd.DataFrame,
    dir_salida: str,
    etiqueta_modo: str,
    modo_transformacion_objetivos: str = 'neg',
    pesos_usuario: Optional[np.ndarray] = None,
    dpi: int = 300,
    emitir_log: bool = True,
) -> List[Tuple[str, str]]:
    """
    Ejecuta el análisis MCDM (Compromise, Pseudo-Weights, Knee Points).
    Produce análisis global y per-configuración.
    """
    artefactos = []

    if pesos_usuario is None:
        pesos_usuario = np.array([0.25, 0.25, 0.25, 0.25])

    cols_req = {
        'f1_compactness', 'f2_transformed_tolerance',
        'f3_transformed_hamming_avg', 'f4_balance_var',
    }
    if df_fronts.empty or not cols_req.issubset(set(df_fronts.columns)):
        if emitir_log:
            print("      • ⚠️  MCDM omitido: datos de frentes insuficientes o columnas faltantes.")
        return artefactos

    os.makedirs(dir_salida, exist_ok=True)
    espacios = " " * 6
    filas_csv = []

    # =======================================================================
    # 1. ANÁLISIS AGREGADO (GLOBAL)
    # =======================================================================
    df_feas, F_real = _decodificar_y_filtrar(df_fronts, modo_transformacion_objetivos)

    if len(F_real) < _MIN_SOLUCIONES:
        if emitir_log:
            print(f"{espacios}• ⚠️  MCDM agregado omitido: sólo {len(F_real)} soluciones factibles.")
        return artefactos

    F_norm = _normalizar_minimizacion(F_real)
    
    # Extraer selecciones
    I_asf = np.array([ASF().do(F_norm, weights=pesos_usuario).argmin()])
    I_pw_val, pw_mat = PseudoWeights(pesos_usuario).do(F_norm, return_pseudo_weights=True)
    I_pw = np.array([I_pw_val])
    I_knee = _detectar_knee_points(F_norm)
    
    if emitir_log:
        print(f"{espacios}• Soluciones en frente agregado: {len(F_real)}")
        
        # 1. Caja ASF
        idx_asf = I_asf[0]
        row_asf = df_feas.iloc[idx_asf]
        algo_asf = row_asf.get('algorithm', 'N/A')
        init_asf = row_asf.get('init', 'N/A')
        sol_asf = F_real[idx_asf]
        lineas_asf = [
            f"Solución #{idx_asf}: Compacidad={sol_asf[0]:.0f}, Tolerancia={sol_asf[1]:.2f},",
            f"Hamming={sol_asf[2]:.2f}, Balance={sol_asf[3]:.4f}",
            f"Config: {algo_asf} ({init_asf})"
        ]
        _imprimir_caja_info("Compromiso (ASF)", lineas_asf, espacios)
        
        # 2. Caja Knee
        n_knee = len(I_knee)
        if n_knee > 0:
            rangos_knee = F_norm[I_knee].mean(axis=1)
            mejor_knee_local = rangos_knee.argmin()
            idx_knee = I_knee[mejor_knee_local]
            row_knee = df_feas.iloc[idx_knee]
            algo_knee = row_knee.get('algorithm', 'N/A')
            init_knee = row_knee.get('init', 'N/A')
            lineas_knee = [
                f"{n_knee} knee points detectados",
                f"Mejor knee: Solución #{idx_knee} ({algo_knee}, {init_knee})"
            ]
        else:
            lineas_knee = ["0 knee points detectados"]
        _imprimir_caja_info("Punto de Mayor Trade-off (Knee)", lineas_knee, espacios)
        
        # 3. Caja Pseudo-Pesos
        idx_pw = I_pw[0]
        pesos_impl = pw_mat[idx_pw]
        pesos_str = f"[{pesos_impl[0]:.2f}, {pesos_impl[1]:.2f}, {pesos_impl[2]:.2f}, {pesos_impl[3]:.2f}]"
        lineas_pw = [
            f"Solución más afín a preferencias: #{idx_pw}",
            f"Pesos implícitos: {pesos_str}"
        ]
        _imprimir_caja_info("Pseudo-Pesos", lineas_pw, espacios)
        print()

    # Graficar Scatter
    destacados_dict = {
        'Knee': (I_knee, '#e41a1c', 'D', 'Knee Point'),
        'ASF': (I_asf, '#377eb8', '*', 'Compromise (ASF)'),
        'PW': (I_pw, '#4daf4a', 's', 'Pseudo-Weights Match'),
    }
    ruta_scatter = os.path.join(dir_salida, f'mcdm_scatter_agregado_{etiqueta_modo}.png')
    _graficar_scatter_mcdm(F_real, destacados_dict, 'Análisis de Decisión MCDM — Agregado Global', ruta_scatter, dpi)
    artefactos.append((ruta_scatter, "Scatter MCDM (Agregado Global)"))
    if emitir_log:
        imprimir_grafico_guardado(ruta_scatter, "Scatter MCDM (Agregado Global)")

    # Graficar Radar de Objetivos
    ruta_radar_obj = os.path.join(dir_salida, f'mcdm_radar_objetivos_agregado_{etiqueta_modo}.png')
    _graficar_radar_mcdm(F_norm, destacados_dict, 'Rendimiento de Soluciones MCDM (Radar)', ruta_radar_obj, dpi)
    artefactos.append((ruta_radar_obj, "Radar Objetivos MCDM (Agregado)"))
    if emitir_log:
        imprimir_grafico_guardado(ruta_radar_obj, "Radar Objetivos MCDM (Agregado)")

    # Graficar Petal de Objetivos
    ruta_petal_obj = os.path.join(dir_salida, f'mcdm_petal_objetivos_agregado_{etiqueta_modo}.png')
    _graficar_petal_mcdm(F_norm, destacados_dict, 'Rendimiento de Soluciones MCDM (Petal)', ruta_petal_obj, dpi)
    artefactos.append((ruta_petal_obj, "Petal Objetivos MCDM (Agregado)"))
    if emitir_log:
        imprimir_grafico_guardado(ruta_petal_obj, "Petal Objetivos MCDM (Agregado)")

    # Graficar Radar Pseudo-Weights
    ruta_radar = os.path.join(dir_salida, f'mcdm_radar_pseudow_{etiqueta_modo}.png')
    _graficar_radar_pseudo_pesos(
        pw_mat[I_pw[0]], pesos_usuario,
        'Afiniadad de Pseudo-Pesos (Mejor Solución)',
        ruta_radar, dpi
    )
    artefactos.append((ruta_radar, "Radar Pseudo-Pesos (Agregado Global)"))
    if emitir_log:
        imprimir_grafico_guardado(ruta_radar, "Radar Pseudo-Pesos (Agregado Global)")

    # Crear resumen tabular
    def registrar_recomendacion(indices, motivo, is_global=True):
        for idx in indices:
            row = df_feas.iloc[idx].to_dict()
            row['Criterio'] = motivo
            row['Alcance'] = 'Global' if is_global else 'Local'
            for i, obj in enumerate(_NOMBRES_OBJETIVOS):
                row[obj] = F_real[idx, i]
                
            # Agregar al CSV global
            fila_csv = {
                'tipo_decision': motivo,
                'alcance': 'Global' if is_global else 'Local',
                'algorithm': row.get('algorithm', ''),
                'init': row.get('init', ''),
                'run': row.get('run', ''),
                'seed': row.get('seed', '')
            }
            for obj in _NOMBRES_OBJETIVOS:
                fila_csv[obj] = row[obj]
            filas_csv.append(fila_csv)
            
            return row

    registrar_recomendacion(I_asf, 'Compromise (ASF)')
    registrar_recomendacion(I_pw, 'Pseudo-Weights Match')
    if len(I_knee) > 0:
        mejor_knee_idx = I_knee[F_norm[I_knee].mean(axis=1).argmin()]
        registrar_recomendacion([mejor_knee_idx], 'Mejor Knee Point')

    # =======================================================================
    # 2. ANÁLISIS PER-CONFIGURACIÓN
    # =======================================================================
    if 'algorithm' in df_fronts.columns and 'init' in df_fronts.columns:
        configs = sorted(df_fronts.groupby(['algorithm', 'init']).groups.keys())

        if emitir_log and len(configs) > 1:
            print(f"\n{espacios}• Análisis per-configuración ({len(configs)} configuraciones):")

        for algo, init in configs:
            df_cfg = df_fronts[(df_fronts['algorithm'] == algo) & (df_fronts['init'] == init)].copy()
            cfg_str = f"{algo} ({init})"
            file_cfg_str = f"{algo.lower()}_{init}"

            df_cfg_feas, F_cfg_real = _decodificar_y_filtrar(df_cfg, modo_transformacion_objetivos)

            if len(F_cfg_real) < _MIN_SOLUCIONES:
                if emitir_log:
                    print(f"{espacios}    [{cfg_str}] omitido ({len(F_cfg_real)} soluciones factibles)")
                continue

            F_cfg_norm = _normalizar_minimizacion(F_cfg_real)
            
            I_cfg_asf = np.array([ASF().do(F_cfg_norm, weights=pesos_usuario).argmin()])
            I_cfg_pw, _ = PseudoWeights(pesos_usuario).do(F_cfg_norm, return_pseudo_weights=True)
            I_cfg_pw = np.array([I_cfg_pw])
            I_cfg_knee = _detectar_knee_points(F_cfg_norm)

            destacados_cfg = {
                'Knee': (I_cfg_knee, '#e41a1c', 'D', 'Knee Point'),
                'ASF': (I_cfg_asf, '#377eb8', '*', 'Compromise (ASF)'),
                'PW': (I_cfg_pw, '#4daf4a', 's', 'Pseudo-Weights Match'),
            }

            ruta_cfg_scatter = os.path.join(dir_salida, f'mcdm_scatter_{file_cfg_str}_{etiqueta_modo}.png')
            _graficar_scatter_mcdm(F_cfg_real, destacados_cfg, f'MCDM: {cfg_str}', ruta_cfg_scatter, dpi)
            desc_scatter = f"Scatter MCDM {cfg_str}"
            artefactos.append((ruta_cfg_scatter, desc_scatter))
            if emitir_log:
                imprimir_grafico_guardado(ruta_cfg_scatter, desc_scatter)

            # Graficar Radar de Objetivos per-config
            ruta_cfg_radar = os.path.join(dir_salida, f'mcdm_radar_objetivos_{file_cfg_str}_{etiqueta_modo}.png')
            _graficar_radar_mcdm(F_cfg_norm, destacados_cfg, f'Radar Objetivos: {cfg_str}', ruta_cfg_radar, dpi)
            desc_radar = f"Radar MCDM {cfg_str}"
            artefactos.append((ruta_cfg_radar, desc_radar))
            if emitir_log:
                imprimir_grafico_guardado(ruta_cfg_radar, desc_radar)

            # Graficar Petal de Objetivos per-config
            ruta_cfg_petal = os.path.join(dir_salida, f'mcdm_petal_objetivos_{file_cfg_str}_{etiqueta_modo}.png')
            _graficar_petal_mcdm(F_cfg_norm, destacados_cfg, f'Petal Objetivos: {cfg_str}', ruta_cfg_petal, dpi)
            desc_petal = f"Petal MCDM {cfg_str}"
            artefactos.append((ruta_cfg_petal, desc_petal))
            if emitir_log:
                imprimir_grafico_guardado(ruta_cfg_petal, desc_petal)

            # Registrar en CSV
            if len(I_cfg_asf) > 0:
                registrar_recomendacion([I_cfg_asf[0]], 'Compromise (ASF)', is_global=False)
            if len(I_cfg_pw) > 0:
                registrar_recomendacion([I_cfg_pw[0]], 'Pseudo-Weights Match', is_global=False)
            if len(I_cfg_knee) > 0:
                mejor_knee_idx = I_cfg_knee[F_cfg_norm[I_cfg_knee].mean(axis=1).argmin()]
                registrar_recomendacion([mejor_knee_idx], 'Mejor Knee Point', is_global=False)

    # =======================================================================
    # 3. EXPORTACIÓN CSV
    # =======================================================================
    if filas_csv:
        df_csv = pd.DataFrame(filas_csv).drop_duplicates()
        ruta_csv = os.path.join(dir_salida, f'mcdm_recomendaciones_{etiqueta_modo}.csv')
        df_csv.to_csv(ruta_csv, index=False)
        artefactos.append((ruta_csv, "CSV Recomendaciones MCDM"))
        if emitir_log:
            imprimir_grafico_guardado(ruta_csv, "CSV Recomendaciones MCDM")

    return artefactos
