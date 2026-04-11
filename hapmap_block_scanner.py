import numpy as np
import pandas as pd
import os
import json
from tfg_tagsnp_pymoo import _ensure_hapmap_phase2_files, _detectar_bloques_ld

# ─────────────────────────────────────────────────────────────────────────────
# Parámetros — deben mantenerse idénticos a los usados en _plot_ld_blocks
# ─────────────────────────────────────────────────────────────────────────────
TAMANO_VENTANA     = 1032   # Número de SNPs por ventana de análisis
PASO_ESCANER       = 50     # Paso entre inicios de ventana consecutivos
VENTANA_SUAVIZADO  = 11     # Media móvil para suprimir ruido genético
UMBRAL_FRONTERA    = 0.10   # r² suavizado por debajo del cual se declara frontera
MIN_SNPS_BLOQUE    = 10     # Anchura mínima en SNPs para descartar micro-bloques
MAX_POS_POR_CLAVE  = 200    # Máximo de posiciones almacenadas por clave


def ejecutar_scanner():
    """
    Escanea todas las ventanas de TAMANO_VENTANA SNPs del cromosoma 21 y
    construye el mapa {n_bloques_reales: [lista de posiciones de inicio válidas]}.

    Utiliza exactamente el mismo algoritmo que _detectar_bloques_ld en
    tfg_tagsnp_pymoo.py para garantizar coherencia entre el escáner y
    la visualización en tiempo de ejecución.
    """
    hapmap_dir = os.path.join("data", "hapmap_phase2")
    paths = _ensure_hapmap_phase2_files(
        hapmap_dir=hapmap_dir, chr_num=21, pop="CEU", release="r21", auto_download=False
    )

    print("Leyendo Legend...")
    legend = pd.read_csv(paths["legend"], sep=r"\s+", compression="gzip")
    print("Leyendo Phased...")
    phased_raw = pd.read_csv(
        paths["phased"], sep=r"\s+", header=None, compression="gzip",
        dtype=np.int8, engine="python"
    ).to_numpy(dtype=np.int8, copy=False)

    # Orientar como (n_haplotipos, n_snps_total)
    if phased_raw.shape[0] == legend.shape[0]:
        H = phased_raw.T
    elif phased_raw.shape[1] == legend.shape[0]:
        H = phased_raw
    else:
        raise ValueError("Inconsistencia en dimensiones phased/legend.")

    n_hap, n_snps_total = H.shape
    print(f"Matriz completa: {n_hap} haplotipos, {n_snps_total} SNPs")

    n_ventanas = len(range(0, n_snps_total - TAMANO_VENTANA, PASO_ESCANER))
    print(
        f"Escaneando {n_ventanas} ventanas de {TAMANO_VENTANA} SNPs "
        f"(paso={PASO_ESCANER}, suavizado={VENTANA_SUAVIZADO} SNPs, "
        f"umbral r²={UMBRAL_FRONTERA}, mín. bloque={MIN_SNPS_BLOQUE} SNPs)..."
    )

    mapa_bloques = {}

    for inicio in range(0, n_snps_total - TAMANO_VENTANA, PASO_ESCANER):
        ventana   = H[:, inicio:inicio + TAMANO_VENTANA]
        segmentos = _detectar_bloques_ld(
            ventana,
            ventana_suavizado=VENTANA_SUAVIZADO,
            umbral=UMBRAL_FRONTERA,
            min_snps_bloque=MIN_SNPS_BLOQUE
        )
        n_bloques = len(segmentos)
        clave     = str(n_bloques)

        if clave not in mapa_bloques:
            mapa_bloques[clave] = []
        if len(mapa_bloques[clave]) < MAX_POS_POR_CLAVE:
            mapa_bloques[clave].append(inicio)

    # Guardar JSON
    archivo_salida = os.path.join(hapmap_dir, "block_map_1032.json")
    with open(archivo_salida, "w") as f:
        json.dump(mapa_bloques, f, indent=4)

    print(f"\nMapa guardado en: {archivo_salida}")
    claves_ord = sorted([int(k) for k in mapa_bloques.keys()])
    for k in claves_ord:
        print(f"  Bloques={k:>4d}: {len(mapa_bloques[str(k)]):>4d} posiciones válidas")

    print(f"\nRango de bloques detectados: [{min(claves_ord)}, {max(claves_ord)}]")


if __name__ == '__main__':
    ejecutar_scanner()
