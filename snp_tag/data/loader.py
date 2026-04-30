"""
Módulo de Carga de Datos (loader.py)
-----------------------------------
Implementa la lógica para la adquisición de datasets haplotípicos, ya sea mediante
la carga del bloque histórico de Hinds et al. (2005) o la generación estocástica
de bloques con estructura de desequilibrio de ligamiento (LD).
"""

import os
import numpy as np
import pandas as pd
import json
from snp_tag.config import ConfiguracionExperimento
from snp_tag.utils.terminal import imprimir_estado, imprimir_grafico_guardado
from snp_tag.utils.filesystem import crear_arbol_directorios_dataset


def filtrar_snps_monomorficos(H: np.ndarray):
    """
    Filtra SNPs monomórficos con el criterio de Ting: 0 < suma_columna < n_haplotipos.
    """
    if H.size == 0:
        return H, np.array([], dtype=int)

    n_haplotipos = int(H.shape[0])
    suma_columnas = H.sum(axis=0)
    mascara_polimorficos = (suma_columnas > 0) & (suma_columnas < n_haplotipos)
    indices_utiles = np.where(mascara_polimorficos)[0].astype(int)
    return H[:, mascara_polimorficos], indices_utiles

def generar_bloque_haplotipico_ld(n_haplotipos=40, n_snps=1200, tam_bloque=50, 
                                 prob_flip=0.02, semilla=42, 
                                 dif_min_pares: int = 0, intentos_max: int = 1000):
    """
    Genera una matriz binaria (Haplotipo x SNP) con estructura de bloques LD.
    Garantiza opcionalmente una distancia Hamming mínima entre individuos.
    """
    rng = np.random.default_rng(semilla)
    n_bloques = int(np.ceil(n_snps / tam_bloque))
    patrones_base = rng.integers(0, 2, size=(n_haplotipos, n_bloques), dtype=np.int8)
    X = np.zeros((n_haplotipos, n_snps), dtype=np.int8)
    
    snp_idx = 0
    for b in range(n_bloques):
        restante = n_snps - snp_idx
        ancho = min(tam_bloque, restante)
        lider = patrones_base[:, b].copy()
        for _ in range(ancho):
            col = lider.copy()
            mascara_flip = rng.random(n_haplotipos) < prob_flip
            col[mascara_flip] = 1 - col[mascara_flip]
            X[:, snp_idx] = col
            snp_idx += 1

    if dif_min_pares <= 0:
        return X, 0, 0

    def _distancias_pares(H):
        n = H.shape[0]
        i_idx, j_idx = np.triu_indices(n, k=1)
        diferencias = (H[i_idx] != H[j_idx]).sum(axis=1)
        return diferencias, i_idx, j_idx

    intentos = 0
    H = X.copy()
    diffs, i_idx, j_idx = _distancias_pares(H)
    min_actual = int(diffs.min()) if diffs.size > 0 else 0
    
    while min_actual < dif_min_pares and intentos < intentos_max:
        arg = int(np.argmin(diffs))
        i, j = int(i_idx[arg]), int(j_idx[arg])
        
        # Seleccionar haplotipo a mutar
        media_i = ((H[i] != H).sum(axis=1)).mean()
        media_j = ((H[j] != H).sum(axis=1)).mean()
        hap_a_mutar = i if media_i <= media_j else j
        
        pos_idénticas = np.where(H[i] == H[j])[0]
        if pos_idénticas.size == 0:
            pos = rng.integers(0, n_snps, size=1)
        else:
            necesitados = dif_min_pares - min_actual
            num_flips = min(pos_idénticas.size, max(1, int(np.ceil(necesitados))))
            pos = rng.choice(pos_idénticas, size=num_flips, replace=False)
            
        H[hap_a_mutar, pos] = 1 - H[hap_a_mutar, pos]
        diffs, i_idx, j_idx = _distancias_pares(H)
        min_actual = int(diffs.min()) if diffs.size > 0 else 0
        intentos += 1

    return H, intentos, min_actual

def cargar_bloque_hinds2005(ruta_fichero: str):
    """
    Carga el bloque haplotípico histórico de Hinds et al. (2005) desde el formato Ting 2010.
    Dimensiones: 48 patrones x 1032 SNPs.
    """
    if not os.path.exists(ruta_fichero):
        raise FileNotFoundError(f"Dataset de Hinds 2005 no hallado en: '{ruta_fichero}'")
        
    with open(ruta_fichero) as f:
        filas = [l.strip() for l in f if l.strip()]
    if not filas:
        raise ValueError(f"El archivo '{ruta_fichero}' carece de datos válidos.")
        
    H = np.array([[int(c) for c in fila] for fila in filas], dtype=np.int8)
    n_patrones, n_snps_entrada = H.shape

    H_filtrado, idx_utiles = filtrar_snps_monomorficos(H)
    n_patrones, n_snps = H_filtrado.shape

    snp_ids = [f"snp_{i}" for i in idx_utiles.tolist()]
    posiciones_snp = idx_utiles.copy()
    haplotipo_ids = [f"patron_{i}" for i in range(n_patrones)]
    
    ruta_rel = os.path.relpath(ruta_fichero)
    print(
        "       \033[92m✅  Hinds 2005 cargado y filtrado: "
        f"{n_patrones} patrones alelicos × {n_snps} SNPs polimórficos "
        f"(de {n_snps_entrada} originales) desde '{ruta_rel}'\033[0m"
    )
    return H_filtrado, snp_ids, posiciones_snp, haplotipo_ids

def exportar_dataset(H, snp_ids, posiciones_snp, haplotipo_ids, carpetas, modo_etiqueta):
    """
    Sincroniza la matriz haplotípica y sus metadatos en archivos CSV.
    """
    df = pd.DataFrame(H, index=haplotipo_ids, columns=snp_ids)
    ruta_csv = os.path.join(carpetas['datos'], f'matriz_haplotipos_seleccionada_{modo_etiqueta}.csv')
    df.to_csv(ruta_csv)
    
    df_meta = pd.DataFrame({'snp_id': snp_ids, 'posicion': posiciones_snp})
    ruta_meta = os.path.join(carpetas['datos'], f'metadatos_snps_seleccionados_{modo_etiqueta}.csv')
    df_meta.to_csv(ruta_meta, index=False)
    
    print(f"       \033[92m✅  Dataset exportado: {len(snp_ids)} SNPs x {len(haplotipo_ids)} haplotipos\033[0m")
    imprimir_grafico_guardado(ruta_csv, "Matriz de haplotipos (CSV)")
    imprimir_grafico_guardado(ruta_meta, "Metadatos de SNPs (CSV)")

def cargar_dataset_objetivo(cfg: ConfiguracionExperimento):
    """
    Orquesta la carga o generación del dataset según la configuración global.
    """
    if cfg.origen_datos == "synthetic":
        H, intentos, min_logrado = generar_bloque_haplotipico_ld(
            n_haplotipos=cfg.n_haplotipos, n_snps=cfg.n_snps,
            tam_bloque=cfg.tam_bloque_sintetico, prob_flip=cfg.prob_flip_sintetico, semilla=cfg.semilla_maestra,
            dif_min_pares=cfg.dif_min_pares_sintetico, intentos_max=cfg.intentos_max_sintetico
        )
        
        n_snps_entrada = int(H.shape[1])
        H, idx_utiles = filtrar_snps_monomorficos(H)
        snp_ids = [f"snp_{i}" for i in idx_utiles.tolist()]
        posiciones_snp = idx_utiles.copy()
        haplotipo_ids = [f"hap_{i}" for i in range(cfg.n_haplotipos)]

        print(
            "       \033[92m✅  Synthetic filtrado: "
            f"{H.shape[0]} haplotipos × {H.shape[1]} SNPs polimórficos "
            f"(de {n_snps_entrada} originales)\033[0m"
        )
        
        # Gestión de directorios: solo crear si no fueron configurados por el orquestador
        if not getattr(cfg, 'carpetas', None):
            _, carpetas_locales = crear_arbol_directorios_dataset(cfg, "synthetic")
            cfg.carpetas = carpetas_locales
        
        exportar_dataset(H, snp_ids, posiciones_snp, haplotipo_ids, cfg.carpetas, cfg.modo_ejecucion)
        
        # Guardar metadatos en JSON
        n = H.shape[0]
        dists = []
        for a in range(n):
            for b in range(a+1, n):
                dists.append(int((H[a] != H[b]).sum()))
        dists = np.array(dists) if dists else np.array([0])
        
        metadatos = {
            'min_dist': int(dists.min()),
            'max_dist': int(dists.max()),
            'mean_dist': float(dists.mean()),
            'attempts': int(intentos),
            'achieved_min': int(min_logrado)
        }
        ruta_json = os.path.join(cfg.carpetas['datos'], 'metadata.json')
        with open(ruta_json, 'w') as jf:
            json.dump(metadatos, jf, indent=2)
            
    elif cfg.origen_datos == "hinds2005":
        H, snp_ids, posiciones_snp, haplotipo_ids = cargar_bloque_hinds2005(cfg.ruta_hinds2005)
        cfg.n_haplotipos = int(H.shape[0])
    else:
        raise ValueError(f"Fuente de datos '{cfg.origen_datos}' no reconocida.")
        
    cfg.n_snps = int(H.shape[1])
    cfg.pm = 1.0 / max(1, cfg.n_snps)
    return H, snp_ids, posiciones_snp, haplotipo_ids
