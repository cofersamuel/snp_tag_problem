"""
Módulo del Motor de Ejecución (runner.py)
----------------------------------------
Orquesta la ejecución paralela y concurrente de los experimentos evolutivos,
gestionando la distribución de carga y la persistencia de resultados.
"""

import time
import hashlib
import concurrent.futures
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.core.callback import Callback

from snp_tag.config import ConfiguracionExperimento
from snp_tag.core.problem import ProblemaTagSNP, evaluar_poblacion_vectorizado
from snp_tag.core.algorithm import fabricar_algoritmo
from snp_tag.utils.terminal import imprimir_subseccion, imprimir_estado
from snp_tag.utils.runtime import calcular_max_workers_paralelo

@dataclass
class ResultadoEjecucion:
    """Contenedor de resultados para una réplica experimental."""
    algoritmo: str
    inicializacion: str
    replica: int
    semilla: int
    tiempo_seg: float
    X_final: np.ndarray
    F_final: np.ndarray
    historial_F: List[np.ndarray]

class CallbackLigero(Callback):
    """Callback optimizado para capturar el historial de objetivos sin saturar la memoria."""
    def __init__(self, H, pair_idx, modo_transformacion_objetivos='neg', modo_evaluacion='absoluta'):
        super().__init__()
        self._H = H
        self._pair_idx = pair_idx
        self._modo_transformacion_objetivos = str(modo_transformacion_objetivos or 'neg')
        self._modo_evaluacion = str(modo_evaluacion or 'absoluta')
        self.historial_F = []

    def notify(self, algoritmo):
        pop = algoritmo.pop
        if pop is None: return
        try:
            X_gen = np.array(pop.get('X'), dtype=bool)
        except:
            return
        if X_gen.size == 0: return
        # Evaluación vectorizada directa para el historial
        from snp_tag.core.problem import evaluar_poblacion_vectorizado
        # Necesitamos la matriz de discrepancia
        prob = algoritmo.problem
        F_gen = evaluar_poblacion_vectorizado(
            X_gen,
            prob.matriz_discrepancia,
            modo_transformacion=self._modo_transformacion_objetivos,
            modo_evaluacion=self._modo_evaluacion,
        )
        self.historial_F.append(F_gen[:, :4].copy())

def _ejecutar_replica_individual(args: Dict[str, Any]) -> ResultadoEjecucion:
    """Punto de entrada para procesos hijos (multiprocessing)."""
    nombre_algo = args['algo']
    nombre_init = args['init']
    num_replica  = args['replica']
    semilla     = args['semilla']
    H           = args['H']
    pair_idx    = args['pair_idx']
    cfg         = args['cfg']
    dirs_ref    = args['dirs_ref']

    problema = ProblemaTagSNP(
        H,
        pair_idx,
        normalizar_busqueda=nombre_algo.startswith('MOEAD_'),
        modo_transformacion_objetivos=cfg.modo_transformacion_objetivos,
        modo_evaluacion=cfg.modo_evaluacion,
    )
    algo = fabricar_algoritmo(problema, H, nombre_algo, nombre_init, cfg, semilla, dirs_ref)
    finalizacion = get_termination('n_gen', cfg.n_generaciones)
    
    cb = CallbackLigero(H, pair_idx, modo_transformacion_objetivos=cfg.modo_transformacion_objetivos, modo_evaluacion=cfg.modo_evaluacion)
    t0 = time.time()
    res = minimize(problema, algo, finalizacion, seed=semilla, verbose=False, 
                   save_history=False, callback=cb)
    delta_t = time.time() - t0

    Xf = np.array(res.X, dtype=bool) if res.X is not None else np.empty((0, problema.n_var), dtype=bool)
    if Xf.shape[0] > 0:
        Ff = evaluar_poblacion_vectorizado(
            Xf,
            problema.matriz_discrepancia,
            modo_transformacion=cfg.modo_transformacion_objetivos,
            modo_evaluacion=cfg.modo_evaluacion,
        )
    else:
        Ff = np.empty((0, 4), dtype=float)

    return ResultadoEjecucion(
        algoritmo=nombre_algo, inicializacion=nombre_init, replica=num_replica,
        semilla=semilla, tiempo_seg=float(delta_t),
        X_final=Xf, F_final=Ff, historial_F=cb.historial_F
    )

def ejecutar_suite_completa(H, pair_idx, cfg: ConfiguracionExperimento):
    """
    Ejecuta el conjunto total de experimentos en paralelo.
    """
    def _offset_estable(algo: str, init: str) -> int:
        clave = f"{algo}::{init}".encode('utf-8')
        return int(hashlib.sha256(clave).hexdigest()[:8], 16)

    if cfg.modo_semillas == 'deterministic':
        rng_semillas = np.random.default_rng(cfg.semilla_maestra)
    else:
        rng_semillas = np.random.default_rng(None)

    base_semillas = rng_semillas.integers(0, 2_000_000_000, size=cfg.n_ejecuciones)
    dirs_ref, _ = fabricar_algoritmo.__globals__['construir_direcciones_referencia'](cfg.tam_poblacion)
    
    algoritmos = [str(a) for a in cfg.algoritmos_activos]
    trabajos = []
    for algo in algoritmos:
        for init in cfg.opciones_init:
            for r in range(cfg.n_ejecuciones):
                semilla = int((base_semillas[r] + _offset_estable(algo, init)) % 2_147_483_647)
                trabajos.append({
                    'algo': algo, 'init': init, 'replica': r + 1,
                    'semilla': semilla, 'H': H, 'pair_idx': pair_idx,
                    'cfg': cfg, 'dirs_ref': dirs_ref
                })

    imprimir_subseccion("Fase Evolutiva", icono="🧬")
    total = len(trabajos)
    print(f"    • Iniciando {total} experimentos en modo paralelo seguro")
    
    max_workers = calcular_max_workers_paralelo()
    print(f"      • Paralelizando con hasta {max_workers} procesos en paralelo")
    
    resultados_dict = {}
    completados = 0
    ancho = len(str(total))

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_ejecutar_replica_individual, t): t for t in trabajos}
        for future in concurrent.futures.as_completed(futures):
            t = futures[future]
            try:
                rr = future.result()
                completados += 1
                print(f"      • [Progreso: {completados:>{ancho}}/{total}] | [{rr.algoritmo}-{rr.inicializacion}] ejecución {rr.replica}/{cfg.n_ejecuciones} ({rr.tiempo_seg:.1f} s)")
                resultados_dict[(rr.algoritmo, rr.inicializacion, rr.replica)] = rr
            except Exception as e:
                print(f"      • ⚠️  Error en [{t['algo']}-{t['init']}] ejecución {t['replica']}: {e}")

    # Reordenar para consistencia
    lista_final = []
    for algo in algoritmos:
        for init in cfg.opciones_init:
            for r in range(1, cfg.n_ejecuciones + 1):
                res = resultados_dict.get((algo, init, r))
                if res: lista_final.append(res)
                
    return lista_final
