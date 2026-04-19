"""
Módulo de Construcción de Algoritmos (algorithm.py)
--------------------------------------------------
Fabrica las instancias de los algoritmos evolutivos (NSGA-II, NSGA-III, 
SPEA2, MOEA/D) configuradas para el problema de Tag SNPs.
"""

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.spea2 import SPEA2
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.operators.crossover.ux import UX
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.util.ref_dirs import get_reference_directions

from snp_tag.core.sampling import (
    MuestreoAleatorioDisperso, MuestreoGreedyHibrido, MuestreoGreedyPuro
)
from snp_tag.config import ConfiguracionExperimento

def construir_direcciones_referencia(tam_poblacion, n_obj=4):
    """
    Genera direcciones de referencia balanceadas para algoritmos basados en descomposición.
    """
    dirs_ref = get_reference_directions('das-dennis', n_obj, n_partitions=1)
    particiones = 1
    p = 1
    while True:
        cand = get_reference_directions('das-dennis', n_obj, n_partitions=p + 1)
        if len(cand) > tam_poblacion:
            break
        dirs_ref = cand
        particiones = p + 1
        p += 1
    return dirs_ref, particiones

def fabricar_algoritmo(problema, H, nombre_algo, nombre_init, cfg: ConfiguracionExperimento, 
                       semilla=42, dirs_ref=None):
    """
    Instancia un algoritmo específico con su estrategia de muestreo y operadores.
    """
    # Selección de la estrategia de muestreo
    nombre_init = str(nombre_init)
    base_init = nombre_init

    if base_init in ['random', 'random_sparse']:
        prob_esperada = max(0.01, min(0.5, 70.0 / problema.n_var))
        sampling = MuestreoAleatorioDisperso(prob=prob_esperada, semilla=semilla)
    elif base_init == 'random_dense':
        sampling = BinaryRandomSampling()
    elif base_init == 'greedy_hybrid':
        sampling = MuestreoGreedyHibrido(H, problema.pair_idx, cobertura_max=cfg.cobertura_max_greedy, semilla=semilla)
    elif base_init == 'greedy_pure':
        sampling = MuestreoGreedyPuro(H, problema.pair_idx, cobertura_max=cfg.cobertura_max_greedy, semilla=semilla)
    else:
        raise ValueError(f"Estrategia de muestreo no soportada: {nombre_init}")

    cruce = UX(prob=cfg.pc)
    mutacion = BitflipMutation(prob=cfg.pm)

    if nombre_algo == 'NSGA2':
        return NSGA2(pop_size=cfg.tam_poblacion, sampling=sampling, crossover=cruce, 
                     mutation=mutacion, eliminate_duplicates=True, n_offsprings=cfg.n_descendencia)
    
    if nombre_algo == 'SPEA2':
        return SPEA2(pop_size=cfg.tam_poblacion, sampling=sampling, crossover=cruce, 
                     mutation=mutacion, eliminate_duplicates=True)

    if dirs_ref is None:
        dirs_ref, _ = construir_direcciones_referencia(cfg.tam_poblacion, n_obj=4)

    if nombre_algo == 'NSGA3':
        return NSGA3(pop_size=cfg.tam_poblacion, ref_dirs=dirs_ref, sampling=sampling, 
                     crossover=cruce, mutation=mutacion, eliminate_duplicates=True, 
                     n_offsprings=cfg.n_descendencia)

    if nombre_algo == 'MOEAD':
        return MOEAD(ref_dirs=dirs_ref, n_neighbors=cfg.vecinos_moead, sampling=sampling, 
                     crossover=cruce, mutation=mutacion)

    raise ValueError(f"Algoritmo no reconocido: {nombre_algo}")
