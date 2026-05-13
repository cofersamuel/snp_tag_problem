"""
Módulo de Construcción de Algoritmos (algorithm.py)
--------------------------------------------------
Fabrica las instancias de los algoritmos evolutivos (NSGA-II, NSGA-III, 
SPEA2, MOEA/D) configuradas para el problema de Tag SNPs.
"""

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.spea2 import SPEA2, SPEA2Survival
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.age2 import AGEMOEA2
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.algorithms.moo.rvea import RVEA
from pymoo.decomposition.tchebicheff import Tchebicheff
from pymoo.decomposition.pbi import PBI
from pymoo.decomposition.weighted_sum import WeightedSum
from pymoo.algorithms.moo.nsga3 import HyperplaneNormalization
from pymoo.operators.crossover.ux import UX
from pymoo.operators.crossover.hux import HUX
from pymoo.operators.crossover.pntx import SinglePointCrossover, TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.core.repair import Repair
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.util.dominator import Dominator
from pymoo.util.misc import vectorized_cdist

import numpy as np

from snp_tag.core.sampling import (
    MuestreoAleatorioDisperso, MuestreoGreedyMultiCobertura,
    MuestreoGreedyTing, MuestreoGreedyHolistico
)
from snp_tag.config import ConfiguracionExperimento


class SPEA2SurvivalSeguro(SPEA2Survival):
    """
    Variante de SPEA2 con normalización robusta para evitar divisiones por cero
    cuando algún objetivo presenta rango nulo (nadir == ideal).
    """

    def _do(self, problem, pop, *args, n_survive=None, **kwargs):
        F = pop.get("F").astype(float, copy=False)

        M = Dominator().calc_domination_matrix(F)
        S = (M == 1).sum(axis=0)
        R = ((M == -1) * S).sum(axis=1)

        k = int(np.sqrt(len(pop)))
        if k >= len(pop):
            k = len(pop) - 1

        if self.normalize:
            if self.norm is None:
                self.norm = HyperplaneNormalization(F.shape[1])
            self.norm.update(F)

            ideal, nadir = self.norm.ideal_point, self.norm.nadir_point
            denominador = nadir - ideal
            denominador_seguro = np.where(np.abs(denominador) > 1e-12, denominador, 1.0)

            _F = (F - ideal) / denominador_seguro
            dists = vectorized_cdist(_F, _F, fill_diag_with_inf=True)
        else:
            dists = vectorized_cdist(F, F, fill_diag_with_inf=True)

        sdists = np.sort(dists, axis=1)
        D = 1 / (sdists[:, k] + 2)
        SPEA_F = R + D

        pop.set(SPEA_F=SPEA_F, SPEA_R=R, SPEA_D=D)

        survivors = list(np.where(np.all(M >= 0, axis=1))[0])

        if self.normalize:
            I = vectorized_cdist(self.norm.extreme_points, F[survivors]).argmin(axis=1)
            pop[survivors][I].set("SPEA_F", -1.0)

        H = set(survivors)
        rem = np.array([k for k in range(len(pop)) if k not in H])

        if len(survivors) < n_survive:
            rem_by_F = rem[SPEA_F[rem].argsort()]
            survivors.extend(rem_by_F[:n_survive - len(survivors)])
        elif len(survivors) > n_survive:
            while len(survivors) > n_survive:
                i = dists[survivors][:, survivors].min(axis=1).argmin()
                survivors = [survivors[j] for j in range(len(survivors)) if j != i]

        return pop[survivors]

def construir_direcciones_referencia(tam_poblacion, n_obj=4):
    """
    Genera direcciones de referencia balanceadas para algoritmos basados en descomposición.
    Fuerza compatibilidad estricta con los tamaños de población de Das-Dennis.
    """
    tam_poblacion = int(max(1, tam_poblacion))
    p = 1
    while True:
        cand = get_reference_directions('das-dennis', n_obj, n_partitions=p)
        if len(cand) == tam_poblacion:
            return cand, p
        elif len(cand) > tam_poblacion:
            prev_cand = get_reference_directions('das-dennis', n_obj, n_partitions=p-1) if p > 1 else []
            prev_len = len(prev_cand) if p > 1 else 1
            error_msg = (
                f"\n=== [ERROR DE CONFIGURACIÓN] ===\n"
                f"El tamaño de población ({tam_poblacion}) no es compatible "
                f"con una distribución geométrica Das-Dennis para {n_obj} objetivos.\n"
                f"Los tamaños válidos más cercanos son {prev_len} o {len(cand)}.\n"
                f"Por favor, ajusta 'tam_poblacion' en tu configuración o usa otro algoritmo.\n"
                f"================================\n"
            )
            print(error_msg)
            raise ValueError(error_msg)
        p += 1

class ReparacionSNP(Repair):
    """
    Operador de reparación que intercepta individuos vacíos (k=0) y activa 
    un SNP aleatorio para mantener la validez del fenotipo y la diversidad.
    """
    def _do(self, problem, Z, **kwargs):
        k = Z.sum(axis=1)
        vacios = np.where(k == 0)[0]
        if len(vacios) > 0:
            for idx in vacios:
                Z[idx, np.random.randint(0, problem.n_var)] = True
        return Z

def fabricar_algoritmo(problema, H, nombre_algo, nombre_init, nombre_crossover, cfg: ConfiguracionExperimento, 
                       semilla=42, dirs_ref=None):
    """
    Instancia un algoritmo específico con su estrategia de muestreo y operadores.
    """
    # Selección de la estrategia de muestreo
    nombre_init = str(nombre_init)
    base_init = nombre_init

    if base_init == 'random_sparse':
        prob_esperada = max(0.01, min(0.5, 70.0 / problema.n_var))
        sampling = MuestreoAleatorioDisperso(prob=prob_esperada, semilla=semilla)
    elif base_init == 'random_dense':
        sampling = BinaryRandomSampling()

    elif base_init == 'greedy_multi':
        sampling = MuestreoGreedyMultiCobertura(
            H, 
            problema.pair_idx, 
            max_cobertura_objetivo=cfg.max_cobertura_objetivo, 
            semilla=semilla
        )
    elif base_init == 'greedy_ting':
        sampling = MuestreoGreedyTing(
            H,
            problema.pair_idx,
            ratio_greedy=cfg.ratio_greedy_ting,
            semilla=semilla,
        )
    elif base_init == 'greedy_holistic':
        sampling = MuestreoGreedyHolistico(
            H,
            problema.pair_idx,
            max_k=cfg.max_k_holistic,
            semilla=semilla,
        )
    else:
        raise ValueError(f"Estrategia de muestreo no soportada: {nombre_init}")

    nombre_crossover = str(nombre_crossover).upper()
    if nombre_crossover == 'HUX':
        cruce = HUX(prob=cfg.pc)
    elif nombre_crossover == '1P':
        cruce = SinglePointCrossover(prob=cfg.pc)
    elif nombre_crossover == '2P':
        cruce = TwoPointCrossover(prob=cfg.pc)
    else:
        cruce = UX(prob=cfg.pc)
        
    mutacion = BitflipMutation(prob=cfg.pm)
    reparador = ReparacionSNP()

    if nombre_algo == 'NSGA2':
        return NSGA2(pop_size=cfg.tam_poblacion, sampling=sampling, crossover=cruce, 
                     mutation=mutacion, repair=reparador, eliminate_duplicates=True, n_offsprings=cfg.n_descendencia)
    
    if nombre_algo == 'SPEA2':
        return SPEA2(
            pop_size=cfg.tam_poblacion,
            sampling=sampling,
            crossover=cruce,
            mutation=mutacion,
            repair=reparador,
            survival=SPEA2SurvivalSeguro(normalize=True),
            eliminate_duplicates=True,
        )

    if nombre_algo == 'AGEMOEA2':
        return AGEMOEA2(
            pop_size=cfg.tam_poblacion,
            sampling=sampling,
            crossover=cruce,
            mutation=mutacion,
            repair=reparador,
            eliminate_duplicates=True
        )

    if nombre_algo == 'SMSEMOA':
        return SMSEMOA(
            pop_size=cfg.tam_poblacion,
            sampling=sampling,
            crossover=cruce,
            mutation=mutacion,
            repair=reparador,
            eliminate_duplicates=True
        )

    if dirs_ref is None:
        dirs_ref, _ = construir_direcciones_referencia(cfg.tam_poblacion, n_obj=4)

    if nombre_algo == 'NSGA3':
        return NSGA3(pop_size=cfg.tam_poblacion, ref_dirs=dirs_ref, sampling=sampling, 
                     crossover=cruce, mutation=mutacion, repair=reparador, eliminate_duplicates=True, 
                     n_offsprings=cfg.n_descendencia)

    if nombre_algo == 'RVEA':
        return RVEA(
            pop_size=cfg.tam_poblacion, 
            ref_dirs=dirs_ref, 
            sampling=sampling, 
            crossover=cruce, 
            mutation=mutacion, 
            repair=reparador,
            eliminate_duplicates=True
        )

    if nombre_algo in {'MOEAD_TCHE', 'MOEAD_PBI', 'MOEAD_WS'}:
        if nombre_algo == 'MOEAD_TCHE':
            descomposicion = Tchebicheff()
        elif nombre_algo == 'MOEAD_PBI':
            descomposicion = PBI(theta=cfg.theta_moead_pbi)
        else:
            descomposicion = WeightedSum()

        algoritmo = MOEAD(
            ref_dirs=dirs_ref,
            n_neighbors=cfg.vecinos_moead,
            prob_neighbor_mating=cfg.prob_vecindad_moead,
            decomposition=descomposicion,
            sampling=sampling,
            crossover=cruce,
            mutation=mutacion,
            repair=reparador,
        )

        # Salvaguardas: elimina duplicados si el backend lo soporta.
        # (MOEA/D puede colapsar a puntos repetidos en problemas discretos binarios)
        try:
            setattr(algoritmo, 'eliminate_duplicates', True)
        except Exception:
            pass

        return algoritmo

    raise ValueError(f"Algoritmo no reconocido: {nombre_algo}")
