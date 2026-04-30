"""
Módulo de Definición del Problema (problem.py)
---------------------------------------------
Define la clase TagSNPProblem compatible con PyMoo, implementando funciones 
de evaluación vectorizadas para optimizar el rendimiento computacional.
"""

import numpy as np
from pymoo.core.problem import Problem

def transformar_objetivos_a_minimizacion(
    k: np.ndarray,
    min_cobertura: np.ndarray,
    hamming_med: np.ndarray,
    varianza: np.ndarray,
    modo_transformacion: str = 'neg',
    epsilon: float = 1e-9,
    mask_infactible: np.ndarray | None = None,
    penalizacion_infactible: float = 1e3,
) -> np.ndarray:
    """
    Convierte objetivos de maximización a minimización según el modo configurado.
    """
    modo = str(modo_transformacion or 'neg').strip().lower()
    if modo == 'inverse':
        # En inverse, usar dominio positivo estable para evitar explosiones numéricas
        f2 = 1.0 / np.maximum(min_cobertura, 1.0)
        f3 = 1.0 / np.maximum(hamming_med, epsilon)
    else:
        tolerancia = min_cobertura - 1.0
        f2 = -tolerancia
        f3 = -hamming_med

    F = np.column_stack([
        k,
        f2,
        f3,
        varianza,
    ]).astype(float)

    # Penalización fuerte y compatible con algoritmos sin soporte de restricciones
    if mask_infactible is not None and np.any(mask_infactible):
        deficit = 1.0 - np.minimum(min_cobertura[mask_infactible], 1.0)
        penalizacion = penalizacion_infactible + deficit
        F[mask_infactible, :] += penalizacion[:, None]

    return F


def evaluar_poblacion_vectorizado(
    X_bool: np.ndarray,
    matriz_discrepancia: np.ndarray,
    modo_transformacion: str = 'neg',
    devolver_min_cobertura: bool = False,
    modo_evaluacion: str = 'absoluta',
) -> np.ndarray:
    """
    Evaluación vectorizada de la población sobre los cuatro objetivos del problema.
    """
    # k: número de SNPs seleccionados por cada individuo
    k = X_bool.sum(axis=1).astype(float)
    
    # Prevenir divisiones por cero en individuos vacíos
    sin_seleccion = (k == 0)
    if sin_seleccion.any():
        X_bool = X_bool.copy()
        X_bool[sin_seleccion, 0] = True
        k[sin_seleccion] = 1.0

    # Distancias de Hamming por par mediante producto matricial
    # D shape: (tam_poblacion, n_pares)
    D = (matriz_discrepancia.astype(np.int32) @ X_bool.T.astype(np.int32)).T.astype(float)

    # f2: Cobertura mínima entre pares (base robusta para transformación)
    min_cobertura = D.min(axis=1)
    
    if modo_evaluacion == 'proportional':
        # NORMALIZACIÓN PROPORCIONAL (Ting et al.)
        hamming_med = D.mean(axis=1) / k
        varianza = D.var(axis=1) / (k ** 2)
    else:
        # MÉTRICA ABSOLUTA
        # f3: Distancia media
        hamming_med = D.mean(axis=1)
        # f4: Varianza (Balance)
        varianza = D.var(axis=1)

    mask_infactible = min_cobertura < 1.0

    # Retorno en formato de minimización (PyMoo standard)
    F = transformar_objetivos_a_minimizacion(
        k,
        min_cobertura,
        hamming_med,
        varianza,
        modo_transformacion=modo_transformacion,
        mask_infactible=mask_infactible,
    )
    if devolver_min_cobertura:
        return F, min_cobertura
    return F

class ProblemaTagSNP(Problem):
    """
    Formulación multiobjetivo del problema de selección de Tag SNPs.
    """
    def __init__(
        self,
        H: np.ndarray,
        pair_idx: np.ndarray,
        normalizar_busqueda: bool = False,
        modo_transformacion_objetivos: str = 'neg',
        modo_evaluacion: str = 'absoluta',
    ):
        self.H = H
        self.pair_idx = pair_idx
        self.normalizar_busqueda = bool(normalizar_busqueda)
        self.modo_transformacion_objetivos = str(modo_transformacion_objetivos or 'neg').strip().lower()
        self.modo_evaluacion = str(modo_evaluacion or 'absoluta').strip().lower()
        
        # Precomputación de la matriz de discrepancia (diferencias bit a bit)
        self.matriz_discrepancia = (H[pair_idx[:, 0], :] != H[pair_idx[:, 1], :]).astype(np.int16)
        
        n_var = H.shape[1]
        D_completa = self.matriz_discrepancia.sum(axis=1).astype(float)
        
        self._escala_f1 = max(1.0, float(n_var))
        
        if self.modo_evaluacion == 'proportional':
            self._escala_f4 = 0.25 # La varianza de una proporción no pasa de 0.25
            if self.modo_transformacion_objetivos == 'inverse':
                self._escala_f2 = 1.0
                self._escala_f3 = 10.0 # Nadir empírico seguro para inverse
            else: # 'neg'
                self._escala_f2 = max(1.0, float(D_completa.min() - 1.0))
                self._escala_f3 = 1.0 # La proporción máxima es 1.0, |-1.0| = 1.0
        else:
            if self.modo_transformacion_objetivos == 'inverse':
                # En inverse, f2 y f3 ya viven en escala ~[0, 1]; usar 1 evita
                # sobrecomprimir objetivos en MOEA/D durante la búsqueda.
                self._escala_f2 = 1.0
                self._escala_f3 = 1.0
            else:
                self._escala_f2 = max(1.0, float(D_completa.min() - 1.0))
                self._escala_f3 = max(1.0, float(D_completa.mean()))
            self._escala_f4 = max(1.0, float(D_completa.var()))
        
        super().__init__(n_var=n_var, n_obj=4, n_ieq_constr=0, xl=0, xu=1, vtype=bool)

    def _evaluate(self, X, out, *args, **kwargs):
        X_bool = X.astype(bool)
        F_crudo, min_cobertura = evaluar_poblacion_vectorizado(
            X_bool,
            self.matriz_discrepancia,
            modo_transformacion=self.modo_transformacion_objetivos,
            devolver_min_cobertura=True,
            modo_evaluacion=self.modo_evaluacion,
        )
        
        if self.normalizar_busqueda:
            F_escalado = F_crudo.copy()
            F_escalado[:, 0] /= self._escala_f1
            F_escalado[:, 1] /= self._escala_f2
            F_escalado[:, 2] /= self._escala_f3
            F_escalado[:, 3] /= self._escala_f4
            out['F'] = F_escalado
        else:
            out['F'] = F_crudo

def calcular_distinguibilidad_snps(H, pair_idx):
    """
    Cuantifica la capacidad de cada SNP para discriminar entre pares de haplotipos.
    """
    a = H[pair_idx[:, 0], :]
    b = H[pair_idx[:, 1], :]
    discrepancia = (a != b).astype(np.int8)
    return discrepancia.sum(axis=0).astype(float)
