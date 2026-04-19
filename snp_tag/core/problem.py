"""
Módulo de Definición del Problema (problem.py)
---------------------------------------------
Define la clase TagSNPProblem compatible con PyMoo, implementando funciones 
de evaluación vectorizadas para optimizar el rendimiento computacional.
"""

import numpy as np
from pymoo.core.problem import Problem

def evaluar_poblacion_vectorizado(X_bool: np.ndarray, matriz_discrepancia: np.ndarray) -> np.ndarray:
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

    # f2: Resolución (Tolerancia) -> min(D) - 1
    tolerancia = (D.min(axis=1) - 1.0)
    # f3: Distancia media
    hamming_med = D.mean(axis=1)
    # f4: Varianza (Balance)
    varianza = D.var(axis=1)

    # Retorno en formato de minimización (PyMoo standard)
    return np.column_stack([
        k,               # f1: Optimizar compacidad
        -tolerancia,     # f2: Maximizar resolución
        -hamming_med,    # f3: Maximizar distancia media
        varianza,        # f4: Minimizar varianza
    ]).astype(float)

class ProblemaTagSNP(Problem):
    """
    Formulación multiobjetivo del problema de selección de Tag SNPs.
    """
    def __init__(self, H: np.ndarray, pair_idx: np.ndarray, normalizar_busqueda: bool = False):
        self.H = H
        self.pair_idx = pair_idx
        self.normalizar_busqueda = bool(normalizar_busqueda)
        
        # Precomputación de la matriz de discrepancia (diferencias bit a bit)
        self.matriz_discrepancia = (H[pair_idx[:, 0], :] != H[pair_idx[:, 1], :]).astype(np.int16)
        
        n_var = H.shape[1]
        D_completa = self.matriz_discrepancia.sum(axis=1).astype(float)
        
        self._escala_f1 = max(1.0, float(n_var))
        self._escala_f2 = max(1.0, float(D_completa.min() - 1.0))
        self._escala_f3 = max(1.0, float(D_completa.mean()))
        self._escala_f4 = max(1.0, float(D_completa.var()))
        
        super().__init__(n_var=n_var, n_obj=4, n_ieq_constr=0, xl=0, xu=1, vtype=bool)

    def _evaluate(self, X, out, *args, **kwargs):
        X_bool = X.astype(bool)
        F_crudo = evaluar_poblacion_vectorizado(X_bool, self.matriz_discrepancia)
        
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
