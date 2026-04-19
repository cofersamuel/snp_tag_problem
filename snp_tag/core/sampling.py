"""
Módulo de Estrategias de Muestreo (sampling.py)
----------------------------------------------
Implementa diversas tácticas de inicialización de la población, incluyendo
métodos aleatorios dispersos y construcciones heurísticas tipo Greedy.
"""

import numpy as np
from pymoo.core.sampling import Sampling
from snp_tag.core.problem import calcular_distinguibilidad_snps

def construir_solucion_greedy(H, pair_idx, cobertura_objetivo=1, indices_ordenados=None):
    """
    Construye una solución mediante una aproximación voraz.
    """
    n_snps = H.shape[1]
    n_pares = pair_idx.shape[0]
    
    if indices_ordenados is None:
        puntuacion = calcular_distinguibilidad_snps(H, pair_idx)
        indices_ordenados = np.argsort(-puntuacion)
        
    seleccionados = np.zeros(n_snps, dtype=bool)
    cubiertos = np.zeros(n_pares, dtype=np.int32)
    
    a = H[pair_idx[:, 0], :]
    b = H[pair_idx[:, 1], :]
    discrepancia = (a != b)
    
    for s in indices_ordenados:
        if np.all(cubiertos >= cobertura_objetivo):
            break
        contribucion = discrepancia[:, s].astype(np.int32)
        if np.any((cubiertos < cobertura_objetivo) & (contribucion > 0)):
            seleccionados[s] = True
            cubiertos += contribucion
            
    if not seleccionados.any():
        seleccionados[indices_ordenados[0]] = True
    return seleccionados

def _agrupar_por_distinguibilidad(indices_desc, puntuaciones):
    """Agrupa SNPs con idéntica capacidad de discriminación."""
    grupos = []
    if len(indices_desc) == 0:
        return grupos
    actual = [int(indices_desc[0])]
    puntuacion_actual = float(puntuaciones[indices_desc[0]])
    for idx in indices_desc[1:]:
        s = float(puntuaciones[idx])
        if s == puntuacion_actual:
            actual.append(int(idx))
        else:
            grupos.append(np.array(actual, dtype=int))
            actual = [int(idx)]
            puntuacion_actual = s
    grupos.append(np.array(actual, dtype=int))
    return grupos

def _ordenar_con_desempate_aleatorio(grupos_puntuacion, rng):
    """Mantiene el orden greedy pero introduce estocasticidad en los empates."""
    partes = []
    for grupo in grupos_puntuacion:
        partes.append(rng.permutation(grupo) if len(grupo) > 1 else grupo)
    return np.concatenate(partes).astype(int)

class MuestreoAleatorioDisperso(Sampling):
    """Genera soluciones binarias con baja densidad de activos."""
    def __init__(self, prob: float = 0.05, semilla=42):
        super().__init__()
        self.prob = prob
        self.rng = np.random.default_rng(semilla)
    def _do(self, problem, n_samples, **kwargs):
        X = self.rng.random((n_samples, problem.n_var)) < self.prob
        vacíos = ~X.any(axis=1)
        if vacíos.any():
            for i in np.where(vacíos)[0]:
                X[i, self.rng.integers(0, problem.n_var)] = True
        return X

class MuestreoGreedyHibrido(Sampling):
    """Combina construcción greedy con relleno aleatorio disperso."""
    def __init__(self, H, pair_idx, cobertura_max=5, ratio_aleatorio=0.2, semilla=42):
        super().__init__()
        self.H = H
        self.pair_idx = pair_idx
        self.cobertura_max = cobertura_max
        self.ratio_aleatorio = ratio_aleatorio
        self.rng = np.random.default_rng(semilla)
        puntuacion = calcular_distinguibilidad_snps(H, pair_idx)
        self.indices_ordenados = np.argsort(-puntuacion)
        self.grupos = _agrupar_por_distinguibilidad(self.indices_ordenados, puntuacion)

    def _do(self, problem, n_samples, **kwargs):
        X = np.zeros((n_samples, problem.n_var), dtype=bool)
        n_aleatorio = int(n_samples * self.ratio_aleatorio)
        n_greedy = n_samples - n_aleatorio
        coberturas = np.linspace(1, self.cobertura_max, n_greedy).astype(int) if n_greedy > 0 else []
        for i in range(n_greedy):
            orden = _ordenar_con_desempate_aleatorio(self.grupos, self.rng)
            X[i] = construir_solucion_greedy(self.H, self.pair_idx, max(1, int(coberturas[i])), orden)
        for i in range(n_greedy, n_samples):
            fila = self.rng.random(problem.n_var) < 0.05
            if not fila.any(): fila[self.rng.integers(0, problem.n_var)] = True
            X[i] = fila
        return X

class MuestreoGreedyPuro(Sampling):
    """Inicialización basada íntegramente en la heurística voraz."""
    def __init__(self, H, pair_idx, cobertura_max=5, semilla=42):
        super().__init__()
        self.H = H
        self.pair_idx = pair_idx
        self.cobertura_max = cobertura_max
        self.rng = np.random.default_rng(semilla)
        puntuacion = calcular_distinguibilidad_snps(H, pair_idx)
        self.indices_ordenados = np.argsort(-puntuacion)
        self.grupos = _agrupar_por_distinguibilidad(self.indices_ordenados, puntuacion)

    def _do(self, problem, n_samples, **kwargs):
        X = np.zeros((n_samples, problem.n_var), dtype=bool)
        coberturas = np.linspace(1, self.cobertura_max, n_samples).astype(int)
        for i in range(n_samples):
            orden = _ordenar_con_desempate_aleatorio(self.grupos, self.rng)
            X[i] = construir_solucion_greedy(self.H, self.pair_idx, max(1, int(coberturas[i])), orden)
        return X
