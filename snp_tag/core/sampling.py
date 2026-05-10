"""
Módulo de Estrategias de Muestreo (sampling.py)
----------------------------------------------
Implementa diversas tácticas de inicialización de la población, incluyendo
métodos aleatorios dispersos y construcciones heurísticas tipo Greedy.
"""

import numpy as np
from pymoo.core.sampling import Sampling
from snp_tag.core.problem import calcular_distinguibilidad_snps

def _construir_tabla_cobertura(H, pair_idx):
    """
    Construye la matriz de cobertura por SNP (pares x SNPs).
    """
    if pair_idx.size == 0 or H.size == 0:
        return np.zeros((0, H.shape[1]), dtype=bool)
    a = H[pair_idx[:, 0], :]
    b = H[pair_idx[:, 1], :]
    return (a != b)

def _agrupar_por_cobertura(cover_table):
    """
    Agrupa SNPs con identica cobertura sobre pares.
    """
    if cover_table.size == 0:
        return []
    packed = np.packbits(cover_table.astype(np.uint8), axis=0)
    grupos_dict = {}
    for s in range(cover_table.shape[1]):
        key = packed[:, s].tobytes()
        grupos_dict.setdefault(key, []).append(s)
    return [np.array(v, dtype=int) for v in grupos_dict.values()]

def _orden_greedy_ting(H, rng):
    """
    Ordena SNPs por balance de alelos, con desempates aleatorios.
    """
    if H.size == 0:
        return np.array([], dtype=int)
    n_hap = H.shape[0]
    suma = H.sum(axis=0).astype(float)
    balance = (n_hap / 2.0) - np.abs(suma - (n_hap / 2.0))
    ruido = rng.normal(0.0, 1e-6, size=balance.shape[0])
    orden = np.argsort(balance + ruido)

    columnas = H.T.astype(np.uint8)
    vistos = set()
    unicos = []
    for idx in orden:
        key = columnas[idx].tobytes()
        if key in vistos:
            continue
        vistos.add(key)
        unicos.append(int(idx))
    return np.array(unicos, dtype=int)

def _greedy_por_orden(orden, cover_table):
    """
    Selecciona SNPs segun el orden dado hasta cubrir todos los pares.
    """
    n_snps = cover_table.shape[1]
    seleccionados = np.zeros(n_snps, dtype=bool)
    if cover_table.size == 0:
        if n_snps > 0:
            idx = int(orden[0]) if orden.size > 0 else 0
            seleccionados[idx] = True
        return seleccionados

    cubiertos = np.zeros(cover_table.shape[0], dtype=bool)
    for s in orden[::-1]:
        if cubiertos.all():
            break
        contrib = cover_table[:, s]
        if np.any((~cubiertos) & contrib):
            seleccionados[s] = True
            cubiertos |= contrib
    if not seleccionados.any() and n_snps > 0:
        idx = int(orden[0]) if orden.size > 0 else 0
        seleccionados[idx] = True
    return seleccionados

def _greedy_grupos(cover_table, grupos, rng):
    """
    Selecciona 1 SNP por grupo hasta cubrir todos los pares.
    """
    n_snps = cover_table.shape[1]
    seleccionados = np.zeros(n_snps, dtype=bool)
    if cover_table.size == 0:
        if n_snps > 0:
            seleccionados[rng.integers(0, n_snps)] = True
        return seleccionados
    cubiertos = np.zeros(cover_table.shape[0], dtype=bool)
    grupos_orden = grupos.copy()
    rng.shuffle(grupos_orden)
    for grupo in grupos_orden:
        if cubiertos.all():
            break
        idx = int(rng.integers(0, len(grupo)))
        s = int(grupo[idx])
        seleccionados[s] = True
        cubiertos |= cover_table[:, s]
    if not seleccionados.any() and n_snps > 0:
        seleccionados[rng.integers(0, n_snps)] = True
    return seleccionados

def _unique_grupos(grupos, n_snps, rng):
    """
    Selecciona exactamente 1 SNP por grupo.
    """
    seleccionados = np.zeros(n_snps, dtype=bool)
    for grupo in grupos:
        idx = int(rng.integers(0, len(grupo)))
        seleccionados[int(grupo[idx])] = True
    if not seleccionados.any() and n_snps > 0:
        seleccionados[rng.integers(0, n_snps)] = True
    return seleccionados

def construir_solucion_greedy(H, pair_idx, indices_ordenados=None):
    """
    Construye una solución mediante una aproximación voraz.
    """
    n_snps = H.shape[1]
    n_pares = pair_idx.shape[0]
    
    if indices_ordenados is None:
        puntuacion = calcular_distinguibilidad_snps(H, pair_idx)
        indices_ordenados = np.argsort(-puntuacion)
        
    seleccionados = np.zeros(n_snps, dtype=bool)
    cubiertos = np.zeros(n_pares, dtype=bool)
    
    a = H[pair_idx[:, 0], :]
    b = H[pair_idx[:, 1], :]
    discrepancia = (a != b)
    
    for s in indices_ordenados:
        if np.all(cubiertos):
            break
        contribucion = discrepancia[:, s]
        if np.any((~cubiertos) & contribucion):
            seleccionados[s] = True
            cubiertos = cubiertos | contribucion
            
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
    def __init__(self, H, pair_idx, ratio_greedy=0.5, prob_aleatoria=0.5, semilla=42):
        super().__init__()
        self.H = H
        self.pair_idx = pair_idx
        self.ratio_greedy = float(ratio_greedy)
        self.prob_aleatoria = float(prob_aleatoria)
        self.rng = np.random.default_rng(semilla)
        puntuacion = calcular_distinguibilidad_snps(H, pair_idx)
        self.indices_ordenados = np.argsort(-puntuacion)
        self.grupos = _agrupar_por_distinguibilidad(self.indices_ordenados, puntuacion)

    def _do(self, problem, n_samples, **kwargs):
        X = np.zeros((n_samples, problem.n_var), dtype=bool)
        n_greedy = int(round(n_samples * self.ratio_greedy))
        n_greedy = min(max(n_greedy, 0), n_samples)
        n_aleatorio = n_samples - n_greedy
        for i in range(n_greedy):
            orden = _ordenar_con_desempate_aleatorio(self.grupos, self.rng)
            X[i] = construir_solucion_greedy(self.H, self.pair_idx, orden)
        for i in range(n_greedy, n_samples):
            fila = self.rng.random(problem.n_var) < self.prob_aleatoria
            if not fila.any(): fila[self.rng.integers(0, problem.n_var)] = True
            X[i] = fila
        return X

def construir_solucion_multicobertura(H, pair_idx, target_k, rng):
    """
    Construye una solución voraz dinámica exigiendo que cada par de haplotipos
    se distinga 'target_k' veces (si es biológicamente posible).
    """
    n_snps = H.shape[1]
    n_pares = pair_idx.shape[0]
    
    seleccionados = np.zeros(n_snps, dtype=bool)
    cubiertos = np.zeros(n_pares, dtype=int)
    
    a = H[pair_idx[:, 0], :]
    b = H[pair_idx[:, 1], :]
    discrepancia = (a != b).astype(int)
    
    # Límite biológico para prevenir bucles infinitos
    cobertura_maxima_biologica = discrepancia.sum(axis=1)
    # Salvaguarda: el objetivo para cada par no puede superar lo que el dataset permite
    objetivo_real = np.minimum(target_k, cobertura_maxima_biologica)
    
    while True:
        # Encontrar pares que aún no han alcanzado su cobertura objetivo real
        necesitan_cobertura = cubiertos < objetivo_real
        if not np.any(necesitan_cobertura):
            break # Todos los pares han alcanzado su objetivo_real
            
        # Puntuación: número de pares insatisfechos que cada SNP puede distinguir
        # Sólo sumamos las filas de discrepancia donde necesitan_cobertura es True
        puntuacion_dinamica = discrepancia[necesitan_cobertura].sum(axis=0)
        
        # Excluir SNPs que ya han sido seleccionados
        puntuacion_dinamica[seleccionados] = -1
        
        max_score = np.max(puntuacion_dinamica)
        if max_score <= 0:
            break # Ningún SNP adicional puede aportar cobertura útil
            
        candidatos = np.where(puntuacion_dinamica == max_score)[0]
        
        # Desempate estocástico
        mejor_snp = rng.choice(candidatos)
        
        seleccionados[mejor_snp] = True
        cubiertos += discrepancia[:, mejor_snp]
        
    # Garantizar que al menos un SNP es seleccionado en caso de objetivos degenerados
    if not seleccionados.any() and n_snps > 0:
        seleccionados[rng.integers(0, n_snps)] = True
        
    return seleccionados

class MuestreoGreedyMultiCobertura(Sampling):
    """
    Inicialización basada en una heurística voraz de cobertura múltiple progresiva.
    Fuerza al algoritmo a seleccionar SNPs redundantes distribuyendo un objetivo
    de cobertura desde 1 hasta max_cobertura_objetivo entre los individuos.
    """
    def __init__(self, H, pair_idx, max_cobertura_objetivo=5, semilla=42):
        super().__init__()
        self.H = H
        self.pair_idx = pair_idx
        self.max_cobertura_objetivo = int(max_cobertura_objetivo)
        self.rng = np.random.default_rng(semilla)

    def _do(self, problem, n_samples, **kwargs):
        X = np.zeros((n_samples, problem.n_var), dtype=bool)
        
        # Distribución lineal de los objetivos de cobertura en la población.
        # Va desde 1 hasta max_cobertura_objetivo (ej. 1, 1, 2, 2, 3, 3...)
        k_targets = np.linspace(1, self.max_cobertura_objetivo, n_samples).astype(int)
        
        for i in range(n_samples):
            target_k = k_targets[i]
            X[i] = construir_solucion_multicobertura(self.H, self.pair_idx, target_k, self.rng)
            
        return X

class MuestreoGreedyTing(Sampling):
    """
    Inicializacion mixta inspirada en Ting (GreedyInit + Greedy_init + Unique_init).
    """
    def __init__(self, H, pair_idx, ratio_greedy=0.5, semilla=42):
        super().__init__()
        self.H = H
        self.pair_idx = pair_idx
        self.ratio_greedy = float(ratio_greedy)
        self.rng = np.random.default_rng(semilla)

        self.cover_table = _construir_tabla_cobertura(H, pair_idx)
        self.grupos_cobertura = _agrupar_por_cobertura(self.cover_table)
        self.orden_ting = _orden_greedy_ting(H, self.rng)

        self.semilla_greedy = _greedy_por_orden(self.orden_ting, self.cover_table)
        self.semilla_densa = np.zeros(H.shape[1], dtype=bool)
        self.semilla_densa[self.orden_ting] = True

    def _do(self, problem, n_samples, **kwargs):
        n_samples = int(n_samples)
        X = np.zeros((n_samples, problem.n_var), dtype=bool)

        n_greedy_total = int(np.ceil(n_samples * self.ratio_greedy))
        n_anchor = int(np.ceil(self.ratio_greedy * 10.0))

        pos = 0
        for _ in range(0, n_anchor, 2):
            if pos >= n_samples:
                break
            X[pos] = self.semilla_greedy
            pos += 1
            if pos >= n_samples:
                break
            X[pos] = self.semilla_densa
            pos += 1

        while pos < min(n_greedy_total, n_samples):
            X[pos] = _greedy_grupos(self.cover_table, self.grupos_cobertura, self.rng)
            pos += 1
            if pos >= min(n_greedy_total, n_samples):
                break
            X[pos] = _unique_grupos(self.grupos_cobertura, problem.n_var, self.rng)
            pos += 1

        if pos < n_samples:
            X[pos:] = self.rng.random((n_samples - pos, problem.n_var)) < 0.5

        return X
