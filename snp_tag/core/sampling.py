"""
Módulo de Estrategias de Muestreo (sampling.py)
----------------------------------------------
Implementa diversas tácticas de inicialización de la población, incluyendo
métodos aleatorios dispersos y construcciones heurísticas tipo Greedy.
"""

import numpy as np
from pymoo.core.sampling import Sampling
from snp_tag.core.problem import calcular_distinguibilidad_snps
from snp_tag.engine.diagnostics_logic import detectar_bloques_ld

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
    Ordena SNPs por balance de alelos, filtrando duplicados exactos vectorizadamente.
    """
    if H.size == 0:
        return np.array([], dtype=int)
        
    # 1. Filtrar columnas idénticas (SNPs redundantes) de forma 100% vectorizada
    _, indices_unicos = np.unique(H, axis=1, return_index=True)
    
    n_hap = H.shape[0]
    
    # 2. Calcular la suma sólo para los representantes únicos
    suma = H[:, indices_unicos].sum(axis=0).astype(float)
    
    # 3. Calcular balance (cercanía a n_hap / 2)
    balance = (n_hap / 2.0) - np.abs(suma - (n_hap / 2.0))
    
    # 4. Añadir ruido mínimo para desempates estocásticos
    ruido = rng.normal(0.0, 1e-6, size=balance.shape[0])
    
    # 5. Ordenar los índices únicos basándose en el balance perturbado
    orden_relativo = np.argsort(balance + ruido)
    
    # Devolver los índices originales ordenados
    return indices_unicos[orden_relativo].astype(int)

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
        """
        Inicializa el muestreo aleatorio disperso.

        Args:
            prob (float): Probabilidad de que un SNP sea seleccionado.
            semilla (int): Semilla para el generador de números aleatorios.
        """
        super().__init__()
        self.prob = prob
        self.rng = np.random.default_rng(semilla)
    def _do(self, problem, n_samples, **kwargs):
        """
        Genera una población de soluciones aleatorias dispersas.

        Args:
            problem: Problema de optimización.
            n_samples (int): Número de soluciones a generar.
            **kwargs: Argumentos adicionales.

        Returns:
            np.ndarray: Población de soluciones binarias.
        """
        X = self.rng.random((n_samples, problem.n_var)) < self.prob
        vacíos = ~X.any(axis=1)
        if vacíos.any():
            for i in np.where(vacíos)[0]:
                X[i, self.rng.integers(0, problem.n_var)] = True
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
        """
        Inicializa el muestreo voraz de cobertura múltiple.

        Args:
            H (np.ndarray): Matriz de haplotipos.
            pair_idx (np.ndarray): Índices de pares de haplotipos.
            max_cobertura_objetivo (int): Cobertura máxima objetivo.
            semilla (int): Semilla para el generador de números aleatorios.
        """
        super().__init__()
        self.H = H
        self.pair_idx = pair_idx
        self.max_cobertura_objetivo = int(max_cobertura_objetivo)
        self.rng = np.random.default_rng(semilla)

    def _do(self, problem, n_samples, **kwargs):
        """
        Genera una población de soluciones voraces de cobertura múltiple.

        Args:
            problem: Problema de optimización.
            n_samples (int): Número de soluciones a generar.
            **kwargs: Argumentos adicionales.

        Returns:
            np.ndarray: Población de soluciones binarias.
        """
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
        """
        Inicializa el muestreo voraz de cobertura múltiple.

        Args:
            H (np.ndarray): Matriz de haplotipos.
            pair_idx (np.ndarray): Índices de pares de haplotipos.
            max_cobertura_objetivo (int): Cobertura máxima objetivo.
            semilla (int): Semilla para el generador de números aleatorios.
        """
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
        """
        Genera una población de soluciones voraces de cobertura múltiple.

        Args:
            problem: Problema de optimización.
            n_samples (int): Número de soluciones a generar.
            **kwargs: Argumentos adicionales.

        Returns:
            np.ndarray: Población de soluciones binarias.
        """
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

# ============================================================
# Helpers para MuestreoGreedyHolistico
# ============================================================

def _ancla_max_hamming_medio(discrepancia, distinguibilidad, n_snps, n_pares, rng):
    """
    Construye una solución que maximiza la distancia media de Hamming.
    Selecciona SNPs en orden descendente de contribución media, continuando
    más allá de la cobertura mínima hasta que la mejora marginal sea despreciable.
    """
    seleccionados = np.zeros(n_snps, dtype=bool)
    D_acum = np.zeros(n_pares, dtype=float)

    ruido = rng.normal(0.0, 1e-6, size=n_snps)
    orden = np.argsort(-(distinguibilidad + ruido))

    cobertura_alcanzada = False
    for s in orden:
        contrib = float(discrepancia[:, s].sum())
        if contrib <= 0:
            continue

        seleccionados[s] = True
        D_acum += discrepancia[:, s].astype(float)

        if not cobertura_alcanzada:
            if D_acum.min() >= 1:
                cobertura_alcanzada = True
            continue

        # Tras cobertura: parar cuando la mejora marginal < 0.5%
        media_actual = D_acum.mean()
        if media_actual > 0 and (contrib / n_pares) / media_actual < 0.005:
            break

    if not seleccionados.any():
        seleccionados[int(orden[0])] = True
    return seleccionados


def _ancla_min_varianza(discrepancia, H, pair_idx, n_snps, n_pares, rng):
    """
    Construye una solución con mínima varianza en distancias de Hamming.
    Parte de una cobertura mínima y añade SNPs que equilibran las distancias
    entre pares.
    """
    distinguibilidad = discrepancia.sum(axis=0).astype(float)
    indices_base = np.argsort(-distinguibilidad)
    seleccionados = construir_solucion_greedy(H, pair_idx, indices_base).copy()
    D_acum = discrepancia[:, seleccionados].sum(axis=1).astype(float)

    k_inicial = int(seleccionados.sum())
    max_extras = min(k_inicial * 2, n_snps - k_inicial)

    for _ in range(max_extras):
        candidatos = np.where(~seleccionados)[0]
        if len(candidatos) == 0:
            break

        var_actual = float(D_acum.var())
        if var_actual < 1e-9:
            break

        # Varianza resultante tras añadir cada candidato
        D_cand = discrepancia[:, candidatos].astype(float)
        D_nuevas = D_acum[:, None] + D_cand
        varianzas = D_nuevas.var(axis=0)

        mejor_idx = int(np.argmin(varianzas))
        if varianzas[mejor_idx] >= var_actual * 0.99:
            break

        seleccionados[candidatos[mejor_idx]] = True
        D_acum = D_nuevas[:, mejor_idx].copy()

    return seleccionados


def _construir_solucion_bloques(discrepancia, bloques, n_snps, n_pares,
                                 indices_bloques, rng):
    """
    Construye una solución seleccionando representantes de un subconjunto de
    bloques LD mediante mini-greedy restringido a los SNPs de cada bloque.
    Incorpora un mecanismo de reparación para garantizar cobertura total.
    """
    seleccionados = np.zeros(n_snps, dtype=bool)
    cubiertos = np.zeros(n_pares, dtype=bool)

    for b_idx in indices_bloques:
        snps_bloque = bloques[b_idx]
        if len(snps_bloque) == 0:
            continue

        # Puntuación: pares aún no cubiertos que cada SNP del bloque distingue
        pendientes = ~cubiertos
        scores = discrepancia[pendientes][:, snps_bloque].sum(axis=0) if pendientes.any() \
            else discrepancia[:, snps_bloque].sum(axis=0)

        orden_local = np.argsort(-scores.astype(float))
        for idx_local in orden_local:
            s_global = snps_bloque[idx_local]
            contrib = discrepancia[:, s_global].astype(bool)
            if np.any((~cubiertos) & contrib):
                seleccionados[s_global] = True
                cubiertos |= contrib
            if cubiertos.all():
                break
        if cubiertos.all():
            break

    # Reparación greedy si los bloques seleccionados fueron insuficientes
    if not cubiertos.all():
        pendientes = ~cubiertos
        scores_globales = discrepancia[pendientes].sum(axis=0).astype(float)
        scores_globales[seleccionados] = -1
        
        ruido = rng.normal(0.0, 1e-6, size=n_snps)
        orden_reparacion = np.argsort(-(scores_globales + ruido))
        
        for s_global in orden_reparacion:
            if scores_globales[s_global] <= 0:
                continue
            contrib = discrepancia[:, s_global].astype(bool)
            if np.any((~cubiertos) & contrib):
                seleccionados[s_global] = True
                cubiertos |= contrib
            if cubiertos.all():
                break

    if not seleccionados.any() and n_snps > 0:
        seleccionados[rng.integers(0, n_snps)] = True
    return seleccionados


def _construir_complemento(discrepancia, solucion_base, n_snps, n_pares, rng):
    """
    Construye el complemento de una solución existente: apunta a los pares
    peor cubiertos por la base y, tras satisfacerlos, continúa la selección
    para garantizar una solución 100% factible.
    """
    D_base = discrepancia[:, solucion_base].sum(axis=1).astype(float)

    if D_base.sum() == 0:
        sol = np.zeros(n_snps, dtype=bool)
        sol[rng.integers(0, n_snps)] = True
        return sol

    mediana = float(np.median(D_base))
    mal_cubiertos = D_base <= mediana

    # Puntuar por contribución a los pares mal cubiertos, excluyendo SNPs de la base
    scores = discrepancia[mal_cubiertos].sum(axis=0).astype(float)
    scores[solucion_base] = -1

    ruido = rng.normal(0.0, 1e-6, size=n_snps)
    orden = np.argsort(-(scores + ruido))

    seleccionados = np.zeros(n_snps, dtype=bool)
    cubiertos_obj = np.zeros(n_pares, dtype=bool)

    # Primera fase: cubrir los pares débiles (mal_cubiertos)
    for s in orden:
        if scores[s] <= 0:
            break
        contrib = discrepancia[:, s].astype(bool)
        if np.any((~cubiertos_obj) & contrib):
            seleccionados[s] = True
            cubiertos_obj |= contrib
        if cubiertos_obj[mal_cubiertos].all():
            break

    # Segunda fase: completar la cobertura total para evitar soluciones infactibles
    if not cubiertos_obj.all():
        pendientes = ~cubiertos_obj
        scores_resto = discrepancia[pendientes].sum(axis=0).astype(float)
        scores_resto[seleccionados] = -1
        
        orden_resto = np.argsort(-(scores_resto + rng.normal(0.0, 1e-6, size=n_snps)))
        for s in orden_resto:
            if scores_resto[s] <= 0:
                continue
            contrib = discrepancia[:, s].astype(bool)
            if np.any((~cubiertos_obj) & contrib):
                seleccionados[s] = True
                cubiertos_obj |= contrib
            if cubiertos_obj.all():
                break

    if not seleccionados.any():
        candidatos = np.where(~solucion_base)[0]
        if len(candidatos) > 0:
            seleccionados[rng.choice(candidatos)] = True
        else:
            seleccionados[rng.integers(0, n_snps)] = True
    return seleccionados


def _muestreo_guiado_disperso(distinguibilidad, discrepancia, n_snps, n_pares, k_objetivo, rng):
    """
    Muestreo ponderado por distinguibilidad con cardinalidad objetivo,
    seguido de una reparación greedy para asegurar cobertura total.
    """
    total = distinguibilidad.sum()
    if total <= 0:
        p = np.full(n_snps, 1.0 / n_snps)
    else:
        p = distinguibilidad / total

    p = p * k_objetivo
    p = np.clip(p, 0.01, 0.8)

    sol = rng.random(n_snps) < p
    
    # Evaluar cobertura
    cubiertos = discrepancia[:, sol].any(axis=1) if sol.any() else np.zeros(n_pares, dtype=bool)
    
    # Reparación greedy
    if not cubiertos.all():
        pendientes = ~cubiertos
        scores = discrepancia[pendientes].sum(axis=0).astype(float)
        scores[sol] = -1
        
        ruido = rng.normal(0.0, 1e-6, size=n_snps)
        orden = np.argsort(-(scores + ruido))
        
        for s in orden:
            if scores[s] <= 0:
                continue
            contrib = discrepancia[:, s].astype(bool)
            if np.any((~cubiertos) & contrib):
                sol[s] = True
                cubiertos |= contrib
            if cubiertos.all():
                break

    if not sol.any():
        sol[rng.integers(0, n_snps)] = True
    return sol


class MuestreoGreedyHolistico(Sampling):
    """
    Inicialización de cinco niveles para optimización multiobjetivo de Tag SNPs.

    Siembra la población cubriendo sistemáticamente las cuatro dimensiones del
    frente de Pareto con soluciones estructuralmente diversas:

    Tier 1: Anclas de Pareto           (~5-10%)  - Extremos de cada objetivo
    Tier 2: Barrido k-Cover            (~25%)    - Multicobertura progresiva
    Tier 3: Ensamblaje por bloques LD  (~25%)    - Diversidad estructural genómica
    Tier 4: Inyección de complementos  (~20%)    - Soluciones que parchean debilidades
    Tier 5: Exploración guiada dispersa(~20-25%) - Muestreo ponderado por importancia

    Post-procesado: deduplicación fenotípica (siempre activo).
    """

    def __init__(self, H, pair_idx, max_k=5, semilla=42):
        super().__init__()
        self.H = H
        self.pair_idx = pair_idx
        self.max_k = int(max_k)
        self.rng = np.random.default_rng(semilla)

        # Precomputación
        self.discrepancia = (H[pair_idx[:, 0]] != H[pair_idx[:, 1]]).astype(np.int16)
        self.distinguibilidad = self.discrepancia.sum(axis=0).astype(float)
        self.n_snps = H.shape[1]
        self.n_pares = pair_idx.shape[0]

        # Detección de bloques LD para Tier 3 (basada en datos,
        # funciona con cualquier dataset).
        segmentos = detectar_bloques_ld(H)
        # Fallback: si la estructura LD es demasiado uniforme y produce
        # muy pocos bloques, dividir posicionalmente para garantizar diversidad.
        if len(segmentos) < 5:
            n_bloques_min = min(10, max(5, self.n_snps // 50))
            segmentos_pos = [(i * self.n_snps // n_bloques_min,
                              (i + 1) * self.n_snps // n_bloques_min)
                             for i in range(n_bloques_min)]
            segmentos = segmentos_pos
        self.bloques = [np.arange(s, e) for s, e in segmentos]

    def _construir_anclas(self):
        """Tier 1: construye una solución extrema por cada objetivo."""
        anclas = []

        # Ancla 1: Min-k (mínima cardinalidad con cobertura)
        indices = np.argsort(-self.distinguibilidad)
        anclas.append(construir_solucion_greedy(self.H, self.pair_idx, indices))

        # Ancla 2: Max-tolerancia (k-cover al máximo)
        anclas.append(construir_solucion_multicobertura(
            self.H, self.pair_idx, self.max_k, self.rng
        ))

        # Ancla 3: Max distancia media de Hamming
        anclas.append(_ancla_max_hamming_medio(
            self.discrepancia, self.distinguibilidad,
            self.n_snps, self.n_pares, self.rng
        ))

        # Ancla 4: Min varianza
        anclas.append(_ancla_min_varianza(
            self.discrepancia, self.H, self.pair_idx,
            self.n_snps, self.n_pares, self.rng
        ))

        return anclas

    def _sweep_k_cover(self, n):
        """Tier 2: barrido de cobertura progresiva con spacing geométrico."""
        soluciones = []
        k_vals = np.unique(np.geomspace(1, self.max_k, max(n, 2)).astype(int))
        # Distribuir n individuos entre los valores de k
        repeticiones = max(1, n // len(k_vals))
        for k in k_vals:
            for _ in range(repeticiones):
                if len(soluciones) >= n:
                    break
                soluciones.append(construir_solucion_multicobertura(
                    self.H, self.pair_idx, int(k), self.rng
                ))
        # Rellenar si faltan
        while len(soluciones) < n:
            k = int(self.rng.integers(1, self.max_k + 1))
            soluciones.append(construir_solucion_multicobertura(
                self.H, self.pair_idx, k, self.rng
            ))
        return soluciones[:n]

    def _bloques_assembly(self, n):
        """Tier 3: ensamblaje de soluciones por subconjuntos de bloques LD."""
        soluciones = []
        n_bloques = len(self.bloques)
        if n_bloques == 0:
            return soluciones

        for _ in range(n):
            # Subconjunto aleatorio de bloques (entre 40% y 100% de los bloques)
            n_sel = self.rng.integers(max(1, n_bloques * 2 // 5), n_bloques + 1)
            indices = self.rng.choice(n_bloques, size=n_sel, replace=False)
            indices.sort()
            soluciones.append(_construir_solucion_bloques(
                self.discrepancia, self.bloques, self.n_snps,
                self.n_pares, indices, self.rng
            ))
        return soluciones

    def _complementos(self, soluciones_existentes, n):
        """Tier 4: construye complementos de soluciones existentes."""
        resultados = []
        n_base = len(soluciones_existentes)
        if n_base == 0:
            return resultados

        for i in range(n):
            base = soluciones_existentes[i % n_base]
            resultados.append(_construir_complemento(
                self.discrepancia, base, self.n_snps, self.n_pares, self.rng
            ))
        return resultados

    def _guided_sparse(self):
        """Tier 5: un individuo con muestreo disperso ponderado y reparación."""
        # Cardinalidad objetivo: entre el min-k observado y algo moderado
        k_obj = self.rng.integers(
            max(1, int(self.distinguibilidad.size * 0.01)),
            max(2, int(self.distinguibilidad.size * 0.15))
        )
        return _muestreo_guiado_disperso(
            self.distinguibilidad, self.discrepancia, self.n_snps, self.n_pares, k_obj, self.rng
        )

    def _deduplicar(self, X):
        """Post-procesado: deduplicación fenotípica mediante mutaciones monótonas."""
        huellas = set()
        for i in range(len(X)):
            fp = X[i].tobytes()
            if fp in huellas:
                # Mutar de forma segura: convertir de 1 a 3 bits 'False' a 'True'.
                # Añadir SNPs matemáticamente preserva la cobertura existente.
                candidatos = np.where(~X[i])[0]
                if len(candidatos) > 0:
                    n_flips = min(len(candidatos), int(self.rng.integers(1, 4)))
                    bits = self.rng.choice(candidatos, n_flips, replace=False)
                    X[i, bits] = True
            huellas.add(X[i].tobytes())
        return X

    def _do(self, problem, n_samples, **kwargs):
        n_samples = int(n_samples)
        X = np.zeros((n_samples, problem.n_var), dtype=bool)
        pos = 0

        # === Tier 1: Anclas de Pareto (~5-10%) ===
        anclas = self._construir_anclas()
        n_anclas = min(len(anclas), max(2, n_samples // 10))
        for i in range(n_anclas):
            if pos >= n_samples:
                break
            X[pos] = anclas[i]
            pos += 1

        restantes = n_samples - pos
        if restantes <= 0:
            return self._deduplicar(X)

        # === Tier 2: Barrido k-Cover (~25%) ===
        n_t2 = max(1, int(round(restantes * 0.30)))
        for sol in self._sweep_k_cover(n_t2):
            if pos >= n_samples:
                break
            X[pos] = sol
            pos += 1

        # === Tier 3: Ensamblaje por bloques LD (~25%) ===
        n_t3 = max(1, int(round(restantes * 0.30)))
        for sol in self._bloques_assembly(n_t3):
            if pos >= n_samples:
                break
            X[pos] = sol
            pos += 1

        # === Tier 4: Inyección de complementos (~20%) ===
        n_t4 = max(1, int(round(restantes * 0.20)))
        existentes = X[:pos].copy()
        for sol in self._complementos(existentes, n_t4):
            if pos >= n_samples:
                break
            X[pos] = sol
            pos += 1

        # === Tier 5: Exploración guiada dispersa (resto) ===
        while pos < n_samples:
            X[pos] = self._guided_sparse()
            pos += 1

        # === Post-procesado: deduplicación fenotípica ===
        return self._deduplicar(X)
