"""
Módulo de Construcción de Algoritmos (algorithm.py)
--------------------------------------------------
Fabrica las instancias de los algoritmos evolutivos (NSGA-II, NSGA-III, 
SPEA2, MOEA/D) configuradas para el problema de Tag SNPs.
"""

# =============================================================================
# LIBRERÍAS DE TERCEROS
# =============================================================================
import numpy as np
from pymoo.algorithms.moo.age2 import AGEMOEA2
from pymoo.algorithms.moo.ctaea import CTAEA
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3, HyperplaneNormalization
from pymoo.algorithms.moo.rvea import RVEA
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.algorithms.moo.spea2 import SPEA2, SPEA2Survival
from pymoo.algorithms.moo.unsga3 import UNSGA3
from pymoo.core.repair import Repair
from pymoo.decomposition.pbi import PBI
from pymoo.decomposition.tchebicheff import Tchebicheff
from pymoo.decomposition.weighted_sum import WeightedSum
from pymoo.operators.crossover.hux import HUX
from pymoo.operators.crossover.pntx import (SinglePointCrossover,
                                            TwoPointCrossover)
from pymoo.operators.crossover.ux import UX
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.util.dominator import Dominator
from pymoo.util.misc import vectorized_cdist
from pymoo.util.ref_dirs import get_reference_directions

# =============================================================================
# MÓDULOS LOCALES (snp_tag)
# =============================================================================
from snp_tag.config import ConfiguracionExperimento
from snp_tag.core.sampling import (MuestreoAleatorioDisperso,
                                   MuestreoGreedyHolistico,
                                   MuestreoGreedyMultiCobertura,
                                   MuestreoGreedyTing)



def construir_direcciones_referencia(tam_poblacion, n_obj=4): # Función para generar las direcciones de referencia de Das-Dennis
    """
    Genera direcciones de referencia balanceadas para algoritmos basados en descomposición.
    Fuerza compatibilidad estricta con los tamaños de población de Das-Dennis.
    """
    tam_poblacion = int(max(1, tam_poblacion)) # Asegura que el tamaño de población sea un entero mayor o igual a 1
    p = 1 # Inicializa el número de particiones del simplex
    while True: # Inicia un bucle infinito para buscar particiones que coincidan con el tamaño deseado
        cand = get_reference_directions('das-dennis', n_obj, n_partitions=p) # Genera direcciones de referencia candidatas para p particiones
        if len(cand) == tam_poblacion: # Comprueba si el número de direcciones candidatas coincide exactamente con tam_poblacion
            return cand, p # Retorna las direcciones de referencia encontradas y el número de particiones
        elif len(cand) > tam_poblacion: # Si las direcciones candidatas superan el tamaño de población objetivo
            prev_cand = get_reference_directions('das-dennis', n_obj, n_partitions=p-1) if p > 1 else [] # Genera direcciones candidatas anteriores
            prev_len = len(prev_cand) if p > 1 else 1 # Define la longitud previa para mostrar un error amigable
            error_msg = ( # Inicia la construcción de la cadena de error formateada en varias líneas
                f"\n=== [ERROR DE CONFIGURACIÓN] ===\n" # Añade el encabezado decorativo del error
                f"El tamaño de población ({tam_poblacion}) no es compatible " # Explica que la población no coincide con el simplex
                f"con una distribución geométrica Das-Dennis para {n_obj} objetivos.\n" # Detalla los objetivos de Das-Dennis
                f"Los tamaños válidos más cercanos son {prev_len} o {len(cand)}.\n" # Ofrece las alternativas correctas de tamaño
                f"Por favor, ajusta 'tam_poblacion' en tu configuración o usa otro algoritmo.\n" # Sugiere modificar la configuración de entrada
                f"================================\n" # Añade el pie de página decorativo del error
            ) # Cierra los paréntesis del bloque de asignación de la variable error_msg
            print(error_msg) # Imprime el mensaje de error en la consola
            raise ValueError(error_msg) # Lanza una excepción ValueError con el mensaje detallado
        p += 1 # Incrementa el número de particiones para la siguiente iteración del bucle



class ReparacionSNP(Repair): # Clase de reparación de soluciones binarias
    """
    Operador de reparación que intercepta individuos vacíos (k=0) y activa 
    un SNP aleatorio para mantener la validez del fenotipo y la diversidad.
    """
    def _do(self, problem, Z, **kwargs): # Método principal que implementa la lógica de la reparación
        k = Z.sum(axis=1) # Suma las variables de decisión activas (True) de cada solución en la matriz Z
        vacios = np.where(k == 0)[0] # Encuentra los índices de aquellas soluciones que no tienen ningún SNP activo (k = 0)
        if len(vacios) > 0: # Comprueba si existe al menos un individuo vacío en el conjunto
            for idx in vacios: # Itera sobre cada uno de los índices correspondientes a individuos vacíos
                Z[idx, np.random.randint(0, problem.n_var)] = True # Activa un SNP al azar para evitar que la solución se quede sin Tag SNPs
        return Z # Retorna la matriz de variables de decisión reparada

def fabricar_algoritmo(problema, H, nombre_algo, nombre_init, nombre_crossover, cfg: ConfiguracionExperimento, # Definición de la función constructora del algoritmo evolutivo
                       semilla=42, dirs_ref=None): # Continuación de la firma de la función constructora con parámetros adicionales
    """
    Instancia un algoritmo específico con su estrategia de muestreo y operadores.
    """
    # Selección de la estrategia de muestreo
    nombre_init = str(nombre_init) # Convierte el nombre del método de inicialización a tipo string
    base_init = nombre_init # Asigna el nombre a una variable local para facilitar su comprobación

    if base_init == 'random_sparse': # Comprueba si el método de inicialización es muestreo aleatorio disperso
        prob_esperada = max(0.01, min(0.5, 70.0 / problema.n_var)) # Calcula la probabilidad de activación adaptativa basada en el número de variables
        sampling = MuestreoAleatorioDisperso(prob=prob_esperada, semilla=semilla) # Instancia el operador de muestreo aleatorio disperso con la probabilidad calculada
    elif base_init == 'random_dense': # Comprueba si se solicita muestreo denso aleatorio estándar
        sampling = BinaryRandomSampling() # Instancia la clase de muestreo binario aleatorio estándar de pymoo

    elif base_init == 'greedy_multi': # Comprueba si el método seleccionado es el muestreo greedy multi-cobertura
        sampling = MuestreoGreedyMultiCobertura( # Instancia la clase MuestreoGreedyMultiCobertura
            H, # Pasa el diccionario H con la información de los desequilibrios de ligamiento
            problema.pair_idx, # Pasa los índices de pares de SNPs correlacionados del problema
            max_cobertura_objetivo=cfg.max_cobertura_objetivo, # Asigna el límite de cobertura objetivo a partir de la configuración
            semilla=semilla # Pasa la semilla para reproducibilidad en la generación de números pseudoaleatorios
        ) # Cierre del inicializador de MuestreoGreedyMultiCobertura
    elif base_init == 'greedy_ting': # Comprueba si el método de inicialización es el greedy de Ting
        sampling = MuestreoGreedyTing( # Instancia la clase MuestreoGreedyTing
            H, # Pasa el diccionario H de desequilibrios de ligamiento
            problema.pair_idx, # Pasa los índices de pares del problema de optimización
            ratio_greedy=cfg.ratio_greedy_ting, # Configura el porcentaje de decisiones codiciosas frente a aleatorias
            semilla=semilla, # Pasa la semilla para inicializar el generador aleatorio interno
        ) # Cierre del constructor de MuestreoGreedyTing
    elif base_init == 'greedy_holistic': # Comprueba si el método de inicialización es el greedy holístico
        sampling = MuestreoGreedyHolistico( # Instancia la clase MuestreoGreedyHolistico
            H, # Pasa el diccionario H de desequilibrios de ligamiento
            problema.pair_idx, # Proporciona la estructura de índices de pares de SNPs
            max_k=cfg.max_k_holistic, # Establece el número máximo de SNPs a seleccionar en la inicialización
            semilla=semilla, # Establece la semilla del generador pseudoaleatorio
        ) # Cierre del constructor de MuestreoGreedyHolistico
    else: # En caso de que el nombre del método de muestreo no coincida con ninguno disponible
        raise ValueError(f"Estrategia de muestreo no soportada: {nombre_init}") # Lanza una excepción detallando que el método es desconocido

    nombre_crossover = str(nombre_crossover).upper() # Normaliza el nombre del operador de cruce a mayúsculas
    if nombre_crossover == 'HUX': # Comprueba si el operador seleccionado es Half Uniform Crossover
        cruce = HUX(prob=cfg.pc) # Instancia el operador HUX con la probabilidad de cruce configurada
    elif nombre_crossover == '1P': # Comprueba si se solicita cruce en un punto
        cruce = SinglePointCrossover(prob=cfg.pc) # Instancia el operador SinglePointCrossover con su probabilidad
    elif nombre_crossover == '2P': # Comprueba si se solicita cruce en dos puntos
        cruce = TwoPointCrossover(prob=cfg.pc) # Instancia el operador TwoPointCrossover con su probabilidad
    else: # Si no se solicita ninguno de los anteriores, se asume cruce uniforme clásico
        cruce = UX(prob=cfg.pc) # Instancia el operador UX con la probabilidad de cruce configurada
        
    mutacion = BitflipMutation(prob=cfg.pm) # Instancia el operador de mutación por inversión de bits (BitflipMutation)
    reparador = ReparacionSNP() # Crea una instancia del operador de reparación personalizado para corregir soluciones vacías

    if nombre_algo == 'NSGA2': # Comprueba si el algoritmo solicitado es NSGA-II
        return NSGA2(pop_size=cfg.tam_poblacion, sampling=sampling, crossover=cruce, # Instancia y retorna el algoritmo NSGA-II con la población, muestreo y cruce especificados
                     mutation=mutacion, repair=reparador, eliminate_duplicates=True, n_offsprings=cfg.n_descendencia) # Pasa la mutación, reparación, eliminación de duplicados y número de descendientes
    
    if nombre_algo == 'SPEA2': # Comprueba si el algoritmo solicitado es SPEA2
        return SPEA2( # Llama e instancia la clase SPEA2 de pymoo
            pop_size=cfg.tam_poblacion, # Define el tamaño de la población del algoritmo
            sampling=sampling, # Asigna el operador de muestreo de inicialización
            crossover=cruce, # Asigna el operador de cruzamiento configurado
            mutation=mutacion, # Asigna el operador de mutación configurado
            repair=reparador, # Pone el operador personalizado de reparación de cromosomas vacíos
            survival=SPEA2Survival(normalize=True), # Instancia la herramienta nativa de PyMoo (exponiéndola a la división por cero)
            eliminate_duplicates=True, # Habilita la eliminación estricta de soluciones idénticas duplicadas
        ) # Cierre del constructor de SPEA2

    if nombre_algo == 'AGEMOEA2': # Comprueba si el algoritmo solicitado es AGE-MOEA2
        return AGEMOEA2( # Instancia la clase AGEMOEA2 de pymoo
            pop_size=cfg.tam_poblacion, # Establece el tamaño de la población de soluciones
            sampling=sampling, # Establece el método de inicialización seleccionado
            crossover=cruce, # Establece el operador de cruzamiento de padres
            mutation=mutacion, # Establece el operador de mutación genética
            repair=reparador, # Incorpora la reparación de cromosomas sin genes activos
            eliminate_duplicates=True # Habilita el control de duplicados en la población
        ) # Cierre de la inicialización de AGEMOEA2

    if nombre_algo == 'SMSEMOA': # Comprueba si el algoritmo solicitado es SMS-EMOA
        return SMSEMOA( # Instancia la clase SMSEMOA de pymoo
            pop_size=cfg.tam_poblacion, # Pasa el tamaño de la población de la configuración
            sampling=sampling, # Configura la inicialización de soluciones
            crossover=cruce, # Configura el cruce de individuos
            mutation=mutacion, # Configura la mutación para la descendencia
            repair=reparador, # Aplica la reparación de cromosomas no factibles
            eliminate_duplicates=True # Previene la coexistencia de copias idénticas en la población
        ) # Cierre de la inicialización de SMSEMOA

    if dirs_ref is None: # Si no se proporcionaron direcciones de referencia en los parámetros
        dirs_ref, _ = construir_direcciones_referencia(cfg.tam_poblacion, n_obj=4) # Genera las direcciones de referencia basándose en la configuración de la población

    if nombre_algo == 'NSGA3': # Comprueba si el algoritmo a utilizar es NSGA-III
        return NSGA3(pop_size=cfg.tam_poblacion, ref_dirs=dirs_ref, sampling=sampling, # Instancia y retorna el algoritmo NSGA-III pasando la población, direcciones y muestreo
                     crossover=cruce, mutation=mutacion, repair=reparador, eliminate_duplicates=True, # Pasa el cruce, mutación, reparación y eliminación de duplicados
                     n_offsprings=cfg.n_descendencia) # Pasa el número de descendientes por generación al algoritmo

    if nombre_algo == 'RVEA': # Comprueba si el algoritmo solicitado es RVEA
        return RVEA( # Instancia la clase RVEA de pymoo
            pop_size=cfg.tam_poblacion, # Configura el tamaño de la población
            ref_dirs=dirs_ref, # Pasa las direcciones de referencia generadas
            sampling=sampling, # Configura la inicialización de las soluciones binarias
            crossover=cruce, # Asigna el operador de cruce seleccionado
            mutation=mutacion, # Asigna el operador de mutación por flip
            repair=reparador, # Incorpora el operador para corregir soluciones vacías
            eliminate_duplicates=True # Evita la replicación exacta de cromosomas en la población
        ) # Cierre de la inicialización de RVEA

    if nombre_algo == 'CTAEA': # Comprueba si el algoritmo solicitado es C-TAEA
        return CTAEA( # Instancia la clase CTAEA de pymoo
            ref_dirs=dirs_ref, # Pasa las direcciones de referencia para descomposición
            sampling=sampling, # Configura la inicialización de la población
            crossover=cruce, # Asigna el operador de cruzamiento
            mutation=mutacion, # Asigna el operador de mutación
            repair=reparador, # Incorpora la reparación de cromosomas vacíos
            eliminate_duplicates=True # Habilita la eliminación de individuos duplicados
        ) # Cierre de la inicialización de CTAEA

    if nombre_algo == 'UNSGA3': # Comprueba si el algoritmo solicitado es U-NSGA-III
        return UNSGA3( # Instancia la clase UNSGA3 de pymoo
            ref_dirs=dirs_ref, # Proporciona el entramado de direcciones geométricas
            pop_size=cfg.tam_poblacion, # Establece el tamaño poblacional total
            sampling=sampling, # Asigna el operador de muestreo de inicialización
            crossover=cruce, # Asigna el cruzamiento recombinador
            mutation=mutacion, # Asigna la mutación exploratoria
            repair=reparador, # Pone el operador personalizado de reparación
            eliminate_duplicates=True, # Previene la saturación de individuos clónicos
            n_offsprings=cfg.n_descendencia # Configura cuántos hijos se generan por generación
        ) # Cierre de la inicialización de UNSGA3

    if nombre_algo in {'MOEA/D_TCHE', 'MOEA/D_PBI', 'MOEA/D_WS'}: # Verifica si el algoritmo es una variante de MOEA/D basada en descomposición
        if nombre_algo == 'MOEA/D_TCHE': # Comprueba si la variante de MOEA/D elegida utiliza la descomposición Tchebycheff
            descomposicion = Tchebicheff() # Instancia la función de descomposición Tchebycheff de pymoo
        elif nombre_algo == 'MOEA/D_PBI': # Comprueba si la variante elegida es la basada en Boundary Intersection penalizada (PBI)
            descomposicion = PBI(theta=cfg.theta_moead_pbi) # Instancia la descomposición PBI utilizando el parámetro theta de la configuración
        else: # Si no coincide con las anteriores, se selecciona suma ponderada simple
            descomposicion = WeightedSum() # Instancia la función de descomposición de suma ponderada (WeightedSum)

        algoritmo = MOEAD( # Instancia la clase MOEAD original importada de pymoo
            ref_dirs=dirs_ref, # Pasa las direcciones de referencia requeridas por MOEA/D
            n_neighbors=cfg.vecinos_moead, # Configura el tamaño de la vecindad de cada subproblema
            prob_neighbor_mating=cfg.prob_vecindad_moead, # Configura la probabilidad de seleccionar padres dentro de la vecindad
            decomposition=descomposicion, # Asigna la función de descomposición instanciada previamente
            sampling=sampling, # Asigna el método de inicialización de la población
            crossover=cruce, # Asigna el operador de cruzamiento de individuos vecinos
            mutation=mutacion, # Asigna el operador de mutación para la exploración
            repair=reparador, # Asigna el reparador de cromosomas para evitar k=0
        ) # Cierre del constructor del algoritmo MOEAD


        return algoritmo # Retorna la instancia configurada del algoritmo MOEA/D

    raise ValueError(f"Algoritmo no reconocido: {nombre_algo}") # Lanza una excepción indicando que el nombre del algoritmo no coincide con ninguno soportado
