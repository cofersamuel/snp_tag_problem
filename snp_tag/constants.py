"""
Módulo de Constantes Globales (constants.py)
--------------------------------------------
Define enumeraciones y listas de configuración inmutables compartidas 
por todo el pipeline, mitigando el uso de "magic strings".
"""

# Orden preferido de algoritmos para la visualización conjunta
PREFERRED_ALGORITHMS_ORDER = [
    'NSGA3', 'MOEAD_TCHE', 'MOEAD_PBI', 'MOEAD_WS', 'NSGA2', 'SPEA2',
    'AGEMOEA2', 'SMSEMOA', 'RVEA'
]

# Definición de las métricas que se intentan maximizar (el resto se minimiza)
HIGHER_IS_BETTER_METRICS = [
    'Hypervolume', 
    'Range', 
    'MaxToleranceRate', 
    'AvgToleranceRate', 
    'AvgHammingDistance'
]

# Lista base de todas las métricas procesadas en el pipeline estadístico
BASE_METRICS = [
    'MinSum', 'Range', 'SumMin', 
    'MaxToleranceRate', 'AvgToleranceRate', 'AvgHammingDistance', 
    'Hypervolume', 'IGD+', 'GD+'
]

# Mapeo de identificador de métrica a título descriptivo para gráficos
METRICS_DISPLAY_NAMES = {
    'Range': 'Rango (Range): Diversidad Geométrica',
    'MinSum': 'MinSum: Convergencia Central',
    'SumMin': 'SumMin: Convergencia Marginal',
    'MaxToleranceRate': 'Tasa de Tolerancia Máxima',
    'AvgToleranceRate': 'Tasa de Tolerancia Promedio',
    'AvgHammingDistance': 'Distancia Hamming Promedio',
    'Hypervolume': 'Hipervolumen (HV)',
    'IGD+': 'Distancia Generacional Invertida Plus (IGD+)',
    'GD+': 'Distancia Generacional Plus (GD+)'
}
