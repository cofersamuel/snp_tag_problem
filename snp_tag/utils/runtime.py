"""
Módulo de Utilidades de Runtime (runtime.py)
-------------------------------------------
Centraliza heurísticas operativas de ejecución, como el cálculo adaptativo
de workers en paralelo en función de CPU y memoria disponible.
"""

import os


def calcular_max_workers_paralelo(ram_por_worker_mb: int = 350, ram_reservada_mb: int = 2048) -> int:
    """Calcula un número seguro de workers en base a CPU y RAM disponible."""
    cpu_limit = max(1, (os.cpu_count() or 2) - 2)
    try:
        with open('/proc/meminfo', 'r') as memf:
            for line in memf:
                if line.startswith('MemAvailable:'):
                    available_mb = int(line.split()[1]) / 1024  # kB -> MB
                    break
            else:
                available_mb = 8000
        ram_limit = max(1, int((available_mb - ram_reservada_mb) / max(1, ram_por_worker_mb)))
    except Exception:
        ram_limit = 4
    return min(cpu_limit, ram_limit)
