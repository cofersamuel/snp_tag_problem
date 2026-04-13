# 📋 Protocolos y Directrices - Proyecto TFG Tag SNPs

Este documento contiene las reglas críticas y flujos de trabajo para cualquier asistente de IA que trabaje en este repositorio.

## 🛡️ Política de Seguridad y Backups
- **Backup Obligatorio**: ANTES de realizar cualquier modificación en el archivo principal del notebook (`tfg_tagsnp_pymoo.ipynb`), se DEBE crear una copia de seguridad.
- **Nomenclatura**: El nombre del backup debe seguir el patrón `tfg_tagsnp_pymoo_backup_YYYYMMDD_HHMM.ipynb` o similar para permitir rastreabilidad.
- **Reversión**: En caso de error o si el usuario no está satisfecho, la primera acción es restaurar desde el último backup funcional.

## 🧬 Contexto del Proyecto
- **Objetivo**: Selección de Tag SNPs mediante algoritmos evolutivos multiobjetivo (NSGA-II, NSGA-III, MOEA/D).
- **Métricas**:
    - **Calidad (Normalizadas [0,1])**: Range, SumMin, MinSum, Hypervolume (HV).
    - **Biológicas (Escala original)**: MaxToleranceRate, AvgToleranceRate, AvgHammingDistance.
- **Interpretación**: 
    - Range, SumMin, MinSum: **Menor es mejor**.
    - Hypervolume (HV): **Mayor es mejor** (indica mayor dominancia y cobertura).
## 💻 Guía de Implementación en el Notebook
- **Documentación**: Cada nueva métrica o cambio lógico debe ir acompañado de una celda Markdown explicativa (bloque matemático y justificación biológica).
