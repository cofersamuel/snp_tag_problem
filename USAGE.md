Uso (CLI)

Instalación de dependencias (desde la raíz del repositorio):

```bash
pip install -r requirements.txt
```

Ejecución del pipeline desde la línea de comandos (ejemplo):

```bash
python -m snp_tag.main --mode fast --data-source synthetic
```

Opciones:
- `--mode` / `-m`: modo de ejecución (`fast`, `medium`, `high`, `full`).
- `--data-source` / `-d`: fuente de datos (`synthetic` o `hinds2005`).
- `--report-only-csv`: ejecuta solo **REPORTES Y VISUALIZACIÓN** usando automáticamente los CSV más recientes encontrados en `snp_tag/input/`.

CSVs generados en ejecuciones normales (directorio `.../1_ejecuciones/`):
- `resultados_detallados_<modo>.csv`
- `historico_generacional_<modo>.csv`
- `frentes_pareto_<modo>.csv` (necesario para regenerar los gráficos del frente de Pareto en modo report-only)

Comportamiento del modo `--report-only-csv` para los gráficos de Pareto:
- Si existe `frentes_pareto_*.csv` y tiene el esquema esperado, se generan los gráficos de Pareto.
- Si no está disponible (o faltan columnas requeridas), la sección de Pareto se omite con un aviso explícito, pero el resto de reportes continúa.

Los parámetros tunables se configuran en `user_config.ini`.
Formato recomendado:
`[Seccion]` + `clave = valor ; explicación`
