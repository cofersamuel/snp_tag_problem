# 🤖 Pipeline de Automatización y Telemetría

Este documento detalla la configuración y las instrucciones operativas para la ejecución desatendida (*unattended*) y el pipeline de telemetría del proyecto `snp_tag`.

---

## 🔑 Configuración del Bot de Telegram

El sistema de telemetría utiliza un bot de Telegram dedicado para transmitir los artifacts experimentales directamente a tu dispositivo personal al finalizar la ejecución.

Antes de ejecutar el pipeline, debes configurar tus credenciales del bot de forma segura:

1. Duplica el archivo de plantilla ubicado en `automation/telemetry_credentials.env.example`.
2. Renombra el archivo duplicado a `telemetry_credentials.env`.
3. Abre `telemetry_credentials.env` e introduce tu Token del Bot y tu Chat ID únicos.

> **Nota de Seguridad:** El archivo `telemetry_credentials.env` está explícitamente ignorado por git (a través de `.gitignore`). **Nunca** subas tus credenciales activas a un repositorio público.

---

## 🚀 Instrucciones de Ejecución

Debido a que el script apaga automáticamente la máquina remota al finalizar la ejecución, debe ejecutarse con privilegios elevados. Además, para garantizar que el script detecte tu entorno activo de Python (Conda o `venv`), **debes** pasar el flag `-E` a `sudo` para preservar tus variables de entorno.

Ejecuta el siguiente comando desde la raíz del proyecto:

```bash
sudo -E ./automation/automation_script.sh
```

---

## 📦 Extracción de Artifacts

Telegram impone un límite de subida estricto de 50 MB. Para evitarlo, el script de automatización divide los resultados comprimidos en chunks de 45 MB (`.part_aa`, `.part_ab`, etc.) almacenados dentro del directorio `automation/archives/`.

Para unir y extraer estos artifacts en tu ordenador personal, ejecuta uno de los siguientes comandos desde el **directorio raíz del proyecto** (`snp_tag_tfg`):

### Opción A: Extraer en el Directorio de Origen Original
Este comando reconstruye la carpeta `snp_tag/results/` exactamente como se generó originalmente en la máquina remota:

```bash
cat automation/archives/experiment_results.tar.gz.part_* | tar -xzf -
```

### Opción B: Extraer en un Subdirectorio de Automatización Dedicado
Si prefieres mantener la salida extraída aislada de tu código fuente principal, este comando descomprimirá los contenidos directamente en un directorio `automation/results/`:

```bash
cat automation/archives/experiment_results.tar.gz.part_* | tar -xzf - -C automation/results/
```
