#!/bin/bash

# Identificar dinámicamente el directorio del script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/telemetry_credentials.env"

# Códigos de color ANSI para la terminal
GREEN='\033[0;32m'
CYAN='\033[0;36m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # Sin color

# Funciones de logging visual
log_info() { echo -e "${CYAN}[⚙️  SYSTEM] $1${NC}"; }
log_success() { echo -e "${GREEN}[✅ SUCCESS] $1${NC}"; }
log_error() { echo -e "${RED}[❌ ERROR] $1${NC}"; }
log_warning() { echo -e "${YELLOW}[⚠️  WARNING] $1${NC}"; }

log_info "Iniciando el pipeline de ejecución unattended..."

# Pre-flight check 1: Verificar que el archivo de credenciales existe
if [ ! -f "$ENV_FILE" ]; then
    log_error "No se encuentra el archivo de credenciales: $ENV_FILE"
    log_error "Por favor, renombra 'telemetry_credentials.env.example' a '.env' e introduce tus datos."
    exit 1
fi

# Cargar las credenciales de Telegram de forma segura
source "$ENV_FILE"

# Variables de configuración de directorios
TARGET_DIR="snp_tag/results/"
ARCHIVE_DIR="automation/archives"
ARCHIVE_PREFIX="${ARCHIVE_DIR}/experiment_results.tar.gz"

# Resolución dinámica del ejecutable de Python para compatibilidad universal
# (Soporta Conda, venv, y sistema, sorteando la restricción secure_path de sudo)
if [ -n "$VIRTUAL_ENV" ]; then
    PYTHON_EXEC="$VIRTUAL_ENV/bin/python"
elif [ -n "$CONDA_PREFIX" ]; then
    PYTHON_EXEC="$CONDA_PREFIX/bin/python"
else
    PYTHON_EXEC="$(command -v python3 || command -v python)"
fi

# Navegar a la raíz del proyecto dinámicamente
cd "$SCRIPT_DIR/.." || { log_error "No se pudo localizar el directorio raíz del proyecto."; exit 1; }

# Pre-flight check 2: Verificar que Python está disponible y el entorno es correcto
if ! command -v "$PYTHON_EXEC" &> /dev/null; then
    log_error "No se encuentra el comando '$PYTHON_EXEC'. Asegúrate de tener Python instalado."
    exit 1
fi

if ! "$PYTHON_EXEC" -c "import snp_tag" &> /dev/null; then
    log_error "El entorno de Python actual no tiene acceso a 'snp_tag' o faltan dependencias."
    log_error "Si usas un entorno (venv/Conda) y ejecutas con sudo, DEBES preservar el entorno activo."
    log_error "Por favor, ejecuta el script utilizando la flag -E:  sudo -E ./automation/automation_script.sh"
    exit 1
fi

# Ejecutar el módulo principal para 30 runs independientes
# Comando real: "$PYTHON_EXEC" -m snp_tag --mode full_30 # Comando real
# Comando de test: "$PYTHON_EXEC" -m snp_tag --mode fast # Comando de test

"$PYTHON_EXEC" -m snp_tag --mode fast # Comando de test

# Verificar la ejecución exitosa antes de continuar
if [ $? -eq 0 ]; then
    log_success "Ejecución completada con éxito. Comenzando proceso de archiving..."
    
    # Asegurar que el directorio dedicado a los archives existe y limpiar artifacts previos
    mkdir -p "$ARCHIVE_DIR"
    rm -f "${ARCHIVE_PREFIX}.part_"*
    
    # Comprimir y dividir el output en chunks de 45MB para evitar límites de Telegram
    tar -czf - "$TARGET_DIR" | split -b 45M - "${ARCHIVE_PREFIX}.part_"

    log_info "Archiving completado. Transmitiendo artifacts al canal de telemetría..."

    # Enviar cada chunk secuencialmente a través de la API del Bot de Telegram
    for file in "${ARCHIVE_PREFIX}.part_"*; do
        if [ -f "$file" ]; then
            log_info "Enviando chunk: $file..."
            curl -s -F document=@"$file" "https://api.telegram.org/bot${BOT_TOKEN}/sendDocument?chat_id=${CHAT_ID}" > /dev/null
            sleep 2 # Breve pausa para mitigar el rate limiting de los servidores
        fi
    done

    log_success "Transmisión de artifacts completada."
else
    log_error "Fallo en la ejecución. Alertando al canal de telemetría..."
    # Enviar mensaje de texto notificando el fallo de forma genérica
    curl -s -d "chat_id=${CHAT_ID}&text=WARNING: La ejecución del pipeline snp_tag ha fallado en la máquina remota." "https://api.telegram.org/bot${BOT_TOKEN}/sendMessage" > /dev/null
fi

log_warning "Iniciando system halt para apagar la máquina..."
# El sistema requiere privilegios elevados para ejecutar la instrucción de halt
# shutdown -h now # Deshabilitado para testing
