#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# INSTALADOR PARA LINUX
# ──────────────────────────────────────────────────────────────────────────────

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="audio_gpu"
YML_PATH="$SCRIPT_DIR/environment.yml"
APP_DIR="$SCRIPT_DIR/Prototipo5"
LAUNCHER="$SCRIPT_DIR/lanzar.sh"

echo ""
echo "════════════════════════════════════════════════════════"
echo "  INSTALADOR - Predictor de parámetros FM (TFG)"
echo "════════════════════════════════════════════════════════"
echo ""
echo "Este script:"
echo "  1. Comprueba si tienes Conda instalado"
echo "  2. Si no lo tienes, descarga e instala Miniconda3"
echo "  3. Crea el entorno '$ENV_NAME' con todas las dependencias"
echo "  4. Genera el archivo lanzar.sh para ejecutar la app"
echo ""
echo "AVISO: La primera instalación puede tardar 15-25 minutos"
echo "       y requiere conexión a internet (~3 GB de descarga)."
echo ""
read -rp "Pulsa ENTER para continuar..."

# ──────────────────────────────────────────────────────────────────────────────
# PASO 1: Buscar conda
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "[1/4] Buscando Conda..."

CONDA_EXE=""

for candidate in \
    "$HOME/miniconda3/bin/conda" \
    "$HOME/anaconda3/bin/conda" \
    "/opt/conda/bin/conda" \
    "/opt/miniconda3/bin/conda" \
    "/usr/local/anaconda3/bin/conda"; do
    if [ -f "$candidate" ]; then
        CONDA_EXE="$candidate"
        break
    fi
done

if [ -z "$CONDA_EXE" ] && command -v conda &>/dev/null; then
    CONDA_EXE="$(command -v conda)"
fi

# ──────────────────────────────────────────────────────────────────────────────
# PASO 2: Si no se encuentra conda, descargar e instalar Miniconda3
# ──────────────────────────────────────────────────────────────────────────────
if [ -z "$CONDA_EXE" ]; then
    echo "[!] Conda no encontrado en el sistema."
    echo ""
    echo "Descargando Miniconda3 para Linux (puede tardar según tu conexión)..."
    echo ""

    MINICONDA_INSTALLER="/tmp/Miniconda3-latest-Linux-x86_64.sh"
    MINICONDA_DIR="$HOME/miniconda3"

    if command -v curl &>/dev/null; then
        curl -fsSL "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" \
            -o "$MINICONDA_INSTALLER"
    elif command -v wget &>/dev/null; then
        wget -q "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" \
            -O "$MINICONDA_INSTALLER"
    else
        echo "[ERROR] No se encontró curl ni wget. Instala uno de ellos e inténtalo de nuevo."
        exit 1
    fi

    bash "$MINICONDA_INSTALLER" -b -p "$MINICONDA_DIR"
    rm -f "$MINICONDA_INSTALLER"

    CONDA_EXE="$MINICONDA_DIR/bin/conda"
    echo "[OK] Miniconda3 instalado en $MINICONDA_DIR"
fi

echo "[OK] Conda encontrado: $CONDA_EXE"

CONDA_BASE="$("$CONDA_EXE" info --base)"
echo "[OK] Base de Conda: $CONDA_BASE"

# ──────────────────────────────────────────────────────────────────────────────
# PASO 3: Crear el entorno desde environment.yml
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "[3/4] Creando entorno '$ENV_NAME'..."
echo "      (Descargará ~3 GB de paquetes - es normal que tarde)"
echo ""

if [ ! -f "$YML_PATH" ]; then
    echo "[ERROR] No se encuentra el archivo environment.yml"
    echo "        Asegúrate de que está en la misma carpeta que instalar.sh"
    exit 1
fi

if "$CONDA_EXE" env list | grep -q "^$ENV_NAME "; then
    echo "[!] El entorno '$ENV_NAME' ya existe. Actualizando dependencias..."
    echo ""
    "$CONDA_EXE" env update -f "$YML_PATH" --prune
else
    "$CONDA_EXE" env create -f "$YML_PATH"
fi

if [ $? -ne 0 ]; then
    echo ""
    echo "[ERROR] Fallo al crear/actualizar el entorno."
    echo "        Revisa los mensajes de error anteriores."
    echo "        Consejo: Si hay conflictos de paquetes, borra el entorno con:"
    echo "          conda env remove -n $ENV_NAME"
    echo "        y vuelve a ejecutar este instalador."
    exit 1
fi

echo ""
echo "[OK] Entorno '$ENV_NAME' configurado correctamente."

# ──────────────────────────────────────────────────────────────────────────────
# PASO 4: Generar el launcher lanzar.sh
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "[4/4] Generando lanzar.sh..."

cat > "$LAUNCHER" << EOF
#!/usr/bin/env bash
# Lanzador generado automáticamente por instalar.sh
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate $ENV_NAME
cd "$APP_DIR"
python main.py
EOF

chmod +x "$LAUNCHER"

echo "[OK] Archivo lanzar.sh generado."
echo ""
echo "════════════════════════════════════════════════════════"
echo "  Instalación completada con éxito!"
echo "════════════════════════════════════════════════════════"
echo ""
echo "Para iniciar la aplicación, ejecuta:"
echo "  bash lanzar.sh"
echo ""
