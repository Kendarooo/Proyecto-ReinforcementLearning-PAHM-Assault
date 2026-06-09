#!/usr/bin/env bash
# setup_env.sh — v1.1
# Crea el entorno virtual con uv (Python 3.11) e instala todas las dependencias.
# Las ROMs de Atari se instalan via ale-py[roms] sin pasos adicionales.
#
# Uso:
#   chmod +x setup_env.sh
#   ./setup_env.sh

set -e  # Abortar ante cualquier error

echo ""
echo "=== Setup Etapa 0 — Boxing DQN v1.1 ==="
echo ""

# ── 1. Verificar que uv está instalado ────────────────────────────────────────
if ! command -v uv &> /dev/null; then
    echo "[ERROR] 'uv' no está instalado."
    echo "        Instálalo con: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi
echo "[OK] uv encontrado: $(uv --version)"

# ── 2. Crear entorno virtual con Python 3.11 ──────────────────────────────────
echo ""
echo "[1] Creando entorno virtual con Python 3.11..."
uv venv .venv --python 3.11
echo "[OK] Entorno virtual creado en .venv/"

# ── 3. Instalar dependencias desde pyproject.toml ────────────────────────────
echo ""
echo "[2] Instalando dependencias (esto puede tardar unos minutos)..."
uv pip install --python .venv/bin/python -e ".[dev]"
echo "[OK] Dependencias instaladas (ROMs incluidas via ale-py[roms])"

# ── 4. Verificar ale-py ───────────────────────────────────────────────────────
echo ""
echo "[3] Verificando ale-py..."
.venv/bin/python -c "import ale_py; print(f'[OK] ale-py {ale_py.__version__}')"

# ── 5. Ejecutar sanity check ──────────────────────────────────────────────────
echo ""
echo "[4] Ejecutando sanity check..."
.venv/bin/python sanity_check.py

echo ""
echo "=== Setup completado. Activa el entorno con: ==="
echo "    source .venv/bin/activate.fish   # fish shell"
echo "    source .venv/bin/activate        # bash/zsh"
echo ""