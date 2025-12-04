# --- Configuration ---
set shell := ["bash", "-c"]
project_file := "pmsm_project.py"
venv_dir := ".venv"

# Default target
default: help

# --- Menu ---
help:
    @echo "🤖 PMSM Control Simulation - Workflow Manager"
    @echo "-------------------------------------------"
    @echo "just install   -> 🏗️  Initialize virtual environment & Install dependencies"
    @echo "just dev       -> 👨‍💻 Start Marimo Editor (Auto-opens Browser)"
    @echo "just app       -> 🚀 Run as Read-Only Web Dashboard"
    @echo "just clean     -> 🧹 Remove cache files"
    @echo "just nuke      -> ☢️  Destroy environment & reset everything"

# --- Commands ---

# 1. Setup: Create venv and install packages
install:
    @echo "📦 Setting up Python virtual environment..."
    # create venv if not exists
    [ -d {{venv_dir}} ] || python3 -m venv {{venv_dir}}
    @echo "Installing dependencies from requirements.txt..."
    {{venv_dir}}/bin/python -m pip install --upgrade pip
    {{venv_dir}}/bin/python -m pip install -r requirements.txt

# 2. Development: Run marimo in virtualenv
dev:
    @echo "🔌 Launching Development Environment..."
    {{venv_dir}}/bin/python -m marimo edit {{project_file}}

# 3. Production/Presentation Mode
app:
    @echo "🌟 Launching Dashboard..."
    {{venv_dir}}/bin/python -m marimo run {{project_file}}

# 4. Utilities
clean:
    rm -rf __pycache__
    rm -rf .ipynb_checkpoints

nuke:
    @echo "WARNING: Deleting virtual environment..."
    rm -rf {{venv_dir}}
    rm -f requirements.txt
