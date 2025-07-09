#!/bin/bash

ASCII_ART="
██████╗  █████╗ ██████╗ ██╗   ██╗██╗     ███╗   ███╗      ███████╗███████╗ ██████╗ ██╗     ███████╗███╗   ██╗
██╔══██╗██╔══██║██╔══██╗╚██╗ ██╔╝██║     ████╗ ████║      ██╔════╝██╔════╝██╔═══██╗██║     ██╔════╝████╗  ██║
██████╔╝███████║██████╔╝ ╚████╔╝ ██║     ██╔████╔██║█████╗███████╗█████╗  ██║   ██║██║     █████╗  ██╔██╗ ██║
██╔══██╗██╔══██║██╔══██╗  ╚██╔╝  ██║     ██║╚██╔╝██║╚════╝╚════██║██╔══╝  ██║▄▄ ██║██║     ██╔══╝  ██║╚██╗██║
██████╔╝██║  ██║██████╔╝   ██║   ███████╗██║ ╚═╝ ██║      ███████║███████╗╚██████╔╝███████╗███████╗██║ ╚████║
╚═════╝ ╚═╝  ╚═╝╚═════╝    ╚═╝   ╚══════╝╚═╝     ╚═╝      ╚══════╝╚══════╝ ╚══▀▀═╝ ╚══════╝╚══════╝╚═╝  ╚═══╝"

# Print the ASCII art
echo -e "\033[1;36m$ASCII_ART\033[0m"

module load cuda/11.8
module load gcc/11

# -------------------- AUTHENTICATE WITH HUGGING FACE --------------------
if ! command -v huggingface-cli &> /dev/null; then
    echo "Hugging Face CLI not found. Installing..."
    pip install --user huggingface-hub[cli] # install huggingface-cli globally
fi

echo "🔑 Authenticating with Hugging Face..."
if [ -f ~/.huggingface/token ]; then
    echo "✓ Already logged in to Hugging Face"
else
    huggingface-cli login
fi

# -------------------- AUTHENTICATE WITH WEIGHTS & BIASES --------------------

if ! command -v wandb &> /dev/null; then
    echo "WandB not found. Installing..."
    pip install --user wandb # install wandb globally
fi

echo "🔑 Authenticating with Weights & Biases..."
if wandb status &>/dev/null; then
    echo "✓ Already logged in to Weights & Biases"
else
    wandb login
fi

# -------------------- POETRY SETUP --------------------
echo "Setting up Poetry..."

if ! command -v poetry &> /dev/null; then
    echo "Poetry not found. Installing..."
    curl -sSL https://install.python-poetry.org | python3 - 
    export PATH="$HOME/.local/bin:$PATH"
fi

# Verify Poetry Installation
if ! command -v poetry &> /dev/null; then
    echo "❌ Poetry installation failed or not in PATH."
    echo "Please add \$HOME/.local/bin to your PATH and try again."
    exit 1
else
    echo "✓ Poetry is installed."
fi

# -------------------- CREATE/UPDATE VIRTUAL ENV --------------------
echo "📦 Setting up virtual environment..."

# Ensure .venv is created in the project directory; not necessary but whatever

poetry env use $(which python)
poetry install --no-root

# Activate the virtual environment
source .venv/bin/activate

# -------------------- FINAL SETUP --------------------
echo "✅ Setup complete! Ready to train models!"
