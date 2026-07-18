#!/bin/bash

set -e
cd "$(dirname "$0")"

if ! command -v python3 >/dev/null 2>&1; then
    echo "Python 3 não foi encontrado."
    echo "Instale-o em https://www.python.org/downloads/macos/ e tente novamente."
    read -r -p "Pressione Enter para fechar."
    exit 1
fi

if [ ! -d ".venv" ]; then
    echo "Criando o ambiente Python..."
    python3 -m venv .venv
fi

source .venv/bin/activate
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements_interface.txt

echo "Abrindo a interface EXATA do multilayer k-SEP..."
python -m streamlit run interface_multilayer.py --server.headless false --browser.gatherUsageStats false

