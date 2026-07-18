#!/usr/bin/env bash

set -e
cd "$(dirname "$0")"

PYTHON=""
for candidato in python3.12 python3.13 python3.11 python3.10 python3.9 python3.14 python3; do
    if command -v "$candidato" >/dev/null 2>&1; then
        PYTHON="$(command -v "$candidato")"
        break
    fi
done

if [ -z "$PYTHON" ]; then
    echo "Python 3.9 ou superior não foi encontrado."
    echo "Instale Python 3.9 ou superior e o pacote python3-venv."
    exit 1
fi


echo "Usando: $($PYTHON --version)"

if [ ! -d ".venv" ]; then
    echo "Criando o ambiente Python..."
    "$PYTHON" -m venv .venv
fi

source .venv/bin/activate
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements_interface.txt

echo "Abrindo a interface EXATA do multilayer k-SEP..."
python -m streamlit run interface_multilayer.py --server.headless false --browser.gatherUsageStats false
