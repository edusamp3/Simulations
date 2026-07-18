@echo off
setlocal
cd /d "%~dp0"

py -3.12 --version >nul 2>&1
if not errorlevel 1 (set "PYTHON=py -3.12"& goto :python_encontrado)
py -3.13 --version >nul 2>&1
if not errorlevel 1 (set "PYTHON=py -3.13"& goto :python_encontrado)
py -3.11 --version >nul 2>&1
if not errorlevel 1 (set "PYTHON=py -3.11"& goto :python_encontrado)
py -3.10 --version >nul 2>&1
if not errorlevel 1 (set "PYTHON=py -3.10"& goto :python_encontrado)
py -3.9 --version >nul 2>&1
if not errorlevel 1 (set "PYTHON=py -3.9"& goto :python_encontrado)
py -3.14 --version >nul 2>&1
if not errorlevel 1 (set "PYTHON=py -3.14"& goto :python_encontrado)

where python >nul 2>&1
if not errorlevel 1 (
    set "PYTHON=python"
    goto :python_encontrado
)

echo Python 3.9 ou superior nao foi encontrado.
echo Instale-o em https://www.python.org/downloads/windows/
echo Durante a instalacao, marque "Add Python to PATH".
pause
exit /b 1

:python_encontrado
%PYTHON% --version
if not exist ".venv\Scripts\python.exe" (
    echo Criando o ambiente Python...
    %PYTHON% -m venv .venv
    if errorlevel 1 goto :erro
)

call ".venv\Scripts\activate.bat"
echo Instalando ou verificando as dependencias...
python -m pip install --quiet --upgrade pip
if errorlevel 1 goto :erro
python -m pip install --quiet -r requirements_interface.txt
if errorlevel 1 goto :erro

echo Abrindo a interface EXATA do multilayer k-SEP...
python -m streamlit run interface_multilayer.py --server.headless false --browser.gatherUsageStats false
if errorlevel 1 goto :erro
exit /b 0

:erro
echo.
echo Ocorreu um erro. Copie ou fotografe as mensagens acima.
pause
exit /b 1
