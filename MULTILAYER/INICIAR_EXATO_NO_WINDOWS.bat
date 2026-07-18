@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>&1
if not errorlevel 1 (
    set "PYTHON=py"
    goto :python_encontrado
)

where python >nul 2>&1
if not errorlevel 1 (
    set "PYTHON=python"
    goto :python_encontrado
)

echo Python 3 nao foi encontrado.
echo Instale-o em https://www.python.org/downloads/windows/
echo Durante a instalacao, marque "Add Python to PATH".
pause
exit /b 1

:python_encontrado
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

