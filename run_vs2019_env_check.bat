@echo off
setlocal EnableExtensions

set "VSDEVCMD=C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\Common7\Tools\VsDevCmd.bat"
set "CONDA_BAT=C:\Users\CGlab\anaconda3\condabin\conda.bat"
set "CONDA_ENV=EarLandMarking_DeepLA"
set "REPO_DIR=%~dp0"

echo [INFO] Activating conda environment: %CONDA_ENV%
if not exist "%CONDA_BAT%" (
    echo [ERROR] conda.bat not found: %CONDA_BAT%
    exit /b 1
)
call "%CONDA_BAT%" activate "%CONDA_ENV%"
if errorlevel 1 exit /b %errorlevel%

echo [INFO] Loading VS2019 Native Tools...
if not exist "%VSDEVCMD%" (
    echo [ERROR] VsDevCmd.bat not found.
    exit /b 1
)
call "%VSDEVCMD%" -arch=amd64 -host_arch=amd64
if errorlevel 1 exit /b %errorlevel%

cd /d "%REPO_DIR%"
echo [INFO] Working directory: %CD%

echo [CHECK] python path
where python

echo [CHECK] python version
python --version

echo [CHECK] cl path
where cl

echo [CHECK] nvcc version
nvcc --version

echo [CHECK] torch / CUDA
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"

echo [PASS] VS2019 + conda environment check completed.
endlocal
