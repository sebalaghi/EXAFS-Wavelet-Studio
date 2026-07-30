@echo off
setlocal
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo Python was not found on PATH. Install Python 3.10 or newer first.
    pause
    exit /b 1
)

if not exist ".venv-build\Scripts\python.exe" (
    echo Creating isolated build environment...
    python -m venv .venv-build
    if errorlevel 1 goto :fail
)

echo Installing build dependencies...
call ".venv-build\Scripts\activate.bat"
python -m pip install --upgrade pip
if errorlevel 1 goto :fail
python -m pip install -r requirements-build.txt
if errorlevel 1 goto :fail

echo Running numerical self-test...
python xafs_wavelet_studio.py --self-test
if errorlevel 1 goto :fail

echo Building one-file Windows application...
python -m PyInstaller --noconfirm --clean --onefile --windowed ^
    --name EXAFS_Wavelet_Studio ^
    --icon EXAFS_Wavelet_Studio.ico ^
    --version-file version_info.txt ^
    --add-data "EXAFS_Wavelet_Studio.png;." ^
    --hidden-import tkinter ^
    --hidden-import matplotlib.backends.backend_tkagg ^
    --exclude-module pandas ^
    --exclude-module scipy ^
    --exclude-module PyQt5 ^
    --exclude-module PyQt6 ^
    --exclude-module PySide2 ^
    --exclude-module PySide6 ^
    xafs_wavelet_studio.py
if errorlevel 1 goto :fail

echo.
echo Build complete:
echo %CD%\dist\EXAFS_Wavelet_Studio.exe
pause
exit /b 0

:fail
echo.
echo Build failed. Read the message above for the exact cause.
pause
exit /b 1
