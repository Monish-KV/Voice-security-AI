@echo off
setlocal
title VoiceShield AI - Setup

set "ENV=%LOCALAPPDATA%\VoiceShieldVenv"
echo.
echo ==========================================
echo       VoiceShield AI Security Setup
echo ==========================================
echo.
echo Using a short virtual-environment path:
echo %ENV%
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo ERROR: Python was not found.
    echo Install Python 3.12 and make sure "Add Python to PATH" is enabled.
    pause
    exit /b 1
)

if not exist "%ENV%\Scripts\python.exe" (
    echo Creating virtual environment...
    python -m venv "%ENV%"
    if errorlevel 1 (
        echo ERROR: Could not create the virtual environment.
        pause
        exit /b 1
    )
)

echo.
echo Upgrading pip...
"%ENV%\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 (
    echo ERROR: pip upgrade failed.
    pause
    exit /b 1
)

echo.
echo Installing VoiceShield dependencies...
"%ENV%\Scripts\python.exe" -m pip install -r "%~dp0requirements.txt"
if errorlevel 1 (
    echo.
    echo ERROR: Dependency installation failed.
    echo The project itself is OK. Please send me the error shown above.
    pause
    exit /b 1
)

echo.
echo ==========================================
echo Setup completed successfully.
echo ==========================================
echo.
echo Run "run.bat" to start VoiceShield.
echo.
pause
