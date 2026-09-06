@echo off
setlocal
title VoiceShield AI Security

set "ENV=%LOCALAPPDATA%\VoiceShieldVenv"

if not exist "%ENV%\Scripts\python.exe" (
    echo VoiceShield environment not found.
    echo Running setup first...
    call "%~dp0setup.bat"
    if errorlevel 1 exit /b 1
)

echo.
echo Starting VoiceShield AI Security...
echo Open: http://127.0.0.1:5000
echo.
echo Press CTRL+C in this window to stop the server.
echo.

"%ENV%\Scripts\python.exe" "%~dp0app.py"
pause
