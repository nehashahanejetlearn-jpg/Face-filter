@echo off
echo ========================================================
echo   Starting VisionAR Face Filter App on Localhost...
echo ========================================================
echo.
echo Opening http://localhost:8080 in your default browser...
cd /d "%~dp0web"

where node >nul 2>nul
if %ERRORLEVEL% equ 0 (
    node server.js
) else (
    python server.py
)
pause
