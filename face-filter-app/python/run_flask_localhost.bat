@echo off
echo ========================================================
echo   Launching VisionAR Flask Localhost Server...
echo ========================================================
echo.
echo Installing dependencies if needed...
pip install -r requirements.txt
echo.
echo Starting Flask server on http://localhost:5000 ...
python flask_app.py
pause
