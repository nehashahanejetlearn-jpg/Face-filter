@echo off
echo ========================================================
echo   Launching VisionAR AI Face Filter (Python OpenCV)
echo ========================================================
echo.
echo Installing dependencies if needed...
pip install -r requirements.txt
echo.
echo Launching Face Filter Studio...
python face_filter.py
pause
