@echo off
REM Quick start script for Fake News Detection project

echo.
echo ============================================================
echo Fake News Detection System - Quick Start
echo ============================================================
echo.

echo [Step 1] Installing dependencies...
echo.
pip install -r requirements.txt

echo.
echo [Step 2] Training the model...
echo This will create a trained model and save it to models/
echo.
python train.py

echo.
echo [Step 3] Done! Next steps:
echo.
echo To start the API server, run:
echo   python app.py
echo.
echo Then in another terminal, test the API:
echo   python test_api.py
echo.
echo For detailed information, read README.md
echo.
pause
