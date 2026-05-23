@echo off
REM Alzheimer Disease Prediction System - Windows Startup Script

echo.
echo ================================
echo Alzheimer Disease Prediction
echo IoMT & Deep Learning System
echo ================================
echo.

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install requirements
echo Installing dependencies...
pip install -r requirements.txt -q

REM Initialize database if needed
if not exist "alzheimer_db.db" (
    echo Initializing database...
    python setup.py
)

REM Start the application
echo.
echo ================================
echo Starting application...
echo ================================
echo.
echo Access the application at: http://localhost:5000
echo.
echo Admin Login:
echo   Username: admin
echo   Password: admin123
echo.
echo Demo User Login:
echo   Username: demo
echo   Password: demo123
echo.

python run.py

pause
