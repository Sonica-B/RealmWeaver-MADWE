@echo off
REM Setup script for MADWE project on Windows

echo Setting up MADWE development environment...

REM Check Python version
python --version 2>&1 | findstr /C:"3.11" >nul
if errorlevel 1 (
    echo Error: Python 3.11.x is required
    exit /b 1
)

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo Installing requirements...
pip install --pre torch torchvision torchaudio 
pip install -r requirements.txt
pip install torch-tensorrt --no-deps



REM Install package in development mode
pip install -e .

REM Create necessary directories
echo Creating data directories...
mkdir data\raw\textures 2>nul
mkdir data\raw\sprites 2>nul
mkdir data\raw\gameplay 2>nul
mkdir data\processed\train 2>nul
mkdir data\processed\val 2>nul
mkdir data\processed\test 2>nul
mkdir data\models\checkpoints 2>nul
mkdir data\models\final 2>nul
mkdir logs 2>nul
mkdir outputs 2>nul

REM Create .env file from example
if not exist .env (
    copy .env.example .env
    echo Created .env file from .env.example
)

echo.
echo Setup complete! 

@REM Activating the virtual environment
venv\Scripts\activate.bat

echo Virtual environment activated.