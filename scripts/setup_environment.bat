@echo off
REM Unity ML-Agents Environment Setup for MADWE Project
echo Setting up Unity ML-Agents environment for MADWE...

REM Check Python version
python --version 2>&1 | findstr /C:"3.10" >nul
if errorlevel 1 (
    echo Error: Python 3.10.12 is required
    exit /b 1
)

REM Create conda environment
echo Creating conda environment 'madwe'...
conda create -n madwe python=3.10.12 -y
call conda activate madwe

REM Clone ML-Agents
echo Cloning Unity ML-Agents...
git clone --branch release_22 https://github.com/Unity-Technologies/ml-agents.git

REM Install ML-Agents Python package
cd ml-agents
pip install -e ./ml-agents-envs
pip install -e ./ml-agents

REM Install additional requirements
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install mlagents-envs==1.1.0

REM Create Unity project structure
cd ..
mkdir unity_project
cd unity_project
mkdir Assets Scripts Prefabs Materials

REM Create .gitignore for Unity
echo Library/ > .gitignore
echo Temp/ >> .gitignore
echo Obj/ >> .gitignore
echo Build/ >> .gitignore
echo Builds/ >> .gitignore
echo Logs/ >> .gitignore
echo UserSettings/ >> .gitignore

echo.
echo Unity ML-Agents environment setup complete!
echo Next steps:
echo 1. Open Unity Hub and create new project with Unity 2023.2 LTS
echo 2. Import ML-Agents package (com.unity.ml-agents@3.0.0)
echo 3. Import Sentis package (com.unity.sentis@2.0.0)