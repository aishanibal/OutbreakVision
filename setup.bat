@echo off
echo Setting up OutbreakVision environment...

:: Create virtual environment
python -m venv venv

:: Activate virtual environment
call venv\Scripts\activate

:: Upgrade pip
python -m pip install --upgrade pip

:: Install dependencies
pip install -r requirements.txt

echo Setup complete! Run "venv\Scripts\activate" to activate the environment.
