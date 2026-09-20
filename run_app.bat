@echo off
REM ============================================================
REM  Precision Agriculture Advisor - App Launcher
REM  Double-click this file to start the Streamlit app.
REM ============================================================

REM Move to the folder this .bat file lives in, regardless of
REM where it's double-clicked from (Desktop shortcut, etc.)
cd /d "%~dp0"

echo.
echo ============================================
echo  Precision Agriculture Advisor
echo ============================================
echo.

REM Check that the virtual environment exists
if not exist ".venv\Scripts\activate.bat" (
    echo ERROR: Could not find .venv\Scripts\activate.bat
    echo Make sure this .bat file is in the project root folder,
    echo next to the ".venv" folder.
    echo.
    pause
    exit /b 1
)

echo Activating virtual environment...
call .venv\Scripts\activate.bat

REM Check that app.py exists
if not exist "app.py" (
    echo ERROR: Could not find app.py in this folder.
    echo Make sure this .bat file sits in the project root,
    echo next to app.py.
    echo.
    pause
    exit /b 1
)

REM Check that streamlit is actually installed in THIS environment
python -m streamlit --version >nul 2>&1
if errorlevel 1 (
    echo Streamlit not found in this environment - installing dependencies...
    echo (This can take a few minutes the first time, especially TensorFlow.)
    echo.
    python -m pip install -r requirements.txt
    echo.
)

echo Starting Streamlit app...
echo (This will open in your default browser. Keep this window open.)
echo.
python -m streamlit run app.py

REM If Streamlit exits or crashes, keep the window open so you can read the error
echo.
echo ============================================
echo  App has stopped.
echo ============================================
pause
