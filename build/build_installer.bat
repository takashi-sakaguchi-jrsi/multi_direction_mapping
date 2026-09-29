@echo off
setlocal EnableExtensions
REM 3-direction colormap product build (PyInstaller onedir)
REM Usage: build\build_installer.bat  (from repo root, or double-click)
REM Output: dist\main_twopass_<VERSION>\

pushd "%~dp0.."
if errorlevel 1 (
    echo [ERROR] Cannot change directory to repository root.
    exit /b 1
)

echo ========================================
echo 3-direction CameraCarSim build
echo ========================================
echo.

echo [Step 1/7] Environment Check...
set "PY_CMD=python"
python --version >nul 2>&1
if not errorlevel 1 goto have_python
py -3 --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Install Python 3.10+ and add it to PATH.
    popd
    exit /b 1
)
set "PY_CMD=py -3"
:have_python
echo   - Python: OK  [%PY_CMD%]

%PY_CMD% -m PyInstaller --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] PyInstaller not found. Install with:
    echo   %PY_CMD% -m pip install pyinstaller
    popd
    exit /b 1
)
echo   - PyInstaller: OK

%PY_CMD% -c "import numpy, cv2, pandas, scipy, matplotlib, pytesseract, openpyxl" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Required packages missing. Install with:
    echo   %PY_CMD% -m pip install -r requirements.txt
    popd
    exit /b 1
)
echo   - Required packages: OK
echo.

echo [Step 1.5/7] Check required files...
if not exist data\config\default_config.json (
    echo [ERROR] data\config\default_config.json not found
    popd
    exit /b 1
)
echo   - data\config\default_config.json: OK
if not exist main_twopass.spec (
    echo [ERROR] main_twopass.spec not found
    popd
    exit /b 1
)
echo   - main_twopass.spec: OK
echo.

echo [Step 2/7] Cleanup...
if exist build\main_twopass rmdir /s /q build\main_twopass
if exist dist\main_twopass rmdir /s /q dist\main_twopass
echo   - Cleanup complete
echo.

echo [Step 3/7] Version...
for /f "usebackq delims=" %%i in (`%PY_CMD% -c "import datetime; print(datetime.datetime.now().strftime('%%Y%%m%%d_%%H%%M%%S'))"`) do set VERSION=%%i
if not defined VERSION (
    echo [ERROR] Failed to get version timestamp
    popd
    exit /b 1
)
echo   - Version: %VERSION%
echo.

echo [Step 4/7] PyInstaller...
%PY_CMD% -m PyInstaller --clean --noconfirm main_twopass.spec
if errorlevel 1 (
    echo [ERROR] PyInstaller failed
    popd
    exit /b 1
)
echo   - Build complete
echo.

echo [Step 5/7] Layout product folders...
if not exist dist\main_twopass\main_twopass.exe (
    echo [ERROR] dist\main_twopass\main_twopass.exe missing
    popd
    exit /b 1
)

if not exist dist\main_twopass\data mkdir dist\main_twopass\data
if not exist dist\main_twopass\data\config mkdir dist\main_twopass\data\config
if not exist dist\main_twopass\data\calibration mkdir dist\main_twopass\data\calibration
if not exist dist\main_twopass\data\input mkdir dist\main_twopass\data\input
if not exist dist\main_twopass\data\output mkdir dist\main_twopass\data\output
if not exist dist\main_twopass\debug mkdir dist\main_twopass\debug
if not exist dist\main_twopass\Log mkdir dist\main_twopass\Log
if not exist dist\main_twopass\progress mkdir dist\main_twopass\progress
if not exist dist\main_twopass\reports mkdir dist\main_twopass\reports

copy /Y data\config\default_config.json dist\main_twopass\data\config\default_config.json >nul
if errorlevel 1 (
    echo [ERROR] Failed to copy default_config.json
    popd
    exit /b 1
)

if exist data\calibration (
    for %%F in (data\calibration\*.json) do copy /Y "%%F" dist\main_twopass\data\calibration\ >nul
)

echo. > dist\main_twopass\data\input\.keep
echo. > dist\main_twopass\data\output\.keep
echo. > dist\main_twopass\Log\.keep
echo. > dist\main_twopass\progress\.keep
echo. > dist\main_twopass\reports\.keep
echo. > dist\main_twopass\debug\.keep
echo   - Folders ready
echo.

echo [Step 5.5/7] Progress viewer...
dotnet publish progress_viewer\ProgressViewer.csproj -c Release -o dist\main_twopass
if errorlevel 1 (
    echo   - WARNING: ProgressViewer publish failed. Continuing without it.
) else (
    echo   - ProgressViewer.exe: OK
)
echo.

echo [Step 6/7] Rename output...
set OUTPUT_DIR=dist\main_twopass_%VERSION%
if exist %OUTPUT_DIR% rmdir /s /q %OUTPUT_DIR%
move dist\main_twopass %OUTPUT_DIR%
if errorlevel 1 (
    echo [ERROR] Failed to rename output directory
    popd
    exit /b 1
)
echo   - %OUTPUT_DIR%
echo.

echo [Step 7/7] README.txt...
%PY_CMD% -c "from pathlib import Path; t=Path('build/README_product.txt').read_text(encoding='utf-8'); Path(r'%OUTPUT_DIR%').joinpath('README.txt').write_text(t.replace('{VERSION}', r'%VERSION%'), encoding='utf-8-sig')"
if errorlevel 1 (
    echo   - WARNING: README.txt write failed
) else (
    echo   - README.txt created
)

echo.
echo Build Success: %OUTPUT_DIR%\main_twopass.exe
popd
exit /b 0
