@echo off
REM 3方向合成カラーマップ 製品版ビルド（PyInstaller onedir）
REM 使い方: build\build_installer.bat（リポジトリルート、またはこの bat から起動）
REM 出力: dist\main_twopass_<VERSION>\

cd /d "%~dp0.."

echo ========================================
echo 3-direction CameraCarSim build
echo ========================================
echo.

echo [Step 1/7] Environment Check...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found.
    exit /b 1
)
echo   - Python: OK

pyinstaller --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] PyInstaller not found. pip install pyinstaller
    exit /b 1
)
echo   - PyInstaller: OK

python -c "import numpy, cv2, pandas, scipy, matplotlib, pytesseract, openpyxl" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Required packages missing. pip install -r requirements.txt
    exit /b 1
)
echo   - Required packages: OK
echo.

echo [Step 1.5/7] Check required files...
if not exist data\config\default_config.json (
    echo [ERROR] data\config\default_config.json not found
    exit /b 1
)
echo   - data\config\default_config.json: OK
echo.

echo [Step 2/7] Cleanup...
if exist build\main_twopass rmdir /s /q build\main_twopass
if exist dist\main_twopass rmdir /s /q dist\main_twopass
echo   - Cleanup complete
echo.

echo [Step 3/7] Version...
for /f "usebackq delims=" %%i in (`powershell -Command "Get-Date -Format 'yyyyMMdd_HHmmss'"`) do set VERSION=%%i
echo   - Version: %VERSION%
echo.

echo [Step 4/7] PyInstaller...
pyinstaller --clean --noconfirm main_twopass.spec
if errorlevel 1 (
    echo [ERROR] PyInstaller failed
    exit /b 1
)
echo   - Build complete
echo.

echo [Step 5/7] Layout product folders...
if not exist dist\main_twopass\main_twopass.exe (
    echo [ERROR] dist\main_twopass\main_twopass.exe missing
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
    exit /b 1
)

if exist data\calibration\*.json (
    copy /Y data\calibration\*.json dist\main_twopass\data\calibration\ >nul
)

echo. > dist\main_twopass\data\input\.keep
echo. > dist\main_twopass\data\output\.keep
echo. > dist\main_twopass\Log\.keep
echo. > dist\main_twopass\progress\.keep
echo. > dist\main_twopass\reports\.keep
echo. > dist\main_twopass\debug\.keep
echo   - Folders ready
echo.

echo [Step 6/7] Rename output...
set OUTPUT_DIR=dist\main_twopass_%VERSION%
if exist %OUTPUT_DIR% rmdir /s /q %OUTPUT_DIR%
move dist\main_twopass %OUTPUT_DIR%
echo   - %OUTPUT_DIR%
echo.

echo [Step 7/7] README.txt...
echo 管内カメラカーシミュレータ 3方向合成カラーマップ > %OUTPUT_DIR%\README.txt
echo Version: %VERSION% >> %OUTPUT_DIR%\README.txt
echo. >> %OUTPUT_DIR%\README.txt
echo フォルダ: >> %OUTPUT_DIR%\README.txt
echo   data\calibration   レンズ校正 JSON >> %OUTPUT_DIR%\README.txt
echo   data\config        default_config.json >> %OUTPUT_DIR%\README.txt
echo   data\input         U.mp4 / R.mp4 / L.mp4 >> %OUTPUT_DIR%\README.txt
echo   data\output        最終カラーマップ >> %OUTPUT_DIR%\README.txt
echo   debug              デバッグ画像 >> %OUTPUT_DIR%\README.txt
echo   Log\process.log    ローテーション付きログ >> %OUTPUT_DIR%\README.txt
echo   progress\progress.json >> %OUTPUT_DIR%\README.txt
echo   reports\report_yyyymmdd_hhmmss.xlsx >> %OUTPUT_DIR%\README.txt
echo. >> %OUTPUT_DIR%\README.txt
echo 使い方: >> %OUTPUT_DIR%\README.txt
echo   main_twopass.exe --help >> %OUTPUT_DIR%\README.txt
echo   main_twopass.exe --config data\config\default_config.json >> %OUTPUT_DIR%\README.txt
echo   main_twopass.exe --input data\input --output data\output --start 1 --end 500 >> %OUTPUT_DIR%\README.txt
echo   main_twopass.exe --pi 250 --ppm 2.55 --debug >> %OUTPUT_DIR%\README.txt
echo. >> %OUTPUT_DIR%\README.txt
echo 要件: Windows 11 64bit, Tesseract-OCR 4.0+ >> %OUTPUT_DIR%\README.txt

echo.
echo Build Success: %OUTPUT_DIR%\main_twopass.exe
exit /b 0
