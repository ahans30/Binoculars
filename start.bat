@echo off
chcp 65001 > nul
title Binoculars WebUI Launcher

cd /d "%~dp0"

echo ========================================================
echo   Binoculars WebUI を初期化・起動しています...
echo ========================================================

if not exist "venv\Scripts\python.exe" (
    echo [情報] 仮想環境 ^(venv^) を作成しています...
    py -V:3.14 -m venv venv
    if errorlevel 1 (
        echo [エラー] Python 3.14 が見つからないか、venvの作成に失敗しました。
        pause
        exit /b 1
    )
)

call venv\Scripts\activate.bat

echo [情報] 依存ライブラリの整合性を確認しています...
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo [エラー] ライブラリのインストールに失敗しました。
    pause
    exit /b 1
)

echo ========================================================
echo   WebUI をブラウザで起動します ^(http://127.0.0.1:7860^)
echo ========================================================
python app.py

pause
