@echo off
chcp 65001 > nul
cd /d "%~dp0"
where py >nul 2>&1
if %ERRORLEVEL%==0 (
    py -m pip install Pillow
    py review_freshneg.py
) else (
    python -m pip install Pillow
    python review_freshneg.py
)
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Gap loi khi chay. Hay cai Python 3.10+ tu python.org va chon Add Python to PATH.
    pause
)
