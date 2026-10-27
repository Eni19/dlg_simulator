@echo off
if exist "%~dp0.venv\Scripts\python.exe" (
  "%~dp0.venv\Scripts\python.exe" "%~dp0scripts\local.py" %*
  exit /b
)
where py >nul 2>nul
if errorlevel 1 goto fallback
py -3.12 "%~dp0scripts\local.py" %*
exit /b %errorlevel%
:fallback
python "%~dp0scripts\local.py" %*
exit /b %errorlevel%
