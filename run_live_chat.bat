@echo off
title Spatial Droplet Live Desktop Copilot
echo =========================================================================
echo       SPATIAL DROPLET -- LIVE DESKTOP COPILOT AND TRANSPARENT HUD
echo =========================================================================
echo Starting Python backend...
"C:\Users\NANDHA K\AppData\Local\Programs\Python\Python310\python.exe" "c:\Users\NANDHA K\Documents\antigravity\amazing-bose\live_copilot.py"
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo =========================================================================
    echo Application stopped with code %ERRORLEVEL%
    echo =========================================================================
    pause
)
