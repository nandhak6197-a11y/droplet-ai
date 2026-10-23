@echo off
title Spatial Droplet -- Live Hardware Chat Console
echo =========================================================================
echo       SPATIAL DROPLET -- LIVE HARDWARE REASONING & CHAT CONSOLE
echo =========================================================================
echo Starting live desktop hardware interaction...
"C:\Users\NANDHA K\AppData\Local\Programs\Python\Python310\python.exe" "c:\Users\NANDHA K\Documents\antigravity\amazing-bose\live_droplet_chat.py"
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo =========================================================================
    echo Console exited with code %ERRORLEVEL%
    echo =========================================================================
    pause
)
