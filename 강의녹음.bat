@echo off
cd /d "%~dp0"
rem Launch the GUI without a console window via the Python launcher (pyw).
start "" pyw "%~dp0lecture_gui.py"
