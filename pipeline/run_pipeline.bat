@echo off
REM Activate photo-prep environment before running pipeline
call conda activate photo-prep
python "%~dp0\2D_to_3D_pipeline.py"
pause