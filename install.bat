@echo off
REM ============================================
REM 2D-to-3D Pipeline Installation Script
REM ============================================
REM This script sets up all required conda environments
REM and installs necessary packages.
REM ============================================

echo.
echo ============================================
echo   2D-to-3D Pipeline Installation
echo ============================================
echo.
echo This will:
echo   1. Create 4 conda environments
echo   2. Install required packages
echo.
echo Note: Marigold model (~2GB) will be downloaded
echo automatically on first use.
echo.
echo This may take 10-20 minutes.
echo.

pause

echo.
echo [1/8] Creating imagen environment...
echo ----------------------------------------
call conda create -n imagen python=3.10 -y
if errorlevel 1 (
    echo [ERROR] Failed to create imagen environment
    pause
    exit /b 1
)

echo.
echo [2/8] Installing imagen packages...
call conda activate imagen
pip install -r environments\requirements_imagen.txt
if errorlevel 1 (
    echo [ERROR] Failed to install imagen packages
    pause
    exit /b 1
)
call conda deactivate

echo.
echo [3/8] Creating marigold environment...
echo ----------------------------------------
call conda create -n marigold python=3.10 -y
if errorlevel 1 (
    echo [ERROR] Failed to create marigold environment
    pause
    exit /b 1
)

echo.
echo [4/8] Installing marigold packages...
call conda activate marigold
pip install -r environments\requirements_marigold.txt
if errorlevel 1 (
    echo [ERROR] Failed to install marigold packages
    pause
    exit /b 1
)
call conda deactivate

echo.
echo [5/8] Creating depth-to-3d environment...
echo ----------------------------------------
call conda create -n depth-to-3d python=3.10 -y
if errorlevel 1 (
    echo [ERROR] Failed to create depth-to-3d environment
    pause
    exit /b 1
)

echo.
echo [6/8] Installing depth-to-3d packages...
call conda activate depth-to-3d
pip install -r environments\requirements_depth_to_3d.txt
if errorlevel 1 (
    echo [ERROR] Failed to install depth-to-3d packages
    pause
    exit /b 1
)
call conda deactivate

echo.
echo [7/8] Creating photo-prep environment...
echo ----------------------------------------
call conda create -n photo-prep python=3.10 -y
if errorlevel 1 (
    echo [ERROR] Failed to create photo-prep environment
    pause
    exit /b 1
)

echo.
echo [8/8] Installing photo-prep packages...
call conda activate photo-prep
pip install -r environments\requirements_photo_prep.txt
if errorlevel 1 (
    echo [ERROR] Failed to install photo-prep packages
    pause
    exit /b 1
)
call conda deactivate

echo.
echo ============================================
echo   Installation Complete!
echo ============================================
echo.
echo Created environments:
echo   - imagen        (image generation)
echo   - marigold      (depth estimation)
echo   - depth-to-3d   (3D model creation)
echo   - photo-prep    (photo enhancement)
echo.
echo Note: Marigold model (~2GB) will download automatically
echo       on first use. This is a one-time download.
echo.
echo Next steps:
echo   1. Set up credentials (see CREDENTIALS_SETUP.md)
echo   2. Run: pipeline\run_pipeline.bat
echo.
echo ============================================

pause