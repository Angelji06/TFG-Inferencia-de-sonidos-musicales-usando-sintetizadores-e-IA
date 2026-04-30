@echo off
:: ============================================================
:: INSTALADOR PARA WINDOWS
:: ============================================================
setlocal EnableDelayedExpansion
title Instalador - Predictor de parametros FM

echo.
echo ================================================
echo   INSTALADOR - Predictor de parametros FM (TFG)
echo ================================================
echo.
echo Este script:
echo   1. Comprueba si tienes Conda instalado
echo   2. Si no lo tienes, descarga e instala Miniconda3
echo   3. Crea el entorno 'audio_gpu' con todas las dependencias
echo   4. Genera el archivo lanzar.bat para ejecutar la app
echo.
echo AVISO: La primera instalacion puede tardar 15-25 minutos
echo        y requiere conexion a internet (~3 GB de descarga).
echo.
pause

:: ============================================================
:: PASO 1: Buscar conda en ubicaciones habituales de Windows
:: ============================================================
echo [1/4] Buscando Conda...
echo.

set "CONDA_EXE="

:: Rutas comunes de Miniconda / Anaconda
set "SEARCH_PATHS="
set "SEARCH_PATHS=!SEARCH_PATHS! C:\ProgramData\miniconda3\Scripts\conda.exe"
set "SEARCH_PATHS=!SEARCH_PATHS! C:\ProgramData\Miniconda3\Scripts\conda.exe"
set "SEARCH_PATHS=!SEARCH_PATHS! C:\Users\%USERNAME%\miniconda3\Scripts\conda.exe"
set "SEARCH_PATHS=!SEARCH_PATHS! C:\Users\%USERNAME%\Miniconda3\Scripts\conda.exe"
set "SEARCH_PATHS=!SEARCH_PATHS! C:\Users\%USERNAME%\anaconda3\Scripts\conda.exe"
set "SEARCH_PATHS=!SEARCH_PATHS! C:\Users\%USERNAME%\Anaconda3\Scripts\conda.exe"
set "SEARCH_PATHS=!SEARCH_PATHS! C:\ProgramData\anaconda3\Scripts\conda.exe"

for %%P in (!SEARCH_PATHS!) do (
    if exist "%%P" (
        set "CONDA_EXE=%%P"
        goto :conda_found
    )
)

:: Comprobar si conda esta en el PATH
where conda >nul 2>&1
if !ERRORLEVEL! EQU 0 (
    for /f "delims=" %%i in ('where conda') do (
        set "CONDA_EXE=%%i"
        goto :conda_found
    )
)

:: ============================================================
:: PASO 2: Conda no encontrado -> Descargar e instalar Miniconda3
:: ============================================================
echo [!] Conda no encontrado en el sistema.
echo.
echo Descargando Miniconda3 para Windows (puede tardar segun tu conexion)...
echo.

set "MINICONDA_INSTALLER=%TEMP%\Miniconda3-latest-Windows-x86_64.exe"
set "MINICONDA_DIR=C:\Users\%USERNAME%\miniconda3"

powershell -NoProfile -Command ^
    "Invoke-WebRequest -Uri 'https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe' -OutFile '%MINICONDA_INSTALLER%' -UseBasicParsing"

if !ERRORLEVEL! NEQ 0 (
    echo.
    echo [ERROR] No se pudo descargar Miniconda3.
    echo         Comprueba tu conexion a internet e intentalo de nuevo.
    echo         Si el problema persiste, descarga manualmente desde:
    echo         https://docs.anaconda.com/miniconda/
    pause & exit /b 1
)

echo Instalando Miniconda3 en %MINICONDA_DIR% ...
echo (Esto puede tardar 2-3 minutos)
echo.

start /wait "" "%MINICONDA_INSTALLER%" /S /AddToPath=0 /RegisterPython=0 /D=%MINICONDA_DIR%

if !ERRORLEVEL! NEQ 0 (
    echo.
    echo [ERROR] La instalacion de Miniconda3 ha fallado.
    pause & exit /b 1
)

set "CONDA_EXE=%MINICONDA_DIR%\Scripts\conda.exe"
echo [OK] Miniconda3 instalado en %MINICONDA_DIR%
echo.

:conda_found
echo [OK] Conda encontrado: !CONDA_EXE!
echo.

:: Guardar la ruta base de conda para el launcher
for /f "delims=" %%i in ('"!CONDA_EXE!" info --base') do set "CONDA_BASE=%%i"
echo [OK] Base de Conda: !CONDA_BASE!
echo.

:: ============================================================
:: PASO 3: Crear el entorno desde environment.yml
:: ============================================================
echo [3/4] Creando entorno 'audio_gpu'...
echo       (Descargara ~3 GB de paquetes - es normal que tarde)
echo.

set "YML_PATH=%~dp0environment.yml"

if not exist "%YML_PATH%" (
    echo [ERROR] No se encuentra el archivo environment.yml
    echo         Asegurate de que esta en la misma carpeta que instalar.bat
    pause & exit /b 1
)

:: Comprobar si el entorno ya existe
"!CONDA_EXE!" env list | findstr /C:"audio_gpu" >nul 2>&1
if !ERRORLEVEL! EQU 0 (
    echo [!] El entorno 'audio_gpu' ya existe.
    echo     Actualizando dependencias...
    echo.
    call "!CONDA_EXE!" env update -f "%YML_PATH%" --prune
) else (
    call "!CONDA_EXE!" env create -f "%YML_PATH%"
)

if !ERRORLEVEL! NEQ 0 (
    echo.
    echo [ERROR] Fallo al crear/actualizar el entorno.
    echo         Revisa los mensajes de error anteriores.
    echo         Consejo: Si hay conflictos de paquetes, borra el entorno con:
    echo           conda env remove -n audio_gpu
    echo         y vuelve a ejecutar este instalador.
    pause & exit /b 1
)

echo.
echo [OK] Entorno 'audio_gpu' configurado correctamente.
echo.

:: ============================================================
:: PASO 4: Generar el archivo lanzar.bat
:: ============================================================
echo [4/4] Generando lanzar.bat...

set "LAUNCHER=%~dp0lanzar.bat"
set "APP_DIR=%~dp0Prototipo5"

(
    echo @echo off
    echo title Predictor de parametros FM - TFG
    echo cd /d "%APP_DIR%"
    echo call "!CONDA_BASE!\Scripts\activate.bat" audio_gpu
    echo python main.py
    echo if %%ERRORLEVEL%% NEQ 0 ^(
    echo     echo.
    echo     echo [ERROR] La aplicacion ha terminado con errores.
    echo     pause
    echo ^)
) > "%LAUNCHER%"

echo [OK] Archivo lanzar.bat generado.
echo.
echo ================================================
echo   Instalacion completada con exito!
echo ================================================
echo.
echo Para iniciar la aplicacion, ejecuta:
echo   lanzar.bat
echo.
echo (Puedes crear un acceso directo a lanzar.bat en el escritorio)
echo.
pause
endlocal
