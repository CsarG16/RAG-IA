@echo off
title AulaIA - FastAPI + SSE Tutor Virtual
color 0B
echo ===================================================
echo      INICIANDO AULAIA (FASTAPI + SSE CLIENT)
echo ===================================================
echo.
echo [1/2] Verificando dependencias...
python -c "import fastapi, uvicorn, sse_starlette, faiss, google.genai; print('Entorno OK')" >nul 2>&1
if %errorlevel% neq 0 (
    echo Instalando dependencias faltantes...
    pip install -r requirements.txt
)

echo.
echo [2/2] Abriendo navegador en http://localhost:8000 ...
start http://localhost:8000

echo Iniciando servidor Uvicorn en http://127.0.0.1:8000 ...
echo Presiona Ctrl + C en esta ventana para detener el servidor.
echo ===================================================
echo.
python -m uvicorn api:app --app-dir src --host 127.0.0.1 --port 8000 --reload
pause
