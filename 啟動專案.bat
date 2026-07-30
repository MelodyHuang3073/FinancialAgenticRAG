@echo off
echo.
echo ==========================================
echo  Starting FinAgent-RAG...
echo ==========================================
echo.

echo [1/2] Starting Backend FastAPI Server (Port 8000)...
start "FinAgent-RAG Backend" cmd /k "cd /d %~dp0backend && python run.py"

timeout /t 3 /nobreak > nul

echo [2/2] Starting Frontend Vite Dev Server (Port 5173)...
start "FinAgent-RAG Frontend" cmd /k "cd /d %~dp0frontend && npm run dev"

timeout /t 4 /nobreak > nul

echo.
echo [3/3] Opening Browser...
start http://localhost:5173

echo.
echo ==========================================
echo  Started! Check your browser:
echo  Frontend: http://localhost:5173
echo  Backend API: http://localhost:8000/docs
echo ==========================================
echo.
pause
