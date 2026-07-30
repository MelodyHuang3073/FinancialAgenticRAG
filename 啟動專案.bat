@echo off
chcp 65001 > nul
echo.
echo ==========================================
echo  FinAgent-RAG 財報問答系統 啟動中...
echo ==========================================
echo.

:: 啟動後端 (開新視窗)
echo [1/2] 啟動後端 FastAPI Server (Port 8000)...
start "FinAgent-RAG Backend" cmd /k "cd /d %~dp0backend && python run.py"

:: 稍等後端啟動
timeout /t 3 /nobreak > nul

:: 啟動前端 (開新視窗)
echo [2/2] 啟動前端 Vite Dev Server (Port 5173)...
start "FinAgent-RAG Frontend" cmd /k "cd /d %~dp0frontend && npm run dev"

:: 稍等前端啟動
timeout /t 4 /nobreak > nul

:: 開啟瀏覽器
echo.
echo [3/3] 開啟瀏覽器...
start http://localhost:5173

echo.
echo ==========================================
echo  啟動完成！請至瀏覽器查看：
echo  前端：http://localhost:5173
echo  後端 API：http://localhost:8000/docs
echo ==========================================
echo.
pause
