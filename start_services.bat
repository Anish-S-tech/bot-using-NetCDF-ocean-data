@echo off
echo Starting Ocean ARGO Chatbot Services...
echo.

echo Starting Python FastAPI backend...
start "Python Backend" cmd /k "cd bot-using-NetCDF-ocean-data && python scripts/api_server.py"

echo Waiting for Python backend to start...
timeout /t 5 /nobreak > nul

echo Starting Node.js frontend...
start "Node.js Frontend" cmd /k "npm start"

echo.
echo Services starting...
echo Python Backend: http://localhost:8000
echo Node.js Frontend: http://localhost:3000
echo.
echo Run 'python test_endpoints.py' to verify connections
pause
