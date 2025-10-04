Write-Host "Starting Ocean ARGO Chatbot Services..." -ForegroundColor Green
Write-Host ""

Write-Host "Starting Python FastAPI backend..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd bot-using-NetCDF-ocean-data; python scripts/api_server.py"

Write-Host "Waiting for Python backend to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

Write-Host "Starting Node.js frontend..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "npm start"

Write-Host ""
Write-Host "Services starting..." -ForegroundColor Green
Write-Host "Python Backend: http://localhost:8000" -ForegroundColor Cyan
Write-Host "Node.js Frontend: http://localhost:3000" -ForegroundColor Cyan
Write-Host ""
Write-Host "Run 'python test_endpoints.py' to verify connections" -ForegroundColor Yellow
Write-Host "Press any key to continue..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
