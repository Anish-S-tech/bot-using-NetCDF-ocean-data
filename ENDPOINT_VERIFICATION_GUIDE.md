# Endpoint Verification Guide

This guide helps you verify that the frontend and LLaMA3 backend are properly connected and working.

## Architecture Overview

```
Frontend (Handlebars) → Node.js Express Server → Python FastAPI → LLaMA3 Model
     ↓                        ↓                      ↓
  Port 3000              Port 3000              Port 8000
```

## Quick Start

### 1. Start Services

**Option A: Using the startup scripts**
```bash
# Windows Batch
start_services.bat

# Windows PowerShell
.\start_services.ps1
```

**Option B: Manual startup**
```bash
# Terminal 1: Start Python FastAPI backend
cd bot-using-NetCDF-ocean-data
python scripts/api_server.py

# Terminal 2: Start Node.js frontend
npm start
```

### 2. Verify Connections

Run the automated test:
```bash
python test_endpoints.py
```

## Manual Verification Steps

### Step 1: Check Python Backend (Port 8000)

```bash
# Health check
curl http://localhost:8000/health

# Expected response:
{
  "status": "healthy",
  "model_loaded": true,
  "vectorstore_loaded": true
}
```

### Step 2: Check Node.js Backend (Port 3000)

```bash
# Health check
curl http://localhost:3000/api/llm/health

# Expected response:
{
  "ok": true,
  "raw": {
    "status": "healthy",
    "model_loaded": true,
    "vectorstore_loaded": true
  }
}
```

### Step 3: Test Query Flow

```bash
# Test direct Python backend
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the average temperature in the Arabian Sea?"}'

# Test Node.js proxy
curl -X POST http://localhost:3000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the average temperature in the Arabian Sea?"}'
```

### Step 4: Test Frontend

1. Open http://localhost:3000 in your browser
2. Scroll down to the "Ask FloatChat" section
3. Enter a question like "What is the average temperature in the Arabian Sea?"
4. Click "Ask"
5. Verify the response appears in the output area

## Troubleshooting

### Common Issues

**1. Python Backend Not Starting**
- Check if the LLaMA model file exists at the specified path
- Verify all Python dependencies are installed: `pip install -r requirements.txt`
- Check if port 8000 is available

**2. Node.js Backend Not Starting**
- Check if port 3000 is available
- Verify Node.js dependencies: `npm install`
- Check if the views directory exists

**3. Connection Refused Errors**
- Ensure both services are running
- Check firewall settings
- Verify the correct ports are being used

**4. LLaMA Model Not Loading**
- Check if the model file exists: `models/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf`
- Verify sufficient RAM (8GB+ recommended)
- Check if the vectorstore is properly initialized

### Debug Commands

```bash
# Check if ports are in use
netstat -an | findstr :3000
netstat -an | findstr :8000

# Check Python process
tasklist | findstr python

# Check Node.js process
tasklist | findstr node
```

## API Endpoints

### Python FastAPI Backend (Port 8000)

- `GET /` - Root endpoint with status
- `GET /health` - Health check
- `POST /query` - Query the LLaMA3 model

### Node.js Express Backend (Port 3000)

- `GET /` - Dashboard page
- `GET /api/data` - Mock data API
- `GET /api/llm/health` - LLM health check
- `POST /ask` - Chatbot query proxy

## Expected Data Flow

1. **User Input**: User types question in frontend form
2. **Frontend**: JavaScript sends POST request to `/ask`
3. **Node.js**: Express server forwards request to Python backend
4. **Python**: FastAPI processes request with LLaMA3 model
5. **Response**: Answer flows back through the chain to frontend
6. **Display**: Response is shown in the chat output area

## Performance Notes

- LLaMA3 model loading takes 30-60 seconds on first startup
- Query processing takes 5-15 seconds depending on complexity
- The model uses significant RAM (4-8GB)
- Vectorstore queries are fast (< 1 second)

## Success Indicators

✅ All services start without errors
✅ Health checks return "healthy" status
✅ Test queries return meaningful responses
✅ Frontend form submits and displays responses
✅ No connection refused errors in logs

## Next Steps

Once verification is complete, you can:
1. Customize the LLaMA3 prompts in `api_server.py`
2. Add more sophisticated error handling
3. Implement response caching
4. Add user authentication
5. Deploy to production servers
