/**
 * Ocean ARGO Chatbot - Node.js Express Server
 * 
 * This server acts as a web frontend and API proxy for the Ocean ARGO Chatbot system.
 * It serves Handlebars templates, handles static files, and proxies requests to the
 * Python FastAPI backend that runs the LLaMA3 model.
 * 
 * Architecture:
 * - Frontend: Handlebars templates (HTML/CSS/JS)
 * - Web Server: Node.js Express (this file)
 * - AI Backend: Python FastAPI + LLaMA3 (port 8000)
 * - Database: PostgreSQL + ChromaDB
 * 
 * @author Ocean ARGO Team
 * @version 1.0.0
 */

// ============================================================================
// IMPORTS AND DEPENDENCIES
// ============================================================================

import express from "express";
import path from "path";
import { fileURLToPath } from "url";
import { engine as handlebarsEngine } from "express-handlebars";
import fetch from "node-fetch";

// ============================================================================
// EXPRESS APP INITIALIZATION
// ============================================================================

const app = express();

// Middleware for parsing JSON requests
app.use(express.json());

// ============================================================================
// PATH CONFIGURATION (ES Modules compatibility)
// ============================================================================

// Resolve __dirname in ES modules (since __dirname is not available in ES modules)
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// ============================================================================
// VIEW ENGINE CONFIGURATION (Handlebars)
// ============================================================================

// Set up Handlebars as the template engine
const viewsPath = path.join(__dirname, "bot-using-NetCDF-ocean-data", "views");

app.engine(
  "hbs",
  handlebarsEngine({
    extname: ".hbs",                    // Use .hbs extension for templates
    defaultLayout: "main",              // Default layout template
    layoutsDir: path.join(viewsPath, "layouts"),  // Layouts directory
  })
);

// Configure view engine and views directory
app.set("view engine", "hbs");
app.set("views", viewsPath);

// ============================================================================
// STATIC FILES CONFIGURATION
// ============================================================================

// Serve static files from public directory (CSS, JS, images)
app.use(express.static(path.join(__dirname, "public")));

// ============================================================================
// MOCK DATA (Temporary - will be replaced with real database queries)
// ============================================================================

/**
 * Mock ocean data for demonstration purposes
 * In production, this will be replaced with real PostgreSQL queries
 */
const demoData = [
  { 
    id: 1, 
    location: "Arabian Sea", 
    temperatureC: 26.4, 
    salinityPsu: 35.1, 
    timestamp: "2025-09-30T08:20:00Z" 
  },
  { 
    id: 2, 
    location: "Bay of Bengal", 
    temperatureC: 28.2, 
    salinityPsu: 33.8, 
    timestamp: "2025-09-30T08:25:00Z" 
  },
  { 
    id: 3, 
    location: "Indian Ocean", 
    temperatureC: 24.9, 
    salinityPsu: 34.7, 
    timestamp: "2025-09-30T08:30:00Z" 
  },
  { 
    id: 4, 
    location: "Arabian Sea", 
    temperatureC: 26.1, 
    salinityPsu: 35.0, 
    timestamp: "2025-09-30T09:00:00Z" 
  },
  { 
    id: 5, 
    location: "Bay of Bengal", 
    temperatureC: 27.9, 
    salinityPsu: 33.6, 
    timestamp: "2025-09-30T09:05:00Z" 
  }
];

// ============================================================================
// API ROUTES
// ============================================================================

/**
 * GET /api/data
 * 
 * Ocean data API endpoint for filtering by location
 * Used by the Data Access page for interactive queries
 * 
 * Query Parameters:
 * - location (optional): Filter results by location name (case-insensitive)
 * 
 * Response:
 * - count: Number of results
 * - results: Array of ocean data objects
 */
app.get("/api/data", (req, res) => {
  try {
    // Extract and sanitize location query parameter
    const locationQuery = String(req.query.location || "").trim().toLowerCase();
    
    // Filter data based on location query (case-insensitive contains search)
    const results = locationQuery
      ? demoData.filter(r => r.location.toLowerCase().includes(locationQuery))
      : demoData;
    
    // Return filtered results with count
    res.json({ 
      count: results.length, 
      results 
    });
  } catch (error) {
    res.status(500).json({ 
      error: "Failed to fetch ocean data", 
      details: String(error) 
    });
  }
});

/**
 * GET /api/llm/health
 * 
 * Health check endpoint for the LLaMA3 AI backend
 * Verifies connectivity to Python FastAPI server and model status
 * 
 * Response:
 * - ok: Boolean indicating if AI backend is healthy
 * - raw: Raw response from Python backend
 */
app.get("/api/llm/health", async (req, res) => {
  try {
    // Set up timeout controller for health check (5 second timeout)
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 5000);
    
    // Make health check request to Python FastAPI backend
    const response = await fetch("http://localhost:8000/health", {
      method: "GET",
      headers: { "Content-Type": "application/json" },
      signal: controller.signal,
    });
    
    // Clear timeout and parse response
    clearTimeout(timeout);
    const data = await response.json();
    
    // Check if backend is healthy (model loaded and vectorstore ready)
    const ok = data.status === "healthy" && data.model_loaded && data.vectorstore_loaded;
    
    res.json({ 
      ok, 
      raw: data 
    });
  } catch (error) {
    // Return 502 Bad Gateway if Python backend is unreachable
    res.status(502).json({ 
      ok: false, 
      error: String(error) 
    });
  }
});

/**
 * POST /ask
 * 
 * Chatbot query endpoint - proxies requests to Python LLaMA3 backend
 * This is the main AI interaction endpoint used by the frontend
 * 
 * Request Body:
 * - question: String - User's question about ocean data
 * 
 * Response:
 * - answer: String - AI-generated response
 * - sources: Array - Source floats used for the answer
 * - status: String - Success/error status
 */
app.post("/ask", async (req, res) => {
  try {
    // Extract question from request body
    const { question } = req.body;
    
    // Validate question input
    if (!question || typeof question !== 'string') {
      return res.status(400).json({ 
        error: "Question is required and must be a string" 
      });
    }
    
    // Forward request to Python FastAPI backend
    const response = await fetch("http://localhost:8000/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    
    // Check if Python backend responded successfully
    if (!response.ok) {
      throw new Error(`Python backend responded with status: ${response.status}`);
    }
    
    // Parse and return the AI response
    const data = await response.json();
    res.json(data);
    
  } catch (error) {
    // Return 500 Internal Server Error for any processing failures
    res.status(500).json({ 
      error: "Failed to query AI backend", 
      details: String(error) 
    });
  }
});

// ============================================================================
// TEMPLATE ROUTES (Frontend Pages)
// ============================================================================

/**
 * GET / and GET /dashboard
 * 
 * Main dashboard page - displays ocean data overview and chatbot interface
 * This is the primary landing page of the application
 */
app.get(["/", "/dashboard"], (req, res) => {
  const isDashboard = req.path === "/dashboard" || req.path === "/";
  res.render("dashboard", { isDashboard });
});

/**
 * GET /data-access
 * 
 * Data Access page - interactive interface for querying ocean data
 * Features map visualization, filters, and search functionality
 */
app.get("/data-access", (req, res) => {
  res.render("data-access", { isDataAccess: true });
});

/**
 * GET /insights-alerts
 * 
 * Insights & Alerts page - displays ocean data trends and anomaly alerts
 * Shows real-time monitoring data and alert configurations
 */
app.get("/insights-alerts", (req, res) => {
  res.render("insights-alerts", { isInsightsAlerts: true });
});

/**
 * GET /help
 * 
 * Help & Documentation page - user guide and API documentation
 * Provides troubleshooting information and usage examples
 */
app.get("/help", (req, res) => {
  res.render("help", { isHelp: true });
});

// ============================================================================
// ERROR HANDLING MIDDLEWARE
// ============================================================================

/**
 * Global error handler for unhandled routes and errors
 * This should be the last middleware in the stack
 */
app.use((req, res) => {
  res.status(404).render("error", { 
    message: "Page not found",
    error: { status: 404 }
  });
});

// Global error handler for uncaught exceptions
app.use((err, req, res, next) => {
  console.error("Unhandled error:", err);
  res.status(500).render("error", { 
    message: "Internal server error",
    error: process.env.NODE_ENV === 'development' ? err : {}
  });
});

// ============================================================================
// SERVER STARTUP
// ============================================================================

/**
 * Start the Express server
 * Server will listen on the specified PORT (default: 3000)
 */
const PORT = process.env.PORT || 3000;

app.listen(PORT, () => {
  console.log("🌊 Ocean ARGO Chatbot Server Started");
  console.log(`📍 Server running on http://localhost:${PORT}`);
  console.log(`🤖 AI Backend expected at http://localhost:8000`);
  console.log(`📊 Dashboard: http://localhost:${PORT}/dashboard`);
  console.log(`🔍 Data Access: http://localhost:${PORT}/data-access`);
  console.log(`📈 Insights: http://localhost:${PORT}/insights-alerts`);
  console.log(`❓ Help: http://localhost:${PORT}/help`);
});
