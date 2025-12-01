require('dotenv').config();
const express = require('express');
const cors = require('cors');
const { notFound, errorHandler } = require('./middleware/errorHandler');
const argoRoutes = require('./routes/argo.routes');

// Initialize Express app
const app = express();
const port = process.env.PORT || 3000;
const apiPrefix = process.env.API_PREFIX || '/api';

// Middleware
app.use(cors());
app.use(express.json());

// Test database connection
require('./config/db');

// API Routes
app.get('/', (req, res) => {
  res.json({ 
    message: 'ARGO Float Data API',
    documentation: `${req.protocol}://${req.get('host')}${apiPrefix}/docs`,
    endpoints: {
      floats: `${req.protocol}://${req.get('host')}${apiPrefix}/floats`,
      region: `${req.protocol}://${req.get('host')}${apiPrefix}/floats/region`,
      nearby: `${req.protocol}://${req.get('host')}${apiPrefix}/floats/search/nearby`
    }
  });
});

// API routes
app.use(`${apiPrefix}/floats`, argoRoutes);

// 404 handler
app.use(notFound);

// Error handler
app.use(errorHandler);

// ARGO Data Endpoints
app.get('/api/floats', async (req, res) => {
  try {
    const { page = 1, limit = 100 } = req.query;
    const offset = (page - 1) * limit;
    
    const result = await pool.query(
      'SELECT * FROM argo_core_measurements ORDER BY time DESC OFFSET $1 LIMIT $2',
      [offset, limit]
    );
    
    res.json(result.rows);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Internal server error' });
  }
});

app.get('/api/floats/:id', async (req, res) => {
  try {
    const { id } = req.params;
    const result = await pool.query(
      'SELECT * FROM argo_core_measurements WHERE id = $1',
      [id]
    );
    
    if (result.rows.length === 0) {
      return res.status(404).json({ error: 'Float not found' });
    }
    
    res.json(result.rows[0]);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Internal server error' });
  }
});

app.get('/api/floats/region', async (req, res) => {
  try {
    const { lat_min, lat_max, lon_min, lon_max, start_date, end_date } = req.query;
    
    let query = 'SELECT * FROM argo_core_measurements WHERE latitude BETWEEN $1 AND $2 AND longitude BETWEEN $3 AND $4';
    const params = [lat_min, lat_max, lon_min, lon_max];
    let paramCount = 5;
    
    if (start_date) {
      query += ` AND time >= $${paramCount++}`;
      params.push(new Date(start_date).toISOString());
    }
    
    if (end_date) {
      query += ` AND time <= $${paramCount++}`;
      params.push(new Date(end_date).toISOString());
    }
    
    query += ' ORDER BY time DESC';
    
    const result = await pool.query(query, params);
    res.json(result.rows);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Start server
app.listen(port, () => {
  console.log(`Server running on port ${port}`);
});
