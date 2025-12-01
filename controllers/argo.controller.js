const { query } = require('../config/db');

// Get all floats with pagination
const getFloats = async (req, res, next) => {
  try {
    const { page = 1, limit = 100 } = req.query;
    const offset = (page - 1) * limit;
    
    const result = await query(
      'SELECT * FROM argo_core_measurements ORDER BY time DESC OFFSET $1 LIMIT $2',
      [offset, limit]
    );
    
    res.json(result.rows);
  } catch (err) {
    next(err);
  }
};

// Get a single float by ID
const getFloatById = async (req, res, next) => {
  try {
    const { id } = req.params;
    const result = await query(
      'SELECT * FROM argo_core_measurements WHERE id = $1',
      [id]
    );
    
    if (result.rows.length === 0) {
      const error = new Error('Float not found');
      error.statusCode = 404;
      throw error;
    }
    
    res.json(result.rows[0]);
  } catch (err) {
    next(err);
  }
};

// Get floats within a region
const getFloatsInRegion = async (req, res, next) => {
  try {
    const { lat_min, lat_max, lon_min, lon_max, start_date, end_date } = req.query;
    
    let queryText = 'SELECT * FROM argo_core_measurements WHERE latitude BETWEEN $1 AND $2 AND longitude BETWEEN $3 AND $4';
    const params = [lat_min, lat_max, lon_min, lon_max];
    let paramCount = 5;
    
    if (start_date) {
      queryText += ` AND time >= $${paramCount++}`;
      params.push(new Date(start_date).toISOString());
    }
    
    if (end_date) {
      queryText += ` AND time <= $${paramCount++}`;
      params.push(new Date(end_date).toISOString());
    }
    
    queryText += ' ORDER BY time DESC';
    
    const result = await query(queryText, params);
    res.json(result.rows);
  } catch (err) {
    next(err);
  }
};

// Get latest float measurements
const getLatestMeasurements = async (req, res, next) => {
  try {
    const { limit = 100 } = req.query;
    
    const queryText = `
      WITH latest_measurements AS (
        SELECT DISTINCT ON (platform_number) *
        FROM argo_core_measurements
        ORDER BY platform_number, time DESC
      )
      SELECT * FROM latest_measurements
      LIMIT $1
    `;
    
    const result = await query(queryText, [limit]);
    res.json(result.rows);
  } catch (err) {
    next(err);
  }
};

// Find floats near a coordinate
const findNearbyFloats = async (req, res, next) => {
  try {
    const { latitude, longitude, radius_km = 100, limit = 100 } = req.query;
    
    const queryText = `
      SELECT *,
        (6371 * acos(
          cos(radians($1)) * cos(radians(latitude)) *
          cos(radians(longitude) - radians($2)) +
          sin(radians($1)) * sin(radians(latitude))
        )) AS distance_km
      FROM argo_core_measurements
      WHERE (6371 * acos(
        cos(radians($1)) * cos(radians(latitude)) *
        cos(radians(longitude) - radians($2)) +
        sin(radians($1)) * sin(radians(latitude))
      )) < $3
      ORDER BY distance_km
      LIMIT $4
    `;
    
    const result = await query(queryText, [
      latitude,
      longitude,
      radius_km,
      limit
    ]);
    
    res.json(result.rows);
  } catch (err) {
    next(err);
  }
};

module.exports = {
  getFloats,
  getFloatById,
  getFloatsInRegion,
  getLatestMeasurements,
  findNearbyFloats
};
