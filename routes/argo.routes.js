const express = require('express');
const router = express.Router();
const {
  getFloats,
  getFloatById,
  getFloatsInRegion,
  getLatestMeasurements,
  findNearbyFloats
} = require('../controllers/argo.controller');

// ARGO Data Endpoints
router.get('/', getFloats);
router.get('/:id', getFloatById);
router.get('/region', getFloatsInRegion);
router.get('/latest/measurements', getLatestMeasurements);

// Search Endpoints
router.get('/search/nearby', findNearbyFloats);

module.exports = router;
