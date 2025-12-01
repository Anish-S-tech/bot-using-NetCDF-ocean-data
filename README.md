# ARGO Float Data API

A RESTful API for accessing and analyzing ARGO float data, built with Node.js and Express.

## Features

- Retrieve ARGO float data with pagination
- Search floats by region and time range
- Get detailed information about specific floats
- Filter and sort data based on various parameters

## Prerequisites

- Node.js (v14 or later)
- npm (comes with Node.js)
- PostgreSQL (v10 or later)
- ARGO database with the required tables

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd argo-float-api
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Set up environment variables:
   - Copy `.env.example` to `.env`
   - Update the database connection string and other settings in `.env`

4. Start the development server:
   ```bash
   npm run dev
   ```

5. The API will be available at `http://localhost:3000`

## API Endpoints

### ARGO Data Endpoints

- `GET /api/floats` - Get a list of floats (with pagination)
- `GET /api/floats/:id` - Get details for a specific float
- `GET /api/floats/region` - Get floats within a bounding box
- `GET /api/floats/latest` - Get most recent float measurements

### Search Endpoints

- `GET /api/search/nearby` - Find floats near a coordinate
- `POST /api/search/region` - Advanced search within a region

### Visualization Endpoints

- `GET /api/visualization/map` - Get GeoJSON data for mapping
- `GET /api/visualization/timeseries` - Get time series data for plotting

## Environment Variables

- `PORT` - Port to run the server on (default: 3000)
- `NODE_ENV` - Environment (development, production, test)
- `DATABASE_URL` - PostgreSQL connection string
- `DEFAULT_PAGE_SIZE` - Default number of items per page (default: 100)
- `MAX_PAGE_SIZE` - Maximum number of items per page (default: 1000)

## Development

- Run in development mode: `npm run dev`
- Run tests: `npm test`
- Lint code: `npm run lint`

## License

MIT
