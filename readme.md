Ocean ARGO Chatbot 

This project builds an AI-powered chatbot for ARGO Core Float data (temperature, salinity, pressure) near India.
It collects ocean data, stores it in a database, summarizes it, and allows users to ask natural questions like:

“What is the average temperature in the Arabian Sea in 2024?”
“Where was salinity high in the Bay of Bengal?”

A. Project Flow

1. Extract Data

2. Download ARGO float data (NetCDF → Parquet format).

3. Load into Database

4. Store it in PostgreSQL + PostGIS for spatial queries.

5. Summarize & Vector DB

6. Convert monthly data into human summaries.

7. Store summaries in a vector database (Chroma/FAISS) for AI search.

8. Chatbot Interface

9. A Dashboard + Chatbot lets users ask questions.

10. Backend converts questions → SQL/vector queries → Results.

B. Setup Steps

1. Install Requirements

pip install -r requirements.txt


2. Create a .env File (Project Settings)

# Database
DB_URI=postgresql://postgres:Anish%401718@localhost:5432/argo_db

# Vector Database Location
CHROMA_PATH=vectorstore/chroma_db

# Embedding Model (for AI understanding)
EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2


3. Load Data into PostgreSQL

python db/load_parquet_to_postgres.py


4. Build Summaries for AI

python vectorstore/build_summaries.py


5. View Stored Summaries

python vectorstore/view_summaries.py


6. Run the Chatbot


C. What Can This Chatbot Answer?

1. Average sea temperature in a region & month

2. Salinity variations over time

3. Pressure vs. depth (thermocline detection)

4. Yearly/seasonal trends in the Indian Ocean

D. Real-World Applications

1. Disaster Management – Detect unusual temperature/salinity (cyclone/tsunami early indicators)
2. Fisheries – Identify nutrient-rich regions
3. Climate Research – Track ocean warming trends
4. Education & Awareness – Students and researchers query ocean data in natural language

In short:
This project transforms complex ARGO ocean datasets into a chatbot anyone can use for insights.
