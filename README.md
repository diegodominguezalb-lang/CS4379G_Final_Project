---
title: NOAA Storm Events Dashboard
emoji: ⛈️
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# NOAA Storm Events Dashboard

Interactive Streamlit dashboard exploring how disaster type, scale, and geography
relate to the economic impact of storm events in the United States (1950–present).

## Setup

### Requirements
- Python 3.12
- pip

### Install dependencies

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Run locally

```bash
streamlit run dashboard.py
```

The app opens at http://localhost:8501.

## Data

The dashboard reads NOAA Storm Events CSV files from the `DataForProject/` folder.

### Fetch / refresh the data

Use the included Python script (works on Windows, macOS, and Linux — no extra dependencies):

```bash
# Download all years (1950–present), skipping files already on disk
python fetch_data.py

# Download a specific range
python fetch_data.py --start 2000 --end 2020

# Force re-download even if files exist
python fetch_data.py --no-skip-existing
```

Data is fetched directly from the NOAA FTP server:
`https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles`

Files follow the naming convention:
`StormEvents_details-ftp_v1.0_d{YEAR}_c{release_date}.csv`

**Data source:** NOAA National Centers for Environmental Information (NCEI)  
**License:** U.S. Government Open Data — public domain

## Docker

```bash
# Build
docker build -t noaa-storm-dashboard .

# Run
docker run -p 8501:8501 -v "$(pwd)/DataForProject:/app/DataForProject" noaa-storm-dashboard
```

## Deploy (Streamlit Community Cloud)

1. Push this repo to a **public** GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Select your repo, branch `main`, and set **Main file path** to `dashboard.py`.
4. Click **Deploy** — the app rebuilds automatically on every `git push`.

> **Note:** Streamlit Community Cloud has a 1 GB RAM limit. If the full CSV dataset
> exceeds memory, consider loading a subset of years or hosting the data externally.