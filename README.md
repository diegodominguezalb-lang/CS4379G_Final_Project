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

Interactive Streamlit dashboard exploring 75 years of U.S. weather disasters — analyzing
how storm type, geography, and season relate to economic damage and human impact (1950–2025).

**Live demo:** [huggingface.co/spaces/diegodomin/CS4379FinalProject](https://huggingface.co/spaces/diegodomin/CS4379FinalProject)

## Dashboard Tabs

| Tab | Contents |
|-----|----------|
| **Overview** | KPI cards, top event types bar, property damage histogram (log scale, USD labels), human impact bar |
| **Damage Analysis** | Total damage stacked bar, average damage per event, frequency vs. damage bubble scatter, property vs. crop scatter |
| **Regional Analysis** | Choropleth map by state, top-15 states stacked bar, event-count/damage correlation histogram, state drill-down |
| **Time Series** | Annual event frequency line, annual damage bar, top-8 event trends, seasonal patterns (filterable by state + event type) |
| **Multivariate** | Pearson correlation matrix heatmap, normalized risk-profile heatmap (top 15 event types) |

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

The dashboard loads data from Hugging Face Datasets (`diegodomin/noaa-storm-events → storms.parquet`)
on first run and caches it locally. Raw NOAA CSV files can also be placed in `DataForProject/`
for offline use or to rebuild the parquet.

**Source:** NOAA National Centers for Environmental Information (NCEI)  
`https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles`  
**License:** U.S. Government Open Data — public domain

> **Note:** Pre-1996 records cover only tornado, thunderstorm, and hail events due to
> NOAA's narrower event taxonomy at the time. Post-1996 data includes ~60 distinct event types.

## Project Structure

```
dashboard.py          # Streamlit app entry point — page config, sidebar, tab routing
requirements.txt      # Python dependencies
Dockerfile            # Container definition (Hugging Face Spaces, port 7860)
tabs/
  overview.py         # Tab 1 — KPI cards, event bar, histogram, human impact
  damage.py           # Tab 2 — Damage breakdown and scatter plots
  regional.py         # Tab 3 — Geographic analysis and choropleth map
  timeseries.py       # Tab 4 — Trends over time and seasonal patterns
  multivariate.py     # Tab 5 — Correlation matrix and risk profile heatmap
utils/
  data_loader.py      # HF Hub fetch, local CSV pipeline, parquet caching
  formatting.py       # _fmt_usd(), _dollar_yaxis(), Okabe-Ito color palette
DataForProject/       # Raw NOAA CSV files (not committed — gitignored)
```

## Docker

```bash
# Build
docker build -t noaa-storm-dashboard .

# Run
docker run -p 7860:7860 noaa-storm-dashboard
```

## Deployment

The app is deployed on **Hugging Face Spaces** (Docker SDK).  
Every push to the `space` remote redeploys automatically.

```bash
git remote add space https://huggingface.co/spaces/diegodomin/CS4379FinalProject
git push space main
```