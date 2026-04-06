# 🚖 TaxiPulse

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Runtime](https://img.shields.io/badge/runtime-Python%203.9-blue)
![License](https://img.shields.io/badge/license-Unspecified-lightgrey)

**Problem:** NYC taxi drivers and fleet operators struggle to identify profitable routes and time windows in fast-changing city demand.

## Table of Contents 📑

- [About the Project 📚](#about-the-project-)
- [Screenshots / Demo 📷](#screenshots--demo-)
- [Technologies Used ☕️ 🐍 ⚛️](#technologies-used-️--⚛️)
- [Setup / Installation 💻](#setup--installation-)
- [Approach 🚶](#approach-)
- [Example Usage / Output](#example-usage--output)
- [Project Structure 📁](#project-structure-)
- [Status 📶](#status-)
- [Limitations ⚠️](#limitations-️)
- [Improvements / Roadmap 🚀](#improvements--roadmap-)
- [Credits 📝](#credits-)
- [Author](#author)

## About the Project 📚

TaxiPulse is a geospatial analytics dashboard for NYC Yellow Taxi trip data. It combines data ingestion, cleaning, spatial zone mapping, and interactive dashboards so drivers can quickly answer: **where should I drive, and when, to maximize earnings?**

It was built to solve the operational blind spot between raw trip logs and real driving decisions. Instead of manually exploring CSV data, users get route, borough, hourly earnings, demand/supply, hotspot, and tip behavior insights in one place.

It is built for:
- Individual taxi drivers optimizing shift plans
- Fleet operators planning supply allocation by borough/time
- Data analysts exploring demand/revenue movement patterns

## Screenshots / Demo 📷

- Dashboard walkthrough video: [YouTube Demo](https://youtu.be/jt1x2oKrZ_0)
- Live app: [TaxiPulse on Render](https://taxipulse.onrender.com)

Image/GIF placeholders:

```md
![Dashboard Overview](docs/assets/dashboard-overview.png)
![Hotspots Heatmap](docs/assets/hotspots-heatmap.gif)
![Analytics View](docs/assets/analytics-page.png)
```

Backend sample output (health endpoint):

```json
{
  "status": "healthy",
  "timestamp": "2026-04-06 11:50:46"
}
```

## Technologies Used ☕️ 🐍 ⚛️

| Layer | Stack |
|---|---|
| App framework | Dash, Flask, Dash Bootstrap Components |
| Visualization | Plotly, Folium |
| Data processing | Pandas, NumPy |
| Geospatial | GeoPandas, Shapely, Fiona, PyProj, Rtree |
| Serving | Waitress |
| Packaging / infra | Docker, Python 3.9 |
| Data source | Google Sheets CSV + NYC taxi zone shapefile |

## Setup / Installation 💻

### Local

```bash
git clone https://github.com/raghuveer9303/TaxiPulse.git
cd TaxiPulse
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

App runs on: `http://localhost:8080`

### Docker

```bash
docker build -t taxipulse .
docker run --rm -p 8080:8080 taxipulse
```

## Approach 🚶

TaxiPulse follows a **data pipeline + dashboard app** pattern:

1. **Ingest** trip records from a hosted CSV source.
2. **Clean & normalize** timestamps, coordinates, fares, and tips.
3. **Spatially enrich** pickup/dropoff points via NYC taxi zone polygons.
4. **Feature engineer** hour/day, zone, borough, and tip metrics.
5. **Serve interactive insights** using Dash callbacks and Plotly/Folium visualizations.

Key design decisions:
- Use geospatial joins early so every downstream chart can filter by real zones/boroughs.
- Keep filtering logic centralized (`filter_df`) to maintain metric consistency across pages.
- Run with Waitress for production-safe serving instead of development server.

## Example Usage / Output

### 1) Filter-based earnings discovery

```text
input  -> day_type=Weekdays, borough=Manhattan, time_of_day=Evening (4PM-7PM)
output -> peak-hour: 18:00
          peak-hour-value: $32.47 / hr
          best-day: Friday
```

### 2) High-value route lookup

```text
input  -> top-routes analysis for Weekends + All Boroughs + Night (8PM-11PM)
output -> Pickup Zone: Upper East Side South
          Dropoff Zone: JFK Airport
          Avg Fare: $58.20
          Avg Tip %: 21.3%
          Rides: 142
```

### 3) Service health check

```bash
curl http://localhost:8080/health
```

```json
{"status":"healthy","timestamp":"2026-04-06 11:50:46"}
```

## Project Structure 📁

```text
TaxiPulse/
├── app.py                           # Main Dash + Flask app with callbacks, routing, and health endpoint
├── create_csv.py                    # Offline preprocessing pipeline for cleaning and spatial enrichment
├── preprocessed_taxi_data.parquet   # Preprocessed dataset used for faster local analysis
├── requirements.txt                 # Python dependencies
├── Dockerfile                       # Container build and runtime config
├── .dockerignore                    # Docker context exclusions
├── taxi_zones/                      # NYC taxi zone shapefile assets
│   ├── taxi_zones.shp               # Zone geometry
│   ├── taxi_zones.shx               # Shape index
│   ├── taxi_zones.dbf               # Attribute table
│   └── taxi_zones.prj               # Coordinate reference metadata
└── README.md                        # Project documentation
```

## Status 📶

**Maintained / Active.**

Stable:
- Dashboard routing and filters
- Core metrics and charts
- Geospatial zone mapping
- Dockerized deployment path

Experimental / evolving:
- Data source reliability (external sheet dependency)
- Advanced forecasting or recommendation logic

## Limitations ⚠️

- Depends on an external CSV source; outages or schema drift can break ingestion.
- In-memory processing can become slow with significantly larger datasets.
- Spatial join quality depends on coordinate quality and polygon boundaries.
- No historical model-based prediction yet (current insights are descriptive/diagnostic).
- Limited automated test coverage in the current repository.

## Improvements / Roadmap 🚀

- Add a scheduled ETL job + versioned dataset snapshots (S3/GCS) to remove runtime data-source coupling.
- Add caching/materialized aggregates for low-latency filtering at larger scale.
- Introduce automated tests for filter logic, callbacks, and data contracts.
- Add trip-demand forecasting (hourly/zone-level) for proactive driver recommendations.
- Add CI workflow for linting, dependency checks, and container build validation.

## Credits 📝

- NYC Taxi & Limousine Commission (TLC) zone definitions and trip data ecosystem
- GeoPandas documentation: https://geopandas.org/
- Plotly + Dash documentation: https://dash.plotly.com/
- Folium documentation: https://python-visualization.github.io/folium/

## Author

Raghuveer — [LinkedIn](https://www.linkedin.com/) · [GitHub](https://github.com/raghuveer9303)
