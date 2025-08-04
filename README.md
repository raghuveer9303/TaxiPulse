# 🚖 TaxiPulse - NYC Taxi Intelligence Dashboard

**TaxiPulse** is a powerful web-based dashboard that transforms NYC Yellow Taxi trip data into valuable insights for maximizing earnings.  
It combines spatial analysis, dynamic charts, and interactive maps to help taxi drivers and fleet managers **optimize their strategies**.

🔗 **Live Demo**: [https://taxipulse.onrender.com](https://taxipulse.onrender.com)

Video Link: https://youtu.be/jt1x2oKrZ_0

---

## 📄 Project Description

TaxiPulse performs:

- 📥 **Live Data Download**: Pulls trip data from a Google Sheets source.
- 🧹 **Data Cleaning**: Validates coordinates, filters outliers.
- 🌐 **Spatial Join**: Matches trips to official NYC Taxi Zones.
- ✨ **Feature Engineering**: Adds pickup/dropoff zones, boroughs, hours, days, and tip percentages.
- 📊 **Data Visualization**: Interactive graphs and maps to explore trip patterns.
- 📦 **Data Storage**: Saves processed data as fast, efficient Parquet files.

---

## 🚀 Features

- 📈 Dynamic Dashboard for earnings and demand/supply analysis
- 🗺️ Interactive Revenue Choropleth Map and Pickup Heatmap
- 📊 Analytics Page for payment trends, passenger counts, weekday performance
- 📦 Efficient Parquet storage for preprocessed datasets
- 🐳 Dockerized and cloud-ready deployment
- 🛡️ Robust error handling for missing data and failures

---

## 🛠️ Tech Stack

| Layer              | Technologies                             |
|--------------------|------------------------------------------|
| Backend/API        | Python 3.9, Flask, Waitress              |
| Frontend Dashboard | Dash, Plotly, Dash Bootstrap Components  |
| Spatial Analysis   | GeoPandas, Folium, Shapely               |
| Data Storage       | Parquet (via Pandas)                     |
| Deployment         | Docker, Render Cloud                    |

---

## ⚙️ Installation

### Local Setup

```bash
git clone https://github.com/yourusername/taxipulse.git
cd taxipulse
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

## Docker 
docker build -t taxipulse .
docker run -d -p 8080:8080 taxipulse
