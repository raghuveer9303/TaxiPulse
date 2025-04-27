# **TaxiPulse** - *NYC Taxi Intelligence*


This project processes taxi trip data with spatial information, combining it with NYC taxi zone data to create an enriched dataset for analysis.

## Project Description

The project performs the following operations:
1. Downloads taxi trip data from a Google Sheets source
2. Cleans and validates coordinate data
3. Performs spatial joins with NYC taxi zones
4. Calculates additional features like pickup hour, day, and tip percentages
5. Saves the preprocessed data in Parquet format

## Prerequisites

- Python 3.7 or higher
- Required Python packages (see requirements.txt)
- Taxi zones shapefile (included in the `taxi_zones/` directory)

## Installation

1. Clone this repository
2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```
3. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Ensure the taxi_zones folder containing the shapefile is in the project directory
2. Run the preprocessing script:
```bash
python create_csv.py
```

The script will:
- Download the latest taxi data
- Process and clean the data
- Perform spatial analysis
- Save the results as `preprocessed_taxi_data.parquet`

## Data Output

The processed data includes the following additional columns:
- pickup_zone: The NYC taxi zone where the pickup occurred
- pickup_borough: The borough where the pickup occurred
- dropoff_zone: The NYC taxi zone where the dropoff occurred
- dropoff_borough: The borough where the dropoff occurred
- pickup_hour: Hour of the day when pickup occurred
- pickup_day: Day of the week
- tip_pct: Tip percentage calculated from the fare

## Error Handling

The script includes comprehensive error handling for:
- Missing data files
- Invalid coordinates
- Missing columns
- Data type conversion issues
- File saving errors

## Dependencies

See requirements.txt for a complete list of Python package dependencies.
