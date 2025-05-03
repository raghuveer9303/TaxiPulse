# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import geopandas as gpd
import folium
from folium.plugins import HeatMap
from shapely.geometry import Point
from waitress import serve
import logging
from datetime import datetime
from flask import jsonify

import dash
from dash import Dash, dcc, html, dash_table, Input, Output, State, callback
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from dash.exceptions import PreventUpdate
import plotly.figure_factory as ff

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Data Loading and Preparation ---
# Data sources (Assuming files are accessible)
DATA_URL = (
    'https://docs.google.com/spreadsheets/d/e/2PACX-1vTZPBWbFSSjfPsKpEK3p1RNIqWbli3lpnZ_OUjQQf-6ygIjCXlZX7TKHAsC3rlqIRecbEE1C0-DLNk6'
    '/pub?gid=1860100186&single=true&output=csv'
)
# --- !!! IMPORTANT: Make sure this path is correct relative to where you run the script !!! ---
TAXI_ZONE_SHAPEFILE_PATH = './taxi_zones/taxi_zones.shp'
# Download shapefile if it doesn't exist (Optional helper)
import os
import requests
import zipfile
import io

if not os.path.exists(TAXI_ZONE_SHAPEFILE_PATH):
    print("Taxi zone shapefile not found. Attempting to download...")
    try:
        shapefile_url = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip"
        r = requests.get(shapefile_url, stream=True)
        r.raise_for_status() # Raise an exception for bad status codes
        z = zipfile.ZipFile(io.BytesIO(r.content))
        os.makedirs('./taxi_zones', exist_ok=True) # Create directory if needed
        z.extractall('./taxi_zones')
        print(f"Shapefile downloaded and extracted to ./taxi_zones/")
        if not os.path.exists(TAXI_ZONE_SHAPEFILE_PATH):
             print(f"Error: Expected file {TAXI_ZONE_SHAPEFILE_PATH} not found after extraction.")
             # Handle error appropriately - maybe exit or disable map features
    except requests.exceptions.RequestException as e:
        print(f"Error downloading shapefile: {e}")
    except zipfile.BadZipFile:
        print("Error: Downloaded file is not a valid zip archive.")
    except Exception as e:
        print(f"An unexpected error occurred during download/extraction: {e}")

# Load ride data
try:
    df = pd.read_csv(DATA_URL)
    # Basic Data Cleaning/Type Conversion
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'], errors='coerce') # Coerce errors
    numeric_cols = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude',
                    'dropoff_latitude', 'total_amount', 'tip_amount', 'trip_distance']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Add pickup_weekday column
    df['pickup_weekday'] = df['tpep_pickup_datetime'].dt.day_name()

    # Drop rows where essential data is missing after coercion
    df.dropna(subset=['tpep_pickup_datetime', 'pickup_longitude', 'pickup_latitude',
                     'total_amount', 'tip_amount'], inplace=True)

    # Filter unreasonable values (example)
    df = df[(df['pickup_latitude'].between(40.4, 41.0)) & (df['pickup_longitude'].between(-74.3, -73.6))]
    df = df[df['total_amount'].between(0, 1000)] # Example fare range
    df = df[df['tip_amount'] >= 0] # Ensure tip is not negative

except Exception as e:
    print(f"Error loading or processing ride data from URL: {e}")
    # Provide fallback or exit if essential data is missing
    df = pd.DataFrame(columns=[
        'tpep_pickup_datetime', 'pickup_longitude', 'pickup_latitude',
        'dropoff_longitude', 'dropoff_latitude', 'total_amount', 'tip_amount',
        'trip_distance', 'VendorID', 'LocationID', 'DOLocationID' # Add potential zone IDs
    ])


# Load taxi zone shapes
try:
    gdf_zones = gpd.read_file(TAXI_ZONE_SHAPEFILE_PATH)
    gdf_zones = gdf_zones[['LocationID','borough','zone','geometry']]
    gdf_zones['LocationID'] = gdf_zones['LocationID'].astype(int)
    # Ensure CRS is WGS84 (lat/lon) which folium expects
    gdf_zones = gdf_zones.to_crs('EPSG:4326')
except Exception as e:
     print(f"Error loading or processing taxi zone shapefile from {TAXI_ZONE_SHAPEFILE_PATH}: {e}")
     # Handle missing shapefile - perhaps exit or disable map features
     gdf_zones = gpd.GeoDataFrame(columns=['LocationID', 'borough', 'zone', 'geometry'], crs="EPSG:4326") # Empty placeholder

# --- Spatial Join Optimization ---
# Perform spatial join ONLY IF both df and gdf_zones are loaded successfully and df has coordinates
if not df.empty and not gdf_zones.empty and 'pickup_latitude' in df.columns:

    # Create GeoDataFrame from pickup points
    geometry_pickup = [Point(xy) for xy in zip(df['pickup_longitude'], df['pickup_latitude'])]
    gdf_pickups = gpd.GeoDataFrame(df[['VendorID', 'tpep_pickup_datetime']], geometry=geometry_pickup, crs="EPSG:4326")

    # Spatial join for pickups
    gdf_pickups = gpd.sjoin(gdf_pickups, gdf_zones[['LocationID', 'zone', 'borough', 'geometry']], how='left', predicate='within')

    # Add pickup zone info back to original df using index
    df['PULocationID'] = gdf_pickups['LocationID'].astype('Int64') # Use nullable integer type
    df['pickup_zone'] = gdf_pickups['zone']
    df['pickup_borough'] = gdf_pickups['borough']


    # Create GeoDataFrame from dropoff points (handle missing dropoff coords)
    df_dropoff = df.dropna(subset=['dropoff_longitude', 'dropoff_latitude']).copy()
    if not df_dropoff.empty:
        geometry_dropoff = [Point(xy) for xy in zip(df_dropoff['dropoff_longitude'], df_dropoff['dropoff_latitude'])]
        gdf_dropoffs = gpd.GeoDataFrame(df_dropoff[['VendorID']], geometry=geometry_dropoff, crs="EPSG:4326")
        # Spatial join for dropoffs
        gdf_dropoffs = gpd.sjoin(gdf_dropoffs, gdf_zones[['LocationID', 'zone', 'borough', 'geometry']], how='left', predicate='within')
        # Add dropoff zone info back to original df using index
        df['DOLocationID'] = gdf_dropoffs['LocationID'].reindex(df.index).astype('Int64') # Align index and use nullable integer
        df['dropoff_zone'] = gdf_dropoffs['zone'].reindex(df.index)
        df['dropoff_borough'] = gdf_dropoffs['borough'].reindex(df.index)


# --- Feature Engineering ---
if not df.empty and 'tpep_pickup_datetime' in df.columns:
    df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour
    df['pickup_day'] = df['tpep_pickup_datetime'].dt.day_name()
    # Calculate tip percentage safely, avoiding division by zero or negative bases
    df['tip_base'] = df['total_amount'] - df['tip_amount']
    df['tip_pct'] = np.where(df['tip_base'] > 0, (df['tip_amount'] / df['tip_base']) * 100, 0)
    df.drop(columns=['tip_base'], inplace=True) # Clean up intermediate column

# Fill potential NaNs created during join (or if spatial join failed)
for col in ['pickup_zone', 'pickup_borough', 'dropoff_zone', 'dropoff_borough']:
    if col not in df.columns:
        df[col] = 'Unknown' # Create column if it doesn't exist
    else:
        df[col].fillna('Unknown', inplace=True)

# Filter out 'Unknown' borough from dropdown options if desired
valid_boroughs = []
if 'pickup_borough' in df.columns:
     valid_boroughs = sorted(df[df['pickup_borough'] != 'Unknown']['pickup_borough'].dropna().unique())


# --- Helper Functions ---
def filter_df(data, day_type, borough, time_of_day):
    """Filters the DataFrame based on selected criteria."""
    if data.empty:
        return data

    d = data.copy() # Work on a copy

    # Ensure necessary columns exist
    if 'pickup_day' not in d.columns or 'pickup_hour' not in d.columns or 'pickup_borough' not in d.columns:
        print("Warning: Filtering columns missing, returning original data.")
        return data # Return original if filtering cols are missing

    if day_type=='Weekdays':
        d = d[d['pickup_day'].isin(['Monday','Tuesday','Wednesday','Thursday','Friday'])]
    elif day_type=='Weekends':
        d = d[d['pickup_day'].isin(['Saturday','Sunday'])]

    if borough!='All Boroughs':
        d = d[d['pickup_borough']==borough]

    tod = {'Morning (5AM-11AM)':(5,11),'Midday (12PM-3PM)':(12,15),
           'Evening (4PM-7PM)':(16,19),'Night (8PM-11PM)':(20,23),'Late Night (12AM-4AM)':(0,4)}
    if time_of_day in tod:
        lo,hi = tod[time_of_day]
        # Handle Late Night wrap-around correctly
        if time_of_day == 'Late Night (12AM-4AM)':
             # This logic includes 0, 1, 2, 3, 4 but not higher hours
             d = d[(d['pickup_hour']>=lo) & (d['pickup_hour']<=hi)]
        else:
             d = d[(d['pickup_hour']>=lo) & (d['pickup_hour']<=hi)]
    return d

def filter_by_specific_time(df_to_filter, time_period):
    """Applies a specific time filter (Morning, Afternoon, etc.) to a DataFrame."""
    # Ensure pickup_hour column exists
    if time_period == 'All Day' or 'pickup_hour' not in df_to_filter.columns:
        return df_to_filter

    # Define hour ranges explicitly, ensuring end hour is inclusive for filtering
    # Example: Morning is 6:00 AM up to 11:59 AM (hours 6 through 11)
    hour_map = {
        'Morning': (6, 11),    # 6 AM to 11:59 AM
        'Afternoon': (12, 16), # 12 PM to 4:59 PM
        'Evening': (17, 20),   # 5 PM to 8:59 PM
        'Night': (21, 5)      # 9 PM to 5:59 AM (includes wrap-around)
    }

    if time_period in hour_map:
        start_hour, end_hour = hour_map[time_period]
        if start_hour <= end_hour: # Morning, Afternoon, Evening case
            mask = (df_to_filter['pickup_hour'] >= start_hour) & (df_to_filter['pickup_hour'] <= end_hour)
        else: # Night case (wrap-around)
            mask = (df_to_filter['pickup_hour'] >= start_hour) | (df_to_filter['pickup_hour'] <= end_hour)
        return df_to_filter[mask]
    else:
        # If time_period is not recognized (shouldn't happen with RadioItems), return unfiltered
        return df_to_filter

# Removed filter_by_hour_range as it's not used in the current layout

# --- Dash App Initialization ---
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],
           meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}])
server = app.server
app.title = "NYC Supercharge Your Taxi Earnings Dashboard"

# --- Layout Definition ---
# Controls (Same as before)
controls = dbc.Card(dbc.CardBody(dbc.Row([
    dbc.Col([html.Label('Day Type'), dcc.Dropdown(id='day-type', value='All Days', clearable=False,
        options=[{'label':v,'value':v} for v in ['All Days','Weekdays','Weekends']])], md=4),
    dbc.Col([html.Label('Borough'), dcc.Dropdown(id='borough', value='All Boroughs', clearable=False,
        options=[{'label':b,'value':b} for b in ['All Boroughs']+valid_boroughs])], md=4),
    dbc.Col([html.Label('Time of Day'), dcc.Dropdown(id='time-of-day', value='All Times', clearable=False,
        options=[{'label':'All Times','value':'All Times'}]+[
            {'label':k,'value':k} for k in ['Morning (5AM-11AM)','Midday (12PM-3PM)',
                                             'Evening (4PM-7PM)','Night (8PM-11PM)','Late Night (12AM-4AM)']
        ])], md=4)
])), className='mb-4 shadow-sm')

# Metric Cards (Same as before)
card_peak = lambda title,id,hint: dbc.Card(dbc.CardBody([
    html.H6(title,className='card-title text-muted'),
    html.H3(id=id, className='text-primary'),
    html.P(id=hint,className='text-success small')
]), className='m-1 flex-fill shadow-sm')

card_peak_hour = card_peak('Peak Earning Hour','peak-hour','peak-hour-value')
card_tip_area = card_peak('Highest Avg Tip Zone','top-tip-area','top-tip-value')
card_best_day = card_peak('Best Earning Day','best-day','best-day-value')

# Navbar with video link
navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Dashboard", href="/", id="dashboard-link")),
        dbc.NavItem(dbc.NavLink("Hotspots", href="/hotspots", id="hotspots-link")),
        dbc.NavItem(dbc.NavLink("Analytics", href="/analytics", id="analytics-link")),
        dbc.NavItem(
            dbc.NavLink(
                "📹 Watch Demo Video", 
                href="https://indiana-my.sharepoint.com/:v:/g/personal/ragvenk_iu_edu/Eas6Utk2eM5IoKGrsl8OUacBOgfnFyfMnnRJ0i7j3GuqMg?nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJTdHJlYW1XZWJBcHAiLCJyZWZlcnJhbFZpZXciOiJTaGFyZURpYWxvZy1MaW5rIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXcifX0%3D&e=IEZwYJ",
                target="_blank",
                style={"border": "1px solid white", "border-radius": "4px", "padding": "4px 8px", "margin-left": "10px", "color": "white", "font-weight": "bold"}
            )
        ),
    ],
    brand='🚕 NYC Taxi Earnings Maximizer',
    brand_href="/",
    color='dark',
    dark=True,
    sticky="top",
    className="mb-4"
)

# Dashboard Layout (Same as before, no changes needed here)
dashboard_layout = dbc.Container([
    controls,
    dbc.Row([
        dbc.Col([
             html.H5('Hourly Earnings & Demand/Supply'),
             dcc.Loading(dcc.Graph(id='hourly-earnings'), type="circle"),
             html.P("See at a glance how your earnings rise and dip over the course of the day, so you can choose the most profitable hours to work. "
                 "And with the demand-to-supply line, you’ll know exactly when the streets are buzzing—meaning shorter waits between fares and more money in your pocket.",className='mt-2 text-muted')
        ], width=12, className='mb-4'),
    ]),
    dbc.Row([
        dbc.Col([
            html.H5('Zone Profitability (Revenue per Mile)'),
            dcc.Loading(dcc.Graph(id='zone-performance'), type="circle"),
            html.P("See which neighborhoods pay off the most per mile so you can steer your shifts toward the streets that boost your earnings. "
                    "On the treemap, larger blocks show where you’ve hauled in the most cash overall, and the lighter the color, the more you’re getting out of every mile.",className='mt-2 text-muted')
        ], md=8, className='mb-4'),
        dbc.Col([
            html.H5('Key Metrics'),
            dbc.Row([dbc.Col(card_peak_hour)], className='mb-2'),
            dbc.Row([dbc.Col(card_tip_area)], className='mb-2'),
            dbc.Row([dbc.Col(card_best_day)], className='mb-2'),
            html.P("These key metrics summarize overall performance: peak earning hour, best tipping zone, and most profitable day. "
                    "Use these insights to optimize your driving strategy at a glance.",
                    className='mt-2 text-muted')
        ], md=4, className='mb-4'),
    ], className='g-3 align-items-stretch'),
    dbc.Row([
         dbc.Col([
            html.H5('Top 5 Most Profitable Routes (by Avg Fare)'),
            dcc.Loading(dash_table.DataTable(
                id='top-routes-table',
                columns=[
                    {'name':'Pickup Zone','id':'Pickup Zone', 'presentation':'markdown'},
                    {'name':'Dropoff Zone','id':'Dropoff Zone', 'presentation':'markdown'},
                    {'name':'Avg Fare','id':'Avg Fare'},
                    {'name':'Avg Tip %','id':'$ Avg Tip %'},
                    {'name':'Rides','id':'Rides'},
                ],
                page_size=5,
                style_cell={'textAlign': 'left', 'padding': '5px'},
                style_header={'backgroundColor': '#f8f9fa', 'fontWeight': 'bold'},
                style_table={'overflowX': 'auto', 'minWidth': '100%'},
                style_data_conditional=[{
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgb(240, 240, 240)',
                }]
            ), type="circle")
        ], width=12)
    ])
], fluid=True)

# --- MODIFIED Hotspots Layout ---
hotspots_layout = dbc.Container([
    controls, # Re-use the same controls card

    dbc.Row([
        # --- Folium Choropleth Map ---
        dbc.Col([
            html.H5('Zone Revenue Map (Choropleth)', className='text-center'),
            # Use dcc.Loading around an html.Iframe to display the folium map
            dcc.Loading(
                html.Iframe(
                    id='pickup-choropleth-map-iframe',
                    style={'border': 'none', 'width': '100%', 'height': '450px'}
                ), type="circle"
            ),
            html.Div(id='pickup-choropleth-stats', className='alert alert-info mt-2 small p-2'),
            html.P("Map highlights total revenue collected in each zone, showing hotspots of high earnings. "
                    "Hover for detailed stats to pinpoint top-performing neighborhoods.",
                    className='mt-2 text-muted')
        ], md=6, className='mb-3'),

        # --- Folium HeatMap ---
        dbc.Col([
            html.H5('Pickup Density Heatmap', className='text-center'),
            dcc.Loading(
                 html.Iframe(
                    id='pickup-heatmap-iframe',
                    # srcDoc=INITIAL_HEATMAP_HTML, # Optional: provide initial placeholder HTML
                    style={'border': 'none', 'width': '100%', 'height': '450px'} # Basic styling
                ), type="circle"
            ),
            html.Div(id='pickup-heatmap-stats', className='alert alert-info mt-2 small p-2'),
            html.P("This heatmap displays pickup density hotspots, indicating where demand concentrates. "
                    "The marker points to the single busiest location for targeted cruising.",
                    className='mt-2 text-muted')
        ], md=6, className='mb-3'),
    ], className='g-3'),
], fluid=True)

# --- New Analytics Page Layout ---
analytics_layout = dbc.Container([  # Reuse the same controls
    dbc.Row([
        # Summary Cards
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Total Trips", className="card-title text-center text-primary"),
                    html.H2(id="total-trips-card", className="text-center font-weight-bold")
                ])
            ], className="mb-4 shadow-sm border-primary")
        ], md=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Avg Fare Per Trip", className="card-title text-center text-success"),
                    html.H2(id="avg-fare-card", className="text-center font-weight-bold")
                ])
            ], className="mb-4 shadow-sm border-success")
        ], md=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Total Distance", className="card-title text-center text-info"),
                    html.H2(id="total-distance-card", className="text-center font-weight-bold")
                ])
            ], className="mb-4 shadow-sm border-info")
        ], md=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Avg Distance/Trip", className="card-title text-center text-warning"),
                    html.H2(id="avg-distance-card", className="text-center font-weight-bold")
                ])
            ], className="mb-4 shadow-sm border-warning")
        ], md=3),
    ], className="mb-4"),
    
    dbc.Row([
        # Donut Charts - Payment Type and Passenger Count
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("Trips by Payment Type", className="text-center")),
                dbc.CardBody([
                    dcc.Loading(dcc.Graph(id="payment-type-donut"), type="circle")
                ]),
                html.P("Donut chart breaks down how passengers pay, helping you anticipate payment methods. "
                    "See which method dominates to manage cash and digital payments accordingly.",
                    className='mt-2 text-muted')
            ], className="shadow")
        ], md=6),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("Trips by Passenger Count", className="text-center")),
                dbc.CardBody([
                    dcc.Loading(dcc.Graph(id="passenger-count-donut"), type="circle"),
                    html.P("Shows distribution of passenger group sizes, aiding in resource planning. "
                        "Identify common ride types, from solo trips to group share opportunities.",
                        className='mt-2 text-muted')
                ])
            ], className="shadow")
        ], md=6),
    ], className="mb-4"),
    
    dbc.Row([
        # Line chart for Trips by Weekday
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("Total Trips by Weekday", className="text-center")),
                dbc.CardBody([
                    dcc.Loading(dcc.Graph(id="trips-by-weekday-line"), type="circle"),
                    html.P("Line chart tracks ride volume across weekdays, revealing peak days for consistent scheduling. "
                        "Highlighted point marks the busiest day to prioritize availability.",
                         className='mt-2 text-muted')
                ])
            ], className="shadow")
        ], md=6),
        # Bar chart for Revenue by Day of Week
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("Total Revenue by Day of Week", className="text-center")),
                dbc.CardBody([
                    dcc.Loading(dcc.Graph(id="revenue-by-weekday-bar"), type="circle"),
                    html.P("Bar chart compares total revenue by day of week, uncovering top-performing days. "
                        "Dashed lines show average weekday and weekend earnings benchmarks.",
                        className='mt-2 text-muted')
                ])
            ], className="shadow")
        ], md=6),
    ], className="mb-4"),
    
    dbc.Row([
        # Additional creative visualization - Heatmap of hourly trips by day
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("Trip Density by Day and Hour", className="text-center")),
                dbc.CardBody([
                    dcc.Loading(dcc.Graph(id="hour-day-heatmap"), type="circle"),
                    html.P("Heatmap visualizes ride count across each hour and weekday, pinpointing peak demand pockets. "
                        "Annotated 'PEAK' shows the single highest surge time for strategic planning.",
                        className='mt-2 text-muted')
                ])
            ], className="shadow")
        ], width=12),
    ]),
], fluid=True)

# Main App Layout (Same as before)
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    navbar,
    html.Div(id='page-content') # Content will be injected here
])

# --- Callbacks ---

# URL Routing
@app.callback(
    [Output('page-content', 'children'),
     Output('dashboard-link', 'active'),
     Output('hotspots-link', 'active'),
     Output('analytics-link', 'active')],
    [Input('url', 'pathname')]
)
def display_page(pathname):
    if pathname == '/hotspots':
        return hotspots_layout, False, True, False
    elif pathname == '/analytics':
        return analytics_layout, False, False, True
    # Default to dashboard layout for "/" or any other path
    return dashboard_layout, True, False, False

# Dashboard Update Callback (Same as before, no changes needed here)
@app.callback(
    [Output('peak-hour','children'),Output('peak-hour-value','children'),
     Output('top-tip-area','children'),Output('top-tip-value','children'),
     Output('best-day','children'),Output('best-day-value','children'),
     Output('hourly-earnings','figure'),Output('zone-performance','figure'),
     Output('top-routes-table','data')],
    [Input('day-type','value'),Input('borough','value'),Input('time-of-day','value')]
)
def update_dashboard(day_type, borough, time_of_day):
    if df.empty or 'pickup_hour' not in df.columns: # Added check for necessary columns
        nodata_fig = go.Figure().update_layout(title="No data available or missing columns", xaxis_visible=False, yaxis_visible=False)
        empty_metrics = ['N/A','','N/A','','N/A','']
        return *empty_metrics, nodata_fig, nodata_fig, []

    d = filter_df(df, day_type, borough, time_of_day)
    # Apply specific time filter if needed
    
    # Check if filtered data is empty
    if d.empty:
        nodata_fig = go.Figure().update_layout(title="No data for selected filters", xaxis_visible=False, yaxis_visible=False)
        empty_metrics = ['N/A','','N/A','','N/A','']
        return *empty_metrics, nodata_fig, nodata_fig, []

    # Hourly Earnings Figure
    hr = d.groupby('pickup_hour').agg(
        avg_earn=('total_amount','mean'),
        count=('total_amount','size'),
        active_taxis=('VendorID', 'nunique')
    ).reset_index()
    # Ensure active_taxis doesn't cause division by zero issues if empty after grouping
    hr['active_taxis'] = hr['active_taxis'].replace(0, np.nan)
    hr['demand_supply'] = (hr['count'] / hr['active_taxis']).fillna(0) # Fill NaN demand/supply with 0

    fig_hr = go.Figure()
    fig_hr.add_trace(go.Bar(
        x=hr['pickup_hour'], y=hr['avg_earn'],
        name='Avg Earnings', marker_color='#88CCEE',  # color-blind safe blue
        hovertemplate='Hour: %{x}:00<br>Avg Earnings: $%{y:.2f}<br>Rides: %{customdata[0]}<extra></extra>',
        customdata=hr[['count']]
    ))
    fig_hr.add_trace(go.Scatter(
        x=hr['pickup_hour'], y=hr['demand_supply'],
        name='Demand/Supply Ratio', mode='lines+markers',
        yaxis='y2', marker_color='#CC6677',  # color-blind safe red
        hovertemplate='Hour: %{x}:00<br>Demand/Supply: %{y:.2f}<extra></extra>', # Format taxis as integer
        customdata=hr[['active_taxis']].fillna(0) # Fill NaN for hover
    ))

    fig_hr.update_layout(
        yaxis=dict(title='Avg Earnings ($)', side='left'),
        yaxis2=dict(title='Demand/Supply (Rides/Taxi)', overlaying='y', side='right', showgrid=False, range=[0, max(1, hr['demand_supply'].max()*1.2)]), # Ensure range starts at 0 and has buffer
        xaxis=dict(title='Hour of Day', tickmode='array', tickvals=list(range(24)), ticktext=[f"{h:02d}" for h in range(24)]),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
        margin=dict(l=50, r=50, t=30, b=50),
        hovermode='x unified',
        height=350,
        plot_bgcolor='white'
    )
    


    fig_hr.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    fig_hr.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')

    max_idx = hr['demand_supply'].idxmax() if not hr.empty else None

    fig_hr.add_annotation(
        x=hr.loc[max_idx, 'pickup_hour'],
        y=hr.loc[max_idx, 'demand_supply'],
        xref='x', yref='y2',  # Attach annotation to the secondary axis
        text="Peak demand here—lots of passengers waiting, time to head out!",
        showarrow=True, arrowhead=2, ax=20, ay=-40,  # Adjusted arrow position to avoid overlap
        font=dict(color="#CC6677", size=12), bgcolor="white", bordercolor="#CC6677", borderwidth=1
    )

    # Zone Performance Treemap
    zm = d[(d['trip_distance'] > 0) & (d['pickup_borough'] != 'Unknown') & (d['pickup_zone'] != 'Unknown')].groupby(['pickup_borough','pickup_zone']).agg(
        total_revenue=('total_amount','sum'),
        total_distance=('trip_distance','sum'),
        ride_count=('total_amount', 'size')
    ).reset_index()

    if not zm.empty:
        zm['revenue_per_mile'] = zm['total_revenue'] / zm['total_distance']
        zm = zm[zm['revenue_per_mile'].notna()] # Remove rows where division resulted in NaN (shouldn't happen with filter > 0)
    else:
        zm['revenue_per_mile'] = None # Add column if zm is empty to avoid errors

    if zm.empty:
         fig_zone = go.Figure().update_layout(title="No zone data for selected filters", xaxis_visible=False, yaxis_visible=False)
    else:
        fig_zone = px.treemap(
            zm,
            path=[px.Constant('All Zones'), 'pickup_borough', 'pickup_zone'],
            values='total_revenue',
            color='revenue_per_mile',
            color_continuous_scale=px.colors.sequential.Viridis,  # color-blind safe continuous scale
            range_color=[0, max(1, zm['revenue_per_mile'].quantile(0.95) if zm['revenue_per_mile'].notna().any() else 1)], # Cap color range safely
            custom_data=['ride_count', 'revenue_per_mile']
        )
        fig_zone.update_traces(
            hovertemplate='<b>%{label}</b> (%{parent})<br>Total Revenue: $%{value:,.0f}<br>Revenue/Mile: $%{customdata[1]:.2f}<br>Rides: %{customdata[0]}<extra></extra>'
        )
        fig_zone.update_layout(
             margin=dict(l=10, r=10, t=30, b=10),
             height=600,
             coloraxis_colorbar=dict(title="Rev/Mile ($)")
        )

    # Metrics Calculation
    ph_row = hr.loc[hr['avg_earn'].idxmax()] if not hr.empty else None
    ph_str = f"{int(ph_row['pickup_hour']):02d}:00" if ph_row is not None else 'N/A'
    ph_val = f"${ph_row['avg_earn']:.2f} / hr" if ph_row is not None else ''

    tz_series = d[(d['pickup_zone'] != 'Unknown') & (d['tip_pct'] >= 0) & (d['tip_pct'] < 200)].groupby('pickup_zone')['tip_pct'].mean() # Filter bad tip pcts
    tz = tz_series.idxmax() if not tz_series.empty else 'N/A' # Use N/A string if empty
    tz_val = f"{tz_series.max():.1f}%" if not tz_series.empty else ''

    bd_series = d.groupby('pickup_day')['total_amount'].mean()
    bd = bd_series.idxmax() if not bd_series.empty else 'N/A'
    bd_val = ''
    if not bd_series.empty and d['total_amount'].mean() > 0:
        overall_mean = d['total_amount'].mean()
        percentage_diff = ((bd_series.max() / overall_mean) - 1) * 100
        bd_val = f"{percentage_diff:.0f}% higher than avg"
    elif not bd_series.empty:
        bd_val = "$0.00 / ride" # If average is 0

    # Top Routes Table
    rt = d[(d['pickup_zone'] != 'Unknown') & (d['dropoff_zone'] != 'Unknown')].groupby(['pickup_zone','dropoff_zone']).agg(
        avg_fare=('total_amount','mean'),
        avg_tip=('tip_pct','mean'),
        count=('total_amount','size')
    ).reset_index()
    rt = rt[rt['count'] > 5] # Filter for significance
    top_routes = rt.nlargest(5,'avg_fare')

    route_records=[]
    for _,r in top_routes.iterrows():
        route_records.append({
            'Pickup Zone':r['pickup_zone'],
            'Dropoff Zone':r['dropoff_zone'],
            'Avg Fare':f"${r['avg_fare']:.2f}",
            '$ Avg Tip %':f"{r['avg_tip']:.1f}%" if pd.notna(r['avg_tip']) else "N/A", # Handle potential NaN tips
            'Rides': f"{r['count']}",
        })

    return ph_str, ph_val, tz, tz_val, bd, bd_val, fig_hr, fig_zone, route_records


# --- MODIFIED Hotspots Visualization Update Callback ---
@app.callback(
    [Output('pickup-choropleth-map-iframe', 'srcDoc'), # Output to Iframe srcDoc
     Output('pickup-heatmap-iframe', 'srcDoc'),       # Output to Iframe srcDoc
     Output('pickup-choropleth-stats', 'children'),
     Output('pickup-heatmap-stats', 'children')],
    [Input('day-type', 'value'),
     Input('borough', 'value'),
     Input('time-of-day', 'value')]
)
def update_hotspots_folium_maps(day_type, borough, time_of_day):
    # --- Initial Data Checks ---
    if df.empty or 'pickup_latitude' not in df.columns or 'PULocationID' not in df.columns:
        no_data_html = "<p>Map data is unavailable.</p>"
        return no_data_html, no_data_html, "No taxi data available.", "No analysis data available."
    if gdf_zones.empty:
        no_zones_html = "<p>Taxi zone boundaries are unavailable.</p>"
        return no_zones_html, no_zones_html, "Taxi zone data missing.", "Taxi zone data missing."

    # 1. Apply general filters (Day Type, Borough, Time of Day)
    d_filtered_initial = filter_df(df, day_type, borough, time_of_day)

    # --- Create placeholder HTML for empty filtered data ---
    no_filter_data_html = "<p>No ride data matches the selected filters.</p>"
    empty_stats = "No data available for selected filters."

    if d_filtered_initial.empty:
        return no_filter_data_html, no_filter_data_html, empty_stats, empty_stats

    # NYC Focus
    nyc_center = [40.7128, -74.0060]
    initial_zoom = 10

    # --- Folium Choropleth Map ---
    try:
        # --- Calculate zone revenue metrics ---
        zone_revenue = d_filtered_initial.groupby('PULocationID').agg({
            'total_amount': 'sum',
            'VendorID': 'size'  # Count for number of rides
        }).reset_index()
        zone_revenue = zone_revenue[zone_revenue['PULocationID'].notna()]
        zone_revenue['PULocationID'] = zone_revenue['PULocationID'].astype(int)

        # Ensure gdf_zones['LocationID'] is also int type
        gdf_zones['LocationID'] = gdf_zones['LocationID'].astype(int)
        gdf_merged_revenue = gdf_zones.merge(zone_revenue, left_on='LocationID', right_on='PULocationID', how='left')
        gdf_merged_revenue['total_amount'].fillna(0, inplace=True)
        gdf_merged_revenue['VendorID'].fillna(0, inplace=True)
        gdf_merged_revenue['avg_ride_revenue'] = gdf_merged_revenue['total_amount'] / gdf_merged_revenue['VendorID'].replace(0, np.nan)
        gdf_merged_revenue['avg_ride_revenue'].fillna(0, inplace=True)

        map_choropleth = folium.Map(location=nyc_center, zoom_start=initial_zoom, tiles='cartodbpositron')

        folium.Choropleth(
            geo_data=gdf_merged_revenue.__geo_interface__,
            name='choropleth',
            data=gdf_merged_revenue,
            columns=['LocationID', 'total_amount'],
            key_on='feature.properties.LocationID',
            fill_color='YlGnBu',
            fill_opacity=0.7,
            line_opacity=0.2,
            legend_name='Total Revenue ($)',
            highlight=True,
            nan_fill_color='grey'
        ).add_to(map_choropleth)

        tooltip = folium.features.GeoJsonTooltip(
            fields=['zone', 'borough', 'total_amount', 'VendorID', 'avg_ride_revenue'],
            aliases=['Zone:', 'Borough:', 'Total Revenue ($):', 'Number of Rides:', 'Avg Revenue/Ride ($):'],
            localize=True,
            sticky=False,
            labels=True,
            max_width=800
        )

        folium.GeoJson(
            gdf_merged_revenue,
            style_function=lambda x: {'fillColor': 'transparent', 'color': 'transparent'},
            tooltip=tooltip,
            name='Zone Information'
        ).add_to(map_choropleth)

        folium.LayerControl().add_to(map_choropleth)

        top_revenue_zones = gdf_merged_revenue.nlargest(3, 'total_amount')
        revenue_stats_list = [f"{row['zone']} (${int(row['total_amount']):,})" for _, row in top_revenue_zones.iterrows() if row['total_amount'] > 0]
        choropleth_stats = f"Top revenue zones: {', '.join(revenue_stats_list) if revenue_stats_list else 'None'}"

        choropleth_html = map_choropleth._repr_html_()

    except KeyError as e:
        print(f"KeyError during Choropleth creation: {e}")
        choropleth_html = f"<p>Error generating choropleth map: KeyError - {e}</p>"
        choropleth_stats = "Error calculating stats due to KeyError."
    except Exception as e:
        print(f"Error creating Folium Choropleth map: {e}")
        choropleth_html = f"<p>Error generating choropleth map: {e}</p>"
        choropleth_stats = "Error calculating stats."

    # --- Folium HeatMap (Uses d_filtered_initial THEN applies hotspot_time_period) ---
    try:
        # Use the initial filtered data for the heatmap
        d_filtered_heatmap = d_filtered_initial

        if d_filtered_heatmap.empty:
            heatmap_html = "<p>No ride data matches the selected time period for the heatmap.</p>"
            heatmap_stats = "No pickups for selected filters."
        else:
            heat_data = d_filtered_heatmap.dropna(subset=['pickup_latitude', 'pickup_longitude'])[['pickup_latitude', 'pickup_longitude']].values.tolist()
            map_heatmap = folium.Map(location=nyc_center, zoom_start=initial_zoom+1, tiles='cartodbpositron')
            
            HeatMap(
                heat_data,
                name='Pickup Density',
                radius=10,
                blur=15
            ).add_to(map_heatmap)

            # Find top pickup location
            if len(heat_data) > 0:
                # Create a temporary DataFrame to cluster and find the hotspot
                pickup_points = pd.DataFrame(heat_data, columns=['lat', 'lon'])
                
                # Rough clustering by rounding coordinates to identify hotspots
                pickup_points['lat_rounded'] = pickup_points['lat'].round(4)
                pickup_points['lon_rounded'] = pickup_points['lon'].round(4)
                
                # Get the top cluster
                top_cluster = pickup_points.groupby(['lat_rounded', 'lon_rounded']).size().reset_index(name='count')
                top_cluster = top_cluster.sort_values('count', ascending=False).iloc[0]
                
                top_lat, top_lon = top_cluster['lat_rounded'], top_cluster['lon_rounded']
                cluster_count = top_cluster['count']
                
                # Get zone information for this hotspot if available
                top_zone = "Unknown"
                top_borough = "Unknown"
                
                # Try to find the zone for this hotspot
                point = Point(top_lon, top_lat)
                for idx, zone in gdf_zones.iterrows():
                    if zone.geometry.contains(point):
                        top_zone = zone['zone']
                        top_borough = zone['borough']
                        break
                
                # Add marker and annotation for top hotspot
                folium.Marker(
                    location=[top_lat, top_lon],
                    tooltip=f"🔥 TOP HOTSPOT<br>Zone: {top_zone}<br>Borough: {top_borough}<br>Pickups: {cluster_count:,}",
                    icon=folium.Icon(color='red', icon='fire', prefix='fa'),
                ).add_to(map_heatmap)
                
                # Add arrow pointing to hotspot with annotation
                annotation_html = f"""
                    <div style="position: fixed; 
                        top: 70px; left: 50px; width: auto; height: auto;
                        background-color: rgba(255,255,255,0.9); border:2px solid red; z-index: 9999;
                        padding: 10px; font-size: 16px; border-radius: 5px; color: red; font-weight: bold;">
                        🔥 HOT SPOT: {top_zone}, {top_borough}<br>
                        <span style="font-size: 14px">{cluster_count:,} pickups in this area</span>
                    </div>
                """
                map_heatmap.get_root().html.add_child(folium.Element(annotation_html))

            folium.LayerControl().add_to(map_heatmap)
            heatmap_stats = f"Displaying density for {len(heat_data):,} pickups with highest concentration in {top_zone} ({top_borough})"
            heatmap_html = map_heatmap._repr_html_()

    except Exception as e:
        print(f"Error creating Folium Heatmap: {e}")
        heatmap_html = f"<p>Error generating heatmap: {e}</p>"
        heatmap_stats = "Error calculating stats."

    return choropleth_html, heatmap_html, choropleth_stats, heatmap_stats


@app.callback(
    [Output("total-trips-card", "children"),
     Output("avg-fare-card", "children"),
     Output("total-distance-card", "children"),
     Output("avg-distance-card", "children"),
     Output("payment-type-donut", "figure"),
     Output("passenger-count-donut", "figure"),
     Output("trips-by-weekday-line", "figure"),
     Output("revenue-by-weekday-bar", "figure"),
     Output("hour-day-heatmap", "figure")],
    [Input("url", "pathname")]
)
def update_analytics_page(pathname):
    if pathname != "/analytics" or df.empty:
        # Empty figures if not on analytics page or no data
        empty_fig = go.Figure().update_layout(
            xaxis={"visible": False},
            yaxis={"visible": False},
            annotations=[{
                "text": "No data available",
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "font": {"size": 20}
            }]
        )
        return "N/A", "N/A", "N/A", "N/A", empty_fig, empty_fig, empty_fig, empty_fig, empty_fig

    # Calculate card values
    total_trips = len(df)
    avg_fare = df["total_amount"].mean()
    total_distance = df["trip_distance"].sum()
    avg_distance = df["trip_distance"].mean()
    
    # Format card values
    total_trips_str = f"{total_trips:,}"
    avg_fare_str = f"${avg_fare:.2f}"
    total_distance_str = f"{total_distance:,.1f} mi"
    avg_distance_str = f"{avg_distance:.2f} mi"
    
    # Create the Payment Type Donut Chart
    # Handle missing or invalid payment_type values
    payment_counts = df['payment_type'].value_counts().reset_index()
    payment_counts.columns = ['payment_type', 'count']
    
    # Map payment codes to descriptions
    payment_counts['payment_name'] = payment_counts['payment_type']
    

    payment_counts = payment_counts[payment_counts['payment_name'].notnull()]
 
    
    # Color palette that's colorblind-friendly
    colors = px.colors.qualitative.Safe
    
    payment_fig = go.Figure(data=[go.Pie(
        labels=payment_counts['payment_name'],
        values=payment_counts['count'],
        hole=.4,
        marker=dict(colors=colors[:len(payment_counts)]),
        textinfo='label+percent',
        hoverinfo='label+value+percent',
        textfont=dict(size=14),
        pull=[0.05 if i == payment_counts['count'].idxmax() else 0 for i in range(len(payment_counts))]
    )])
    
    payment_fig.update_layout(
        title={
            'text': 'Distribution of Payment Methods',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        legend_title="Payment Method",
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        hoverlabel=dict(
            bgcolor="white",
            font_size=14,
            font_family="Arial"
        )
    )
    
    # Create the Passenger Count Donut Chart
    df_passengers = df.copy()
    df_passengers.loc[df_passengers['passenger_count'] > 6, 'passenger_count'] = 7  # Group as "7+"
    df_passengers.loc[df_passengers['passenger_count'] == 0, 'passenger_count'] = 8  # Group as "0 (error)"
    
    passenger_counts = df_passengers['passenger_count'].value_counts().reset_index()
    passenger_counts.columns = ['passenger_count', 'count']
    
    # Map values to readable labels
    passenger_counts['label'] = passenger_counts['passenger_count'].apply(
        lambda x: f"{int(x)} Passengers" if 1 <= x <= 6 else ("7+ Passengers" if x == 7 else "0 (Error)")
    )
    
    # Use sequential color scheme for passengers count (1-6+)
    passenger_colors = px.colors.sequential.Blues[2:9]  # Get enough distinct blues
    passenger_colors.append("#FFD700")  # Add gold for the error category
    
    passenger_fig = go.Figure(data=[go.Pie(
        labels=passenger_counts['label'],
        values=passenger_counts['count'],
        hole=.4,
        marker=dict(
            colors=[passenger_colors[min(int(row['passenger_count']), 8)-1] for _, row in passenger_counts.iterrows()],
            line=dict(color='#FFFFFF', width=2)
        ),
        textinfo='label+percent',
        hoverinfo='label+value+percent',
        textfont=dict(size=14),
        pull=[0.05 if i == passenger_counts['count'].idxmax() else 0 for i in range(len(passenger_counts))]
    )])
    
    passenger_fig.update_layout(
        title={
            'text': 'Distribution of Passenger Counts',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        hoverlabel=dict(
            bgcolor="white",
            font_size=14,
            font_family="Arial"
        )
    )
    
    # Create Trips by Weekday Line Chart with custom point interactions
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekday_counts = df['pickup_weekday'].value_counts().reset_index()
    weekday_counts.columns = ['day', 'trips']
    
    # Ensure days are in correct order
    weekday_counts['day_num'] = weekday_counts['day'].apply(lambda x: weekday_order.index(x) if x in weekday_order else -1)
    weekday_counts = weekday_counts.sort_values('day_num')
    
    weekday_line = go.Figure()
    
    # Add shadow line for depth
    weekday_line.add_trace(go.Scatter(
        x=weekday_counts['day'],
        y=weekday_counts['trips'],
        mode='lines',
        line=dict(color='rgba(0,100,200,0.1)', width=10),
        hoverinfo='none',
        showlegend=False
    ))
    
    # Add main line with points and interactive features
    weekday_line.add_trace(go.Scatter(
        x=weekday_counts['day'],
        y=weekday_counts['trips'],
        mode='lines+markers',
        line=dict(color='#3366cc', width=4),
        marker=dict(
            size=12,
            color='#ffffff',
            line=dict(color='#3366cc', width=2),
            symbol='circle'
        ),
        name='Trips',
        hovertemplate='<b>%{x}</b><br>Trips: %{y:,.0f}<extra></extra>'
    ))
    
    # Find the day with max trips for highlighting
    max_day_idx = weekday_counts['trips'].idxmax()
    max_day = weekday_counts.loc[max_day_idx]
    
    # Add a highlighted point for max day
    weekday_line.add_trace(go.Scatter(
        x=[max_day['day']],
        y=[max_day['trips']],
        mode='markers',
        marker=dict(
            size=16,
            color='#ff9900',
            line=dict(color='#ffffff', width=2),
            symbol='star'
        ),
        name='Peak Day',
        hovertemplate='<b>PEAK DAY</b><br>%{x}<br>Trips: %{y:,.0f}<extra></extra>'
    ))
    
    weekday_line.update_layout(
        xaxis=dict(
            title="Day of Week",
            categoryorder='array',
            categoryarray=weekday_order,
            tickfont=dict(size=14),
            showgrid=True,
            gridcolor='rgba(230,230,230,0.8)'
        ),
        yaxis=dict(
            title="Number of Trips",
            tickfont=dict(size=14),
            showgrid=True,
            gridcolor='rgba(230,230,230,0.8)'
        ),
        height=400,
        hovermode='x unified',
        margin=dict(l=40, r=40, t=40, b=40),
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=False,
        hoverlabel=dict(
            bgcolor="white",
            font_size=14,
            font_family="Arial",
            bordercolor="#333333"
        )
    )
    
    # Add annotation for peak day
    weekday_line.add_annotation(
        x=max_day['day'],
        y=max_day['trips'],
        text=f"Peak: {max_day['trips']:,.0f}",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="#ff9900",
        ax=0,
        ay=-40,
        bordercolor="#ff9900",
        borderwidth=2,
        borderpad=4,
        bgcolor="white",
        font=dict(color="#333333", size=12)
    )
    
    # Create Revenue by Day of Week Bar Chart
    weekday_revenue = df.groupby('pickup_weekday')['total_amount'].sum().reset_index()
    weekday_revenue.columns = ['day', 'revenue']
    
    # Calculate average fare per day for custom data
    weekday_avg_fare = df.groupby('pickup_weekday')['total_amount'].mean().reset_index()
    weekday_avg_fare.columns = ['day', 'avg_fare']
    
    # Merge data
    weekday_merged = weekday_revenue.merge(weekday_avg_fare, on='day')
    
    # Ensure days are in correct order
    weekday_merged['day_num'] = weekday_merged['day'].apply(lambda x: weekday_order.index(x) if x in weekday_order else -1)
    weekday_merged = weekday_merged.sort_values('day_num')
    
    # Find max revenue day for highlighting
    max_rev_day = weekday_merged.loc[weekday_merged['revenue'].idxmax()]['day']
    
    # Create bar colors - highlight weekend vs weekday
    bar_colors = ['#3366cc' if day not in ['Saturday', 'Sunday'] else '#ff9900' for day in weekday_merged['day']]
    
    revenue_bar = go.Figure()
    
    # Add gradient effect bars
    for i, (_, row) in enumerate(weekday_merged.iterrows()):
        revenue_bar.add_trace(go.Bar(
            x=[row['day']],
            y=[row['revenue']],
            marker=dict(
                color=bar_colors[i],
                opacity=0.9,
                line=dict(width=1, color='#ffffff')
            ),
            customdata=[[row['avg_fare']]],
            hovertemplate='<b>%{x}</b><br>Total Revenue: $%{y:,.2f}<br>Avg Fare: $%{customdata[0]:.2f}<extra></extra>',
            showlegend=False
        ))
    
    revenue_bar.update_layout(
        xaxis=dict(
            title="Day of Week",
            categoryorder='array',
            categoryarray=weekday_order,
            tickfont=dict(size=14),
            showgrid=False
        ),
        yaxis=dict(
            title="Total Revenue ($)",
            tickfont=dict(size=14),
            showgrid=True,
            gridcolor='rgba(230,230,230,0.8)',
            tickformat='$,.0f'
        ),
        barmode='group',
        height=400,
        margin=dict(l=40, r=40, t=40, b=40),
        plot_bgcolor='white',
        paper_bgcolor='white',
        hoverlabel=dict(
            bgcolor="white",
            font_size=14,
            font_family="Arial"
        )
    )
    
    # Add annotation for highest revenue day
    max_rev = weekday_merged.loc[weekday_merged['day'] == max_rev_day, 'revenue'].values[0]
    revenue_bar.add_annotation(
        x=max_rev_day,
        y=max_rev,
        text=f"Highest: ${max_rev:,.0f}",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="#ff9900",
        ax=0,
        ay=-40,
        bordercolor="#ff9900",
        borderwidth=2,
        borderpad=4,
        bgcolor="white",
        font=dict(color="#333333", size=12)
    )
    
    # Add helper line for weekend comparison
    weekday_avg = weekday_merged[weekday_merged['day'].isin(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'])]['revenue'].mean()
    weekend_avg = weekday_merged[weekday_merged['day'].isin(['Saturday', 'Sunday'])]['revenue'].mean()
    
    revenue_bar.add_shape(
        type="line",
        x0=-0.5,
        y0=weekday_avg,
        x1=4.5,
        y1=weekday_avg,
        line=dict(
            color="#3366cc",
            width=2,
            dash="dash",
        ),
        name="Weekday Avg"
    )
    
    revenue_bar.add_shape(
        type="line",
        x0=4.5,
        y0=weekend_avg,
        x1=6.5,
        y1=weekend_avg,
        line=dict(
            color="#ff9900",
            width=2,
            dash="dash",
        ),
        name="Weekend Avg"
    )
    
    # Add legend for weekday/weekend
    revenue_bar.add_trace(go.Scatter(
        x=[None], y=[None], mode='markers',
        marker=dict(size=10, color='#3366cc'),
        name='Weekday'
    ))
    revenue_bar.add_trace(go.Scatter(
        x=[None], y=[None], mode='markers',
        marker=dict(size=10, color='#ff9900'),
        name='Weekend'
    ))
    
    # Create Hour-Day Heatmap (Creative Visualization)
    if 'pickup_hour' in df.columns and 'pickup_weekday' in df.columns:
        # Create hour-day matrix
        hour_day_counts = df.groupby(['pickup_weekday', 'pickup_hour']).size().reset_index()
        hour_day_counts.columns = ['day', 'hour', 'count']
        
        # Pivot to create matrix
        hour_day_pivot = hour_day_counts.pivot(index='day', columns='hour', values='count').reset_index()
        
        # Ensure all hours are present
        for hour in range(24):
            if hour not in hour_day_pivot.columns:
                hour_day_pivot[hour] = 0
        
        # Sort weekdays correctly
        hour_day_pivot['day_num'] = hour_day_pivot['day'].apply(lambda x: weekday_order.index(x) if x in weekday_order else -1)
        hour_day_pivot = hour_day_pivot.sort_values('day_num')
        hour_day_pivot = hour_day_pivot.drop('day_num', axis=1)
        
        # Get matrix data
        days = hour_day_pivot['day'].tolist()
        hours = list(range(24))
        z_data = hour_day_pivot[sorted([col for col in hour_day_pivot.columns if col != 'day'])].values
        
        # Create annotation text
        text_matrix = [[f"{value:,.0f} trips" for value in row] for row in z_data]
        
        # Create heatmap with Viridis colorscale
        heatmap = go.Figure(data=go.Heatmap(
            z=z_data,
            x=[f"{h:02d}:00" for h in hours],
            y=days,
            colorscale='Viridis',  # color-blind safe continuous colorscale
            zmin=0,
            zmax=z_data.max() * 0.8,  # Cap for better color distribution
            text=text_matrix,
            hovertemplate='<b>%{y}, %{x}</b><br>Trips: %{z:,.0f}<extra></extra>',
            name='',
            showscale=True,
            colorbar=dict(
                title=dict(text="Trip Count", font=dict(size=14)),
                tickfont=dict(size=12)
            )
        ))
        
        # Find peak hours
        peak_idx = np.unravel_index(z_data.argmax(), z_data.shape)
        peak_day = days[peak_idx[0]]
        peak_hour = hours[peak_idx[1]]
        peak_count = z_data[peak_idx]
        
        heatmap.update_layout(
            title={
                'text': f'Trip Density by Day and Hour (Peak: {peak_day} at {peak_hour:02d}:00 with {int(peak_count):,} trips)',
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(size=16)
            },
            xaxis=dict(
                title="Hour of Day",
                tickmode='array',
                tickvals=[f"{h:02d}:00" for h in range(0, 24, 3)],  # Show every 3 hours
                ticktext=[f"{h:02d}:00" for h in range(0, 24, 3)],
                tickangle=-45,
                tickfont=dict(size=12),
                showgrid=False
            ),
            yaxis=dict(
                title="Day of Week",
                categoryorder='array',
                categoryarray=weekday_order,
                tickfont=dict(size=12),
                showgrid=False
            ),
            height=500,
            margin=dict(l=40, r=40, t=80, b=60),
            plot_bgcolor='white',
            paper_bgcolor='white',
            hoverlabel=dict(
                bgcolor="white",
                font_size=14,
                font_family="Arial"
            )
        )
        
        # Update peak annotation color to be more visible on Viridis scale
        heatmap.add_annotation(
            x=f"{peak_hour:02d}:00",
            y=peak_day,
            text="PEAK",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="#DDCC77",  # color-blind safe gold
            ax=0,
            ay=-30,
            bordercolor="#DDCC77",
            borderwidth=2,
            borderpad=4,
            bgcolor="white",
            opacity=0.8,
            font=dict(color="#333333", size=12, family="Arial")
        )
    else:
        heatmap = go.Figure().update_layout(
            annotations=[{"text": "Hour/day data not available", "showarrow": False}]
        )
    
    return total_trips_str, avg_fare_str, total_distance_str, avg_distance_str, payment_fig, passenger_fig, weekday_line, revenue_bar, heatmap


# --- Run the App ---
if __name__=='__main__':
    print('Starting Dash server...')
    logger.info('Application starting up...')
    print("Dash app running on http://0.0.0.0:8080/")
    serve(server, host='0.0.0.0', port=8080)

# --- Analytics Page Visualizations ---
@app.callback(
    [Output("total-trips-card", "children"),
     Output("avg-fare-card", "children"),
     Output("total-distance-card", "children"),
     Output("avg-distance-card", "children"),
     Output("payment-type-donut", "figure"),
     Output("passenger-count-donut", "figure"),
     Output("trips-by-weekday-line", "figure"),
     Output("revenue-by-weekday-bar", "figure"),
     Output("hour-day-heatmap", "figure")],
    [Input("url", "pathname")]
)
def update_analytics_page(pathname):
    # Color palette that's color-blind safe
    from plotly.colors import qualitative
    colors = qualitative.Safe
    
    # Use color-blind safe categorical for passenger counts
    passenger_colors = qualitative.Safe[:len(passenger_counts)]
    # fallback for error category
    if len(passenger_colors) < len(passenger_counts):
        passenger_colors.append('#DDCC77')  # color-blind safe gold

    if pathname != "/analytics" or df.empty:
        # Empty figures if not on analytics page or no data
        empty_fig = go.Figure().update_layout(
            xaxis={"visible": False},
            yaxis={"visible": False},
            annotations=[{
                "text": "No data available",
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "font": {"size": 20}
            }]
        )
        return "N/A", "N/A", "N/A", "N/A", empty_fig, empty_fig, empty_fig, empty_fig, empty_fig

    # Calculate card values
    total_trips = len(df)
    avg_fare = df["total_amount"].mean()
    total_distance = df["trip_distance"].sum()
    avg_distance = df["trip_distance"].mean()
    
    # Format card values
    total_trips_str = f"{total_trips:,}"
    avg_fare_str = f"${avg_fare:.2f}"
    total_distance_str = f"{total_distance:,.1f} mi"
    avg_distance_str = f"{avg_distance:.2f} mi"
    
    # Create the Payment Type Donut Chart
    # Handle missing or invalid payment_type values
    payment_counts = df['payment_type'].value_counts().reset_index()
    payment_counts.columns = ['payment_type', 'count']
    
    # Map payment codes to descriptions
    payment_counts['payment_name'] = payment_counts['payment_type']
    

    payment_counts = payment_counts[payment_counts['payment_name'].notnull()]
 
    
    # Color palette that's colorblind-friendly
    colors = px.colors.qualitative.Safe
    
    payment_fig = go.Figure(data=[go.Pie(
        labels=payment_counts['payment_name'],
        values=payment_counts['count'],
        hole=.4,
        marker=dict(colors=colors[:len(payment_counts)]),
        textinfo='label+percent',
        hoverinfo='label+value+percent',
        textfont=dict(size=14),
        pull=[0.05 if i == payment_counts['count'].idxmax() else 0 for i in range(len(payment_counts))]
    )])
    
    payment_fig.update_layout(
        title={
            'text': 'Distribution of Payment Methods',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        legend_title="Payment Method",
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        hoverlabel=dict(
            bgcolor="white",
            font_size=14,
            font_family="Arial"
        )
    )
    
    # Create the Passenger Count Donut Chart
    df_passengers = df.copy()
    df_passengers.loc[df_passengers['passenger_count'] > 6, 'passenger_count'] = 7  # Group as "7+"
    df_passengers.loc[df_passengers['passenger_count'] == 0, 'passenger_count'] = 8  # Group as "0 (error)"
    
    passenger_counts = df_passengers['passenger_count'].value_counts().reset_index()
    passenger_counts.columns = ['passenger_count', 'count']
    
    # Map values to readable labels
    passenger_counts['label'] = passenger_counts['passenger_count'].apply(
        lambda x: f"{int(x)} Passengers" if 1 <= x <= 6 else ("7+ Passengers" if x == 7 else "0 (Error)")
    )
    
    # Use sequential color scheme for passengers count (1-6+)
    passenger_colors = px.colors.sequential.Blues[2:9]  # Get enough distinct blues
    passenger_colors.append("#FFD700")  # Add gold for the error category
    
    passenger_fig = go.Figure(data=[go.Pie(
        labels=passenger_counts['label'],
        values=passenger_counts['count'],
        hole=.4,
        marker=dict(
            colors=[passenger_colors[min(int(row['passenger_count']), 8)-1] for _, row in passenger_counts.iterrows()],
            line=dict(color='#FFFFFF', width=2)
        ),
        textinfo='label+percent',
        hoverinfo='label+value+percent',
        textfont=dict(size=14),
        pull=[0.05 if i == passenger_counts['count'].idxmax() else 0 for i in range(len(passenger_counts))]
    )])
    
    passenger_fig.update_layout(
        title={
            'text': 'Distribution of Passenger Counts',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        hoverlabel=dict(
            bgcolor="white",
            font_size=14,
            font_family="Arial"
        )
    )
    
    # Create Trips by Weekday Line Chart with custom point interactions
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekday_counts = df['pickup_weekday'].value_counts().reset_index()
    weekday_counts.columns = ['day', 'trips']
    
    # Ensure days are in correct order
    weekday_counts['day_num'] = weekday_counts['day'].apply(lambda x: weekday_order.index(x) if x in weekday_order else -1)
    weekday_counts = weekday_counts.sort_values('day_num')
    
    weekday_line = go.Figure()
    
    # Add shadow line for depth
    weekday_line.add_trace(go.Scatter(
        x=weekday_counts['day'],
        y=weekday_counts['trips'],
        mode='lines',
        line=dict(color='rgba(0,100,200,0.1)', width=10),
        hoverinfo='none',
        showlegend=False
    ))
    
    # Add main line with points and interactive features
    weekday_line.add_trace(go.Scatter(
        x=weekday_counts['day'],
        y=weekday_counts['trips'],
        mode='lines+markers',
        line=dict(color='#3366cc', width=4),
        marker=dict(
            size=12,
            color='#ffffff',
            line=dict(color='#3366cc', width=2),
            symbol='circle'
        ),
        name='Trips',
        hovertemplate='<b>%{x}</b><br>Trips: %{y:,.0f}<extra></extra>'
    ))
    
    # Find the day with max trips for highlighting
    max_day_idx = weekday_counts['trips'].idxmax()
    max_day = weekday_counts.loc[max_day_idx]
    
    # Add a highlighted point for max day
    weekday_line.add_trace(go.Scatter(
        x=[max_day['day']],
        y=[max_day['trips']],
        mode='markers',
        marker=dict(
            size=16,
            color='#ff9900',
            line=dict(color='#ffffff', width=2),
            symbol='star'
        ),
        name='Peak Day',
        hovertemplate='<b>PEAK DAY</b><br>%{x}<br>Trips: %{y:,.0f}<extra></extra>'
    ))
    
    weekday_line.update_layout(
        xaxis=dict(
            title="Day of Week",
            categoryorder='array',
            categoryarray=weekday_order,
            tickfont=dict(size=14),
            showgrid=True,
            gridcolor='rgba(230,230,230,0.8)'
        ),
        yaxis=dict(
            title="Number of Trips",
            tickfont=dict(size=14),
            showgrid=True,
            gridcolor='rgba(230,230,230,0.8)'
        ),
        height=400,
        hovermode='x unified',
        margin=dict(l=40, r=40, t=40, b=40),
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=False,
        hoverlabel=dict(
            bgcolor="white",
            font_size=14,
            font_family="Arial",
            bordercolor="#333333"
        )
    )
    
    # Add annotation for peak day
    weekday_line.add_annotation(
        x=max_day['day'],
        y=max_day['trips'],
        text=f"Peak: {max_day['trips']:,.0f}",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="#ff9900",
        ax=0,
        ay=-40,
        bordercolor="#ff9900",
        borderwidth=2,
        borderpad=4,
        bgcolor="white",
        font=dict(color="#333333", size=12)
    )
    
    # Create Revenue by Day of Week Bar Chart
    weekday_revenue = df.groupby('pickup_weekday')['total_amount'].sum().reset_index()
    weekday_revenue.columns = ['day', 'revenue']
    
    # Calculate average fare per day for custom data
    weekday_avg_fare = df.groupby('pickup_weekday')['total_amount'].mean().reset_index()
    weekday_avg_fare.columns = ['day', 'avg_fare']
    
    # Merge data
    weekday_merged = weekday_revenue.merge(weekday_avg_fare, on='day')
    
    # Ensure days are in correct order
    weekday_merged['day_num'] = weekday_merged['day'].apply(lambda x: weekday_order.index(x) if x in weekday_order else -1)
    weekday_merged = weekday_merged.sort_values('day_num')
    
    # Find max revenue day for highlighting
    max_rev_day = weekday_merged.loc[weekday_merged['revenue'].idxmax()]['day']
    
    # Create bar colors - highlight weekend vs weekday
    bar_colors = ['#3366cc' if day not in ['Saturday', 'Sunday'] else '#ff9900' for day in weekday_merged['day']]
    
    revenue_bar = go.Figure()
    
    # Add gradient effect bars
    for i, (_, row) in enumerate(weekday_merged.iterrows()):
        revenue_bar.add_trace(go.Bar(
            x=[row['day']],
            y=[row['revenue']],
            marker=dict(
                color=bar_colors[i],
                opacity=0.9,
                line=dict(width=1, color='#ffffff')
            ),
            customdata=[[row['avg_fare']]],
            hovertemplate='<b>%{x}</b><br>Total Revenue: $%{y:,.2f}<br>Avg Fare: $%{customdata[0]:.2f}<extra></extra>',
            showlegend=False
        ))
    
    revenue_bar.update_layout(
        xaxis=dict(
            title="Day of Week",
            categoryorder='array',
            categoryarray=weekday_order,
            tickfont=dict(size=14),
            showgrid=False
        ),
        yaxis=dict(
            title="Total Revenue ($)",
            tickfont=dict(size=14),
            showgrid=True,
            gridcolor='rgba(230,230,230,0.8)',
            tickformat='$,.0f'
        ),
        barmode='group',
        height=400,
        margin=dict(l=40, r=40, t=40, b=40),
        plot_bgcolor='white',
        paper_bgcolor='white',
        hoverlabel=dict(
            bgcolor="white",
            font_size=14,
            font_family="Arial"
        )
    )
    
    # Add annotation for highest revenue day
    max_rev = weekday_merged.loc[weekday_merged['day'] == max_rev_day, 'revenue'].values[0]
    revenue_bar.add_annotation(
        x=max_rev_day,
        y=max_rev,
        text=f"Highest: ${max_rev:,.0f}",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="#ff9900",
        ax=0,
        ay=-40,
        bordercolor="#ff9900",
        borderwidth=2,
        borderpad=4,
        bgcolor="white",
        font=dict(color="#333333", size=12)
    )
    
    # Add helper line for weekend comparison
    weekday_avg = weekday_merged[weekday_merged['day'].isin(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'])]['revenue'].mean()
    weekend_avg = weekday_merged[weekday_merged['day'].isin(['Saturday', 'Sunday'])]['revenue'].mean()
    
    revenue_bar.add_shape(
        type="line",
        x0=-0.5,
        y0=weekday_avg,
        x1=4.5,
        y1=weekday_avg,
        line=dict(
            color="#3366cc",
            width=2,
            dash="dash",
        ),
        name="Weekday Avg"
    )
    
    revenue_bar.add_shape(
        type="line",
        x0=4.5,
        y0=weekend_avg,
        x1=6.5,
        y1=weekend_avg,
        line=dict(
            color="#ff9900",
            width=2,
            dash="dash",
        ),
        name="Weekend Avg"
    )
    
    # Add legend for weekday/weekend
    revenue_bar.add_trace(go.Scatter(
        x=[None], y=[None], mode='markers',
        marker=dict(size=10, color='#3366cc'),
        name='Weekday'
    ))
    revenue_bar.add_trace(go.Scatter(
        x=[None], y=[None], mode='markers',
        marker=dict(size=10, color='#ff9900'),
        name='Weekend'
    ))
    
    # Create Hour-Day Heatmap (Creative Visualization)
    if 'pickup_hour' in df.columns and 'pickup_weekday' in df.columns:
        # Create hour-day matrix
        hour_day_counts = df.groupby(['pickup_weekday', 'pickup_hour']).size().reset_index()
        hour_day_counts.columns = ['day', 'hour', 'count']
        
        # Pivot to create matrix
        hour_day_pivot = hour_day_counts.pivot(index='day', columns='hour', values='count').reset_index()
        
        # Ensure all hours are present
        for hour in range(24):
            if hour not in hour_day_pivot.columns:
                hour_day_pivot[hour] = 0
        
        # Sort weekdays correctly
        hour_day_pivot['day_num'] = hour_day_pivot['day'].apply(lambda x: weekday_order.index(x) if x in weekday_order else -1)
        hour_day_pivot = hour_day_pivot.sort_values('day_num')
        hour_day_pivot = hour_day_pivot.drop('day_num', axis=1)
        
        # Get matrix data
        days = hour_day_pivot['day'].tolist()
        hours = list(range(24))
        z_data = hour_day_pivot[sorted([col for col in hour_day_pivot.columns if col != 'day'])].values
        
        # Create annotation text
        text_matrix = [[f"{value:,.0f} trips" for value in row] for row in z_data]
        
        # Create heatmap with Viridis colorscale
        heatmap = go.Figure(data=go.Heatmap(
            z=z_data,
            x=[f"{h:02d}:00" for h in hours],
            y=days,
            colorscale='Viridis',  # color-blind safe continuous colorscale
            zmin=0,
            zmax=z_data.max() * 0.8,  # Cap for better color distribution
            text=text_matrix,
            hovertemplate='<b>%{y}, %{x}</b><br>Trips: %{z:,.0f}<extra></extra>',
            name='',
            showscale=True,
            colorbar=dict(
                title=dict(text="Trip Count", font=dict(size=14)),
                tickfont=dict(size=12)
            )
        ))
        
        # Find peak hours
        peak_idx = np.unravel_index(z_data.argmax(), z_data.shape)
        peak_day = days[peak_idx[0]]
        peak_hour = hours[peak_idx[1]]
        peak_count = z_data[peak_idx]
        
        heatmap.update_layout(
            title={
                'text': f'Trip Density by Day and Hour (Peak: {peak_day} at {peak_hour:02d}:00 with {int(peak_count):,} trips)',
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(size=16)
            },
            xaxis=dict(
                title="Hour of Day",
                tickmode='array',
                tickvals=[f"{h:02d}:00" for h in range(0, 24, 3)],  # Show every 3 hours
                ticktext=[f"{h:02d}:00" for h in range(0, 24, 3)],
                tickangle=-45,
                tickfont=dict(size=12),
                showgrid=False
            ),
            yaxis=dict(
                title="Day of Week",
                categoryorder='array',
                categoryarray=weekday_order,
                tickfont=dict(size=12),
                showgrid=False
            ),
            height=500,
            margin=dict(l=40, r=40, t=80, b=60),
            plot_bgcolor='white',
            paper_bgcolor='white',
            hoverlabel=dict(
                bgcolor="white",
                font_size=14,
                font_family="Arial"
            )
        )
        
        # Update peak annotation color to be more visible on Viridis scale
        heatmap.add_annotation(
            x=f"{peak_hour:02d}:00",
            y=peak_day,
            text="PEAK",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="#DDCC77",  # color-blind safe gold
            ax=0,
            ay=-30,
            bordercolor="#DDCC77",
            borderwidth=2,
            borderpad=4,
            bgcolor="white",
            opacity=0.8,
            font=dict(color="#333333", size=12, family="Arial")
        )
    else:
        heatmap = go.Figure().update_layout(
            annotations=[{"text": "Hour/day data not available", "showarrow": False}]
        )
    
    return total_trips_str, avg_fare_str, total_distance_str, avg_distance_str, payment_fig, passenger_fig, weekday_line, revenue_bar, heatmap


@server.route('/health')
def health_check():
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logger.info(f'Health check endpoint called at {current_time}')
    return jsonify({'status': 'healthy', 'timestamp': current_time})

# --- Run the App ---
if __name__=='__main__':
    print('Starting Dash server...')
    logger.info('Application starting up...')
    print("Dash app running on http://0.0.0.0:8080/")
    serve(server, host='0.0.0.0', port=8080)

