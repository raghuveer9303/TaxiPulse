import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import numpy as np
import warnings
import os

# --- Configuration ---
DATA_URL = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vTZPBWbFSSjfPsKpEK3p1RNIqWbli3lpnZ_OUjQQf-6ygIjCXlZX7TKHAsC3rlqIRecbEE1C0-DLNk6/pub?gid=1860100186&single=true&output=csv'
TAXI_ZONE_FOLDER = './taxi_zones' # Folder containing the shapefile
TAXI_ZONE_SHAPEFILE_PATH = os.path.join(TAXI_ZONE_FOLDER, 'taxi_zones.shp')
OUTPUT_FILE = './preprocessed_taxi_data.parquet' # Recommended format
# OUTPUT_FILE = './preprocessed_taxi_data.csv' # Alternative format

# --- Main Script ---
if __name__ == "__main__":
    warnings.simplefilter(action='ignore', category=FutureWarning) # Suppress common FutureWarning
    warnings.simplefilter(action='ignore', category=UserWarning) # Suppress Shapely 2.0 warnings if using older geopandas

    print(f"Step 1: Loading taxi data from URL...")
    print(f"URL: {DATA_URL}")
    try:
        df = pd.read_csv(DATA_URL)
        print(f" -> Loaded {len(df):,} rows.")
    except Exception as e:
        print(f"FATAL ERROR loading data from URL: {e}")
        print("Please check the URL and your internet connection.")
        exit()

    print("Step 2: Basic data cleaning and type conversion...")
    # Convert datetime
    try:
        df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'], format='mixed')
    except KeyError:
         print("ERROR: Column 'tpep_pickup_datetime' not found in the data.")
         # Decide how to handle: exit(), continue without time features?
         exit()
    except Exception as e:
         print(f"Error converting 'tpep_pickup_datetime': {e}. Check column format.")
         exit()

    # Define coordinate columns
    coord_cols = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']

    # Check if coordinate columns exist
    missing_cols = [col for col in coord_cols if col not in df.columns]
    if missing_cols:
        print(f"ERROR: Missing required coordinate columns: {missing_cols}")
        exit()

    # Convert coordinate columns to numeric, coercing errors
    for col in coord_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows with invalid or missing coordinates needed for joins
    initial_rows = len(df)
    df.dropna(subset=coord_cols, inplace=True)

    # Filter out typical invalid placeholder coordinates (adjust range if needed)
    df = df[df['pickup_latitude'].between(-90, 90) & df['pickup_longitude'].between(-180, 180)]
    df = df[df['dropoff_latitude'].between(-90, 90) & df['dropoff_longitude'].between(-180, 180)]
    # Optional: Filter out coordinates exactly at (0, 0) if they are known invalid placeholders
    df = df[~((df['pickup_latitude'] == 0) & (df['pickup_longitude'] == 0))]
    df = df[~((df['dropoff_latitude'] == 0) & (df['dropoff_longitude'] == 0))]

    rows_dropped = initial_rows - len(df)
    if rows_dropped > 0:
        print(f" -> Dropped {rows_dropped:,} rows with missing/invalid coordinates.")
    if len(df) == 0:
        print("FATAL ERROR: No valid coordinate data remaining after cleaning.")
        exit()


    print(f"Step 3: Loading taxi zone shapefile...")
    print(f"Path: {TAXI_ZONE_SHAPEFILE_PATH}")
    if not os.path.exists(TAXI_ZONE_SHAPEFILE_PATH):
         print(f"FATAL ERROR: Shapefile not found at {TAXI_ZONE_SHAPEFILE_PATH}")
         print("Please ensure the 'taxi_zones' folder exists and contains the shapefile.")
         exit()
    try:
        gdf_zones = gpd.read_file(TAXI_ZONE_SHAPEFILE_PATH)
        # Keep only necessary columns
        gdf_zones = gdf_zones[['borough', 'zone', 'geometry']] # Add 'LocationID' if needed elsewhere

        # Ensure the geometry column is valid, attempt to fix if not
        if not gdf_zones.geometry.is_valid.all():
             print(" -> Warning: Invalid geometries found in shapefile. Attempting to fix...")
             gdf_zones['geometry'] = gdf_zones.geometry.buffer(0)
             if not gdf_zones.geometry.is_valid.all():
                 print(" -> ERROR: Could not fix invalid geometries in shapefile. Proceeding, but results may be inaccurate.")
                 # Optionally exit() here if valid zones are critical

        print(f" -> Loaded {len(gdf_zones):,} zones. Zone CRS: {gdf_zones.crs}")
    except Exception as e:
        print(f"FATAL ERROR loading shapefile: {e}")
        exit()


    print("Step 4: Performing spatial joins (this may take time)...")
    original_df_cols = df.columns.tolist() # Keep track of original columns

    # Process pickup and dropoff locations
    for pt_type in ['pickup', 'dropoff']:
        print(f" -> Processing {pt_type} locations...")
        lon_col = f'{pt_type}_longitude'
        lat_col = f'{pt_type}_latitude'
        zone_col = f'{pt_type}_zone'
        borough_col = f'{pt_type}_borough'

        # Create GeoDataFrame with points using the original DataFrame's index
        print(f"    -> Creating points...")
        try:
            geometry = [Point(xy) for xy in zip(df[lon_col], df[lat_col])]
            gdf_points = gpd.GeoDataFrame(df.index.to_frame(name='original_index'), geometry=geometry, crs='EPSG:4326') # Set initial CRS as WGS84
        except Exception as e:
            print(f"    -> ERROR creating points for {pt_type}: {e}")
            continue # Skip to next point type or exit

        # Transform CRS to match zones
        print(f"    -> Transforming CRS...")
        try:
            gdf_points = gdf_points.to_crs(gdf_zones.crs)
        except Exception as e:
            print(f"    -> ERROR transforming CRS for {pt_type}: {e}")
            continue

        # Perform spatial join
        print(f"    -> Performing spatial join...")
        try:
            # Use left join to keep all taxi ride points
            joined_gdf = gpd.sjoin(gdf_points, gdf_zones[['zone', 'borough', 'geometry']], how='left', predicate='within') # Use 'predicate' instead of 'op' in newer versions
        except Exception as e:
             print(f"    -> ERROR during spatial join for {pt_type}: {e}")
             continue

        # --- Merge results back to the main DataFrame ---
        print(f"    -> Merging results back...")
        # Keep only the first match per original point (index) in case of overlaps (unlikely with 'within')
        joined_gdf = joined_gdf[~joined_gdf['original_index'].duplicated(keep='first')]

        # Select and rename columns for merging
        zone_info = joined_gdf[['original_index', 'zone', 'borough']].set_index('original_index')
        zone_info = zone_info.rename(columns={'zone': zone_col, 'borough': borough_col})

        # Merge back into the original dataframe, adding the new columns
        df = df.merge(zone_info, left_index=True, right_index=True, how='left')

        # Fill missing zones/boroughs (points that didn't fall 'within' any zone)
        df[zone_col].fillna('Unknown', inplace=True)
        df[borough_col].fillna('Unknown', inplace=True)

        print(f" -> Finished processing {pt_type} locations. Added columns: {zone_col}, {borough_col}")


    print("Step 5: Calculating additional features...")
    df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour
    df['pickup_day'] = df['tpep_pickup_datetime'].dt.day_name()

    # Calculate tip percentage safely
    print(" -> Calculating tip percentage...")
    if 'total_amount' in df.columns and 'tip_amount' in df.columns:
        df['total_amount'] = pd.to_numeric(df['total_amount'], errors='coerce')
        df['tip_amount'] = pd.to_numeric(df['tip_amount'], errors='coerce')
        # Base amount for tip calculation (total excluding tip)
        base_amount = df['total_amount'] - df['tip_amount']
        # Calculate tip_pct only where base_amount is positive and non-zero
        df['tip_pct'] = np.where(base_amount > 1e-6, (df['tip_amount'].fillna(0) / base_amount) * 100, 0)
        # Fill NaN results in tip_pct (e.g., if total_amount or tip_amount was NaN) with 0
        df['tip_pct'].fillna(0, inplace=True)
        # Optional: Cap unreasonable tip percentages if needed
        # df['tip_pct'] = df['tip_pct'].clip(upper=100) # Example: Cap tips at 100%
    else:
        print(" -> Warning: 'total_amount' or 'tip_amount' not found. Skipping 'tip_pct' calculation.")
        df['tip_pct'] = 0.0 # Add the column with default value


    print(f"Step 6: Saving pre-processed data to {OUTPUT_FILE}...")
    final_cols = original_df_cols + ['pickup_zone', 'pickup_borough', 'dropoff_zone', 'dropoff_borough', 'pickup_hour', 'pickup_day', 'tip_pct']
    # Ensure only desired columns are saved, remove duplicates if any occurred
    df_final = df[final_cols].drop_duplicates()

    try:
        if OUTPUT_FILE.endswith('.parquet'):
             # Ensure you have pyarrow installed: pip install pyarrow
             df_final.to_parquet(OUTPUT_FILE, index=False, engine='pyarrow')
             print(f" -> Successfully saved pre-processed data as Parquet.")
        elif OUTPUT_FILE.endswith('.csv'):
             df_final.to_csv(OUTPUT_FILE, index=False)
             print(f" -> Successfully saved pre-processed data as CSV.")
        else:
             print(f" -> Warning: Unknown output file extension. Saving as Parquet to {OUTPUT_FILE}.parquet")
             df_final.to_parquet(f"{OUTPUT_FILE}.parquet", index=False, engine='pyarrow')

        print(f" -> Final dataset has {len(df_final):,} rows.")
        print("\nPre-processing complete!")
        print(f"You can now modify your Dash app to load the pre-processed data from: {os.path.abspath(OUTPUT_FILE)}")

    except ImportError:
         print("\nError: 'pyarrow' library not found. Needed for saving to Parquet.")
         print("Please install it ('pip install pyarrow') and run the script again.")
         print("Alternatively, change OUTPUT_FILE to end with '.csv'.")
    except Exception as e:
        print(f"\nError saving data: {e}")