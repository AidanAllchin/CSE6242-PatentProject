#!/usr/bin/env python3
#-*- coding: utf-8 -*-
"""
Created: Fri Nov 08 2024
Author: Aidan Allchin

Each US county has it's own unique FIPS code. After adding a latitude and 
longitude for each patent we can use those to determine which county the 
patent originated from, and can add the FIPS code (uid for US county) to each
patent, which we will later group on.

Uses the newly added latitudes and longitudes to assign FIPS codes to each 
patent based on the location of the inventor. 

Note that this replaces the `patents.tsv` file with the updated version.
"""
import os, sys
from pathlib import Path

# ALL FILES IN THE PROJECT SHOULD BE IMPORTED RELATIVE TO THE PROJECT ROOT
# the # of `.parent`s should be adjusted based on the file's location
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from typing import Tuple, Dict, Optional, List
import multiprocessing as mp
from multiprocessing import Pool
import requests
import time
from rtree import index
import json
from shapely.geometry import shape, Point
from rich.progress import Progress
import random
from tqdm import tqdm
import pandas as pd
from src.other.logging import PatentLogger

# Initialize logger
logger = PatentLogger.get_logger(__name__)

# Path to the final patents file
FINAL_PATENTS_PATH = os.path.join(project_root, "data", "patents.tsv")
BOUNDARY_GEOJSON_P = os.path.join(project_root, "data", "geolocation", "county_boundaries.geojson")


###############################################################################
#                                                                             #
#                           CENSUS GEOCODING API                              #
#                                                                             #
###############################################################################

# This was going to take 11 days to run, so I'm replacing it with a local 
# version using the R-tree spatial index and the county boundary GeoJSON file.

def get_county_info_from_coords_api(lat: float, lon: float, cache: Dict = None) -> Optional[Tuple[str, str]]:
    """
    Get county name and FIPS code for given coordinates using Census Geocoding 
    API. Uses a cache to avoid repeated API calls for the same coordinates.
    Utilizes the same retry and rate limiting logic as the 
    `get_coordinates_for_city` function in `helpers.py`.
    
    Args:
        lat: Latitude
        lon: Longitude
        cache: Optional dictionary to cache results
        
    Returns:
        Tuple of (county_name, fips_code) or None if not found
    """
    if cache is None:
        cache = {}
        
    # Round coordinates to 4 decimal places for cache key
    coord_key = (round(lat, 4), round(lon, 4))

    if coord_key in cache:
        return cache[coord_key]
    
    # Census Geocoding API endpoint
    url = f"https://geocoding.geo.census.gov/geocoder/geographies/coordinates"
    params = {
        "x": lon,
        "y": lat,
        "benchmark": "Public_AR_Census2020",
        "vintage": "Census2020_Census2020",
        "layers": "Counties",
        "format": "json"
    }

    # params
    max_retries = 10   # max number of retries
    base_delay  = 1    # seconds
    max_delay   = 60   # seconds
    
    for i in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=10)

            # Retry if rate limited
            if response.status_code == 429:
                delay = min(max_delay, base_delay * (2 ** i) + random.uniform(0, 1))
                logger.warning(f"Rate limited. Retrying in {delay:.2f} seconds... (Attempt {i + 1}/{max_retries})")
                time.sleep(delay)
                continue

            # actually not sure if 429 is the only error code for rate limiting
            # so i'm just doing this as well
            elif response.status_code != 200:
                delay = min(max_delay, base_delay * (2 ** i) + random.uniform(0, 1))
                logger.warning(f"Error {response.status_code}. Retrying in {delay:.2f} seconds... (Attempt {i + 1}/{max_retries})")
                time.sleep(delay)
                continue

            data = response.json()
            result = data.get("result", {}).get("geographies", {}).get("Counties", [])
            
            if result:
                county = result[0]
                county_name = county.get("BASENAME")
                state_fips = county.get("STATE")
                county_fips = county.get("COUNTY")
                full_fips = f"{state_fips}{county_fips}"
                
                # Cache the result
                cache[coord_key] = (county_name, full_fips)
                return county_name, full_fips
                    
            return None
            
        except requests.exceptions.ReadTimeout:
            delay = min(max_delay, base_delay * (2 ** i) + random.uniform(0, 1))
            logger.warning(f"Request timed out. Retrying in {delay:.2f} seconds... (Attempt {i + 1}/{max_retries})")
            time.sleep(delay)
            
        except requests.exceptions.ConnectionError:
            delay = min(max_delay, base_delay * (2 ** i) + random.uniform(0, 1))
            logger.warning(f"Connection error. Retrying in {delay:.2f} seconds... (Attempt {i + 1}/{max_retries})")
            time.sleep(delay)
            
        except Exception as e:
            logger.exception(f"Unexpected error getting county info for coords ({lat}, {lon}): {e}")
            return None
        
    logger.error(f"Failed to get county info for ({lat}, {lon}) after {max_retries} attempts")
    return None


###############################################################################
#                                                                             #
#                           LOCAL SPATIAL INDEXING                            #
#                                                                             #
###############################################################################


def initialize_spatial_index(geojson_path: str) -> Tuple[index.Index, list]:
    """
    Initialize an R-tree spatial index from county boundary GeoJSON data.
    
    Args:
        geojson_path: Path to the county boundaries GeoJSON file
        
    Returns:
        Tuple of (rtree_index, county_polygons)
    """
    # Load GeoJSON data
    with open(geojson_path, "r") as f:
        geojson_data = json.load(f)

    # Initialize R-tree index and polygon storage
    idx = index.Index()
    county_polygons = []
    
    # Build out the spatial index
    with Progress() as progress:
        task = progress.add_task(
            "[cyan]Building county boundaries spatial index...", 
            total=len(geojson_data["features"])
        )
        
        for pos, feature in enumerate(geojson_data["features"]):
            # Extract geometry and properties
            geom = shape(feature["geometry"])
            props = feature["properties"]
            
            # Store polygon and properties
            county_polygons.append((props, geom))
            
            # Insert polygon bounds into R-tree
            idx.insert(pos, geom.bounds)
            
            progress.update(task, advance=1)
    
    return idx, county_polygons


###############################################################################
#                                                                             #
#                          MULTIPROCESSING BATCHES                            #
#                                                                             #
###############################################################################


def process_batch(args: Tuple) -> List[Dict]:
    """
    Process a batch of patent records in parallel. Trying this instead of the 
    cached single-threaded version to see if it's faster.
    
    Args:
        args: Tuple containing (batch_df, rtree_idx, county_polygons)
        
    Returns:
        List of dictionaries containing county info for each patent
    """
    batch_df, polygons_data = args
    results = []
    
    # Recreate R-tree index for each process
    idx = index.Index()
    for pos, (props, geom) in enumerate(polygons_data):
        idx.insert(pos, geom.bounds)
    
    # Process each patent in the batch
    for _, row in batch_df.iterrows():
        result = {
            'index': row.name,
            'inventor_county': '',
            'inventor_fips': '',
            'assignee_county': '',
            'assignee_fips': ''
        }

        # Process inventor location
        if (row['inventor_latitude'] != 0.0 and row['inventor_longitude'] != 0.0 and 
            str(row['inventor_latitude']) != "nan" and str(row['inventor_longitude']) != "nan"):
            point = Point(row['inventor_longitude'], row['inventor_latitude'])
            
            # Query R-tree
            for idx_candidate in idx.intersection((point.x, point.y, point.x, point.y)):
                props, polygon = polygons_data[idx_candidate]
                if polygon.contains(point):
                    # Extract county name and FIPS from properties
                    county_name = props.get("name", "")
                    state_fips  = props.get("statefp", "")
                    county_fips = props.get("countyfp", "")
                    
                    # Ensure FIPS codes are padded correctly
                    state_fips  = state_fips.zfill(2)
                    county_fips = county_fips.zfill(3)
                    full_fips   = f"{state_fips}{county_fips}"

                    result['inventor_county'] = county_name
                    result['inventor_fips'] = full_fips
                    break

        # Process assignee location
        if (row['assignee_latitude'] != 0.0 and row['assignee_longitude'] != 0.0 and 
            str(row['assignee_latitude']) != "nan" and str(row['assignee_longitude']) != "nan"):
            point = Point(row['assignee_longitude'], row['assignee_latitude'])
            
            # Query R-tree
            for idx_candidate in idx.intersection((point.x, point.y, point.x, point.y)):
                props, polygon = polygons_data[idx_candidate]
                if polygon.contains(point):
                    # Extract county name and FIPS from properties
                    county_name = props.get("name", "")
                    state_fips  = props.get("statefp", "")
                    county_fips = props.get("countyfp", "")
                    
                    # Ensure FIPS codes are padded correctly
                    state_fips  = state_fips.zfill(2)
                    county_fips = county_fips.zfill(3)
                    full_fips   = f"{state_fips}{county_fips}"

                    result['assignee_county'] = county_name
                    result['assignee_fips'] = full_fips
                    break
                    
        results.append(result)
    
    return results

def add_county_info_parallel(df: pd.DataFrame, rtree_idx: index.Index, county_polygons: list, batch_size: int = 1000, num_processes:int = None) -> pd.DataFrame:
    """
    Add county name and FIPS code columns to patents dataframe using parallel processing.
    
    Args:
        df: Patents dataframe with latitude/longitude columns
        rtree_idx: R-tree spatial index
        county_polygons: List of (properties, polygon) tuples
        batch_size: Number of records to process per batch
        num_processes: Number of processes to use (defaults to CPU count - 1)
        
    Returns:
        DataFrame with added county_name and fips_code columns
    """
    if num_processes is None:
        num_processes = max(1, mp.cpu_count() - 1)

    # Add new columns only if they don't exist
    for col in ['inventor_county', 'inventor_fips', 'assignee_county', 'assignee_fips']:
        if col not in df.columns:
            df[col] = ""
            logger.info(f"Added new column: {col}")
        elif df[col].isnull().all():  # If column exists but is all NULL/NaN
            df[col] = ""
            logger.info(f"Reset empty column: {col}")
        else:
            logger.warning(f"Column {col} already exists with data - skipping creation")

    # Separate into batches
    num_batches = len(df) // batch_size + (1 if len(df) % batch_size else 0)
    logger.info(f"Processing patents in {num_batches} batches of {batch_size} records each...")
    batches = []
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size].copy()
        # Skip batch if all rows already have FIPS codes
        if not batch[batch['inventor_fips'] == ''].empty:
            batches.append((batch, county_polygons))

    # Process batches in parallel
    with Pool(processes=num_processes) as pool:
        results = []
        with tqdm(total=len(batches), desc=f"Processing patents using {num_processes} cores") as pbar:
            for batch_results in pool.imap_unordered(process_batch, batches):
                results.extend(batch_results)
                pbar.update()

    # Update dataframe with results
    for result in results:
        idx = result.pop('index')
        for col, value in result.items():
            df.at[idx, col] = value
    
    return df


###############################################################################
#                                                                             #
#                                    MAIN                                     #
#                                                                             #
###############################################################################


def add_fips_codes():
    """
    Main function to add FIPS codes to the patents dataset.
    """
    if not os.path.exists(BOUNDARY_GEOJSON_P):
        logger.error(f"County boundaries GeoJSON not found at {BOUNDARY_GEOJSON_P}")
        return
        
    # Initialize spatial index
    logger.info("Initializing spatial index...")

    rtree_idx, county_polygons = initialize_spatial_index(BOUNDARY_GEOJSON_P)

    logger.info("Reading patents file...")
    df = pd.read_csv(FINAL_PATENTS_PATH, sep='\t')
    
    # Add county information
    logger.info("Adding county information to patents...")
    #df = add_county_info_single(df, rtree_idx, county_polygons)
    df = add_county_info_parallel(df, rtree_idx, county_polygons, batch_size=500)
    
    # Save final results
    logger.info("\nSaving updated patents file...")
    df.to_csv(FINAL_PATENTS_PATH, sep='\t', index=False)
    
    total_patents    = len(df)
    inventor_success = len(df[df['inventor_fips'] != ''])
    assignee_success = len(df[df['assignee_fips'] != ''])
    
    logger.info(f"\nResults:")
    logger.info(f"Total patents processed: {total_patents}")
    logger.info(f"Patents with inventor FIPS: {inventor_success} ({inventor_success/total_patents*100:.1f}%)")
    logger.info(f"Patents with assignee FIPS: {assignee_success} ({assignee_success/total_patents*100:.1f}%)")

if __name__ == "__main__":
    add_fips_codes()