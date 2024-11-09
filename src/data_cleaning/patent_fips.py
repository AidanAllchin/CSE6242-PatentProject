#!/usr/bin/env python3
#-*- coding: utf-8 -*-
"""
Created: Fri Nov 09 2024
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

from colorama import Fore, Style
import numpy as np
from typing import Tuple, Dict, Optional
import requests
import time
import random
from tqdm import tqdm
import pandas as pd
from src.other.helpers import log

# Path to the final patents file
FINAL_PATENTS_PATH = os.path.join(project_root, "data", "patents.tsv")


def get_county_info_from_coords(lat: float, lon: float, cache: Dict = None) -> Optional[Tuple[str, str]]:
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
                log(f"Rate limited. Retrying in {delay:.2f} seconds... (Attempt {i + 1}/{max_retries})", level="WARNING")
                time.sleep(delay)
                continue

            # actually not sure if 429 is the only error code for rate limiting
            # so i'm just doing this as well
            elif response.status_code != 200:
                delay = min(max_delay, base_delay * (2 ** i) + random.uniform(0, 1))
                #print(response)
                #print(response.text)
                log(f"Error {response.status_code}. Retrying in {delay:.2f} seconds... (Attempt {i + 1}/{max_retries})", level="WARNING")
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
            log(f"Request timed out. Retrying in {delay:.2f} seconds... (Attempt {i + 1}/{max_retries})", level="WARNING")
            time.sleep(delay)
            
        except requests.exceptions.ConnectionError:
            delay = min(max_delay, base_delay * (2 ** i) + random.uniform(0, 1))
            log(f"Connection error. Retrying in {delay:.2f} seconds... (Attempt {i + 1}/{max_retries})", level="WARNING")
            time.sleep(delay)
            
        except Exception as e:
            log(f"Unexpected error getting county info for coords ({lat}, {lon}): {e}", level="WARNING")
            return None
        
    log(f"Failed to get county info for ({lat}, {lon}) after {max_retries} attempts", level="ERROR")
    return None

def add_county_info_to_patents(df: pd.DataFrame, batch_size: int = 25) -> pd.DataFrame:
    """
    Add county name and FIPS code columns to patents dataframe.
    Processes in batches to handle large datasets efficiently.
    
    Args:
        df: Patents dataframe with latitude/longitude columns
        batch_size: Number of records to process per batch
        
    Returns:
        DataFrame with added county_name and fips_code columns
    """
    # Add new columns only if they don't exist
    for col in ['inventor_county', 'inventor_fips', 'assignee_county', 'assignee_fips']:
        if col not in df.columns:
            df[col] = ""
            log(f"Added new column: {col}", color=Fore.CYAN)
        elif df[col].isnull().all():  # If column exists but is all NULL/NaN
            df[col] = ""
            log(f"Reset empty column: {col}", color=Fore.CYAN)
        else:
            log(f"Column {col} already exists with data - skipping creation", level="WARNING")
    
    # Cache for coordinate lookups
    coord_cache = {}
    
    # Process in batches
    log(f"Note that the expected time shown below will decrease as the cache fills up.")
    for i in tqdm(range(0, len(df), batch_size), desc="Adding county info", unit="batch", total=len(df)//batch_size):
        batch = df.iloc[i:i+batch_size]
        
        for idx, row in batch.iterrows():
            # Skip if the county info is already present
            if row['inventor_fips'] != '':
                continue

            # Process inventor location if coordinates exist
            elif row['inventor_latitude'] != 0.0 and row['inventor_longitude'] != 0.0 and str(row['inventor_latitude']) != "nan" and str(row['inventor_longitude']) != "nan":
                #print(f"coords: {row['inventor_latitude']}, {row['inventor_longitude']}")
                #print(f"types: {type(row['inventor_latitude'])}, {type(row['inventor_longitude'])}")

                county_info = get_county_info_from_coords(
                    row['inventor_latitude'], 
                    row['inventor_longitude'],
                    coord_cache
                )
                if county_info:
                    df.at[idx, 'inventor_county'] = county_info[0]
                    df.at[idx, 'inventor_fips']   = county_info[1]
            
            if row['assignee_fips'] != '':
                continue

            # Process assignee location if coordinates exist
            elif row['assignee_latitude'] != 0.0 and row['assignee_longitude'] != 0.0 and str(row['assignee_latitude']) != "nan" and str(row['assignee_longitude']) != "nan":
                #print(f"coords: {row['assignee_latitude']}, {row['assignee_longitude']}")
                #print(f"types: {type(row['assignee_latitude'])}, {type(row['assignee_longitude'])}")

                county_info = get_county_info_from_coords(
                    row['assignee_latitude'], 
                    row['assignee_longitude'],
                    coord_cache
                )
                if county_info:
                    df.at[idx, 'assignee_county'] = county_info[0]
                    df.at[idx, 'assignee_fips']   = county_info[1]
                    
        # Save after each batch
        df.to_csv(FINAL_PATENTS_PATH, sep='\t', index=False)
        
    return df

def add_fips_codes():
    """
    Main function to add FIPS codes to the patents dataset.
    """
    log("Reading patents file...", color=Fore.CYAN)
    df = pd.read_csv(FINAL_PATENTS_PATH, sep='\t')
    
    # Add county information
    log("\nAdding county information to patents...", color=Fore.CYAN)
    df = add_county_info_to_patents(df)
    
    
    # Save final results
    log("\nSaving updated patents file...", color=Fore.CYAN)
    df.to_csv(FINAL_PATENTS_PATH, sep='\t', index=False)
    
    # Print summary statistics
    total_patents = len(df)
    inventor_success = len(df[df['inventor_fips'] != ''])
    assignee_success = len(df[df['assignee_fips'] != ''])
    
    log(f"\nResults:", color=Fore.GREEN)
    log(f"Total patents processed: {total_patents}", color=Fore.GREEN)
    log(f"Patents with inventor FIPS: {inventor_success} ({inventor_success/total_patents*100:.1f}%)", color=Fore.GREEN)
    log(f"Patents with assignee FIPS: {assignee_success} ({assignee_success/total_patents*100:.1f}%)", color=Fore.GREEN)

if __name__ == "__main__":
    add_fips_codes()