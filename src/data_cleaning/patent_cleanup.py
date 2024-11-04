#!/usr/bin/env python3
#-*- coding: utf-8 -*-
"""
Created: Tue Oct 17 2024
Author: Aidan Allchin

Cleans the patent data by replacing/updating fields with patent-specific
terminology to be more readable and consistent.
"""
import os, sys
from pathlib import Path

# ALL FILES IN THE PROJECT SHOULD BE IMPORTED RELATIVE TO THE PROJECT ROOT
# the # of `.parent`s should be adjusted based on the file's location
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from colorama import Fore, Style
from typing import List, Tuple, Set
import pandas as pd
import json
from tqdm import tqdm
from src.objects.patent import Patent
from src.other.helpers import log, get_coordinates_for_city, local_filename

# Conceptual steps:
# 1. Convert Classifications (IPC and CPC) to their meanings
# 2. Convert Dates to a consistent format
# 3. Convert Patent Status to a more readable format
# 4. Add latitude and longitude to all locations


###############################################################################
#                                   SETUP                                     #
###############################################################################

with open(os.path.join(project_root, "config", "config.json"), "r") as f:
    config = json.load(f)

LOCATIONS_TSV_PATH = os.path.join(project_root, config["settings"]["city_coordinates_path"])


###############################################################################
#                              Coordinate Stuff                               #
###############################################################################


def add_coordinates(patents: pd.DataFrame) -> pd.DataFrame:
    """
    For every patent in the list, add the latitude and longitude of the assignee
    and inventor locations.
    
    Args:
        patents (pd.DataFrame): The pandas DataFrame with the patents straight after the merging.
        
    Returns:
        pd.DataFrame: The same patents DataFrame with latitude and longitude 
            information for the primary inventor and primary assignee.
    """

    # Ensure the city coordinates file exists
    if not os.path.exists(LOCATIONS_TSV_PATH):
        log(f"City coordinates file not found at {local_filename(LOCATIONS_TSV_PATH)}.", level="ERROR")
        return patents

    # Read the city coordinates file
    locations_df = pd.read_csv(LOCATIONS_TSV_PATH, sep='\t')
    locations = set(locations_df.apply(lambda row: (row['city'], row['country'], row['state']), axis=1))

    # Add coordinates to each patent
    print()
    for i, patent in tqdm(patents.iterrows(), desc="Adding coordinates from cache"):
        if (patent["inventor_city"].lower(), patent["inventor_country"].lower(), patent["inventor_state"].lower() if patent["inventor_state"] else "") in locations:
            location = locations_df[(locations_df["inventor_city"] == patent["inventor_city"].lower()) & (locations_df["inventor_country"] == patent["inventor_country"].lower()) & (locations_df["inventor_state"] == patent["inventor_state"].lower() if patent["inventor_state"] else "")]
            patents[i, "inventor_latitude"]  = location["latitude"].values[0]
            patents[i, "inventor_longitude"] = location["longitude"].values[0]
        else:
            patents[i, "inventor_latitude"]  = 0.0
            patents[i, "inventor_longitude"] = 0.0

        if (patent["assignee_city"].lower(), patent["assignee_country"].lower(), patent["assignee_state"].lower() if patent["assignee_state"] else "") in locations:
            location = locations_df[(locations_df["assignee_city"] == patent["assignee_city"].lower()) & (locations_df["assignee_country"] == patent["assignee_country"].lower()) & (locations_df["assignee_state"] == patent["assignee_state"].lower() if patent["assignee_state"] else "")]
            patents[i, "assignee_latitude"]  = location["latitude"].values[0]
            patents[i, "assignee_longitude"] = location["longitude"].values[0]
        else:
            patents[i, "assignee_latitude"]  = 0.0
            patents[i, "assignee_longitude"] = 0.0

    num_without_coordinates = len(patents[patents['assignee_latitude' == 0.0] | patents['assignee_longitude'] == 0
                                        | patents['inventor_latitude' == 0.0] | patents['inventor_longitude'] == 0])
    
    t = "WARNING" if num_without_coordinates > 0 else "INFO"
    log(f"After adding coordinates, there are {num_without_coordinates} rows without parsed locations.", level=t)

    return patents

# Function for creating the cache of city coordinates (stored in a CSV file)
# This should not be run by team members, but the result will be shared with the team
# after completion. Not directly part of the pipeline.
def create_city_coordinates_cache(patents: pd.DataFrame) -> None:
    """
    Create a cache of city coordinates by iterating over all patents and
    collecting the latitude and longitude of each city.

    Args:
        patents (pd.DataFrame): The pandas DataFrame containing all the patents
    """
    # Ensure coords dir exists
    os.makedirs(os.path.dirname(LOCATIONS_TSV_PATH), exist_ok=True)

    # Initial parsing (faster than iterating)
    patents['inventor_city']    = patents['inventor_city'].lower()
    patents['inventor_state']   = patents['inventor_state'].lower()
    patents['inventor_country'] = patents['inventor_country'].lower()

    patents['assignee_city']    = patents['assignee_city'].lower()
    patents['assignee_state']   = patents['assignee_state'].lower()
    patents['assignee_country'] = patents['assignee_country'].lower()

    # Create a set of all unique cities
    cities = set()
    no_city_counter = 0
    print()
    for i, patent in tqdm(patents.iterrows(), desc="Collecting unique locations"):
        if patent['inventor_city'] is not None:
            cities.add((patent["inventor_city"], patent["inventor_country"], patent["inventor_state"] if patent["inventor_state"] else ""))
        else:
            no_city_counter += 1
        if patent["assignee_city"] is not None:
            cities.add((patent["assignee_city"], patent["assignee_country"], patent["assignee_state"] if patent["assignee_state"] else ""))
        else:
            no_city_counter += 1

    log(f"Found {len(cities)} unique cities.", color=Fore.CYAN)
    log(f"Unfortunately, {no_city_counter} entries exist with no city, meaning no coordinates will be added.", level="WARNING")
    
    # Read existing entries
    existing_cities = set()
    if os.path.exists(LOCATIONS_TSV_PATH):
        df = pd.read_csv(LOCATIONS_TSV_PATH, sep='\t', header=0, keep_default_na=False)

        # Gotta make this messy because dataframes won't format tuples correctly
        existing_cities = set(df.apply(lambda row: (row['city'], row['country'], row['state']), axis=1))

    log(f"Found {len(existing_cities)} cities already cached.", color=Fore.CYAN)
    log(f"\nCreating cache of city coordinates...", color=Fore.LIGHTBLUE_EX, color_full=True)

    # Create or update the cache of city coordinates
    failed_counter = 0
    mode = 'a' if os.path.exists(LOCATIONS_TSV_PATH) else 'w' # Don't want to overwrite if she already exists
    with open(LOCATIONS_TSV_PATH, mode, newline='', encoding='utf-8') as f:
        if mode == 'w':
            f.write("city\tcountry\tstate\tlatitude\tlongitude\n")
        
        for city, country, state in tqdm(cities, desc="Creating city coordinates cache"):
            if (city, country, state) in existing_cities:
                continue
            lat, lon = get_coordinates_for_city(city, country, state)
            if lat == 0.0 and lon == 0.0:
                failed_counter += 1
            f.write(f"{city}\t{country}\t{state}\t{lat}\t{lon}\n")
            existing_cities.add((city, country, state))
            f.flush()  # Ensure data is written to file immediately

    if failed_counter > 0:
        log(f"Failed to find coordinates for {failed_counter} cities.", level="WARNING")
    log(f"City coordinates cache created/updated at {local_filename(LOCATIONS_TSV_PATH)}", color=Fore.GREEN)

    clean_coordinate_info()

def clean_coordinate_info():
    """
    Clean the coordinate tsv file. This should only be called after the cache
    has been created in the `create_city_coordinates_cache` function.
    """
    # Ensure coords dir exists
    os.makedirs(os.path.dirname(LOCATIONS_TSV_PATH), exist_ok=True)
    if not os.path.exists(LOCATIONS_TSV_PATH):
        log(f"City coordinates file not found at {local_filename(LOCATIONS_TSV_PATH)}.", level="ERROR")
        return

    # Read existing entries
    df = pd.read_csv(LOCATIONS_TSV_PATH, sep='\t')
    length_prior = len(df)
    df = df.drop_duplicates()
    log(f"Removed {length_prior - len(df)} duplicate entries.", color=Fore.CYAN)

    # Print how many have 0.0 for lat/lon
    failed = df[(df["latitude"] == 0.0) & (df["longitude"] == 0.0)]
    log(f"Found {len(failed)} entries with 0.0 for latitude and longitude.", color=Fore.CYAN)
    # print out the failed entries
    # print(failed)

    # Save the failed entries to a new file
    failed_path = os.path.join(project_root, "data", "geolocation", "city_coordinates_failed.tsv")
    failed.to_csv(failed_path, sep='\t', index=False)
    log(f"Failed city coordinates saved to {local_filename(failed_path)}", color=Fore.YELLOW)

    # INput the correct values
    # for i, row in failed.iterrows():
    #     city = row["city"]
    #     country = row["country"]
    #     state = row["state"]
    #     lat = input(f"Enter latitude for {city}, {state}, {country}: ")
    #     if lat == "":
    #         continue
    #     lon = input(f"Enter longitude for {city}, {state}, {country}: ")
    #     df.loc[i, "latitude"] = float(lat)
    #     df.loc[i, "longitude"] = float(lon)

    # Save the cleaned data
    df.to_csv(LOCATIONS_TSV_PATH, sep='\t', index=False)
    log(f"Cleaned city coordinates saved to {local_filename(LOCATIONS_TSV_PATH)}", color=Fore.GREEN)


###############################################################################
#                               Mapping to FIPS                               #
###############################################################################


# Each US county has it's own unique FIPS code. After adding a latitude and 
# longitude for each patent we can use those to determine which county the 
# patent originated from, and can add the FIPS code (uid for US county) to each
# patent, which we will later group on.

def add_fips_codes(patents: pd.DataFrame) -> pd.DataFrame:
    pass # This one is gonna be tough
