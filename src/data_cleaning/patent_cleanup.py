#!/usr/bin/env python3
#-*- coding: utf-8 -*-
"""
Created: Tue Oct 17 2024
Author: Aidan Allchin

Cleans the patent data by adding coordinates to each patent. This is done by
iterating over each unique city/state/country pair in the patent data and
collecting the latitude and longitude of each city. This data is then saved
to a TSV file for future use.
"""
import os, sys
from pathlib import Path

# ALL FILES IN THE PROJECT SHOULD BE IMPORTED RELATIVE TO THE PROJECT ROOT
# the # of `.parent`s should be adjusted based on the file's location
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from colorama import Fore, Style
import subprocess
import pandas as pd
import json
import time
from tqdm import tqdm
from src.other.logging import PatentLogger
from src.other.helpers import get_coordinates_for_city, local_filename, completion_time


###############################################################################
#                                   SETUP                                     #
###############################################################################


# Initialize logger
logger = PatentLogger.get_logger(__name__)

LOCATIONS_TSV_PATH = os.path.join(project_root, 'data', 'geolocation', 'city_coordinates.tsv')
CORRECTIONS_PATH   = os.path.join(project_root, 'data', 'geolocation', 'location_corrections.tsv')
PATENTS_PATH       = os.path.join(project_root, 'data', 'processed_patents.tsv')
FINAL_PATENTS_PATH = os.path.join(project_root, 'data', 'patents.tsv')


###############################################################################
#                              Coordinate Stuff                               #
###############################################################################


def add_coordinates():
    """
    For every patent in the list, add the latitude and longitude of the assignee
    and inventor locations.

    Relies on the cache of city coordinates. If the cache does not exist, it will
    be created, which takes a long time.
    """
    # Ensure the patents file exists
    if not os.path.exists(PATENTS_PATH):
        logger.error("Patents file not found at {}. Try running from `main.py` and selection option (1).".format(local_filename(PATENTS_PATH)))
        return
    
    # Read the patents file
    patents = pd.read_csv(PATENTS_PATH, sep='\t')

    # Ensure the city coordinates file exists
    if not os.path.exists(LOCATIONS_TSV_PATH):
        logger.error("City coordinates file not found at {}. Creating cache (this may take >1hr)...".format(local_filename(LOCATIONS_TSV_PATH)))
        create_city_coordinates_cache(patents=patents)

    locs = pd.read_csv(LOCATIONS_TSV_PATH, sep='\t')
    num_zeros = len(locs[(locs['latitude'] == 0.0) & (locs['longitude'] == 0.0)])
    logger.info(f"% of city coordinates with 0.0 for latitude and longitude: {num_zeros / len(locs) * 100:.2f}%")
    del locs

    if not os.path.exists(CORRECTIONS_PATH):
        logger.error("Location corrections file not found at {}. Creating corrections...".format(local_filename(CORRECTIONS_PATH)))
        i = input("Are you sure you want to continue generating location corrections? (y/n): ")
        if i.lower() != 'y':
            logger.error("Exiting...", level="ERROR")
            sys.exit(1)

        # Wasn't actually going to use subprocess for this one but it's async
        subprocess.run(['python3', 'src/data_cleaning/llm_location_cleaner.py'])

    # Assume we have access to the corrections
    add_coordinates_to_corrections()

    # Update patents file and city_coordinates.tsv
    final_merge_and_clean()

    # Drop all remaining patents with locations of 0.0 for latitude and longitude
    # in either the inventor or assignee locations
    patents       = pd.read_csv(FINAL_PATENTS_PATH, sep='\t')
    num_patents   = len(patents)
    patents       = patents[(patents['inventor_latitude'] != 0.0) & (patents['inventor_longitude'] != 0.0)]
    patents       = patents[(patents['assignee_latitude'] != 0.0) & (patents['assignee_longitude'] != 0.0)]
    num_remaining = len(patents)
    num_removed   = num_patents - num_remaining
    patents.to_csv(FINAL_PATENTS_PATH, sep='\t', index=False)

    logger.info(f"Removed {num_removed} patents with 0.0 for latitude and longitude in either inventor or assignee locations.")
    logger.info(f"Lost {num_removed / num_patents * 100:.2f}% of patents due to missing coordinates.")

def create_city_coordinates_cache(patents: pd.DataFrame):
    """
    Create a cache of city coordinates by iterating over all patents and
    collecting the latitude and longitude of each city.

    This is a pretty messy function, and if I had more time would try to batch
    the requests to the geolocation API and handle asynchronously. As it stands,
    this function takes a long time to run.

    Args:
        patents (pd.DataFrame): The pandas DataFrame containing all the patents
    """
    # Ensure coords dir exists
    os.makedirs(os.path.dirname(LOCATIONS_TSV_PATH), exist_ok=True)

    logger.info("Creating coordinates for each unique city/state/country pair. Estimated completion time: {}.".format(completion_time(3600)))

    # Initial parsing (faster than iterating)
    patents['inventor_city']    = patents['inventor_city'].str.lower()
    patents['inventor_state']   = patents['inventor_state'].str.lower()
    patents['inventor_country'] = patents['inventor_country'].str.lower()

    patents['assignee_city']    = patents['assignee_city'].str.lower()
    patents['assignee_state']   = patents['assignee_state'].str.lower()
    patents['assignee_country'] = patents['assignee_country'].str.lower()

    # Create a set of all unique cities
    cities = set()
    total_len = len(patents)
    no_city_counter = 0
    for i, patent in tqdm(patents.iterrows(), desc="Collecting unique locations", total=total_len):
        if patent['inventor_city'] is not None:
            cities.add((patent["inventor_city"], patent["inventor_country"], patent["inventor_state"] if patent["inventor_state"] else ""))
        else:
            no_city_counter += 1
        if patent["assignee_city"] is not None:
            cities.add((patent["assignee_city"], patent["assignee_country"], patent["assignee_state"] if patent["assignee_state"] else ""))
        else:
            no_city_counter += 1

    logger.info("Found {} unique cities.".format(len(cities)))

    if no_city_counter > 0:
        logger.warning("{} entries exist with no city, meaning no coordinates will be added for {:.2f}% of the patents.".format(no_city_counter, no_city_counter * 100 / total_len))
    
    # Read existing entries
    existing_cities = set()
    if os.path.exists(LOCATIONS_TSV_PATH):
        df = pd.read_csv(LOCATIONS_TSV_PATH, sep='\t', header=0, keep_default_na=False)

        # Gotta make this messy because dataframes won't format tuples correctly
        existing_cities = set(df.apply(lambda row: (row['city'], row['country'], row['state']), axis=1))

    logger.info("Found {} cities already cached.".format(len(existing_cities)))
    st = time.time()
    logger.info("Creating cache of city coordinates...")

    # Create or update the cache of city coordinates
    failed_counter = 0
    mode = 'a' if os.path.exists(LOCATIONS_TSV_PATH) else 'w'  # Don't want to overwrite if she already exists
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
        logger.warning("Failed to find coordinates for {} cities.".format(failed_counter))
    logger.info("City coordinates cache created/updated at {} in {:.2f} minutes.".format(local_filename(LOCATIONS_TSV_PATH), (time.time() - st) / 60))

    clean_coordinate_info()

def add_coordinates_to_corrections():
    """
    Reads the location corrections TSV file, adds coordinates for each corrected
    location, and saves the updated file.
    """
    if not os.path.exists(CORRECTIONS_PATH):
        logger.error("Corrections file not found at {}. Exiting...".format(local_filename(CORRECTIONS_PATH)))
        return

    # Read the corrections file
    df = pd.read_csv(CORRECTIONS_PATH, sep='\t')
    total_rows = len(df)
    num_remaining = len(df[(df['latitude'] == 0.0) & (df['longitude'] == 0.0)])
    if num_remaining / total_rows < 0.25:
        logger.warning("More than 75% of the locations have coordinates. Assuming all locations have been processed.")
        return

    logger.info("Processing coordinates for {} location corrections...".format(total_rows))

    # Add latitude and longitude columns if they don't exist
    if 'latitude' not in df.columns:
        df['latitude'] = 0.0
    if 'longitude' not in df.columns:
        df['longitude'] = 0.0

    # Process each row
    updates = 0
    for idx, row in tqdm(df.iterrows(), total=total_rows, desc="Adding coordinates to LLM-generated locations"):
        # Skip if we already have coordinates for this row
        if row['latitude'] != 0.0 or row['longitude'] != 0.0:
            continue
            
        # Get coordinates for the corrected location
        lat, lon = get_coordinates_for_city(
            city=row['new_city'],
            state=row['new_state'],
            country=row['new_country']
        )
        
        # Update the DataFrame
        df.at[idx, 'latitude'] = lat
        df.at[idx, 'longitude'] = lon
        
        if lat != 0.0 and lon != 0.0:
            updates += 1

        # Save after 10 rows
        if idx % 10 == 0:
            df.to_csv(CORRECTIONS_PATH, sep='\t', index=False)

    df.to_csv(CORRECTIONS_PATH, sep='\t', index=False)

    # Report results
    success_rate = (updates / total_rows) * 100
    failed = total_rows - updates
    logger.info("Added coordinates for {} locations ({:.1f}% success rate)".format(updates, success_rate))
    if failed > 0:
        logger.warning("Failed to get coordinates for {} locations".format(failed))

def final_merge_and_clean(batch_size: int = 10000):
    """
    Consolidates corrected locations and coordinates into the master coordinates file
    and updates patent locations based on corrections.
    1. Load corrections into memory as a lookup dictionary
    2. Update the coordinates file with corrections (in lowercase)
    3. Update patent location names with corrections
    4. Update patent coordinates using the lowercase coordinates file

    Args:
        batch_size (int): Number of rows to process at a time
    """
    start_time = time.time()
    logger.info("Starting final merge and clean process. Estimated completion time: {}.".format(completion_time(205)))

    #######################################################################
    #     Step 1: Load corrections into memory as a lookup dictionary     #
    #######################################################################

    logger.info("Loading corrections into memory...")
    corrections_df = pd.read_csv(CORRECTIONS_PATH, sep='\t')

    # Create lookup dictionary with lowercase, stripped keys
    corrections_map = {
        (
            str(row['bad_city']).lower().strip(),
            str(row['bad_state']).lower().strip(),
            str(row['bad_country']).lower().strip()
        ): {
            'city': row['new_city'],
            'state': row['new_state'],
            'country': row['new_country']
        }
        for _, row in corrections_df.iterrows()
        if row['latitude'] != 0.0 and row['longitude'] != 0.0  # Only include if we have valid coordinates
    }
    num_valid_corrections = len(corrections_map)
    num_total_corrections = len(corrections_df)
    num_filtered = num_total_corrections - num_valid_corrections
    success_rate = (num_valid_corrections / num_total_corrections * 100) if num_total_corrections > 0 else 0
    
    logger.info("Loaded {} valid corrections out of {} total".format(num_valid_corrections, num_total_corrections))
    if num_filtered > 0:
        logger.warning("Filtered out {} corrections without valid coordinates ({:.1f}% success rate)".format(num_filtered, success_rate))

    #######################################################################
    #         Step 2: Update the coordinates file with corrections        #
    #######################################################################

    logger.info("Updating master coordinates file...")
    coords_df = pd.read_csv(LOCATIONS_TSV_PATH, sep='\t')

    # Create set of existing locations to avoid duplicates
    existing_locations = set(coords_df.apply(
        lambda row: (
            str(row['city']).lower().strip(),
            str(row['state']).lower().strip() if pd.notna(row['state']) else "",
            str(row['country']).lower().strip()
        ), 
        axis=1
    ))

    # Convert corrections_df location columns to string
    corrections_df['new_city']    = corrections_df['new_city'].astype(str)
    corrections_df['new_state']   = corrections_df['new_state'].astype(str)
    corrections_df['new_country'] = corrections_df['new_country'].astype(str)
    corrections_df['bad_city']    = corrections_df['bad_city'].astype(str)
    corrections_df['bad_state']   = corrections_df['bad_state'].astype(str)
    corrections_df['bad_country'] = corrections_df['bad_country'].astype(str)

    # Occasionally state, city, country values will be "nan" or "none"
    # We need to convert these to empty strings for comparison
    corrections_df['new_state']   = corrections_df['new_state'].apply(lambda x: "" if x.lower() == "nan" or x.lower() == "none" else x)
    corrections_df['new_city']    = corrections_df['new_city'].apply(lambda x: "" if x.lower() == "nan" or x.lower() == "none" else x)
    corrections_df['new_country'] = corrections_df['new_country'].apply(lambda x: "" if x.lower() == "nan" or x.lower() == "none" else x)

    # Same with the bad columns
    corrections_df['bad_state']   = corrections_df['bad_state'].apply(lambda x: "" if x.lower() == "nan" or x.lower() == "none" else x)
    corrections_df['bad_city']    = corrections_df['bad_city'].apply(lambda x: "" if x.lower() == "nan" or x.lower() == "none" else x)
    corrections_df['bad_country'] = corrections_df['bad_country'].apply(lambda x: "" if x.lower() == "nan" or x.lower() == "none" else x)
    
    # Add new locations from corrections
    new_locations = []
    for _, correction in tqdm(corrections_df.iterrows(), total=num_valid_corrections, desc="Adding corrections to coordinates"):
        if correction['latitude'] == 0.0 or correction['longitude'] == 0.0:
            continue

        # Create a lowercase version for storage
        new_loc = {
            'city': str(correction['new_city']).lower().strip(),
            'country': str(correction['new_country']).lower().strip(),
            'state': str(correction['new_state']).lower().strip(),
            'latitude': correction['latitude'],
            'longitude': correction['longitude']
        }

        # Just using this to check if the corrected location is already in the file
        # - it's not being saved in lowercase/stripped form (good)
        check_key = (new_loc['city'], new_loc['country'], new_loc['state'])
        if check_key not in existing_locations:
            new_locations.append(new_loc)
            existing_locations.add(check_key)
    
    if new_locations:
        new_locations_df = pd.DataFrame(new_locations)
        coords_df = pd.concat([coords_df, new_locations_df], ignore_index=True)
        coords_df.to_csv(LOCATIONS_TSV_PATH, sep='\t', index=False)
        logger.info("Added {} new locations to coordinates file".format(len(new_locations)))
    
    #######################################################################
    #       Step 3: Update patent location names with corrections         #
    #######################################################################

    # Per chunk:
    # - INVENTOR LOCATION:
    #   - Try to update the location name with LLM corrections
    #   - If found, update the coordinate lookup key
    #   - Add the coordinates if they exist
    # - ASSIGNEE LOCATION:
    #   - Try to update the location name with LLM corrections
    #   - If found, update the coordinate lookup key
    #   - Add the coordinates if they exist

    logger.info("Updating patent location names. Estimated completion time: {}.".format(completion_time(70)))
    p_names_start = time.time()
    location_updates = 0
    total_patents = 0

    # Set up the intermediate file
    INTERIM_PATENTS_PATH = PATENTS_PATH.replace('.tsv', '_interim.tsv')
    if os.path.exists(INTERIM_PATENTS_PATH):
        os.remove(INTERIM_PATENTS_PATH)
    
    # Process patents file in chunks to update location names
    for chunk in pd.read_csv(PATENTS_PATH, sep='\t', chunksize=batch_size):
        total_patents += len(chunk)
        
        for idx, row in chunk.iterrows():
            # Update inventor location name if correction exists
            inv_key = (
                str(row['inventor_city']).lower().strip(),
                str(row['inventor_state']).lower().strip() if pd.notna(row['inventor_state']) else "",
                str(row['inventor_country']).lower().strip()
            )
            if inv_key in corrections_map:
                correction = corrections_map[inv_key]
                chunk.at[idx, 'inventor_city'] = correction['city']
                chunk.at[idx, 'inventor_state'] = correction['state']
                chunk.at[idx, 'inventor_country'] = correction['country']

                location_updates += 1
            
            # Update assignee location name if correction exists and not null
            if pd.notna(row['assignee_city']):
                assign_key = (
                    str(row['assignee_city']).lower().strip(),
                    str(row['assignee_state']).lower().strip() if pd.notna(row['assignee_state']) else "",
                    str(row['assignee_country']).lower().strip()
                )
                if assign_key in corrections_map:
                    correction = corrections_map[assign_key]
                    chunk.at[idx, 'assignee_city'] = correction['city']
                    chunk.at[idx, 'assignee_state'] = correction['state']
                    chunk.at[idx, 'assignee_country'] = correction['country']

                    location_updates += 1
        
        # Write chunk to interim file
        chunk.to_csv(INTERIM_PATENTS_PATH, sep='\t', index=False, mode='a', header=not os.path.exists(INTERIM_PATENTS_PATH))

    logger.info("Updated {} location names in {:.2f} minutes.".format(location_updates, (time.time() - p_names_start) / 60))


    #######################################################################
    #       Step 4: Add coordinates to patents using lowercase lookup     #
    #######################################################################


    logger.info("Adding coordinates from master file. Estimated completion time: {}.".format(completion_time(140)))
    coord_add_start = time.time()
    
    # Read and create lowercase coordinate lookup
    coords_df = pd.read_csv(LOCATIONS_TSV_PATH, sep='\t')
    coords_map = {
        (
            str(row['city']).lower().strip(),
            str(row['country']).lower().strip(),
            str(row['state']).lower().strip() if pd.notna(row['state']) else ""
        ): {
            'latitude': row['latitude'],
            'longitude': row['longitude']
        }
        for _, row in coords_df.iterrows()
    }
    
    coordinate_updates = 0
    
    # Process the interim file to add coordinates
    if os.path.exists(FINAL_PATENTS_PATH):
        os.remove(FINAL_PATENTS_PATH)
    
    for chunk in pd.read_csv(INTERIM_PATENTS_PATH, sep='\t', chunksize=batch_size):
        for idx, row in chunk.iterrows():
            # Look up inventor coordinates
            inv_key = (
                str(row['inventor_city']).lower().strip(),
                str(row['inventor_country']).lower().strip(),
                str(row['inventor_state']).lower().strip() if pd.notna(row['inventor_state']) else ""
            )
            if inv_key in coords_map:
                coords = coords_map[inv_key]
                chunk.at[idx, 'inventor_latitude'] = coords['latitude']
                chunk.at[idx, 'inventor_longitude'] = coords['longitude']
                coordinate_updates += 1
            else:
                chunk.at[idx, 'inventor_latitude'] = 0.0
                chunk.at[idx, 'inventor_longitude'] = 0.0
            
            # Look up assignee coordinates if not null
            if pd.notna(row['assignee_city']):
                assign_key = (
                    str(row['assignee_city']).lower().strip(),
                    str(row['assignee_country']).lower().strip(),
                    str(row['assignee_state']).lower().strip() if pd.notna(row['assignee_state']) else ""
                )
                if assign_key in coords_map:
                    coords = coords_map[assign_key]
                    chunk.at[idx, 'assignee_latitude'] = coords['latitude']
                    chunk.at[idx, 'assignee_longitude'] = coords['longitude']
                    coordinate_updates += 1
                else:
                    chunk.at[idx, 'assignee_latitude'] = 0.0
                    chunk.at[idx, 'assignee_longitude'] = 0.0
        
        # Write chunk to final file
        chunk.to_csv(FINAL_PATENTS_PATH, sep='\t', index=False, mode='a', header=not os.path.exists(FINAL_PATENTS_PATH))

    # Clean up interim file
    os.remove(INTERIM_PATENTS_PATH)

    # Report results
    runtime = time.time() - start_time
    location_rate = (location_updates / total_patents) * 100 if total_patents > 0 else 0
    coord_rate = (coordinate_updates / (total_patents * 2)) * 100 if total_patents > 0 else 0

    logger.info("Added coordinates to {} inventor/assignee locations in {:.2f} minutes.".format(coordinate_updates, (time.time() - coord_add_start) / 60))
    
    logger.info("Final merge and clean completed in {:.1f} seconds:".format(runtime))
    logger.info("  > Processed {} patents".format(total_patents))
    logger.info("  > Updated {} location names ({:.1f}% update rate)".format(location_updates, location_rate))
    logger.info("  > Added {} coordinate pairs ({:.1f}% success rate)".format(coordinate_updates, coord_rate))

def drop_unusable():
    """
    Very simple final step dropping the patents without a FIPS code for the 
    inventor. This is only 126 patents, but I want the final .tsv super clean.
    """
    # Ensure the patents file exists
    if not os.path.exists(FINAL_PATENTS_PATH):
        logger.error("Patents file not found at {}. Try running from `main.py` and selection option (1).".format(local_filename(FINAL_PATENTS_PATH)))
        return

    # Read the patents file
    patents       = pd.read_csv(FINAL_PATENTS_PATH, sep='\t')
    num_patents   = len(patents)
    patents       = patents[patents['inventor_fips'].notnull()]
    num_remaining = len(patents)
    num_removed   = num_patents - num_remaining
    patents.to_csv(FINAL_PATENTS_PATH, sep='\t', index=False)
    
    logger.info("Removed {} patents without a FIPS code for the inventor.".format(num_removed))
    logger.info("Lost {:.2f}% of patents due to missing FIPS codes.".format(num_removed / num_patents * 100))

def clean_coordinate_info():
    """
    Clean the coordinate tsv file. This should only be called after the cache
    has been created in the `create_city_coordinates_cache` function, and 
    before the corrections/final merge process.
    """
    # Ensure coords dir exists
    os.makedirs(os.path.dirname(LOCATIONS_TSV_PATH), exist_ok=True)
    if not os.path.exists(LOCATIONS_TSV_PATH):
        logger.error("City coordinates file not found at {}. Exiting...".format(local_filename(LOCATIONS_TSV_PATH)))
        return

    # Read existing entries
    df = pd.read_csv(LOCATIONS_TSV_PATH, sep='\t')
    length_prior = len(df)
    df = df.drop_duplicates()
    logger.info("Removed {} duplicate location entries.".format(length_prior - len(df)))

    # Print how many have 0.0 for lat/lon
    failed = df[(df["latitude"] == 0.0) & (df["longitude"] == 0.0)]
    logger.info("Found {} entries with 0.0 for latitude and longitude.".format(len(failed)))

    # Save the failed entries to a new file
    failed_path = os.path.join(project_root, "data", "geolocation", "city_coordinates_failed.tsv")
    failed.to_csv(failed_path, sep='\t', index=False)
    logger.info(f"Failed city coordinates saved to {local_filename(failed_path)}")

    # Save the cleaned data
    df.to_csv(LOCATIONS_TSV_PATH, sep='\t', index=False)
    logger.info("Cleaned city coordinates saved to {}.".format(local_filename(LOCATIONS_TSV_PATH)))