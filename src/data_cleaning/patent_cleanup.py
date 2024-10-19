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


###############################################################################
#                           CLEAN PATENT DATA                                 #
###############################################################################


# def add_assignee_location(patent: Patent) -> Patent:
#     """
#     Add the latitude and longitude of the assignee location to the patent object.
    
#     Args:
#         patent: Patent object

#     Returns:
#         Patent object with latitude and longitude added
#     """
#     for i, assignee in enumerate(patent.assignee_info):
#         if assignee["city"] and assignee["country"]:
#             lat, lon = get_coordinates_for_city(assignee["city"], assignee["country"], assignee["state"] if assignee["state"] else None)
#             patent.assignee_info[i]["latitude"]  = lat
#             patent.assignee_info[i]["longitude"] = lon

#     return patent

# def add_inventor_location(patent: Patent) -> Patent:
#     """
#     Add the latitude and longitude of the inventor location to the patent object.
    
#     Args:
#         patent: Patent object

#     Returns:
#         Patent object with latitude and longitude added
#     """
#     for i, inventor in enumerate(patent.inventor_info):
#         if inventor["city"] and inventor["country"]:
#             lat, lon = get_coordinates_for_city(inventor["city"], inventor["country"], inventor["state"] if inventor["state"] else None)
#             patent.inventor_info[i]["latitude"]  = lat
#             patent.inventor_info[i]["longitude"] = lon

#     return patent

def add_coordinates(patents: List[Patent]) -> List[Patent]:
    """
    For every patent in the list, add the latitude and longitude of the assignee
    and inventor locations.
    
    Args:
        patents: List of Patent objects
        
    Returns:
        List of Patent objects with latitude and longitude added
    """
    locations_tsv_path = os.path.join(project_root, config["settings"]["city_coordinates_path"])

    # Ensure the city coordinates file exists
    if not os.path.exists(locations_tsv_path):
        log(f"City coordinates file not found at {local_filename(locations_tsv_path)}.", level="ERROR")
        return patents

    # Read the city coordinates file
    locations_df = pd.read_csv(locations_tsv_path, sep='\t')
    locations = set(locations_df.apply(lambda row: (row['city'], row['country'], row['state']), axis=1))

    # Add coordinates to each patent
    print()
    for _, patent in enumerate(tqdm(patents, desc="Adding coordinates from cache")):
        for j, assignee in enumerate(patent.assignee_info):
            if assignee["city"] and assignee["country"]:
                # Just to ensure I don't mess anything up with my pre-processing for the tsv,
                # I'm replacing tabs with spaces here too
                assignee["city"] = assignee["city"].replace("\t", " ")
                assignee["country"] = assignee["country"].replace("\t", " ")
                assignee["state"] = assignee["state"].replace("\t", " ") if assignee["state"] else None

                if (assignee["city"], assignee["country"], assignee["state"] if assignee["state"] else "") in locations:
                    location = locations_df[(locations_df["city"] == assignee["city"]) & (locations_df["country"] == assignee["country"]) & (locations_df["state"] == assignee["state"] if assignee["state"] else "")]
                    patent.assignee_info[j]["latitude"]  = location["latitude"].values[0]
                    patent.assignee_info[j]["longitude"] = location["longitude"].values[0]

        for j, inventor in enumerate(patent.inventor_info):
            if inventor["city"] and inventor["country"]:
                inventor["city"] = inventor["city"].replace("\t", " ")
                inventor["country"] = inventor["country"].replace("\t", " ")
                inventor["state"] = inventor["state"].replace("\t", " ") if inventor["state"] else None
                if (inventor["city"], inventor["country"], inventor["state"] if inventor["state"] else "") in locations:
                    location = locations_df[(locations_df["city"] == inventor["city"]) & (locations_df["country"] == inventor["country"]) & (locations_df["state"] == inventor["state"] if inventor["state"] else "")]
                    patent.inventor_info[j]["latitude"]  = location["latitude"].values[0]
                    patent.inventor_info[j]["longitude"] = location["longitude"].values[0]
    
    return patents


# Temporary function for creating the cache of city coordinates (stored in a CSV file)
# This should not be run by team members, but will be shared with the team
# after completion.
def create_city_coordinates_cache(patents: List[Patent]) -> None:
    """
    Create a cache of city coordinates by iterating over all patents and
    collecting the latitude and longitude of each city.

    Args:
        patents: List of Patent objects
    """
    tsv_path = os.path.join(project_root, "data", "geolocation", "city_coordinates.tsv")
    
    # Ensure coords dir exists
    os.makedirs(os.path.dirname(tsv_path), exist_ok=True)

    # Create a set of all unique cities
    cities = set()
    no_city_counter = 0
    print()
    for patent in tqdm(patents, desc="Collecting cities"):
        for assignee in patent.assignee_info:
            if assignee["state"]:
                assignee["state"] = assignee["state"].replace("\t", " ")
            if assignee["country"]:
                assignee["country"] = assignee["country"].replace("\t", " ")
            if assignee["city"]:
                assignee["city"] = assignee["city"].replace("\t", " ")
                cities.add((assignee["city"], assignee["country"], assignee["state"] if assignee["state"] else ""))
            else:
                no_city_counter += 1
        for inventor in patent.inventor_info:
            if inventor["state"]:
                inventor["state"] = inventor["state"].replace("\t", " ")
            if inventor["country"]:
                inventor["country"] = inventor["country"].replace("\t", " ")
            if inventor["city"]:
                inventor["city"] = inventor["city"].replace("\t", " ")
                cities.add((inventor["city"], inventor["country"], inventor["state"] if inventor["state"] else ""))
            else:
                no_city_counter += 1

    log(f"Found {len(cities)} unique cities.", color=Fore.CYAN)
    log(f"Unfortunately, {no_city_counter} entries exist with no city, meaning no coordinates will be added.", level="WARNING")
    
    # Read existing entries
    existing_cities = set()
    if os.path.exists(tsv_path):
        df = pd.read_csv(tsv_path, sep='\t', header=0, keep_default_na=False)
        # Gotta make this messy because dataframes won't format tuples correctly
        existing_cities = set(df.apply(lambda row: (row['city'], row['country'], row['state']), axis=1))

    log(f"Found {len(existing_cities)} cities already cached.", color=Fore.CYAN)
    log(f"\nCreating cache of city coordinates...", color=Fore.LIGHTBLUE_EX, color_full=True)

    # Create or update the cache of city coordinates
    failed_counter = 0
    mode = 'a' if os.path.exists(tsv_path) else 'w'
    with open(tsv_path, mode, newline='', encoding='utf-8') as f:
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
    log(f"City coordinates cache created/updated at {local_filename(tsv_path)}", color=Fore.GREEN)

    clean_coordinate_info()

def clean_coordinate_info():
    """
    Clean the coordinate tsv file. This should only be called after the cache
    has been created in the `create_city_coordinates_cache` function.
    """
    tsv_path = os.path.join(project_root, config["settings"]["city_coordinates_path"])
    
    # Ensure coords dir exists
    os.makedirs(os.path.dirname(tsv_path), exist_ok=True)
    if not os.path.exists(tsv_path):
        log(f"City coordinates file not found at {local_filename(tsv_path)}.", level="ERROR")
        return

    # Read existing entries
    df = pd.read_csv(tsv_path, sep='\t')
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
    df.to_csv(tsv_path, sep='\t', index=False)
    log(f"Cleaned city coordinates saved to {local_filename(tsv_path)}", color=Fore.GREEN)
