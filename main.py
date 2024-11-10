#!/usr/bin/env python3
#-*- coding: utf-8 -*-
"""
Created: Mon Nov 04 2024
Author: Aidan Allchin

Ensure `pip install -r requirements.txt` has been run prior to running this script.

This script is the main entry point for cleaning the patent data. It downloads
the required tables from the USPTO site, merges the appropriate tables,
adds coordinates to each of the patents, and then sorts them by county.
Provides a (gross looking, sure) menu to navigate the creation of predictive
factors for our Innovation Hub Predictor Model.
"""
import os, sys
from pathlib import Path

# ALL FILES IN THE PROJECT SHOULD BE IMPORTED RELATIVE TO THE PROJECT ROOT
# the # of `.parent`s should be adjusted based on the file's location
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

import json
import subprocess

# This does all the setup steps for the project - don't modify it
subprocess.run(['python3', '__init__.py'])

from colorama import Fore, Style
from src.other.logging import PatentLogger
from src.data_cleaning.patent_cleanup import add_coordinates, drop_unusable
from src.data_cleaning.patent_fips import add_fips_codes


###############################################################################
#                               CONFIGURATION                                 #
###############################################################################


# Initialize logger
logger = PatentLogger.get_logger(__name__)

# Directories
PATENTS_DIRECTORY    = os.path.join(project_root, 'data', 'patents')
CLEANED_PATENTS_PATH = os.path.join(project_root, 'data', 'patents.tsv')
BEA_DATA_PATH        = os.path.join(project_root, 'data', 'bea', 'bea_predictors.tsv')
CENSUS_DATA_PATH     = os.path.join(project_root, 'data', 'census', 'census_predictors.tsv')


###############################################################################
#                              MAIN FUNCTIONS                                 #
###############################################################################


def get_patent_data():
    """
    This script should handle all patent data download and merging. The end 
    result is `<root>/data/patents.tsv`.
    """
    if os.path.exists(CLEANED_PATENTS_PATH) and os.path.getsize(CLEANED_PATENTS_PATH) > 0:
        logger.info("Patent data already exists. Skipping.")
        return
    else:
        logger.warning("Please consider downloading the patent data and placing it in the appropriate directory.")
        logger.warning("Running this script as an alternative will day upwards of 24 hours.")
        i = input("Continue? (y/n): ")
        if i.lower() != "y":
            return

    try:
        subprocess.run(["python3", "src/data_cleaning/import_data.py"], check=True)
        print()
    except subprocess.CalledProcessError as e:
        print(f"{Style.BRIGHT}{Fore.RED}\n[system]: {Style.NORMAL}Failed to run `import_data.py`: {e}{Style.RESET_ALL}\n")

    # Let em do what it does
    add_coordinates()
    add_fips_codes()
    drop_unusable()

def get_bea_data():
    """
    Downloads and creates a .tsv with the Bureau of Economic Analysis data
    organized by county and grouped by time period.

    Should result in a `<root>/data/bea/bea_predictors.tsv` file.
    """
    try:
        # TODO: Waiting for Kaitlyn to finish
        subprocess.run(["python3", "src/modeling_overlays/bea_data.py"], check=True)
        print()
    except subprocess.CalledProcessError as e:
        print(f"{Style.BRIGHT}{Fore.RED}\n[system]: {Style.NORMAL}Failed to run `bea_data.py`: {e}{Style.RESET_ALL}\n")

def get_census_data():
    """
    Downloads and creates a .tsv with the US Census data
    organized by county and grouped by time period.

    Should result in a `<root>/data/census/census_predictors.tsv` file.
    """
    try:
        # TODO: Waiting for Kaitlyn to finish
        subprocess.run(["python3", "src/modeling_overlays/census_data.py"], check=True)
        print()
    except subprocess.CalledProcessError as e:
        print(f"{Style.BRIGHT}{Fore.RED}\n[system]: {Style.NORMAL}Failed to run `census_data.py`: {e}{Style.RESET_ALL}\n")

def create_model_predictors():
    """
    Uses patent data, BEA data, Fed data, Census data to create predictor
    variables for each county in the US, for each time period. Also generates
    an `innovation_score` for each region/time to use as a response variable.

    Should result in a `<root>/data/to_train.tsv` file.
    """
    pass

def display_menu():
    """Display the main menu options."""
    print(f"{Style.BRIGHT}       Main Menu{Style.RESET_ALL}")
    print("1. ETL patent data")
    print("2. ETL BEA data")
    print("3. ETL census data")
    print("4. Generate Innovation Hub Predictor Model (IHPM) variables")
    print("0. Exit\n")
    return input("Select an option: ")

def main():
    """
    Main function to run the patent processing pipeline.
    """
    logger.info("Starting patent processing pipeline...")

    while True:
        option = display_menu()
        if option == "1":
            user_input = input(f"Please ensure you have downloaded {Style.DIM}city_coordinates.tsv{Style.NORMAL} and {Style.DIM}location_coordinates.tsv{Style.NORMAL} first and are using the correct python environment. Continue? (y/n): ")
            if user_input.lower() == "y":
                get_patent_data()
            else: print()
        elif option == "2":
            get_bea_data()
        elif option == "3":
            get_census_data()
        elif option == "4":
            if not os.path.exists(CLEANED_PATENTS_PATH) or os.path.getsize(CLEANED_PATENTS_PATH) == 0:
                logger.error("Cleaned patents file doesn't exists. Please run step 1 first.")
                continue
            if not os.path.exists(BEA_DATA_PATH) or os.path.getsize(BEA_DATA_PATH) == 0:
                logger.error("BEA predictors file doesn't exists. Please run step 2 first.")
                continue
            if not os.path.exists(CENSUS_DATA_PATH) or os.path.getsize(CENSUS_DATA_PATH) == 0:
                logger.error("Census predictors file doesn't exists. Please run step 3 first.")
                continue

            # If everything we need exists, run it
            create_model_predictors()
        elif option == "0":
            break
        else:
            logger.error("Invalid option. Please try again.")
    
    logger.info("Exiting patent processing pipeline...")

if __name__ == "__main__":
    main()