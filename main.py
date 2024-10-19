#!/usr/bin/env python3
#-*- coding: utf-8 -*-
"""
Created: Tue Oct 17 2024
Author: Aidan Allchin

Ensure `pip install -r requirements.txt` has been run prior to running this script.

This script is the main entry point for cleaning the patent data. It reads the
patent data from the `data/patents` directory, cleans the data by replacing/updating
fields with patent-specific terminology to be more readable and consistent, and
saves the cleaned data to the SQLite database located at `data/patents.db`.
"""
import os, sys
from pathlib import Path

# ALL FILES IN THE PROJECT SHOULD BE IMPORTED RELATIVE TO THE PROJECT ROOT
# the # of `.parent`s should be adjusted based on the file's location
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

import json
import subprocess
import sqlite3

# Test all imports
try:
    from colorama import Fore, Style
    import pandas as pd
    import numpy as np
    import geopy
    import lxml.etree as ET
    from tqdm import tqdm
    from src.other.helpers import log, local_filename
    from src.objects.patent import Patent
except ImportError as e:
    log(f"Error importing modules: {e}", color=Fore.RED)
    log(f"Ensure you have run `pip install -r requirements.txt` or manually run `python __init__.py", color=Fore.RED)


###############################################################################
#                               CONFIGURATION                                 #
###############################################################################


# Directories
PATENTS_DIRECTORY = os.path.join(project_root, 'data', 'patents')
CONFIG_PATH = os.path.join(project_root, 'config', 'config.json')
with open(CONFIG_PATH, 'r') as f:
    config = json.load(f)

# Database
DATABASE_PATH = os.path.join(project_root, config["settings"]["database_path"])


###############################################################################
#                              MAIN FUNCTIONS                                 #
###############################################################################


def init_database():
    """Initialize the database by creating tables and adding patent data."""
    log("Adding patent data to the database...", color=Fore.CYAN)
    subprocess.run(["python3", "src/data_cleaning/create_database.py"], check=True)

def view_demo_patent():
    """Display a demo patent from the database."""
    log("Displaying demo patent...", color=Fore.CYAN)
    with sqlite3.connect(DATABASE_PATH) as conn:
        patent = Patent.from_sqlite_by_id(conn, 20240318365)
        print(patent)

def display_menu():
    """Display the main menu options."""
    print("\nMain Menu:")
    print("1. Initialize database")
    print("2. View demo patent")
    print("3. Exit\n")
    return input("Select an option: ")

def check_data_availability():
    """Check if the required data is available and download if necessary."""
    data_name = config["settings"]["desired_data_release"]
    data_name = f"ipa{data_name[2:4]}{data_name[5:7]}{data_name[8:10]}"
    data_path = os.path.join(PATENTS_DIRECTORY, "..", f"{data_name}.xml")

    if not os.path.exists(data_path):
        log(f"Data for {data_name[:4]} is not downloaded. Running __init__.py...", color=Fore.YELLOW)
        subprocess.run(["python3", "__init__.py"], check=True)
    
    if not os.path.exists(PATENTS_DIRECTORY) or not os.listdir(PATENTS_DIRECTORY):
        log("Data directory is empty. Running xml_splitter.py...", color=Fore.YELLOW)
        subprocess.run(["python3", "src/data_cleaning/xml_splitter.py"], check=True)

def main():
    """Main function to run the patent processing pipeline."""
    check_data_availability()

    while True:
        option = display_menu()
        if option == "1":
            log(f"Note: This will erase the existing database at {local_filename(DATABASE_PATH)}...", level="WARNING")
            user_input = input("Please ensure you have downloaded `city_coordinates.tsv` first. Continue? (y/n): ")
            if user_input.lower() == "y":
                init_database()
        elif option == "2":
            if not os.path.exists(DATABASE_PATH):
                log("Database does not exist. Please initialize the database first.", level="ERROR")
            view_demo_patent()
        elif option == "3":
            break
        else:
            log("Invalid option. Please try again.", color=Fore.RED)

    log("\nAll steps completed.", color=Fore.LIGHTCYAN_EX, color_full=True)

if __name__ == "__main__":
    main()