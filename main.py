#!/usr/bin/env python3
#-*- coding: utf-8 -*-
"""
Created: Tue Oct 17 2024
Author: Aidan Allchin

Ensure __init__.py has been run prior to running this script.

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

try:
    from colorama import Fore, Style
    import json
    import subprocess
    from src.other.helpers import log
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

if __name__ == "__main__":
    # Check if the data is downloaded
    data_name = config["settings"]["desired_data_release"]

    # Parse the data name from YYYY-MM-DD to YYMMDD
    data_name = f"ipa{data_name.split('-')[0][2:]}{data_name.split('-')[1]}{data_name.split('-')[2]}"
    data_path = os.path.join(PATENTS_DIRECTORY, "..", f"{data_name}.xml")
    data_year = data_name.split("-")[0]

    if not os.path.exists(data_path):
        log(f"Data for {data_year} is not downloaded. Running __init__.py...", color=Fore.YELLOW)
        subprocess.run(["python3", "__init__.py"])
    
    if not os.path.exists(PATENTS_DIRECTORY) or len(os.listdir(PATENTS_DIRECTORY)) == 0:
        log("Data directory does not exist. Running xml_splitter.py...", color=Fore.YELLOW)
        subprocess.run(["python3", "src/data_cleaning/xml_splitter.py"])

    # Add data
    log("Adding patent data to the database...", color=Fore.CYAN)
    subprocess.run(["python3", "src/data_cleaning/create_database.py"])

    log("\nAll steps completed.", color=Fore.LIGHTCYAN_EX, color_full=True)