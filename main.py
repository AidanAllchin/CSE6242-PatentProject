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

from colorama import Fore, Style
import sqlite3
from typing import List, Dict
from tqdm import tqdm
import json
from src.data_cleaning.xml_splitter import split_xml_by_patent
from src.data_cleaning.patent_parsing import collect_patent_objects
from src.objects.patent import Patent
from src.other.helpers import log

###############################################################################
#                               CONFIGURATION                                 #
###############################################################################

# Directories
PATENTS_DIRECTORY = os.path.join(project_root, 'data', 'patents')
CONFIG_PATH = os.path.join(project_root, 'config', 'config.json')
with open(CONFIG_PATH, 'r') as f:
    config = json.load(f)

# Database
DATABASE_PATH = os.path.join(project_root, 'data', 'patents.db')
