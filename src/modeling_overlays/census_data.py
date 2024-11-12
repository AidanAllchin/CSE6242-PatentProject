#!/usr/bin/env python3
#-*- coding: utf-8 -*-
"""
Created: Tue Nov 12 2024
Author: Aidan Allchin

Gathers relevant census data for each county we have patent data in, and 
organizes it by county FIPS code and year to the /data directory as a tsv.
"""
import os, sys
from pathlib import Path

# ALL FILES IN THE PROJECT SHOULD BE IMPORTED RELATIVE TO THE PROJECT ROOT
# the # of `.parent`s should be adjusted based on the file's location
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import time
from tqdm import tqdm
from src.other.logging import PatentLogger


###############################################################################
#                               CONFIGURATION                                 #
###############################################################################


# Initialize logger
logger = PatentLogger.get_logger(__name__)

DATA_FOLDER  = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'census')
NAME_TO_FIPS = os.path.join(DATA_FOLDER, 'fips_to_county.tsv')


###############################################################################
#                               DATA CLEANING                                 #
###############################################################################


