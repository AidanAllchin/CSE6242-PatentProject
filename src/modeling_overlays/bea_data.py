#!/usr/bin/env python3
#-*- coding: utf-8 -*-
"""
Created: Tue Oct 29 2024
Author: Kaitlyn Williams

Adds BEA data organized by county FIPS code and year to the /data directory as a tsv.
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

# For access to data here, just do f"{DATA_FOLDER}/and then whatever file name you want"
DATA_FOLDER = os.path.join(os.path.dirname(__file__), '..', '..', 'data')

# Paths
employment_p = os.path.join(DATA_FOLDER, 'raw', 'yearly_employment_county.csv')
gdp_p        = os.path.join(DATA_FOLDER, 'raw', 'yearly_gdp_county.csv')
income_p     = os.path.join(DATA_FOLDER, 'raw', 'yearly_personal_income_county.csv')
more_info_p  = os.path.join(DATA_FOLDER, 'raw', 'yearly_more_county_info.csv')


###############################################################################
#                               DATA CLEANING                                 #
###############################################################################


