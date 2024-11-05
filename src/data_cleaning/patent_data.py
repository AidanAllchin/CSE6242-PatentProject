#!/usr/bin/env python3
#-*- coding: utf-8 -*-
"""
Created: Mon Nov 04 2024
Author: Aidan Allchin

One-stop-shop for patent data cleaning.

Essentially adds all information we will ever need for every patent that we can
"""
import os, sys
from pathlib import Path

# ALL FILES IN THE PROJECT SHOULD BE IMPORTED RELATIVE TO THE PROJECT ROOT
# the # of `.parent`s should be adjusted based on the file's location
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import subprocess; subprocess.run(['python3', 'src/data_cleaning/import_data.py']) # This is hacky - fix eventually
import pandas as pd
from src.other.helpers import log, local_filename, check_internet_connection
from src.data_cleaning.patent_cleanup import add_coordinates, add_fips_codes

# Assumes `import_data` (and by extension `__init__.py`) have been run
PATENT_DATA_PATH = os.path.join(project_root, 'data', 'processed_patents.tsv')

df = pd.read_csv(PATENT_DATA_PATH, sep='\t', header=0)

# Add coordinates from the cities and stuff (assumes cache exists)
df = add_coordinates(patents=df)

# Add the FIPS codes for each patent
df = add_fips_codes(patents=df)

# Save df back to where she was
df.to_csv(PATENT_DATA_PATH, sep='\t', index=False)

