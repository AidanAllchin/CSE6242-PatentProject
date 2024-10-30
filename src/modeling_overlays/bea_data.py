#!/usr/bin/env python3
#-*- coding: utf-8 -*-
"""
Created: Tue Oct 29 2024
Author: Kaitlyn Williams

Adds BEA data organized by MSA region and date to the /data directory as a tsv.
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
from src.other.helpers import log

# For access to data here, just do f"{DATA_FOLDER}/and then whatever file name you want"
DATA_FOLDER = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'bea')
