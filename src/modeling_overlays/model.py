#!/usr/bin/env python3
#-*- coding: utf-8 -*-
"""
Created: Wed Nov 13 2024
Author: Aidan Allchin

Trains the model using the generated data in `generate_predictors.py`. The
model is then saved to disk for later use.
"""
import os, sys
from pathlib import Path

# ALL FILES IN THE PROJECT SHOULD BE IMPORTED RELATIVE TO THE PROJECT ROOT
# the # of `.parent`s should be adjusted based on the file's location
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import joblib
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
from typing import List
from datetime import datetime
from src.other.logging import PatentLogger
from src.other.helpers import local_filename

###############################################################################
#                               CONFIGURATION                                 #
###############################################################################

# Initialize logger
logger = PatentLogger.get_logger(__name__)

# Directories
DATA_PATH = os.path.join(project_root, 'data')
PREDICTORS_PATH = os.path.join(DATA_PATH, 'model', 'training_data.tsv')

# Model
MODEL_PATH = os.path.join(DATA_PATH, 'model', 'model.pkl')
RANDOM_STATE = 42


###############################################################################
#                              MAIN FUNCTIONS                                 #
###############################################################################


