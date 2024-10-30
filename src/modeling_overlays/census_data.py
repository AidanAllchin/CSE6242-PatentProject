#!/usr/bin/env python3
#-*- coding: utf-8 -*-
"""
Created: Tue Oct 29 2024
Author: Kaitlyn Williams

Gathers relevant census data for the patent data, and organizes it by MSA region and date.
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


