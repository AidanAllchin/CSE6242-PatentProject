#!/usr/bin/env python3
#-*- coding: utf-8 -*-
"""
Created: Tue Oct 17 2024
Author: Aidan Allchin

Cleans the patent data by replacing/updating fields with patent-specific
terminology to be more readable and consistent.
"""
import os, sys
from pathlib import Path

# ALL FILES IN THE PROJECT SHOULD BE IMPORTED RELATIVE TO THE PROJECT ROOT
# the # of `.parent`s should be adjusted based on the file's location
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from colorama import Fore, Style


# Conceptual steps:
# 1. Convert Classifications (IPC and CPC) to their meanings
# 2. Convert Dates to a consistent format
# 3. Convert Patent Status to a more readable format
# 4. Add latitude and longitude to all locations


