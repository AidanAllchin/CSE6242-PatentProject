#!/usr/bin/env python3
#-*- coding: utf-8 -*-
"""
Created: Tue Oct 17 2024
Author: Aidan Allchin

Holds helper functions that are used across multiple scripts.
"""
import os, sys
from pathlib import Path

# ALL FILES IN THE PROJECT SHOULD BE IMPORTED RELATIVE TO THE PROJECT ROOT
# the # of `.parent`s should be adjusted based on the file's location
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from colorama import Fore, Style
import re
import inspect

def get_patent_id_from_filename(fname: str) -> str:
    """
    Extracts the patent ID from a file name, if available.
    """
    # If it contains the project root, remove it
    if str(project_root) in fname:
        fname = fname.replace(str(project_root), "")
    
    # Extract the patent ID
    patent_id = re.search(r'\d+', fname)
    if patent_id:
        return patent_id.group()
    return None

def log(message: str, level: str = "INFO", color = Fore.LIGHTMAGENTA_EX, color_full: bool = False):
    """
    Prints a formatted log message to the console.

    Args:
        message (str): The message to be logged.
        level (str): The log level (e.g., "INFO", "WARNING", "ERROR"). Defaults to "INFO".
        color (Fore): The color to use for the log level. Defaults to Fore.LIGHTMAGENTA_EX.
        color_full (bool): Whether to use the color scheme for the whole message or just the tag. Defaults to False.
    """
    # Get the name of the file that called this function
    caller_frame = inspect.stack()[1]
    caller_filename = caller_frame.filename.split("/")[-1]

    # Define color schemes for different log levels
    color_scheme = {
        "INFO": Fore.LIGHTMAGENTA_EX,
        "WARNING": Fore.YELLOW,
        "ERROR": Fore.RED,
    }

    # Correctly handle newlines
    if message.startswith("\n"):
        message = message[1:]
        print()

    # Use the provided color or the color scheme based on the log level
    log_color = color if color != Fore.LIGHTMAGENTA_EX else color_scheme.get(level.upper(), Fore.LIGHTMAGENTA_EX)

    # Format and print the log message
    if level.upper() == "INFO" and not color_full:
        print(f"{Style.BRIGHT}{log_color}[{caller_filename}] {level}: {Style.RESET_ALL}{message}")
    else:
        print(f"{Style.BRIGHT}{log_color}[{caller_filename}] {level}: {Style.NORMAL}{message}{Style.RESET_ALL}")


