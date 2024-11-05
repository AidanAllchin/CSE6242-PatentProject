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
import time
import requests
import random
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable

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
    while message.startswith("\n"):
        message = message[1:]
        print()

    # Use the provided color or the color scheme based on the log level
    log_color = color if color != Fore.LIGHTMAGENTA_EX else color_scheme.get(level.upper(), Fore.LIGHTMAGENTA_EX)

    # Format and print the log message
    if level.upper() == "INFO" and not color_full:
        print(f"{Style.BRIGHT}{log_color}[{caller_filename}] {level}: {Style.RESET_ALL}{message}")
    else:
        print(f"{Style.BRIGHT}{log_color}[{caller_filename}] {level}: {Style.NORMAL}{message}{Style.RESET_ALL}")

def completion_time(s_runtime: int) -> str:
    """
    Returns a string with just the hour, minute, and second values from 
    time.ctime().

    Args:
        s_runtime (int): Number of seconds between now and task completion.

    Returns:
        str: "HH:MM:SS (SSs)" of completion
    """
    s = time.ctime(time.time() + s_runtime)
    s = s.split(':')
    hh = s[0][-2:]
    mm = s[1].replace(':', '')
    ss = s[2][:2]
    
    return hh + ":" + mm + ":" + ss + f" ({s_runtime}s)"

def get_coordinates_for_city(city: str, country: str, state: str = None) -> tuple:
    """
    Get the latitude and longitude of a city with exponential backoff retry logic.

    Args:
        city (str): The name of the city.
        country (str): The name of the country.
        state (str): The name of the state. Defaults to None.

    Returns:
        tuple: The latitude and longitude of the city, or (0.0, 0.0) if not found after all retries.
    """
    geolocator = Nominatim(user_agent="patent_project_geocoder")
    
    # Crazy how QUEUE is just Q with 4 silent letters after it
    query = f"{city}, "
    if state:
        query += f"{state}, "
    query += country

    max_retries = 10  # Maximum number of retries (I may just set this to 1000)
    base_delay  = 1   # Base delay (s)
    max_delay   = 60  # Maximum delay (s)

    for attempt in range(max_retries):
        try:
            location = geolocator.geocode(query, timeout=10)
            if location:
                return (location.latitude, location.longitude)
            else:
                log(f"Could not find coordinates for {query}", level="WARNING")
                return 0.0, 0.0
        except (GeocoderTimedOut, GeocoderUnavailable) as e:
            delay = min(max_delay, base_delay * (2 ** attempt) + random.uniform(0, 1)) # Exponential backoff with jitter (little wiggle to prevent identical requests)
            log(f"Geocoding service error. Retrying in {delay:.2f} seconds... (Attempt {attempt + 1}/{max_retries})", level="WARNING")
            #log(f"Error details: {str(e)}", level="DEBUG")
            time.sleep(delay)

    log(f"Failed to geocode {query} after {max_retries} attempts. Returning default coordinates.", level="ERROR")
    return 0.0, 0.0

def local_filename(global_fname: str | Path) -> str:
    """
    Converts a global file name to a local file name relative to the project root.
    """
    return global_fname.replace(str(project_root), "")


def check_internet_connection(url='http://www.google.com/', timeout=5, retries: int = 0):
    """
    Check if the internet connection is available.

    Args:
        url (str): The URL to check the connection.
        timeout (int): The timeout for the request.
        retries (int): [DEFAULT=0] # of attempts.

    Returns:
        bool: True if the internet connection is available, False otherwise.
    """
    try:
        requests.get(url, timeout=timeout)
        return True
    except requests.ConnectionError:
        return False
    except requests.exceptions.ReadTimeout:
        time.sleep(5)
        if retries < 3:
            return check_internet_connection(url, timeout, retries=retries+1)

        return False
