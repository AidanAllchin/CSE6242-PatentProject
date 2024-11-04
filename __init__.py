#!/usr/bin/env python3
#-*- coding: utf-8 -*-
"""
Created: Tue Oct 01 2024
Author: Aidan Allchin

This script is used to install requirements, create necessary directories, and 
download the data. It should be run before running any other scripts in the
project. This is also the only file that uses relative paths, as it is intended
to be run from the root of the project directory.
"""
import os, sys, subprocess, json, re, time
from src.other.helpers import check_internet_connection

if not check_internet_connection():
    print(f"--- THIS SCRIPT REQUIRES AN INTERNET CONNECTION TO FUNCTION ---")
    sys.exit(0)

# Install the required packages
if not os.path.exists("requirements.txt"):
    print("Error: requirements.txt not found. Please confirm git repository is cloned correctly and you are in the correct directory.")
    sys.exit(1)

#confirm = input("\nThis script will install the required packages, create necessary directories, and download the data.\nPLEASE ENSURE YOU'RE USING YOUR DESIRED ENVIRONMENT...\nContinue? (y/n): ")
#if confirm.lower() != "y":
#    print("Exiting...")
#    sys.exit(0)
print("Installing required packages, creating necessary directories, and downloading the data...")
print("PLEASE ENSURE YOU'RE USING YOUR DESIRED ENVIRONMENT!")
time.sleep(2)

# Check if requirements are already installed
installed_packages = subprocess.run(["pip", "freeze"], stdout=subprocess.PIPE, text=True)
with open("requirements.txt", "r") as f:
    required_packages = f.read()

if required_packages in installed_packages.stdout:
    print(f"All required packages are already installed.")
else:
    print("Installing required packages...")
    subprocess.run(["pip", "install", "-r", "requirements.txt"])

from colorama import Fore, Style # just for styling of console output

def ensure_directory_exists(dir: str):
    """
    Ensures that the specified directory exists.

    Args:
        dir (str): The directory to check.
    """
    try:
        os.makedirs(dir, exist_ok=True)
        print(f"{Fore.GREEN}[__init__.py]: Created directory: {dir}{Style.RESET_ALL}")
    except PermissionError:
        print(f"{Fore.RED}[__init__.py]: {Style.DIM}Permission denied when creating directory: {dir}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}[__init__.py]: Attempting to create directory with sudo...{Style.RESET_ALL}")
        try:
            subprocess.run(['sudo', 'mkdir', '-p', dir], check=True)
            subprocess.run(['sudo', 'chmod', '777', dir], check=True)
            print(f"{Fore.GREEN}[__init__.py]: Successfully created directory with sudo.{Style.RESET_ALL}")
        except subprocess.CalledProcessError as e:
            print(f"{Fore.RED}[__init__.py]: Failed to create directory even with sudo: {e}{Style.RESET_ALL}")
            raise

# Create directories
if not os.path.exists("data"):
    ensure_directory_exists("data")
    ensure_directory_exists("data/patents")
    ensure_directory_exists("data/geolocation")
    ensure_directory_exists("data/census")
    ensure_directory_exists("data/bea")
    ensure_directory_exists("data/raw")
if not os.path.exists("data/patents"):
    ensure_directory_exists("data/patents")
if not os.path.exists("data/geolocation"):
    ensure_directory_exists("data/geolocation")
if not os.path.exists("data/census"):
    ensure_directory_exists("data/census")
if not os.path.exists("data/bea"):
    ensure_directory_exists("data/bea")
if not os.path.exists("data/raw"):
    ensure_directory_exists("data/raw")
if not os.path.exists("config"):
    ensure_directory_exists("config")

CONFIG_PATH = "config/config.json"

if not os.path.exists(CONFIG_PATH):
    print(f"{Fore.YELLOW}[__init__.py]: {Style.NORMAL}Creating config file...{Style.RESET_ALL}")
    with open(CONFIG_PATH, "w") as f:
        json.dump({"settings": {
        "desired_data_release": "2024-09-26", # We can modify this to get data from different releases
        "data_link": "https://bulkdata.uspto.gov/data/patent/application/redbook/fulltext/",
        "database_path": "data/patents.db",
        "city_coordinates_path": "data/geolocation/city_coordinates.tsv"
    }}, f)

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

def os_specific_download(url: str, dest: str):
    sys_type = sys.platform

    if check_internet_connection():
        if sys_type == "linux":
            subprocess.run(["wget", url])
        elif sys_type == "win32":
            subprocess.run(["curl", url, "-o", dest])
        # mac doesn't have wget
        elif sys_type == "darwin":
            subprocess.run(["curl", url, "-o", dest])
        else:
            print(f"{Fore.RED}[__init__.py]: {Style.NORMAL}Unsupported operating system: {sys_type}{Style.RESET_ALL}")
            print(f"{Fore.RED}[__init__.py]: {Style.NORMAL}Please download the data manually from the USPTO website:\n\t{url}{Style.RESET_ALL}")
            sys.exit(1)
    else:
        print(f"{Style.BRIGHT}{Fore.RED}[__init__.py]: {Style.NORMAL}No internet connection - unable to download required files. Aborting...{Style.RESET_ALL}")
        sys.exit(0)

def os_specific_unzip(file: str, dest: str):
    sys_type = sys.platform
    if sys_type == "linux":
        subprocess.run(["unzip", file, "-d", dest])
        subprocess.run(["rm", file])
    elif sys_type == "win32":
        subprocess.run(["tar", "-xf", file, "-C", dest])
        subprocess.run(["del", file], shell=True)
    # mac doesn't have wget
    elif sys_type == "darwin":
        subprocess.run(["unzip", file, "-d", dest])
        subprocess.run(["rm", file])
    else:
        print(f"{Fore.RED}[__init__.py]: {Style.NORMAL}Unsupported operating system: {sys_type}{Style.RESET_ALL}")
        print(f"{Fore.RED}[__init__.py]: {Style.NORMAL}Please download the data manually from the USPTO website: {file}{Style.RESET_ALL}")
        sys.exit(1)

@DeprecationWarning
def download_old():
    desired_data_release = config["settings"]["desired_data_release"]
    base_link            = config["settings"]["data_link"]

    # Structure the 'YYYY-MM-DD' date format to YYMMDD
    desired_data_release = desired_data_release.split("-")
    release_year         = desired_data_release[0]
    assert len(release_year) == 4, "Year must be in 'YYYY' format."
    desired_data_release = desired_data_release[0][2:] + desired_data_release[1] + desired_data_release[2]

    # Download the data from the USPTO website
    dir_contents = os.listdir("data")
    re_match = re.compile(r"ipa\d{6}\.xml")

    all_data_files = [f for f in dir_contents if re_match.match(f)]
    contains_desired = any(desired_data_release in f for f in all_data_files)
    print(f"{Style.BRIGHT}{Fore.CYAN}[__init__.py]: {Style.NORMAL}Found {len(all_data_files)} data files in data directory.{Style.RESET_ALL}")
    print(f"{Style.BRIGHT}{Fore.LIGHTCYAN_EX}[__init__.py]: {Style.RESET_ALL}Desired data release: {desired_data_release}.")
    name = f"ipa{desired_data_release}.zip"

    if not contains_desired:
        print(f"{Fore.YELLOW}[__init__.py]: {Style.NORMAL}Data file not found in data directory.{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}{Style.BRIGHT}[__init__.py]: {Style.NORMAL}Downloading data from USPTO website...\n{Style.RESET_ALL}")
        link = f"{base_link}{release_year}/{name}"
        
        sys_type = sys.platform
        if sys_type == "linux":
            subprocess.run(["wget", link, "-O", f"data/{name}"])
            print(f"\n{Style.BRIGHT}{Fore.LIGHTCYAN_EX}[__init__.py]: {Style.DIM}Downloaded data file: data/{name}. Extracting...{Style.RESET_ALL}\n")
            subprocess.run(["unzip", f"data/{name}", "-d", "data"])
            subprocess.run(["rm", f"data/{name}"])
        elif sys_type == "win32":
            subprocess.run(["curl", link, "-o", f"data/{name}"])
            print(f"\n{Style.BRIGHT}{Fore.LIGHTCYAN_EX}[__init__.py]: {Style.DIM}Downloaded data file: data/{name}. Extracting...{Style.RESET_ALL}\n")
            subprocess.run(["tar", "-xf", f"data/{name}", "-C", "data"])
            subprocess.run(["del", f"data/{name}"], shell=True)
        # mac doesn't have wget
        elif sys_type == "darwin":
            subprocess.run(["curl", link, "-o", f"data/{name}"])
            print(f"\n{Style.BRIGHT}{Fore.LIGHTCYAN_EX}[__init__.py]: {Style.DIM}Downloaded data file: data/{name}. Extracting...{Style.RESET_ALL}\n")
            subprocess.run(["unzip", f"data/{name}", "-d", "data"])
            subprocess.run(["rm", f"data/{name}"])
        else:
            print(f"{Fore.RED}[__init__.py]: {Style.NORMAL}Unsupported operating system: {sys_type}{Style.RESET_ALL}")
            print(f"{Fore.RED}[__init__.py]: {Style.NORMAL}Please download the data manually from the USPTO website: {link}{Style.RESET_ALL}")
            sys.exit(1)

        print(f"\n{Style.BRIGHT}{Fore.LIGHTCYAN_EX}[__init__.py]: {Style.DIM}Extracted data file: data/{name.replace('.zip', '.xml')}{Style.RESET_ALL}")
    else:
        print(f"{Fore.GREEN}[__init__.py]: Data file already exists: data/{name.replace('.zip', '.xml')}{Style.RESET_ALL}")

# download_old()

def download_raw_tables():
    # Downloads the raw tables from USPTO required to run the notebook Reid made
    # Paths (modified to be local)
    patents_g_p   = os.path.join(os.path.dirname(__file__), 'data', 'raw', 'g_patent.tsv')
    applicant_g_p = os.path.join(os.path.dirname(__file__), 'data', 'raw', 'g_inventor_not_disambiguated.tsv')
    location_g_p  = os.path.join(os.path.dirname(__file__), 'data', 'raw', 'g_location_not_disambiguated.tsv')
    assignee_g_p  = os.path.join(os.path.dirname(__file__), 'data', 'raw', 'g_assignee_not_disambiguated.tsv')
    wipo_g_p      = os.path.join(os.path.dirname(__file__), 'data', 'raw', 'g_wipo_technology.tsv')

    def zip_instead(p: str) -> str:
        return p + ".zip"

    # Download the raw tables
    print(f"\n{Style.BRIGHT}{Fore.CYAN}--- DOWNLOADING RAW TABLES ---{Style.RESET_ALL}")
    if not os.path.exists(patents_g_p):
        print(f"\n{Fore.YELLOW}[__init__.py]: Downloading {Style.DIM}g_patent.tsv{Style.NORMAL}...{Style.RESET_ALL}")
        os_specific_download("https://s3.amazonaws.com/data.patentsview.org/download/g_patent.tsv.zip", zip_instead(patents_g_p))
        
        # unzip
        print(f"{Fore.LIGHTMAGENTA_EX}[__init__.py]: Unzipping {Style.DIM}g_patent.tsv{Style.NORMAL}...{Style.RESET_ALL}")
        os_specific_unzip(zip_instead(patents_g_p), os.path.dirname(patents_g_p))
    else:
        print(f"{Fore.GREEN}[__init__.py]: {Style.DIM}g_patent.tsv{Style.NORMAL} already exists.{Style.RESET_ALL}")

    if not os.path.exists(applicant_g_p):
        print(f"\n{Fore.YELLOW}[__init__.py]: Downloading {Style.DIM}g_inventor_not_disambiguated.tsv{Style.NORMAL}...{Style.RESET_ALL}")
        os_specific_download("https://s3.amazonaws.com/data.patentsview.org/download/g_inventor_not_disambiguated.tsv.zip", zip_instead(applicant_g_p))

        # unzip
        print(f"{Fore.LIGHTMAGENTA_EX}[__init__.py]: Unzipping {Style.DIM}g_inventor_not_disambiguated.tsv{Style.NORMAL}...{Style.RESET_ALL}")
        os_specific_unzip(zip_instead(applicant_g_p), os.path.dirname(applicant_g_p))
    else:
        print(f"{Fore.GREEN}[__init__.py]: {Style.DIM}g_inventor_not_disambiguated.tsv{Style.NORMAL} already exists.{Style.RESET_ALL}")

    if not os.path.exists(location_g_p):
        print(f"\n{Fore.YELLOW}[__init__.py]: Downloading {Style.DIM}g_location_not_disambiguated.tsv{Style.NORMAL}...{Style.RESET_ALL}")
        os_specific_download("https://s3.amazonaws.com/data.patentsview.org/download/g_location_not_disambiguated.tsv.zip", zip_instead(location_g_p))

        # unzip
        print(f"{Fore.LIGHTMAGENTA_EX}[__init__.py]: Unzipping {Style.DIM}g_location_not_disambiguated.tsv{Style.NORMAL}...{Style.RESET_ALL}")
        os_specific_unzip(zip_instead(location_g_p), os.path.dirname(location_g_p))
    else:
        print(f"{Fore.GREEN}[__init__.py]: {Style.DIM}g_location_not_disambiguated.tsv{Style.NORMAL} already exists.{Style.RESET_ALL}")

    if not os.path.exists(assignee_g_p):
        print(f"\n{Fore.YELLOW}[__init__.py]: Downloading {Style.DIM}g_assignee_not_disambiguated.tsv{Style.NORMAL}...{Style.RESET_ALL}")
        os_specific_download("https://s3.amazonaws.com/data.patentsview.org/download/g_assignee_not_disambiguated.tsv.zip", zip_instead(assignee_g_p))

        # unzip
        print(f"{Fore.LIGHTMAGENTA_EX}[__init__.py]: Unzipping {Style.DIM}g_assignee_not_disambiguated.tsv{Style.NORMAL}...{Style.RESET_ALL}")
        os_specific_unzip(zip_instead(assignee_g_p), os.path.dirname(assignee_g_p))
    else:
        print(f"{Fore.GREEN}[__init__.py]: {Style.DIM}g_assignee_not_disambiguated.tsv{Style.NORMAL} already exists.{Style.RESET_ALL}")

    if not os.path.exists(wipo_g_p):
        print(f"\n{Fore.YELLOW}[__init__.py]: Downloading {Style.DIM}g_wipo_technology.tsv{Style.NORMAL}...{Style.RESET_ALL}")
        os_specific_download("https://s3.amazonaws.com/data.patentsview.org/download/g_wipo_technology.tsv.zip", zip_instead(wipo_g_p))

        # unzip
        print(f"{Fore.LIGHTMAGENTA_EX}[__init__.py]: Unzipping {Style.DIM}g_wipo_technology.tsv{Style.NORMAL}...{Style.RESET_ALL}")
        os_specific_unzip(zip_instead(wipo_g_p), os.path.dirname(wipo_g_p))
    else:
        print(f"{Fore.GREEN}[__init__.py]: {Style.DIM}g_wipo_technology.tsv{Style.NORMAL} already exists.{Style.RESET_ALL}")

    print(f"{Fore.GREEN}[__init__.py]: {Style.NORMAL}All raw tables downloaded.{Style.RESET_ALL}")


download_raw_tables()
print(f"{Style.BRIGHT}{Fore.GREEN}[__init__.py]:{Style.NORMAL} Initialization complete.{Style.RESET_ALL}")