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
    print(f"--- INIT.PY REQUIRES AN INTERNET CONNECTION TO FUNCTION. SKIPPING ---")
    sys.exit(0)

# Install the required packages
if not os.path.exists("requirements.txt"):
    print("Error: requirements.txt not found. Please confirm git repository is cloned correctly and you are in the correct directory.")
    sys.exit(1)

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
    ensure_directory_exists("data/geolocation")
    ensure_directory_exists("data/census")
    ensure_directory_exists("data/bea")
    ensure_directory_exists("data/raw")
    ensure_directory_exists("data/model")
if not os.path.exists("data/geolocation"):
    ensure_directory_exists("data/geolocation")
if not os.path.exists("data/census"):
    ensure_directory_exists("data/census")
if not os.path.exists("data/bea"):
    ensure_directory_exists("data/bea")
if not os.path.exists("data/raw"):
    ensure_directory_exists("data/raw")
if not os.path.exists("data/model"):
    ensure_directory_exists("data/model")
if not os.path.exists("config"):
    ensure_directory_exists("config")

RAW_DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "raw")

def os_specific_download(url: str, dest: str):
    sys_type = sys.platform

    if check_internet_connection():
        if sys_type == "linux":
            subprocess.run(["wget", url, "-O", dest])
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

def fetch_patent_raw_data():
    # Downloads the raw tables from USPTO required to run the notebook Reid made
    # Paths (modified to be local)
    patents_g_p   = os.path.join(RAW_DATA_PATH, 'g_patent.tsv')
    applicant_g_p = os.path.join(RAW_DATA_PATH, 'g_inventor_not_disambiguated.tsv')
    location_g_p  = os.path.join(RAW_DATA_PATH, 'g_location_not_disambiguated.tsv')
    assignee_g_p  = os.path.join(RAW_DATA_PATH, 'g_assignee_not_disambiguated.tsv')
    wipo_g_p      = os.path.join(RAW_DATA_PATH, 'g_wipo_technology.tsv')

    def zip_instead(p: str) -> str:
        return p + ".zip"

    # Download the raw tables
    print(f"\n{Style.BRIGHT}{Fore.CYAN}--- DOWNLOADING RAW USPTO TABLES ---{Style.RESET_ALL}")
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

def fetch_bea_raw_tables():
    # Downloads the raw tables from BEA needed to create predictors
    # Paths (modified to be local)
    yearly_income_p      = os.path.join(RAW_DATA_PATH, 'yearly_personal_income_county.csv')
    yearly_income_b      = os.path.join(RAW_DATA_PATH, 'CAINC1__ALL_AREAS_1969_2022.csv')
    yearly_gdp_p         = os.path.join(RAW_DATA_PATH, 'yearly_gdp_county.csv')
    yearly_gdp_b         = os.path.join(RAW_DATA_PATH, 'CAGDP1__ALL_AREAS_2001_2022.csv')
    yearly_employment_p  = os.path.join(RAW_DATA_PATH, 'yearly_employment_county.csv')
    yearly_employment_b  = os.path.join(RAW_DATA_PATH, 'CAINC4__ALL_AREAS_1969_2022.csv')
    yearly_more_county_p = os.path.join(RAW_DATA_PATH, 'yearly_more_county_info.csv')
    yearly_more_county_b = os.path.join(RAW_DATA_PATH, 'CAINC30__ALL_AREAS_1969_2022.csv')

    def zip_instead(p: str) -> str:
        return p + ".zip"
    
    # Download the raw tables
    print(f"\n{Style.BRIGHT}{Fore.CYAN}--- DOWNLOADING RAW BEA TABLES ---{Style.RESET_ALL}")
    ran = False

    if not os.path.exists(yearly_income_p):
        print(f"\n{Fore.YELLOW}[__init__.py]: Downloading {Style.DIM}yearly_personal_income_county.zip{Style.NORMAL}...{Style.RESET_ALL}")
        os_specific_download("https://apps.bea.gov/regional/zip/CAINC1.zip", zip_instead(yearly_income_p))

        # unzip
        print(f"{Fore.LIGHTMAGENTA_EX}[__init__.py]: Unzipping {Style.DIM}yearly_personal_income_county.zip{Style.NORMAL}...{Style.RESET_ALL}")
        os_specific_unzip(zip_instead(yearly_income_p), os.path.dirname(yearly_income_b))

        # Rename
        os.rename(yearly_income_b, yearly_income_p)
        ran = True
    else:
        print(f"{Fore.GREEN}[__init__.py]: {Style.DIM}yearly_personal_income_county.zip{Style.NORMAL} already exists.{Style.RESET_ALL}")

    if not os.path.exists(yearly_gdp_p):
        print(f"\n{Fore.YELLOW}[__init__.py]: Downloading {Style.DIM}yearly_gdp_county.zip{Style.NORMAL}...{Style.RESET_ALL}")
        os_specific_download("https://apps.bea.gov/regional/zip/CAGDP1.zip", zip_instead(yearly_gdp_p))

        # unzip
        print(f"{Fore.LIGHTMAGENTA_EX}[__init__.py]: Unzipping {Style.DIM}yearly_gdp_county.zip{Style.NORMAL}...{Style.RESET_ALL}")
        os_specific_unzip(zip_instead(yearly_gdp_p), os.path.dirname(yearly_gdp_b))

        # Rename
        os.rename(yearly_gdp_b, yearly_gdp_p)
        ran = True
    else:
        print(f"{Fore.GREEN}[__init__.py]: {Style.DIM}yearly_gdp_county.zip{Style.NORMAL} already exists.{Style.RESET_ALL}")

    if not os.path.exists(yearly_employment_p):
        print(f"\n{Fore.YELLOW}[__init__.py]: Downloading {Style.DIM}yearly_employment_county.zip{Style.NORMAL}...{Style.RESET_ALL}")
        os_specific_download("https://apps.bea.gov/regional/zip/CAINC4.zip", zip_instead(yearly_employment_p))

        # unzip
        print(f"{Fore.LIGHTMAGENTA_EX}[__init__.py]: Unzipping {Style.DIM}yearly_employment_county.zip{Style.NORMAL}...{Style.RESET_ALL}")
        os_specific_unzip(zip_instead(yearly_employment_p), os.path.dirname(yearly_employment_b))

        # Rename
        os.rename(yearly_employment_b, yearly_employment_p)
        ran = True
    else:
        print(f"{Fore.GREEN}[__init__.py]: {Style.DIM}yearly_employment_county.zip{Style.NORMAL} already exists.{Style.RESET_ALL}")

    if not os.path.exists(yearly_more_county_p):
        print(f"\n{Fore.YELLOW}[__init__.py]: Downloading {Style.DIM}yearly_more_county_info.zip{Style.NORMAL}...{Style.RESET_ALL}")
        os_specific_download("https://apps.bea.gov/regional/zip/CAINC30.zip", zip_instead(yearly_more_county_p))

        # unzip
        print(f"{Fore.LIGHTMAGENTA_EX}[__init__.py]: Unzipping {Style.DIM}yearly_more_county_info.zip{Style.NORMAL}...{Style.RESET_ALL}")
        os_specific_unzip(zip_instead(yearly_more_county_p), os.path.dirname(yearly_more_county_b))

        # Rename
        os.rename(yearly_more_county_b, yearly_more_county_p)
        ran = True
    else:
        print(f"{Fore.GREEN}[__init__.py]: {Style.DIM}yearly_more_county_info.zip{Style.NORMAL} already exists.{Style.RESET_ALL}")

    if ran:
        # Remove all files in the folder that start with CAINC1
        for f in os.listdir(RAW_DATA_PATH):
            if f.startswith('CAINC1') or f.startswith('CAGDP1') or f.startswith('CAINC4') or f.startswith('CAINC30'):
                os.remove(os.path.join(RAW_DATA_PATH, f))

def fetch_helper_tables():
    fips_converter_p    = os.path.join(RAW_DATA_PATH, '..', 'census', 'fips_to_county.tsv')
    county_boundaries_p = os.path.join(RAW_DATA_PATH, '..', 'geolocation', 'county_boundaries.geojson')

    print(f"\n{Style.BRIGHT}{Fore.CYAN}--- DOWNLOADING HELPER TABLES ---{Style.RESET_ALL}")

    if not os.path.exists(fips_converter_p):
        print(f"\n{Fore.YELLOW}[__init__.py]: Downloading {Style.DIM}fips_to_county.tsv{Style.NORMAL}...{Style.RESET_ALL}")
        os_specific_download("https://raw.githubusercontent.com/ChuckConnell/articles/refs/heads/master/fips2county.tsv", fips_converter_p)

        print(f"{Fore.GREEN}[__init__.py]: {Style.NORMAL}Downloaded {Style.DIM}fips_to_county.tsv{Style.NORMAL}.{Style.RESET_ALL}")
    else:
        print(f"{Fore.GREEN}[__init__.py]: {Style.NORMAL}fips_to_county.tsv already exists.{Style.RESET_ALL}")

    if not os.path.exists(county_boundaries_p):
        print(f"\n{Fore.YELLOW}[__init__.py]: Downloading {Style.DIM}county_boundaries.geojson{Style.NORMAL}...{Style.RESET_ALL}")
        os_specific_download("https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/us-county-boundaries/exports/geojson?lang=en&timezone=America%2FNew_York", county_boundaries_p)

        print(f"{Fore.GREEN}[__init__.py]: {Style.NORMAL}Downloaded {Style.DIM}county_boundaries.geojson{Style.NORMAL}.{Style.RESET_ALL}")
    else:
        print(f"{Fore.GREEN}[__init__.py]: {Style.NORMAL}county_boundaries.geojson already exists.{Style.RESET_ALL}")

    print(f"{Fore.GREEN}[__init__.py]: {Style.NORMAL}All helper tables downloaded.{Style.RESET_ALL}")


fetch_patent_raw_data()
fetch_bea_raw_tables()
fetch_helper_tables()

print(f"\n{Style.BRIGHT}{Fore.GREEN}[__init__.py]: Initialization complete.{Style.RESET_ALL}")