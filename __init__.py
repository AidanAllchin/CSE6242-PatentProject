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
import os, sys, subprocess, json, re

# Install the required packages
if not os.path.exists("requirements.txt"):
    print("Error: requirements.txt not found. Please confirm git repository is cloned correctly and you are in the correct directory.")
    sys.exit(1)

confirm = input("\n\nThis script will install the required packages, create necessary directories, and download the data.\nPLEASE ENSURE YOU'RE USING YOUR DESIRED ENVIRONMENT...\nContinue? (y/n): ")
if confirm.lower() != "y":
    print("Exiting...")
    sys.exit(0)

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
        print(f"{Fore.GREEN}[init]: Created directory: {dir}{Style.RESET_ALL}")
    except PermissionError:
        print(f"{Fore.RED}[init]: {Style.DIM}Permission denied when creating directory: {dir}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}[init]: Attempting to create directory with sudo...{Style.RESET_ALL}")
        try:
            subprocess.run(['sudo', 'mkdir', '-p', dir], check=True)
            subprocess.run(['sudo', 'chmod', '777', dir], check=True)
            print(f"{Fore.GREEN}[init]: Successfully created directory with sudo.{Style.RESET_ALL}")
        except subprocess.CalledProcessError as e:
            print(f"{Fore.RED}[init]: Failed to create directory even with sudo: {e}{Style.RESET_ALL}")
            raise

# Create directories
if not os.path.exists("data"):
    ensure_directory_exists("data")
    ensure_directory_exists("data/patents")
elif not os.path.exists("data/patents"):
    ensure_directory_exists("data/patents")
if not os.path.exists("config"):
    ensure_directory_exists("config")

CONFIG_PATH = "config/config.json"

if not os.path.exists(CONFIG_PATH):
    print(f"{Fore.YELLOW}[init]: {Style.NORMAL}Creating config file...{Style.RESET_ALL}")
    with open(CONFIG_PATH, "w") as f:
        json.dump({"settings": {
        "desired_data_release": "2024-09-26" # We can modify this to get data from different releases
    }}, f)

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

desired_data_release = config["settings"]["desired_data_release"]

# Structure the 'YYYY-MM-DD' date format to YYMMDD
desired_data_release = desired_data_release.split("-")
desired_data_release = desired_data_release[0][2:] + desired_data_release[1] + desired_data_release[2]

# Download the data from the USPTO website
dir_contents = os.listdir("data")
re_match = re.compile(r"ipa\d{6}\.xml")

all_data_files = [f for f in dir_contents if re_match.match(f)]
contains_desired = any(desired_data_release in f for f in all_data_files)
print(f"{Style.BRIGHT}{Fore.CYAN}[init]: {Style.NORMAL}Found {len(all_data_files)} data files in data directory.{Style.RESET_ALL}")
print(f"{Style.BRIGHT}{Fore.LIGHTCYAN_EX}[init]: {Style.RESET_ALL}Desired data release: {desired_data_release}.")
name = f"ipa{desired_data_release}.zip"

if not contains_desired:
    print(f"{Fore.YELLOW}[init]: {Style.NORMAL}Data file not found in data directory.{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}{Style.BRIGHT}[init]: {Style.NORMAL}Downloading data from USPTO website...\n{Style.RESET_ALL}")
    link = f"https://bulkdata.uspto.gov/data/patent/application/redbook/fulltext/2024/{name}"
    
    sys_type = sys.platform
    if sys_type == "linux":
        subprocess.run(["wget", link, "-O", f"data/{name}"])
        print(f"\n{Style.BRIGHT}{Fore.LIGHTCYAN_EX}[init]: {Style.DIM}Downloaded data file: data/{name}. Extracting...{Style.RESET_ALL}\n")
        subprocess.run(["unzip", f"data/{name}", "-d", "data"])
        subprocess.run(["rm", f"data/{name}"])
    elif sys_type == "win32":
        subprocess.run(["curl", link, "-o", f"data/{name}"])
        print(f"\n{Style.BRIGHT}{Fore.LIGHTCYAN_EX}[init]: {Style.DIM}Downloaded data file: data/{name}. Extracting...{Style.RESET_ALL}\n")
        subprocess.run(["tar", "-xf", f"data/{name}", "-C", "data"])
        subprocess.run(["del", f"data/{name}"], shell=True)
    # mac doesn't have wget
    elif sys_type == "darwin":
        subprocess.run(["curl", link, "-o", f"data/{name}"])
        print(f"\n{Style.BRIGHT}{Fore.LIGHTCYAN_EX}[init]: {Style.DIM}Downloaded data file: data/{name}. Extracting...{Style.RESET_ALL}\n")
        subprocess.run(["unzip", f"data/{name}", "-d", "data"])
        subprocess.run(["rm", f"data/{name}"])
    else:
        print(f"{Fore.RED}[init]: {Style.NORMAL}Unsupported operating system: {sys_type}{Style.RESET_ALL}")
        print(f"{Fore.RED}[init]: {Style.NORMAL}Please download the data manually from the USPTO website: {link}{Style.RESET_ALL}")
        sys.exit(1)

    print(f"\n{Style.BRIGHT}{Fore.LIGHTCYAN_EX}[init]: {Style.DIM}Extracted data file: data/{name.replace('.zip', '.xml')}{Style.RESET_ALL}")
else:
    print(f"{Fore.GREEN}[init]: Data file already exists: data/{name.replace('.zip', '.xml')}{Style.RESET_ALL}")

print(f"{Style.BRIGHT}{Fore.GREEN}[init]:{Style.NORMAL} Initialization complete.{Style.RESET_ALL}")