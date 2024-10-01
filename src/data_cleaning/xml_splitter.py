#!/usr/bin/env python3
#-*- coding: utf-8 -*-
"""
Created: Tue Oct 01 2024
Author: Aidan Allchin

This script is used to split the XML file containing all patent applications
into individual files, one for each patent application. The script reads the
XML file line by line and saves each patent application to a separate file in
the `data/patents` directory. The patent applications are identified by the
`<doc-number>` tag in the XML file.

This is specifically designed for the USPTO patent XML file located at:
https://developer.uspto.gov/product/patent-application-full-text-dataxml
"""
import os, sys
from pathlib import Path

# ALL FILES IN THE PROJECT SHOULD BE IMPORTED RELATIVE TO THE PROJECT ROOT
# the # of `.parent`s should be adjusted based on the file's location
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from colorama import Fore, Style
from typing import List
import re
import time
from tqdm import tqdm
import json


###############################################################################
#                               CONFIGURATION                                 #
###############################################################################


# Directories
PATENTS_DIRECTORY = os.path.join(project_root, 'data', 'patents')
CONFIG_PATH = os.path.join(project_root, 'config', 'config.json')
with open(CONFIG_PATH, 'r') as f:
    config = json.load(f)
desired_data_release = config["settings"]["desired_data_release"]

# Structure the 'YYYY-MM-DD' date format to YYMMDD
desired_data_release = desired_data_release.split("-")
desired_data_release = desired_data_release[0][2:] + desired_data_release[1] + desired_data_release[2]

# File to split
FILE_NAME = f"ipa{desired_data_release}.xml"


###############################################################################
#                               XML SPLITTER                                  #
###############################################################################


def extract_patent_id(line):
    """
    Extracts the patent ID from a line of XML, if available.
    Adjust the pattern to match the structure of your XML file.

    Args:
        line (str): A line from the XML file.

    Returns:
        str: The extracted patent ID or None if not found.
    """
    # Example pattern: <doc-number>12345678</doc-number>
    match = re.search(r'<doc-number>(\d+)</doc-number>', line)
    if match:
        return match.group(1)
    return None

def save_patent_to_file(document_lines: List[str], patent_id: str) -> int:
    """
    Saves the collected lines of a patent document to a file, skipping empty 
    documents and those without a patent ID (always seems to be empty).

    Args:
        document_lines (list): The lines of the patent XML document.
        patent_id (str): The unique patent ID to use for the file name.

    Returns:
        int: The size of the saved file in KB.
    """
    if patent_id is None:
        # Use a fallback name if no patent ID was found
        # patent_id = f"unknown_{os.urandom(8).hex()}"
        return # Just skip these actually, they all end up being empty

    output_file_path = os.path.join(PATENTS_DIRECTORY, f"patent_{patent_id}.xml")

    if not document_lines or document_lines == ['<?xml version="1.0" encoding="UTF-8"?>']:
        # Skip empty documents
        return
    
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        output_file.writelines(document_lines)

    return os.path.getsize(output_file_path) / 1024 # Kilobytes

def split_xml_by_patent(file_name: str):
    """
    Splits an XML file by patent application. Each patent application is saved
    in its own file in the `patents` directory.

    Args:
        file_name (str): The name of the file to split. Should literally be the 
            name of the file, not the path to the file. Needs to be in the `data`
            directory.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(PATENTS_DIRECTORY, exist_ok=True)

    file_path = os.path.join(project_root, 'data', file_name)

    current_document = []
    inside_document = False
    patent_id = None
    patent_count = 0
    sizes = []
    times = []
    overall_start = time.time()
    doc_start     = time.time()
    
    print(f"{Style.BRIGHT}{Fore.LIGHTMAGENTA_EX}[XML Splitter]: {Style.NORMAL}Loading {file_name} of size {os.path.getsize(file_path) / 1024 / 1024:.2f} MB...{Style.RESET_ALL}")

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in tqdm(file, desc="Processing XML", unit="lines", unit_scale=True):
            # Start of a new patent document
            if '<?xml version="1.0"' in line or '<!DOCTYPE us-patent-application' in line:
                if inside_document:
                    # If already inside a document, close it and save
                    s = save_patent_to_file(current_document, patent_id)
                    if s: sizes.append(s)
                    patent_count += 1
                    current_document = []  # Clear memory for next document
                    times.append(time.time() - doc_start)

                # Start new document
                inside_document = True
                current_document.append(line)
                patent_id = None  # Reset the patent ID for the new document
                doc_start = time.time()

            else:
                if inside_document:
                    current_document.append(line)

                # Try to extract the patent ID to store with the document
                if patent_id is None:
                    patent_id = extract_patent_id(line)

            # End of a patent document
            if inside_document and '</us-patent-application>' in line:
                s = save_patent_to_file(current_document, patent_id)
                if s: sizes.append(s)

                patent_count += 1
                current_document = []  # Clear memory after saving
                inside_document = False
                times.append(time.time() - doc_start)

        # Handle any remaining document that wasn't closed
        if current_document:
            s = save_patent_to_file(current_document, patent_id)
            if s: sizes.append(s)
            patent_count += 1
            times.append(time.time() - doc_start)

    overall_time = time.time() - overall_start
    print(f"{Style.BRIGHT}{Fore.LIGHTMAGENTA_EX}[XML Splitter]: {Style.RESET_ALL}Split {patent_count} patent documents from {file_name}.\n\tAverage size: {sum(sizes) / len(sizes):.2f} KB. Total time: {overall_time:.2f}s. Average time: {sum(times) / len(times) * 1000:.2f}ms.")
    num_in_directory = len(os.listdir(PATENTS_DIRECTORY))
    printable_dir = PATENTS_DIRECTORY.replace(str(project_root), '')
    print(f"\n{Style.BRIGHT}{Fore.MAGENTA}[XML Splitter]: Saved {num_in_directory} non-empty patent documents to {printable_dir}.{Style.RESET_ALL}")


###############################################################################
#                                   MAIN                                      #
###############################################################################


if __name__ == '__main__':
    os.makedirs(PATENTS_DIRECTORY.replace('/patents', ''), exist_ok=True)
    if not os.path.exists(os.path.join(project_root, 'data', FILE_NAME)):
        print(f"{Style.BRIGHT}{Fore.RED}[XML Splitter]: {Style.NORMAL}File {FILE_NAME} not found in `data` directory.\n\tPlease run {Style.DIM}__init.py{Style.NORMAL} before trying again.{Style.RESET_ALL}")
        sys.exit(1)
    
    split_xml_by_patent(FILE_NAME)