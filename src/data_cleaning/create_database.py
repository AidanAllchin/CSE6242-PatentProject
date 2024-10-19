#!/usr/bin/env python3
#-*- coding: utf-8 -*-
"""
Created: Tue Oct 17 2024
Author: Aidan Allchin

This script is used to create a SQLite database and insert patent data into it.
Can be run individually but will be called explicitly by `main.py`.
"""
import os, sys
from pathlib import Path

# ALL FILES IN THE PROJECT SHOULD BE IMPORTED RELATIVE TO THE PROJECT ROOT
# the # of `.parent`s should be adjusted based on the file's location
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from colorama import Fore, Style
import sqlite3
from tqdm import tqdm
import time
from src.objects.patent import Patent
from src.data_cleaning.patent_parsing import collect_patent_objects
from src.data_cleaning.patent_cleanup import add_coordinates, create_city_coordinates_cache
from src.other.helpers import log

# Database path
DATABASE_PATH = os.path.join(project_root, 'data', 'patents.db')


###############################################################################
#                              SETUP DATABASE                                 #
###############################################################################


# Wipe the database if it already exists
if os.path.exists(DATABASE_PATH):
    log("Database already exists. Deleting...", color=Fore.YELLOW)
    os.remove(DATABASE_PATH)

# Collect patent objects
patents = collect_patent_objects()

# Add location of each assignee and inventor to the patent object
failed = True
while failed:
    try:
        create_city_coordinates_cache(patents)
        failed = False
    except Exception as e:
        print(e)
        log("Failed to create city coordinates cache. Retrying...", color=Fore.RED)
        time.sleep(10)

patents = add_coordinates(patents)

# Print example patent
log("\nExample patent:", color=Fore.MAGENTA)
print(patents[0])

print('-' * 80 + '\n')
log("\nSetting up database...", color=Fore.MAGENTA, color_full=True)

# Create a connection to the database
conn = sqlite3.connect(DATABASE_PATH)
cursor = conn.cursor()

# Create the patents table
cursor.execute('''
CREATE TABLE IF NOT EXISTS patents (
    patent_id INTEGER PRIMARY KEY,
    patent_name TEXT,
    assignee_info TEXT,
    inventor_info TEXT,
    dates_info TEXT,
    classifications TEXT,
    application_number INTEGER,
    document_kind TEXT,
    application_type TEXT,
    abstract TEXT,
    claims TEXT,
    description TEXT
)
''')

# Commit the changes and close the connection
conn.commit()
conn.close()

log("Database and table created successfully.", color=Fore.GREEN)

# Reopen the connection for inserting data
conn = sqlite3.connect(DATABASE_PATH)

# Insert patent objects into the database
print()
for patent in tqdm(patents, desc="Inserting patents"):
    patent.to_sqlite(conn)

# Close the connection
conn.close()

log(f"Inserted {len(patents)} patents into the database.", color=Fore.GREEN)
log(f"Database is size {os.path.getsize(DATABASE_PATH) / 1024 / 1024:.2f} MB.")

