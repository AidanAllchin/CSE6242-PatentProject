#!/usr/bin/env python3
#-*- coding: utf-8 -*-
"""
Created: Wed Oct 30 2024
Author: Aidan Allchin

This is a script version of the Jupyter notebook designed for automated data
download, merging, and cleaning. This is some ugly ass code, but she does the 
job.
"""
import os, sys
from pathlib import Path

# ALL FILES IN THE PROJECT SHOULD BE IMPORTED RELATIVE TO THE PROJECT ROOT
# the # of `.parent`s should be adjusted based on the file's location
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# Below the subprocess line assumes __init__.py has been run
import pandas as pd
import time
import warnings
from colorama import Fore, Style
from src.other.helpers import log, local_filename, completion_time

# Shut up
warnings.filterwarnings("ignore")


dest_path = os.path.join(os.getcwd(), 'data', 'processed_patents.tsv')
if os.path.exists(dest_path):
    i = input(f"Looks like this has already been run: {Style.DIM}{local_filename(dest_path)}{Style.NORMAL} already exists. Overwrite? (y/n): ")
    if i.lower() != 'y':
        log("Aborting the initial cleaning steps.", level="WARNING")
        sys.exit()


###############################################################################
#                                                                             #
#                                   Loading                                   #
#                                                                             #
###############################################################################


RUNTIME = 180 # estimating runtime in seconds

start = time.time()
log(f"\nGenerating clean patent data. Estimated completion time: {completion_time(RUNTIME)}", color=Fore.CYAN, color_full=True)


# Paths (modified to be relative to the current working directory)
patents_g_p   = os.path.join(os.getcwd(), 'data', 'raw', 'g_patent.tsv')
applicant_g_p = os.path.join(os.getcwd(), 'data', 'raw', 'g_inventor_not_disambiguated.tsv')
location_g_p  = os.path.join(os.getcwd(), 'data', 'raw', 'g_location_not_disambiguated.tsv')
assignee_g_p  = os.path.join(os.getcwd(), 'data', 'raw', 'g_assignee_not_disambiguated.tsv')
wipo_g_p      = os.path.join(os.getcwd(), 'data', 'raw', 'g_wipo_technology.tsv')

# Dataload (~60s)
#patent information
patents_g = pd.read_csv(patents_g_p, sep='\t')
print(f"  > Loaded table {Style.DIM}patents_g{Style.NORMAL} ({os.path.getsize(patents_g_p) / 1024**3:.2f}MB)")

#applicant information (include multiple per patent doing seq = 0 should be primary)
applicant_g = pd.read_csv(applicant_g_p, sep='\t')
print(f"  > Loaded table {Style.DIM}applicant_g{Style.NORMAL} ({os.path.getsize(applicant_g_p) / 1024**3:.2f}MB)")

# raw location not disambiguated (e.g, no coordinates)
location_g = pd.read_csv(location_g_p, sep='\t')
print(f"  > Loaded table {Style.DIM}location_g{Style.NORMAL} ({os.path.getsize(location_g_p) / 1024**3:.2f}MB)")

#get organization
assignee_g = pd.read_csv(assignee_g_p, sep='\t')
print(f"  > Loaded table {Style.DIM}assignee_g{Style.NORMAL} ({os.path.getsize(assignee_g_p) / 1024**3:.2f}MB)")

#the sector and type of patent
wipo_g = pd.read_csv(wipo_g_p, sep='\t')
print(f"  > Loaded table {Style.DIM}wipo_g{Style.NORMAL} ({os.path.getsize(wipo_g_p) / 1024**3:.2f}MB)")

log(f"Tables loaded to memory in {time.time() - start:.2f} seconds.", color=Fore.LIGHTBLUE_EX)


###############################################################################
#                                                                             #
#                               Initial Clean                                 #
#                                                                             #
###############################################################################


def simplify_applicants():
    global applicant_g
    # Simplifying inventor information to just primary inventor (~10s)
    # Getting count of non-unique patent_id rows
    non_uniques = applicant_g[applicant_g.duplicated(subset='patent_id', keep=False)]
    #print(f"{Style.BRIGHT}Duplicated patent_ids are duplicated because each patent/inventor combo has it's own row:{Style.RESET_ALL}")
    log(f"Number of non-unique patent_id rows: {len(non_uniques)}.")

    just_dupes = non_uniques[non_uniques.duplicated(subset='patent_id', keep='first')]
    num_unique_patents = len(applicant_g['patent_id'].unique())
    print(f"  > Removing all duplicated patent_ids will result in {len(just_dupes)} fewer rows, but we will retain {num_unique_patents} patents (this doesn't change).")

    # Value counts of inventor sequence for full data
    #print(f"\n{Style.BRIGHT}Value counts of inventor_sequence for the full data:{Style.RESET_ALL}")
    #print(applicant_g['inventor_sequence'].value_counts().sort_index())

    # Drop all rows with inventor_sequence > 0
    applicant_g = applicant_g[applicant_g['inventor_sequence'] == 0].drop(columns=['inventor_sequence', 'deceased_flag'])
    print(f"  > {Style.DIM}applicant_g{Style.NORMAL} now has {len(applicant_g)} rows, matching our patent count.")

def simplify_assignees():
    global assignee_g
    # Simplifying assignee information to just primary assignee (~10s)
    # Getting count of non-unique patent_id rows
    non_uniques = assignee_g[assignee_g.duplicated(subset='patent_id', keep=False)]
    #print(f"\n{Style.BRIGHT}Duplicated patent_ids are duplicated because each patent/assignee combo has it's own row:{Style.RESET_ALL}")

    just_dupes = non_uniques[non_uniques.duplicated(subset='patent_id', keep='first')]
    num_unique_patents = len(assignee_g['patent_id'].unique())
    print(f"  > Removing all duplicated patent_ids will result in {len(just_dupes)} fewer rows, but we will retain {num_unique_patents} patents.")

    # Value counts of assignee seq for full data
    #print(f"\n{Style.BRIGHT}Value counts of assignee_sequence for the full data:{Style.RESET_ALL}")
    #print(assignee_g['assignee_sequence'].value_counts().sort_index())

    # Drop all rows with assignee_sequence > 0
    assignee_g = assignee_g[assignee_g['assignee_sequence'] == 0].drop(columns=['assignee_sequence', 'assignee_type'])
    print(f"  > {Style.DIM}assignee_g{Style.NORMAL} now has {len(assignee_g)} rows. Note that this {Style.BRIGHT}doesn't{Style.NORMAL} match our patent count, as not every patent has an assignee.")

simplify_applicants()
simplify_assignees()


###############################################################################
#                                                                             #
#                                  Merging                                    #
#                                                                             #
###############################################################################

def convert_ids():
    # Assure all ids are string type is there is no issue when merging
    patents_g.loc[:, 'patent_id']       = patents_g['patent_id'].astype('string')
    applicant_g.loc[:, 'patent_id']     = applicant_g['patent_id'].astype('string')
    assignee_g.loc[:, 'patent_id']      = assignee_g['patent_id'].astype('string')
    wipo_g.loc[:, 'patent_id']          = wipo_g['patent_id'].astype('string')
    location_g.loc[:, 'rawlocation_id'] = location_g['rawlocation_id'].astype('string')
    # ignore the warning

convert_ids()

###
# Assignee Merging on Location
###

# Start the timer
assignee_merge_start = time.time()
log(f"\nMerging assignee data with location data. Estimated completion time: {completion_time(19)}", color=Fore.LIGHTCYAN_EX)

# Simplifying the table to just what we need (very few first/last names)
assignee_g = assignee_g[['patent_id', 'raw_assignee_organization', 'rawlocation_id']]
assignee_g.loc[:, 'patent_id'] = assignee_g['patent_id'].astype('string')
assignee_g.loc[:, 'rawlocation_id'] = assignee_g['rawlocation_id'].astype('string')

# Merge assignee_g with locations to get assignee_city, assignee_state, assignee_country (~20s)
assignee_g = pd.merge(assignee_g, location_g, on="rawlocation_id", how="left")
assignee_g = assignee_g.rename(columns={
    'raw_city': 'assignee_city', 
    'raw_state': 'assignee_state', 
    'raw_country': 'assignee_country', 
    'raw_assignee_organization': 'assignee',
    'location_id': 'assignee_location_id'
})
assignee_g.drop(columns=['rawlocation_id'], inplace=True)

print(f"  > Assignee data merged with location data in {time.time() - assignee_merge_start:.2f} seconds.")


###
# Applicant Merging on Location
###

# Start the timer
applicant_merge_start = time.time()
log(f"\nMerging applicant data with location data. Estimated completion time: {completion_time(20)}", color=Fore.LIGHTCYAN_EX)

# Simplifying the table to just what we need
applicant_g = applicant_g.drop(columns=['inventor_id'])
applicant_g.loc[:, 'patent_id'] = applicant_g['patent_id'].astype('string')
applicant_g.loc[:, 'rawlocation_id'] = applicant_g['rawlocation_id'].astype('string')

# Merge applicant_g with locations to get inventor_city, inventor_state, inventor_country (~20s)
applicant_g = pd.merge(applicant_g, location_g, on="rawlocation_id", how="left")
applicant_g = applicant_g.rename(columns={
    'raw_city': 'inventor_city', 
    'raw_state': 'inventor_state', 
    'raw_country': 'inventor_country',
    'raw_inventor_name_first': 'inventor_firstname',
    'raw_inventor_name_last': 'inventor_lastname',
    'location_id': 'inventor_location_id'
})
applicant_g.drop(columns=['rawlocation_id'], inplace=True)

print(f"  > Applicant data merged with location data in {time.time() - applicant_merge_start:.2f} seconds.")


# Now we merge both of those together based on patent_id
# Start timer
locations_patents_st = time.time()
log(f"\nMerging assignee and applicant data. Estimated completion time: {completion_time(8)}", color=Fore.LIGHTCYAN_EX)
locations_df = pd.merge(applicant_g, assignee_g, on = 'patent_id', how = 'left')

print(f"  > Merged assignee and applicant data on patent_id in {time.time() - locations_patents_st:.2f}s.")

# Almost done combining (~12s)
# Start timer
merge_start = time.time()
log(f"\nMerging location and patent data together. Estimated completion time: {completion_time(12)}", color=Fore.LIGHTCYAN_EX)
df_no_wipo = pd.merge(patents_g, locations_df, on='patent_id', how='left')
df_no_wipo = df_no_wipo[df_no_wipo['patent_type'] == "utility"].drop(columns=['patent_type'])

print(f"  > Location and patent data merged in {time.time() - merge_start:.2f} seconds.")

# Explore if there are any state or city information for rows missing countries for either assignee or inventor
def explore_missing_locs(colprefix: str):
    missing_countries = df_no_wipo[df_no_wipo[f'{colprefix}_country'].isna()]
    non_na = missing_countries[missing_countries[f'{colprefix}_state'].notna() | missing_countries[f'{colprefix}_city'].notna()]

    print(f"  > Number of rows with missing country for {colprefix} but non-missing state or city: {len(non_na)}")
    if len(missing_countries) != 0:
        print(f"  > This is {len(non_na) / len(missing_countries) * 100:.2f}% of the rows with missing countries, and {len(non_na) / len(df_no_wipo) * 100:.2f}% of the total rows.")
    print(f"  > This is too few to be useful, so we're dropping these.")

log(f"Removing patents with improper location information...", color=Fore.LIGHTCYAN_EX)
explore_missing_locs('assignee')
explore_missing_locs('inventor')

df_no_wipo = df_no_wipo.dropna(subset=['assignee_country'])
df_no_wipo = df_no_wipo.dropna(subset=['inventor_country'])
df_no_wipo = df_no_wipo.dropna(subset=['inventor_city', 'inventor_state']) # Gotta do this too sadly

print(f"\n  > {Style.BRIGHT}Current length of the dataframe (# of unique patents with inventor and assignee locations) = {len(df_no_wipo)}.{Style.RESET_ALL}")

# Start timer
wipo_merge_start = time.time()
log(f"\nMerging WIPO data with patent data. Estimated completion time: {completion_time(12)}", color=Fore.LIGHTCYAN_EX)

# Merge to get WIPO type for each patent (~10s)
df = pd.merge(df_no_wipo, wipo_g, on='patent_id', how='left')

# Drop the few remaining columns we don't need
df = df.drop(columns=['wipo_field_sequence', 'wipo_field_id', 'wipo_kind'])

# Sort by date and drop all before 2001-01-01
df['patent_date'] = pd.to_datetime(df['patent_date'])
df = df[df['patent_date'] >= '2001-01-01'].sort_values('patent_date')

# Also (forgot this earlier) drop all patents not within the US
# TODO: I'm assuming we're using inventor for predictors and not assignee. Revisit.
df = df[df['inventor_country'] == 'US']
#print(df['inventor_country'].value_counts())
duplicates = df[df.duplicated(subset='patent_id', keep=False)]
print(f"  > Number of duplicated patent_ids: {Style.DIM}{len(duplicates)}{Style.RESET_ALL}")

#print(f"\n{Style.BRIGHT}Some of the duplicated patent_ids:{Style.RESET_ALL}")
#print(duplicates.head(10))
print(f"  > Number of unique patent_ids: {Style.DIM}{len(df['patent_id'].unique())}{Style.RESET_ALL}")
print(f"  > Number of null {Style.DIM}wipo_field_title{Style.NORMAL} values: {Style.DIM}{df['wipo_field_title'].isna().sum()}{Style.RESET_ALL}. Dropping...")
df = df.dropna(subset=['wipo_field_title'])

def comma_separate_fields():
    """
    Instead of having multiple rows for every `wipo_field_title` value, these
    should be comma separated in a single cell in list form.
    """
    global df
    combining_st = time.time()
    df['wipo_field_title'] = df.groupby('patent_id')['wipo_field_title'].transform(lambda x: str(list(x)))
    df = df.drop_duplicates(subset='patent_id')
    print(f"  > Combined all {Style.DIM}wipo_field_title{Style.NORMAL} values for each patent into a single cell in {time.time() - combining_st:.2f}s.")

log(f"\nMerging separate WIPO field titles for each patent. Estimated completion time: {completion_time(70)}", color=Fore.LIGHTCYAN_EX)
comma_separate_fields()

print(f"  > Number of unique assignee organizations: {Style.DIM}{len(df['assignee'].unique())}{Style.RESET_ALL}")
print(f"  > Number of non-US assignee countries: {Style.DIM}{len(df[df['assignee_country'] != 'US'])}{Style.RESET_ALL}")

print(f"  > WIPO data merged with patent data in {time.time() - wipo_merge_start:.2f} seconds.")

print(f"\n\n{Style.BRIGHT}{Fore.MAGENTA}This leaves us with a total of {len(df)} patents in the US with full information since January 1st, 2001.{Style.RESET_ALL}")

def safe_format():
    """
    Iterate through the columns of `df` and 
    .replace all '\t' in all string columns
    """
    for col in df.columns:
        if isinstance(df[col].dtype, object):
            df[col] = df[col].str.replace('\t', ' ')

#safe_format() # This isn't working yet

df.to_csv(dest_path, sep='\t', index=False)

log(f"\nData saved to {local_filename(dest_path)}.", color=Fore.LIGHTGREEN_EX)
log(f"Total runtime: {time.time() - start:.2f} seconds.")

