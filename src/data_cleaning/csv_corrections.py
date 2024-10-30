#!/usr/bin/env python3
#-*- coding: utf-8 -*-
"""
Created: Tue Oct 29 2024
Author: Aidan Allchin

Data exploration and correcting for the patent data generated in Jupyter.
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
from src.other.helpers import log, local_filename
from src.objects.patent import Patent

def load():
    # Load the patent data
    REID_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'patent_locations_V1.csv')

    st = time.time()
    d = pd.read_csv(REID_PATH)
    log(f"Loading the patent data took {time.time() - st:.2f} seconds.\n")

    return d

def consolidate_patent_data(df):
    """
    Consolidates patent data by combining multiple assignees into single rows.
    
    Args:
        df (pandas.DataFrame): DataFrame containing patent data with duplicate rows for multiple assignees
    
    Returns:
        pandas.DataFrame: Consolidated DataFrame with one row per patent
    """
    # Columns we're going to keep
    patent_level_cols = [
        'patent_id', 'patent_type', 'patent_date', 'patent_title',
        'wipo_kind', 'num_claims', 'withdrawn', 'filename'
    ]
    
    # Assignee-related columns
    assignee_cols = [
        'assignee_sequence', 'assignee_id', 'raw_assignee_individual_name_first',
        'raw_assignee_individual_name_last', 'raw_assignee_organization',
        'assignee_type', 'rawlocation_id', 'location_id', 'raw_city',
        'raw_state', 'raw_country'
    ]

    no_fname = 0
    no_lname = 0
    no_org   = 0
    
    # Combiniong assignee information
    def combine_assignee_info(group):
        nonlocal no_fname, no_lname, no_org

        assignees = []
        for _, row in group[assignee_cols].iterrows():
            fname = row['raw_assignee_individual_name_first']
            if fname == 'Not listed':
                no_fname += 1
                fname = None
            
            lname = row['raw_assignee_individual_name_last']
            if lname == 'Not listed':
                no_lname += 1
                lname = None

            org = row['raw_assignee_organization']
            if org == 'Not listed':
                no_org += 1
                org = None

            assignee_info = {
                'sequence': row['assignee_sequence'],
                'id': row['assignee_id'],
                #'first_name': fname,
                #'last_name': lname,
                'organization': org,
                'type': row['assignee_type'],
                'location': {
                    'id': row['rawlocation_id'],
                    'location_id': row['location_id'],
                    'city': row['raw_city'],
                    'state': row['raw_state'],
                    'country': row['raw_country']
                }
            }
            assignees.append(assignee_info)
        return assignees
    
    # Then all we gotta do is group by patent and combine assignee information
    result = df.groupby(patent_level_cols).apply(combine_assignee_info).reset_index()
    result.columns = patent_level_cols + ['assignees']

    log(f"Number of assignees with missing first names: {no_fname}")
    log(f"Number of assignees with missing last names: {no_lname}")
    log(f"Number of assignees with missing organizations: {no_org}")

    log(f"\nNot including first and last names in the assignee information, as {max(no_fname, no_lname) / df.shape[0]:.2%} of assignees are missing one or both.")
    
    return result

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the dataframe by removing unnecessary columns and rows with missing
    patent_ids.

    Args:
        df (pd.DataFrame): The dataframe to clean

    Returns:
        pd.DataFrame: The cleaned dataframe
    """
    df = df.replace(r'\t', ' ', regex=True)
    df = df.replace(r'\n', ' ', regex=True)

    explore(df)

    log(f"\nConsolidating assignee information to maintain 1 row per patent.")
    st = time.time()
    df = consolidate_patent_data(df)
    log(f"Consolidation took {time.time() - st:.2f} seconds.")

    # Randomly sample 1000 rows from the full dataframe and save them to a .tsv
    df.sample(n=1000).to_csv(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'sample_patents.tsv'), sep='\t', index=False)

    # Drop rows with missing patent_ids
    none_patent_id = df['patent_id'].isnull().sum()
    if none_patent_id > 0:
        log(f"Dropping {none_patent_id} rows with missing patent_ids.")
        df = df.dropna(subset=['patent_id'])
    
    log(f"num_claims varies from {df['num_claims'].min()} to {df['num_claims'].max()}, with {df['num_claims'].isnull().sum()} missing values.")
    log(f"withdrawn varies from {df['withdrawn'].min()} to {df['withdrawn'].max()}, with {df['withdrawn'].isnull().sum()} missing values.")

    # Drop non utility patent types
    prior_len = df.shape[0]
    df = df[df['patent_type'] == 'utility']
    log(f"Dropped {prior_len - df.shape[0]} rows with non-utility patent types.")

    # Save the cleaned dataframe
    df.to_csv(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'cleaned_patents.tsv'), sep='\t', index=False)

    return df

def explore(df: pd.DataFrame):
    """
    Counts how many duplicate patent_ids there are, and prints the duplicates.
    Compiles a .tsv of all unique city, country, state values.
    Randomly samples 1000 rows from the full dataframe and saves them to a .tsv

    Args:
        df (pandas.DataFrame): DataFrame containing patent data
    """
    # Count the number of duplicate patent_ids
    unique_patent_ids = df['patent_id'].nunique()
    duplicated_ids = df.duplicated(subset='patent_id').sum()
    log(f"Number of duplicate patent_ids: {duplicated_ids} / {unique_patent_ids} = {(duplicated_ids / unique_patent_ids):.2%}")

    # Print the duplicate patent_ids
    #log("Duplicate patent_ids:")
    #print(df[df.duplicated(subset='patent_id', keep=False)].sort_values(by='patent_id'))
    # Save the df of duplicates
    df[df.duplicated(subset='patent_id', keep=False)].sort_values(by='patent_id').to_csv(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'reid_duplicates.tsv'), sep='\t', index=False)

    # Compile a .tsv of all unique city, country, state values
    unique_locations = df[['raw_city', 'raw_country', 'raw_state']].drop_duplicates()
    unique_locations.to_csv(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'geolocation', 'unique_locations.tsv'), sep='\t', index=False)

    log(f"Number of unique locations: {unique_locations.shape[0]}")

    # Determine # of patents within US
    us_patents = df[df['raw_country'] == 'US']
    log(f"Number of patents within the US: {us_patents.shape[0]}")

    # Determine # of patents since 2001 within the US
    us_patents_2001 = us_patents[us_patents['patent_date'] >= '2001-01-01']
    log(f"Number of patents within the US since 2001: {us_patents_2001.shape[0]}")

    log(f"patent_type has {df['patent_type'].isnull().sum()} missing values:")
    print(df['patent_type'].value_counts())

    log(f"wipo_kind has {df['wipo_kind'].isnull().sum()} missing values:")
    print(df['wipo_kind'].value_counts())


if __name__ == "__main__":
    patents = load()
    patents = clean(patents)
    #print(patents.head())
    #print(patents.columns)
    #print(patents.dtypes)
    #print(patents.describe())
    #print(patents.info())
    #print(patents['patent_id'].nunique())
    #print(patents['patent_id'].value_counts())

