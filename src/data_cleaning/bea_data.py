#!/usr/bin/env python3
#-*- coding: utf-8 -*-
"""
Created: Tue Nov 12 2024
Author: Aidan Allchin

Adds BEA data organized into year-based TSVs to the /data/bea directory.
"""
import os, sys
from pathlib import Path

# ALL FILES IN THE PROJECT SHOULD BE IMPORTED RELATIVE TO THE PROJECT ROOT
# the # of `.parent`s should be adjusted based on the file's location
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Dict
from src.other.logging import PatentLogger


###############################################################################
#                               CONFIGURATION                                 #
###############################################################################


# Initialize logger
logger = PatentLogger.get_logger(__name__)

# For access to data here, just do f"{DATA_FOLDER}/and then whatever file name you want"
DATA_FOLDER = os.path.join(os.path.dirname(__file__), '..', '..', 'data')

# Paths
employment_p = os.path.join(DATA_FOLDER, 'raw', 'yearly_employment_county.csv')
gdp_p        = os.path.join(DATA_FOLDER, 'raw', 'yearly_gdp_county.csv')
income_p     = os.path.join(DATA_FOLDER, 'raw', 'yearly_personal_income_county.csv')
more_info_p  = os.path.join(DATA_FOLDER, 'raw', 'yearly_more_county_info.csv')

DATE_CUTOFF  = 2001  # Only include data starting from new GDP calculation method


###############################################################################
#                               DATA CLEANING                                 #
###############################################################################

def gather_employment_df() -> Dict[int, pd.DataFrame]:
    """
    Reads employment data from a CSV file, processes it, and returns a dict of DataFrames, 
    each representing data for a specific year.

    The function performs the following steps:
    1. Reads the CSV file specified by the `employment_p` variable with 'latin1' encoding.
    2. Drops rows with missing 'GeoFIPS' or 'GeoName' values.
    3. Drops unnecessary columns: 'Region', 'TableName', 'LineCode', 'IndustryClassification', 'Unit'.
    4. Filters rows based on specific 'Description' values of interest.
    5. Creates a DataFrame for each year, with one row per county (FIPS) and columns based on 'Description' values.
    6. Renames columns to be more descriptive.
    7. Returns dict of DataFrames, keyed by year.

    Returns:
        Dict[int, pd.DataFrame]: A dictionary of DataFrames, each representing data for a specific year.
    """
    df = pd.read_csv(employment_p, encoding='latin1')
    df = df.dropna(subset=['GeoFIPS'])
    df = df.dropna(subset=['GeoName'])

    # Drop columns that are not needed
    df = df.drop(columns=['Region', 'TableName', 'LineCode', 'IndustryClassification', 'Unit'])

    # Drop description values we don't care about as predictors
    keep_vals = ['Per capita personal income (dollars) 4/', 'Total employment ', 'Population (persons) 3/', 'Equals: Net earnings by place of residence ']
    prior_len = len(df)
    df = df[df['Description'].isin(keep_vals)]
    logger.info(f"Dropped {prior_len - len(df)} rows that were not in the keep_vals list of Descriptions")

    # Create a df for every year with one row per county (FIPS)
    # Columns based on 'Description' value
    dfs = {}
    # Year range should be the min of all integer-castable columns to the max of all integer-castable columns
    col_max = max([int(col) for col in df.columns if col.isnumeric()])
    col_min = min([int(col) for col in df.columns if col.isnumeric()])
    col_min = max(col_min, DATE_CUTOFF)
    logger.info(f"Gathering data from {col_min} to {col_max}")
    for year in tqdm(range(col_min, col_max + 1), desc="Processing years"):
        year_df = df[['GeoFIPS', 'GeoName', 'Description', str(year)]]
        year_df = year_df.rename(columns={str(year): 'Value'})
        year_df['Year'] = year

        # Each value in description becomes a column, each FIPS code becomes a unique row
        year_df = year_df.pivot(index='GeoFIPS', columns='Description', values='Value').reset_index()

        # Rename the columns to be more descriptive
        year_df = year_df.rename(columns={
            #'Personal income (thousands of dollars) ': 'personal_income_thousands',
            'Per capita personal income (dollars) 4/': 'per_capita_income_dollars',
            'Total employment ': 'total_employment_count',
            'Population (persons) 3/': 'population_count',
            'Equals: Net earnings by place of residence ': 'residence_net_earnings_thousands'
        })

        # Ensure correct types
        year_df['per_capita_income_dollars'] = year_df['per_capita_income_dollars'].replace('(NA)', np.nan)
        year_df['per_capita_income_dollars'] = year_df['per_capita_income_dollars'].astype(float)
        year_df['total_employment_count'] = year_df['total_employment_count'].replace('(NA)', np.nan)
        year_df['total_employment_count'] = year_df['total_employment_count'].astype(float)
        year_df['population_count'] = year_df['population_count'].replace('(NA)', np.nan)
        year_df['population_count'] = year_df['population_count'].astype(float)
        year_df['residence_net_earnings_thousands'] = year_df['residence_net_earnings_thousands'].replace('(NA)', np.nan)
        year_df['residence_net_earnings_thousands'] = year_df['residence_net_earnings_thousands'].astype(float)

        # Make things per capita
        year_df['employment_per_capita'] = year_df['total_employment_count'] / year_df['population_count']
        year_df['earnings_per_capita'] = year_df['residence_net_earnings_thousands'] / year_df['population_count']
        year_df = year_df.drop(columns=['total_employment_count', 'residence_net_earnings_thousands'])

        year_df = year_df.reset_index(drop=True)
        dfs[year] = year_df
    
    return dfs

def gather_gdp_df() -> Dict[int, pd.DataFrame]:
    """
    Reads GDP data from a CSV file, processes it, and returns a dictionary of DataFrames, 
    each representing GDP data for a specific year.

    The function performs the following steps:
    1. Reads the CSV file containing GDP data.
    2. Drops rows with missing 'GeoFIPS' and 'GeoName' values.
    3. Drops unnecessary columns.
    4. Filters the data to keep only specific 'Description' values.
    5. Creates a DataFrame for each year, with one row per county (FIPS) and columns based on 'Description' values.
    6. Renames columns to be more descriptive.
    7. Returns a dictionary where keys are years and values are the corresponding DataFrames.

    Returns:
        Dict[int, pd.DataFrame]: A dictionary where keys are years and values are DataFrames containing GDP data for that year.
    """
    df = pd.read_csv(gdp_p, encoding='latin1')
    df = df.dropna(subset=['GeoFIPS'])
    df = df.dropna(subset=['GeoName'])

    # Drop columns we don't need
    df = df.drop(columns=['Region', 'TableName', 'LineCode', 'IndustryClassification', 'Unit'])

    # Drop description values we don't care about as predictors
    keep_vals = ['Real GDP (thousands of chained 2017 dollars) ']  # Only one value in this dataset we want
    prior_len = len(df)
    df = df[df['Description'].isin(keep_vals)]
    logger.info(f"Dropped {prior_len - len(df)} rows that were not in the keep_vals list of Descriptions")

    # Create a df for every year with one row per county (FIPS)
    # Columns based on 'Description' value
    dfs = {}
    # Year range should be the min of all integer-castable columns to the max of all integer-castable columns
    col_max = max([int(col) for col in df.columns if col.isnumeric()])
    col_min = min([int(col) for col in df.columns if col.isnumeric()])
    col_min = max(col_min, DATE_CUTOFF)
    logger.info(f"Gathering data from {col_min} to {col_max}")
    for year in tqdm(range(col_min, col_max + 1), desc="Processing years"):
        year_df = df[['GeoFIPS', 'GeoName', 'Description', str(year)]]
        year_df = year_df.rename(columns={str(year): 'Value'})
        year_df['Year'] = year

        # Each value in description becomes a column, each FIPS code becomes a unique row
        year_df = year_df.pivot(index='GeoFIPS', columns='Description', values='Value').reset_index()

        # Rename the columns to be more descriptive
        year_df = year_df.rename(columns={
            'Real GDP (thousands of chained 2017 dollars) ': 'real_gdp_thousands'
        })

        # Remove the index
        year_df = year_df.reset_index(drop=True)

        dfs[year] = year_df

    # Return the list of dataframes
    return dfs

def get_more_info_df() -> Dict[int, pd.DataFrame]:
    """
    Reads more info data from a CSV file, processes it, and returns a dict of DataFrames, 
    each representing data for a specific year.

    The function performs the following steps:
    1. Reads the CSV file specified by the `more_info_p` variable with 'latin1' encoding.
    2. Drops rows with missing 'GeoFIPS' or 'GeoName' values.
    3. Drops unnecessary columns: 'Region', 'TableName', 'LineCode', 'IndustryClassification', 'Unit'.
    4. Filters rows based on specific 'Description' values of interest.
    5. Creates a DataFrame for each year, with one row per county (FIPS) and columns based on 'Description' values.
    6. Renames columns to be more descriptive.
    7. Returns a dictionary of DataFrames, keyed by year.

    Returns:
        Dict[int, pd.DataFrame]: A dictionary of DataFrames, each representing data for a specific year.
    """
    df = pd.read_csv(more_info_p, encoding='latin1')
    df = df.dropna(subset=['GeoFIPS'])
    df = df.dropna(subset=['GeoName'])

    # Drop columns that are not needed
    df = df.drop(columns=['Region', 'TableName', 'LineCode', 'IndustryClassification', 'Unit'])

    # Drop description values we don't care about as predictors
    keep_vals = ['Average earnings per job (dollars) ']  # Only one value in this dataset we want
    prior_len = len(df)
    df = df[df['Description'].isin(keep_vals)]
    logger.info(f"Dropped {prior_len - len(df)} rows that were not in the keep_vals list of Descriptions")

    # Create a df for every year with one row per county (FIPS)
    # Columns based on 'Description' value
    dfs = {}
    # Year range should be the min of all integer-castable columns to the max of all integer-castable columns
    col_max = max([int(col) for col in df.columns if col.isnumeric()])
    col_min = min([int(col) for col in df.columns if col.isnumeric()])
    col_min = max(col_min, DATE_CUTOFF)
    logger.info(f"Gathering data from {col_min} to {col_max}")
    for year in tqdm(range(col_min, col_max + 1), desc="Processing years"):
        year_df = df[['GeoFIPS', 'GeoName', 'Description', str(year)]]
        year_df = year_df.rename(columns={str(year): 'Value'})
        year_df['Year'] = year

        # Each value in description becomes a column, each FIPS code becomes a unique row
        year_df = year_df.pivot(index='GeoFIPS', columns='Description', values='Value').reset_index()

        # Rename the columns to be more descriptive
        year_df = year_df.rename(columns={
            'Average earnings per job (dollars) ': 'average_earnings_per_job_dollars'
        })

        # Remove the index
        year_df = year_df.reset_index(drop=True)

        dfs[year] = year_df

    # Return the list of dataframes
    return dfs

def collect_bea_predictors():
    dest_path = os.path.join(DATA_FOLDER, 'bea')
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    
    employment_dfs = gather_employment_df()
    gdp_dfs        = gather_gdp_df()
    more_info_dfs  = get_more_info_df()

    # Merge all the dataframes together
    dfs = []
    logger.info("Merging dataframes together")
    for year in employment_dfs:
        df = employment_dfs[year]
        df = df.merge(gdp_dfs[year], on='GeoFIPS', how='left')
        df = df.merge(more_info_dfs[year], on='GeoFIPS', how='left')

        # Remove state-level aggregates for county analysis
        df = df[~df['GeoFIPS'].str.strip("\"").str.endswith('000')]

        dfs.append(df)

    # Save each year's data to a file
    for year, df in zip(employment_dfs, dfs):
        df['GeoFIPS'] = df['GeoFIPS'].replace('\"', '', regex=True)
        df.to_csv(os.path.join(dest_path, f'bea_predictors_{year}.tsv'), sep='\t', index=False)

    logger.info("Generated BEA data for all years")

if __name__ == '__main__':
    collect_bea_predictors()

