#!/usr/bin/env python3
#-*- coding: utf-8 -*-
"""
Created: Tue Nov 12 2024
Author: Kailtyn Williams

Gathers relevant census data for each county we have patent data in, and 
organizes it by county FIPS code and year to the /data directory as a tsv.

NOTE: This file is un-usable for the required time period. The Census started
collecting county-level data in 2005, and the ACS started in 2009. This file
is only useful for 2009-2022, and will only generate TSVs for 2012 onward.
"""
import os, sys
from pathlib import Path

# ALL FILES IN THE PROJECT SHOULD BE IMPORTED RELATIVE TO THE PROJECT ROOT
# the # of `.parent`s should be adjusted based on the file's location
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import time
from census import Census
from tqdm import tqdm
from src.other.logging import PatentLogger


###############################################################################
#                               CONFIGURATION                                 #
###############################################################################


# Initialize logger
logger = PatentLogger.get_logger(__name__)

DATA_FOLDER    = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'census')
NAME_TO_FIPS   = os.path.join(DATA_FOLDER, 'fips_to_county.tsv')
CENSUS_API_KEY = '29e0fb5c56cb38bb77a95450557344c1e217e72f'


###############################################################################
#                               DATA CLEANING                                 #
###############################################################################


def fetch_education_data(c: Census, year: int) -> pd.DataFrame:
    """
    Fetch education attainment data by county.
    
    Variables:
    B15003_001E: Total population 25 years and over
    B15003_022E: Bachelor's degree
    B15003_023E: Master's degree
    B15003_024E: Professional school degree
    B15003_025E: Doctorate degree

    Args:
        c: Census object
        year: Year to fetch data for

    Returns:
        pd.DataFrame: DataFrame with education data
    """
    education_data = c.acs5.get(
        ("NAME",
         "B15003_001E",  # Total population 25+
         "B15003_022E",  # Bachelor's
         "B15003_023E",  # Master's
         "B15003_024E",  # Professional
         "B15003_025E"), # Doctorate
        {'for': 'county:*', 'in': 'state:*'},
        year=year
    )
    
    df = pd.DataFrame(education_data)
    
    # Calculate education score (0-1)
    # Weighted: Bachelor's (0.25), Master's (0.35), Professional (0.2), Doctorate (0.2)
    df['education_score'] = (
        (0.25 * df['B15003_022E'].astype(float) +
         0.35 * df['B15003_023E'].astype(float) +
         0.20 * df['B15003_024E'].astype(float) +
         0.20 * df['B15003_025E'].astype(float)
        ) / df['B15003_001E'].astype(float)
    )
    df = df.rename(columns={
        'B15003_001E': 'population_over_25'
    })
    df = df.drop(columns=['B15003_022E', 'B15003_023E', 'B15003_024E', 'B15003_025E'])
    
    return df

def fetch_demographic_data(c: Census, year: int) -> pd.DataFrame:
    """
    Fetch demographic and economic indicators by county.
    
    Variables:
    B19013_001E: Median household income
    B23025_005E: Unemployment count
    B23025_003E: Workers in labor force
    B01003_001E: Total population

    Args:
        c: Census object
        year: Year to fetch data for

    Returns:
        pd.DataFrame: DataFrame with demographic data
    """
    d = c.acs5.get(
        ("NAME",
         "B19013_001E",  # Median household income
         "B23025_005E",  # Unemployment
         "B23025_003E",  # In labor force
         "B01003_001E"), # Population
        {'for': 'county:*', 'in': 'state:*'},
        year=year
    )
    
    df = pd.DataFrame(d)
    
    # Calculate unemployment rate
    df['unemployment_rate'] = (
        df['B23025_005E'].astype(float) / 
        df['B23025_003E'].astype(float)
    )

    df = df.rename(columns={
        'B19013_001E': 'median_household_income'
    })
    df = df.drop(columns=['B23025_005E', 'B23025_003E'])
    
    return df

def collect_census_predictors(start_year: int = 2001, end_year: int = 2022):
    """
    Main function to collect all census data predictors.
    
    Args:
        start_year: First year to collect data for
        end_year: Last year to collect data for
    """
    c = Census(CENSUS_API_KEY)
    
    # Process each year
    for year in tqdm(range(start_year, end_year + 1), desc="Processing Census years"):
        try:
            # Fetch all data types
            education_df   = fetch_education_data(c, year)
            demographic_df = fetch_demographic_data(c, year)
            
            # Merge dataframes
            merged_df = pd.merge(
                education_df,
                demographic_df,
                on=['state', 'county', 'NAME']
            )
            
            # Create FIPS code
            merged_df['fips'] = merged_df['state'].str.zfill(2) + merged_df['county'].str.zfill(3)

            merged_df = merged_df.drop(columns=['state', 'county'])
            
            # Save to TSV
            output_path = os.path.join(DATA_FOLDER, f'census_predictors_{year}.tsv')
            merged_df.to_csv(output_path, sep='\t', index=False)
            
            #logger.info(f"Saved census data for {year}")
            
        except Exception as e:
            logger.error(f"Error processing year {year}: {str(e)}")
            continue
    
    logger.info("Finished processing all years")

if __name__ == '__main__':
    collect_census_predictors()
