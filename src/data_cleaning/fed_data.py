#!/usr/bin/env python3
#-*- coding: utf-8 -*-
"""
Created: Sun Nov 12 2024
Author: Kaitlyn Williams

Gathers relevant fed data for each county we have patent data in, and 
saves each year to the /data/fed directory as a tsv.
"""
import os, sys
from pathlib import Path

# ALL FILES IN THE PROJECT SHOULD BE IMPORTED RELATIVE TO THE PROJECT ROOT
# the # of `.parent`s should be adjusted based on the file's location
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import time
from fredapi import Fred
import json
import random
from tqdm import tqdm
from typing import Dict, List
from src.other.logging import PatentLogger


###############################################################################
#                               CONFIGURATION                                 #
###############################################################################


# Initialize logger
logger = PatentLogger.get_logger(__name__)

# For access to data here, just do f"{DATA_FOLDER}/and then whatever file name you want"
DATA_FOLDER  = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'fed')
FRED_API_KEY = '51fbd42a3ad7fc528dd32ac27499a7ee'


###############################################################################
#                               DATA CLEANING                                 #
###############################################################################


class FedDataCollector:
    def __init__(self, api_key: str):
        self.fred = Fred(api_key=api_key)
        # Tech sector NAICS codes for QCEW
        self.tech_naics = [
            '5112',  # Software Publishers
            '5179',  # Other Telecommunications
            '5182',  # Data Processing, Hosting
            '5413',  # Engineering Services
            '5415',  # Computer Systems Design
            '5417'   # Scientific R&D Services
        ]
        
    def get_series_id(self, base: str, fips: float, naics: str = None) -> str:
        """
        Construct proper FRED series ID based on type.
        
        Args:
            base: Base series ID
            fips: FIPS code
            naics: NAICS code

        Returns:
            str: FRED series ID for API call
        """
        fips = str(int(fips)).zfill(5)
        if base == 'QCEW' and naics:
            return f"{base}{naics}XXX{fips}"
        elif base == 'UNRATE':
            return f"{base}{fips}"
        elif base == 'MHI':
            return f"{base}XX{fips}"
        elif base == 'POP':
            return f"{base}XX{fips}"
        return None
    
    def fetch_series_data(self, series_id: str, start_date: str, end_date: str) -> pd.Series:
        """
        Fetch data for a single series with error handling and rate limiting.
        
        Args:
            series_id: FRED series ID
            start_date: Start date for data
            end_date: End date for data
        
        Returns:
            pd.Series: Series data
        """
        try:
            logger.info(f"Fetching {series_id} from {start_date} to {end_date}")
            data = self.fred.get_series(
                series_id,
                observation_start=start_date,
                observation_end=end_date
            )
            # Rate limiting with jitter
            time.sleep(0.5 + random.uniform(0, 0.5))
            return data
        except Exception as e:
            logger.warning(f"Failed to fetch {series_id}: {str(e)}")
            return pd.Series()

    def fetch_county_data(self, counties: List[str], year: int) -> pd.DataFrame:
        """
        Fetch data for all indicators for specified counties.
        
        Args:
            counties: List of FIPS codes
            year: Year to fetch data for

        Returns:
            pd.DataFrame: Data for all indicators
        """
        start_date = f"{year}-01-01"
        end_date   = f"{year}-12-31"
        
        data = []
        for fips in tqdm(counties, desc=f"Fetching Fed data for {year}"):
            county_data = {'fips': fips, 'year': year}
            
            # unemployment rate
            series_id = self.get_series_id('UNRATE', fips)
            if series_id:
                series_data = self.fetch_series_data(series_id, start_date, end_date)
                if not series_data.empty:
                    county_data['unemployment_rate'] = series_data.mean()
            
            # tech sector employment
            tech_employment = 0
            for naics in self.tech_naics:
                series_id = self.get_series_id('QCEW', fips, naics)
                if series_id:
                    series_data = self.fetch_series_data(series_id, start_date, end_date)
                    if not series_data.empty:
                        tech_employment += series_data.mean()
            county_data['tech_employment'] = tech_employment
            
            # median household income
            series_id = self.get_series_id('MHI', fips)
            if series_id:
                series_data = self.fetch_series_data(series_id, start_date, end_date)
                if not series_data.empty:
                    county_data['median_household_income'] = series_data.mean()
            
            # population
            series_id = self.get_series_id('POP', fips)
            if series_id:
                series_data = self.fetch_series_data(series_id, start_date, end_date)
                if not series_data.empty:
                    county_data['population'] = series_data.mean()
            
            data.append(county_data)
            
        return pd.DataFrame(data)

def collect_fed_predictors(start_year: int = 2001, end_year: int = 2022):
    """
    Main function to collect Federal Reserve data predictors.
    
    Args:
        start_year: First year to collect data for
        end_year: Last year to collect data for
    """
    collector = FedDataCollector(FRED_API_KEY)
    
    # Get list of counties from patents file
    patents_file = os.path.join(project_root, "data", "patents.tsv")
    if not os.path.exists(patents_file):
        logger.error(f"Patents file not found: {patents_file}")
        return
        
    patents_df = pd.read_csv(patents_file, sep='\t')
    counties = patents_df['inventor_fips'].unique().tolist()
    logger.info(f"Found {len(counties)} unique counties in patents data")
    
    # Process each year
    for year in range(start_year, end_year + 1):
        try:
            df = collector.fetch_county_data(counties, year)
            
            # Calculate tech employment per capita
            if 'tech_employment' in df.columns and 'population' in df.columns:
                df['tech_employment_per_capita'] = df['tech_employment'] / df['population']
            
            print(df.head())

            # Save to TSV
            output_path = os.path.join(DATA_FOLDER, f'fed_predictors_{year}.tsv')
            df.to_csv(output_path, sep='\t', index=False)
            
            logger.info(f"Saved Fed data for {year}")
            
        except Exception as e:
            logger.error(f"Error processing year {year}: {str(e)}")
            continue

if __name__ == '__main__':
    collect_fed_predictors()