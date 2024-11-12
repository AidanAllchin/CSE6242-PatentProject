#!/usr/bin/env python3
#-*- coding: utf-8 -*-
"""
Created: Tue Nov 12 2024
Author: Kaitlyn Williams

Implements the Innovation Hub Prediction model, which predicts future innovation
scores for US counties based on patent and economic indicators. 

This module combines patent data with economic indicators from multiple sources:
- Bureau of Economic Analysis (BEA) for GDP data
- Census Bureau for demographic and education data
- Federal Reserve (FRED) for economic indicators
- USPTO patent data for innovation metrics

The model uses these features to predict which regions are likely to become
innovation hubs in the future.
"""
import os, sys
from pathlib import Path

# ALL FILES IN THE PROJECT SHOULD BE IMPORTED RELATIVE TO THE PROJECT ROOT
project_root = Path(__file__).resolve().parent#.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from datetime import datetime
import asyncio
import aiohttp
import hashlib
import json
from typing import Dict, List, Optional
from census import Census
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from fredapi import Fred
from src.other.logging import PatentLogger

logger = PatentLogger.get_logger(__name__)

# Constants
BEA_API_KEY    = 'F9919422-E7E7-4562-89DB-3051ACF0B6A8'
CENSUS_API_KEY = '29e0fb5c56cb38bb77a95450557344c1e217e72f'
FRED_API_KEY   = '51fbd42a3ad7fc528dd32ac27499a7ee'


###############################################################################
#                                                                             #
#                               DATA COLLECTOR                                #
#                                                                             #
###############################################################################


class EconomicDataCollector:
    """
    Making this asynchronous to be able to run multiple requests at once.
    """
    def __init__(self, bea_key: str, census_key: str, fred_key: str):
        self.bea_key = bea_key
        self.census_key = census_key
        # Initialize FRED API
        self.fred = Fred(api_key=fred_key)
        self.cache_dir = os.path.join(project_root, 'data', 'cache')
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_cache_path(self, data_type: str, year: int) -> str:
        """
        Get cache file path for a specific data type and year.
        This is to simplify not having all the same data in the same file.

        Args:
            data_type (str): Type of data (e.g. 'bea', 'census', 'fred')
            year (int): Year of the data

        Returns:
            str: Cache file path
        """
        return os.path.join(self.cache_dir, f"{data_type}_{year}.json")

    def _load_cache(self, cache_path: str) -> Dict:
        """
        Load cached data if it exists.

        Args:
            cache_path (str): Path to the cache file

        Returns:
            Dict: Cached data
        """
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.warning(f"Corrupted cache file: {cache_path}")
                return {}
        return {}

    def _save_cache(self, cache_path: str, data: Dict):
        """
        Save data to cache file.

        Args:
            cache_path (str): Path to the cache file
            data (Dict): Data to save
        """
        with open(cache_path, 'w') as f:
            json.dump(data, f)

    async def get_gdp_async(self, session: aiohttp.ClientSession, year: int, county_fips: str) -> float:
        """
        Fetch GDP data for a single county asynchronously.

        Args:
            session (aiohttp.ClientSession): HTTP session
            year (int): Year to fetch
            county_fips (str): County FIPS code

        Returns:
            float: GDP value for the county that year
        """
        # Try to get it from the cache first
        cache_path = self._get_cache_path("gdp", year)
        cache = self._load_cache(cache_path)
        
        if county_fips in cache:
            return cache[county_fips]
            
        # Couldn't find it in cache, fetch from the API
        url = "https://apps.bea.gov/api/data/"
        params = {
            'UserID': self.bea_key,
            'method': 'GetData',
            'datasetname': 'Regional',
            'TableName': 'CAGDP2',
            'LineCode': '1',
            'GeoFips': county_fips,
            'Year': str(year),
            'ResultFormat': 'JSON'
        }
        
        try:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    try:
                        gdp = float(data['BEAAPI']['Results']['Data'][0]['DataValue'].replace(',', ''))
                        cache[county_fips] = gdp
                        self._save_cache(cache_path, cache)
                        return gdp
                    except (KeyError, IndexError, ValueError):
                        return np.nan
                else:
                    logger.warning(f"Failed to fetch GDP for county {county_fips}: {response.status}")
                    return np.nan
        except Exception as e:
            logger.warning(f"Error fetching GDP for county {county_fips}: {e}")
            return np.nan
    
    async def get_gdp_batch(self, year: int, counties: List[str]) -> Dict[str, float]:
        """
        Fetch GDP data for multiple counties concurrently.
        
        Args:
            year (int): Year to fetch
            counties (List[str]): List of county FIPS codes

        Returns:
            Dict[str, float]: Dictionary of county FIPS codes to GDP values
        """
        async with aiohttp.ClientSession() as session:
            tasks = [
                self.get_gdp_async(session, year, county)
                for county in counties
            ]
            results = await asyncio.gather(*tasks)
            return dict(zip(counties, results))

    async def get_census_data_async(self, year: int) -> pd.DataFrame:
        """
        Fetch census data with caching.

        Args:
            year (int): Year to fetch

        Returns:
            pd.DataFrame: Census data for all counties
        """
        cache_path = self._get_cache_path("census", year)
        cache = self._load_cache(cache_path)
        
        if cache:
            return pd.DataFrame.from_dict(cache, orient='index')
            
        try:
            # Census API doesn't support async calls, just adding cache here
            c = Census(self.census_key)
            
            county_data = c.acs5.get(
                fields=[
                    'B01003_001E',  # Population
                    'B15003_022E',  # Bachelor's
                    'B15003_023E',  # Master's
                    'B15003_024E',  # Professional
                    'B15003_025E',  # Doctorate
                    'B19013_001E',  # Median income
                    'B23025_002E',  # Labor force
                ],
                geo={'for': 'county:*'},
                year=year
            )
            
            # Convert to DataFrame and process
            df = pd.DataFrame(county_data)

            # Pivot to have one row per county
            df = df.pivot(index='county', columns='state', values=df.columns[1:])
            
            # Calculate higher education percentage
            df = df.apply(pd.to_numeric, errors='ignore')
            df['higher_ed_percent'] = (
                df['B15003_022E'] + df['B15003_023E'] + df['B15003_024E'] + df['B15003_025E']
            ) / df['B01003_001E'] * 100
            
            # Drop unnecessary columns and rename
            df = df.rename(columns={
                'B01003_001E': 'total_population',
                'B19013_001E': 'median_household_income',
                'B23025_002E': 'labor_force_total',
                'state': 'state_fips',
                'county': 'county_fips'
            })
            df = df[['state_fips', 'county_fips', 'total_population', 'median_household_income', 'labor_force_total', 'higher_ed_percent']]
        
            # Cache results
            if df['county_fips'].is_unique:
                cache = df.set_index('county_fips').to_dict(orient='index')
                self._save_cache(cache_path, cache)
            else:
                logger.error("Duplicate county_fips found in census data.")
                # Print all the duplicated county_fips
                logger.error(df[df['county_fips'].duplicated()]['county_fips'])
                raise ValueError("Duplicate county_fips found in census data.")

            return df
        except Exception as e:
            logger.error(f"Failed to fetch census data: {e}")
            raise

    async def get_fred_data_async(self, year: int, county_fips: str) -> Dict[str, float]:
        """
        Fetch FRED economic data for a county asynchronously.
        Currently fetches:
        - Unemployment rate
        - Real GDP growth
        - Personal income growth

        Args:
            year (int): Year to fetch
            county_fips (str): County FIPS code

        Returns:
            Dict[str, float]: Dictionary of economic metrics
        """
        cache_path = self._get_cache_path("fred", year)
        cache      = self._load_cache(cache_path)
        
        if county_fips in cache:
            return cache[county_fips]
        
        start_date = f"{year}-01-01"
        end_date   = f"{year}-12-31"
        metrics    = {}
        
        try:
            # Unemployment rate
            series_id = f'LAUCN{county_fips}0000000003'  # County unemployment rate
            data = self.fred.get_series(
                series_id, 
                observation_start=start_date,
                observation_end=end_date
            )
            metrics['unemployment_rate'] = data.mean() if not data.empty else np.nan
            
            # Real GDP Growth (if available)
            series_id = f'RGMP{county_fips}'  # Real GDP growth
            try:
                data = self.fred.get_series(
                    series_id,
                    observation_start=start_date,
                    observation_end=end_date
                )
                metrics['gdp_growth'] = data.mean() if not data.empty else np.nan
            except Exception:
                metrics['gdp_growth'] = np.nan
            
            # Personal Income Growth (if available)
            series_id = f'PIPCN{county_fips}'  # Personal income per capita
            try:
                data = self.fred.get_series(
                    series_id,
                    observation_start=start_date,
                    observation_end=end_date
                )
                metrics['income_growth'] = data.pct_change().mean() if not data.empty else np.nan
            except Exception:
                metrics['income_growth'] = np.nan
            
            # Cache results
            cache[county_fips] = metrics
            self._save_cache(cache_path, cache)
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Error fetching FRED data for county {county_fips}: {e}")
            return {
                'unemployment_rate': np.nan,
                'gdp_growth': np.nan,
                'income_growth': np.nan
            }

    async def get_fred_batch(self, year: int, counties: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Fetch FRED data for multiple counties concurrently.
        
        Args:
            year (int): Year to fetch
            counties (List[str]): List of county FIPS codes

        Returns:
            Dict[str, Dict[str, float]]: Dictionary of county FIPS codes to economic metrics
        """
        tasks = [
            self.get_fred_data_async(year, county)
            for county in counties
        ]
        results = await asyncio.gather(*tasks)
        return dict(zip(counties, results))

async def collect_economic_data(counties: List[str], year: int, bea_key: str, census_key: str, fred_key: str) -> pd.DataFrame:
    """
    Collect all economic data concurrently.
    
    Args:
        counties: List of county FIPS codes
        year: Year to collect data for
        bea_key: BEA API key
        census_key: Census API key
        fred_key: FRED API key
        
    Returns:
        DataFrame containing all economic indicators
    """
    collector = EconomicDataCollector(bea_key, census_key, fred_key)
    
    # Create tasks for all data collection
    logger.info("Fetching economic data concurrently...")
    gdp_task    = collector.get_gdp_batch(year, counties)
    census_task = collector.get_census_data_async(year)
    fred_task   = collector.get_fred_batch(year, counties)
    
    # Run tasks concurrently
    gdp_data, census_df, fred_data = await asyncio.gather(
        gdp_task, census_task, fred_task
    )
    
    # Combine results
    result_df = pd.DataFrame({'county_fips': counties})
    result_df['gdp'] = result_df['county_fips'].map(gdp_data)
    result_df = pd.merge(result_df, census_df, on='county_fips', how='left')
    
    # Add FRED metrics
    for metric in ['unemployment_rate', 'gdp_growth', 'income_growth']:
        result_df[metric] = result_df['county_fips'].map(
            lambda x: fred_data.get(x, {}).get(metric, np.nan)
        )
    
    # Calculate derived metrics
    result_df['gdp_per_capita'] = result_df['gdp'] / result_df['total_population']
    
    # Log completion stats
    total = len(counties)
    complete = result_df.notna().all(axis=1).sum()
    logger.info(f"Data collection complete: {complete}/{total} counties with complete data")
    
    return result_df

# 1. Fetch GDP Data from BEA API for Counties
def get_county_gdp(bea_api_key, year, county_fips):
    url = "https://apps.bea.gov/api/data/"
    params = {
        'UserID': bea_api_key,
        'method': 'GetData',
        'datasetname': 'Regional',
        'TableName': 'CAGDP2',
        'LineCode': '1',
        'GeoFips': county_fips,
        'Year': year,
        'ResultFormat': 'JSON'
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    try:
        gdp_value = float(data['BEAAPI']['Results']['Data'][0]['DataValue'].replace(',', ''))
    except (KeyError, IndexError, ValueError):
        gdp_value = np.nan
    
    return gdp_value

# 2. Fetch Census Data for Counties
def get_county_census_data(census_api_key, year):
    c = Census(census_api_key, year=year)
    
    county_data = c.acs5.get(
        variables=[
            'B01003_001E',  # Total population
            'B15003_022E',  # Bachelor's degree
            'B15003_023E',  # Master's degree
            'B15003_024E',  # Professional degree
            'B15003_025E',  # Doctorate degree
            'B19013_001E',  # Median household income
            'B23025_002E',  # Labor force total
        ],
        geo={'for': 'county:*'}
    )
    
    df = pd.DataFrame(county_data)
    
    df = df.apply(pd.to_numeric, errors='ignore')
    df['higher_ed_percent'] = (
        df['B15003_022E'] + df['B15003_023E'] + df['B15003_024E'] + df['B15003_025E']
    ) / df['B01003_001E'] * 100
    
    df = df.rename(columns={
        'B01003_001E': 'total_population',
        'B19013_001E': 'median_household_income',
        'B23025_002E': 'labor_force_total',
        'state': 'state_fips',
        'county': 'county_fips'
    })
    df = df[['state_fips', 'county_fips', 'total_population', 'median_household_income', 'labor_force_total', 'higher_ed_percent']]
    
    return df

# 3. Fetch FRED Economic Data for Counties
def get_county_unemployment_rate(county_fips, start_date, end_date):
    try:
        fred_series_id = f'URNST{county_fips}'
        unemployment_data = fred.get_series(fred_series_id, observation_start=start_date, observation_end=end_date)
        avg_unemployment_rate = unemployment_data.mean()
    except Exception as e:
        print(f"Error fetching unemployment rate for county {county_fips}: {e}")
        avg_unemployment_rate = np.nan
        
    return avg_unemployment_rate

# 4. Combine Economic Features
def get_economic_features_with_fred(county_list, year, start_date, end_date):
    economic_df = pd.DataFrame()
    economic_df['county_fips'] = county_list
    economic_df['year'] = year
    
    gdp_values = [get_county_gdp(BEA_API_KEY, str(year), county) for county in county_list]
    economic_df['gdp'] = gdp_values

    census_df = get_county_census_data(CENSUS_API_KEY, year)

    economic_features = pd.merge(economic_df, census_df, on='county_fips', how='left')

    unemployment_rates = [get_county_unemployment_rate(county, start_date, end_date) for county in county_list]
    economic_features['avg_unemployment_rate'] = unemployment_rates

    economic_features['gdp_per_capita'] = economic_features['gdp'] / economic_features['total_population']
    
    return economic_features

# 5. Calculate Patent Features Grouped by `inventor_fips`
def calculate_patent_features(patent_df, period_start, period_end):
    period_df = patent_df[
        (pd.to_datetime(patent_df['patent_date']) >= period_start) &
        (pd.to_datetime(patent_df['patent_date']) < period_end)
    ]
    
    grouped = period_df.groupby('inventor_fips')
    
    features = pd.DataFrame()
    features['num_patents'] = grouped['patent_id'].nunique()
    features['num_inventors'] = grouped['inventor_firstname'].nunique()
    features['num_assignees'] = grouped['assignee'].nunique()
    features['unique_wipo_fields'] = grouped['wipo_field_title'].nunique()
    
    # Diversity Score: Evenness of WIPO field distribution (Shannon Diversity Index)
    def shannon_diversity(group):
        counts = group['wipo_field_title'].value_counts()
        proportions = counts / counts.sum()
        return -sum(proportions * np.log(proportions)) if not counts.empty else 0
    
    features['diversity_score'] = grouped.apply(lambda group: shannon_diversity(group))
    
    # Collaboration Score: Ratio of patents with multiple assignees
    def collaboration_ratio(group):
        total_patents = group['patent_id'].nunique()
        multi_assignee_patents = group[group['assignee'].duplicated(keep=False)]['patent_id'].nunique()
        return multi_assignee_patents / total_patents if total_patents > 0 else 0
    
    features['collaboration_score'] = grouped.apply(lambda group: collaboration_ratio(group))
    
    features.reset_index(inplace=True)
    features = features.rename(columns={'inventor_fips': 'county_fips'})
    
    return features

# 6. Prepare Final Dataset
def prepare_dataset(patent_df, period_start, period_end):
    patent_features = calculate_patent_features(patent_df, period_start, period_end)
    county_list = patent_features['county_fips'].tolist()
    
    year = period_start.year
    economic_features = get_economic_features_with_fred(county_list, year, period_start, period_end)
    
    final_dataset = pd.merge(patent_features, economic_features, on='county_fips', how='left')
    final_dataset.dropna(inplace=True)
    
    # Add additional location details for labeling
    final_dataset['county_name'] = patent_df.groupby('inventor_fips')['inventor_county'].first().values
    final_dataset['county_state'] = patent_df.groupby('inventor_fips')['inventor_state'].first().values
    
    return final_dataset

# 7. Model Training
def train_model(final_dataset):
    X = final_dataset.drop(columns=['county_fips', 'next_period_innovation_score'])
    y = final_dataset['next_period_innovation_score']
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f'RMSE: {rmse}')
    
    return model

# 8. Main Execution
async def main():
    # d = await collect_economic_data(
    #     counties=['01001', '01003', '01005'],
    #     year=2019,
    #     bea_key=BEA_API_KEY,
    #     census_key=CENSUS_API_KEY,
    #     fred_key=FRED_API_KEY
    # )
    ob = EconomicDataCollector(BEA_API_KEY, CENSUS_API_KEY, FRED_API_KEY)
    d = await ob.get_census_data_async(2019)
    print(d)
    sys.exit(0)
    period_start = pd.Timestamp('2017-01-01')
    period_end = pd.Timestamp('2018-01-01')
    
    # Load the real patents data from the data folder
    patent_df = pd.read_csv('patents.tsv', sep='\t')
    
    final_dataset = prepare_dataset(patent_df, period_start, period_end)
    
    model = train_model(final_dataset)

if __name__ == '__main__':
    asyncio.run(main())