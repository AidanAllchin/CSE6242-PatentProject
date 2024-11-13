#!/usr/bin/env python3
#-*- coding: utf-8 -*-
"""
Created: Tue Nov 11 2024
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
import requests
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

fred = Fred(api_key=FRED_API_KEY)


###############################################################################
#                                                                             #
#                               DATA COLLECTOR                                #
#                                                                             #
###############################################################################


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
def main():
    period_start = pd.Timestamp('2017-01-01')
    period_end = pd.Timestamp('2018-01-01')
    
    # Load the real patents data from the data folder
    patent_df = pd.read_csv('patents.tsv', sep='\t')
    
    final_dataset = prepare_dataset(patent_df, period_start, period_end)
    
    model = train_model(final_dataset)

if __name__ == '__main__':
    main()