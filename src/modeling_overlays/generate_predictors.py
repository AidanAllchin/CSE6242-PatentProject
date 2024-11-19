#!/usr/bin/env python3
#-*- coding: utf-8 -*-
"""
Created: Tue Nov 12 2024
Author: Aidan Allchin & Kaitlyn Williams

This script generates predictors for each year from 2001 to 2022. The predictors
are based on patent data and BEA data. The patent data is used to calculate
the number of patents, inventors, assignees, unique WIPO fields, diversity score,
and exports score for each county. The BEA data is used to calculate an innovation
score for each county. The predictors are saved to TSV files in the `data/model`
folder.

Flow:
1. Load patent data and BEA data for each year
2. Calculate patent-based predictors for each year:
    - Number of patents
    - Number of inventors
    - Number of assignees
    - Number of unique WIPO fields
    - Diversity score = Shannon Diversity Index of WIPO fields
    - Exports score = Number of patents with assignee outside inventor's county
3. Add population count to each year's predictors
4. Scale and normalize the predictors
5. Calculate innovation score for each year:
    - Innovation score = (
        0.3 * per_capita_income_dollars +
        0.2 * earnings_per_capita +
        0.2 * employment_per_capita +
        0.4 * real_gdp_thousands +
        0.1 * average_earnings_per_job_dollars
    )
6. Add temporal metrics to each year's predictors:
    - Population percentage change
    - Economic momentum (relative change in GDP per capita)
7. Save predictors to TSV files in the `data/model` folder
"""
import os, sys
from pathlib import Path

# ALL FILES IN THE PROJECT SHOULD BE IMPORTED RELATIVE TO THE PROJECT ROOT
# the # of `.parent`s should be adjusted based on the file's location
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import joblib
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
import json
from sklearn.preprocessing import RobustScaler
from typing import List
from datetime import datetime
from src.other.logging import PatentLogger


###############################################################################
#                               CONFIGURATION                                 #
###############################################################################


# Initialize logger
logger = PatentLogger.get_logger(__name__)

BEA_FOLDER   = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'bea')
PATENTS_PATH = os.path.join(project_root, 'data', 'patents.tsv')
MODEL_FOLDER = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'model')


###############################################################################
#                             HELPER FUNCTIONS                                #
###############################################################################


def scale_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Scales a specified column in a DataFrame using robust scaling methods while handling NAs intelligently.

    The function performs the following steps:
    1. Converts '(NA)' strings to numpy NaN values.
    2. Converts the column to float type, preserving NaN values.
    3. Applies a log transformation to columns representing counts or populations.
    4. Uses robust scaling for financial metrics.
    5. Scales only the non-NA values in the column.
    
    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        column (str): The name of the column to be scaled.

    Returns:
        pd.DataFrame: The DataFrame with the specified column scaled.
        
    """
    # Convert (NA) strings to numpy NaN
    df[column] = df[column].replace('(NA)', np.nan)
    
    # Convert to float (this will preserve NaN values)
    df[column] = df[column].astype(float)
    
    # For columns representing counts/populations, use log transformation
    if any(x in column for x in ['count', 'population']):
        # Add small constant to handle zeros
        df[column] = np.log1p(df[column])
    
    # For financial metrics, use robust scaling
    scaler = RobustScaler()
    
    # Only scale non-NA values
    mask = df[column].notna()
    df.loc[mask, column] = scaler.fit_transform(df.loc[mask, column].values.reshape(-1, 1))
    
    return df

def load_bea_features(year: int) -> pd.DataFrame:
    """
    Load BEA features for a given year.
    
    Args:
        year: Year to load data for
        
    Returns:
        DataFrame with BEA features
    """
    bea_path = os.path.join(BEA_FOLDER, f'bea_predictors_{year}.tsv')
    if not os.path.exists(bea_path):
        raise FileNotFoundError(f"BEA data for year {year} not found at {bea_path}")
        
    df = pd.read_csv(bea_path, sep='\t')
    df['county_fips'] = df['GeoFIPS'].astype(str)
    df = df.drop(columns=['GeoFIPS'])
    
    return df

def load_county_metadata() -> pd.DataFrame:
    """
    Load county metadata from the GeoJSON file, using pre-calculated centroids.
    Because why calculate what's already calculated...
    
    Returns:
        DataFrame with columns:
            - county_fips: Combined state + county FIPS code
            - state_name: Full state name (not abbreviation)
            - county_name: County name without the word "County"
            - latitude: Pre-calculated centroid latitude
            - longitude: Pre-calculated centroid longitude
    """
    county_meta = []
    geojson_path = os.path.join(project_root, "data", "geolocation", "county_boundaries.geojson")
    
    with open(geojson_path, 'r') as f:
        counties = json.load(f)
        
    for feature in counties['features']:
        props = feature['properties']
        
        # Get the pre-calculated centroid
        centroid = props['geo_point_2d']
        
        # Create FIPS from state + county FIPS
        state_fips  = props['statefp'].zfill(2)
        county_fips = props['countyfp'].zfill(3)
        fips        = f"{state_fips}{county_fips}"
        
        # The name in the GeoJSON includes "County"
        county_name = props['name'].replace(' County', '')
        
        county_meta.append({
            'county_fips': fips,
            'state_name': props['state_name'],
            'county_name': county_name,
            'latitude': centroid['lat'],
            'longitude': centroid['lon']
        })
    
    return pd.DataFrame(county_meta)

def add_county_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add state name, county name, and centroid coordinates to the DataFrame.
    Now with 100% less geometry calculations! (god I love this .geojson file)
    
    Args:
        df: DataFrame with county_fips column
        
    Returns:
        DataFrame with added columns:
            - state_name
            - county_name 
            - latitude
            - longitude
    """
    county_meta = load_county_metadata()
    
    # Merge with main DataFrame
    df = df.merge(
        county_meta,
        on='county_fips',
        how='left'
    )
    
    # Fill any missing values (this shouldn't happen)
    df['state_name']  = df['state_name'].fillna('Unknown')
    df['county_name'] = df['county_name'].fillna('Unknown')
    df['latitude']    = df['latitude'].fillna(0.0)
    df['longitude']   = df['longitude'].fillna(0.0)
    
    # Log some stats about the merge
    merge_stats = (
        f"Added location data to {len(df)} rows\n"
        f"Missing state names: {df['state_name'].isna().sum()}\n"
        f"Missing county names: {df['county_name'].isna().sum()}\n"
        f"Missing coordinates: {df['latitude'].isna().sum()}"
    )
    logger.info(merge_stats)
    
    return df

def clean_fips(fips):
    if pd.isna(fips):
        return '00000'
    # Remove any decimal points and trailing zeros
    fips = str(fips).split('.')[0]
    # Pad to 5 digits
    return fips.zfill(5)

def convert_bea_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Convert BEA column from '(NA)' strings to proper float values.
    
    Args:
        df: DataFrame with the column to convert
        column: Column name to convert

    Returns:
        DataFrame with the column converted to float
    """
    df[column] = df[column].replace('(NA)', np.nan)
    df[column] = df[column].astype(float)
    return df[column]


###############################################################################
#                               PREDICTOR GEN                                 #
###############################################################################


def calculate_patent_features(patent_df: pd.DataFrame, period_start: datetime, period_end: datetime) -> pd.DataFrame:
    """
    Generate predictors for the given period.

    Args:
        patents_df (pd.DataFrame): DataFrame with patent data
        period_start (datetime): Start of the period
        period_end (datetime): End of the period
    
    Returns:
        DataFrame with the following columns:
            - county_fips
            - num_patents
            - num_inventors
            - num_assignees
            - unique_wipo_fields
            - diversity_score
            - exports_score
    """
    st = time.time()
    loading_start = time.time()

    # Fill missing FIPS codes with 0 (there shouldn't be any, at least for inventors)
    patent_df['inventor_fips'] = patent_df['inventor_fips'].fillna('00000')
    patent_df['assignee_fips'] = patent_df['assignee_fips'].fillna('00000')

    # Set them to 5-digit strings
    patent_df['inventor_fips'] = patent_df['inventor_fips'].astype(str).str.zfill(5)
    patent_df['assignee_fips'] = patent_df['assignee_fips'].astype(str).str.zfill(5)

    # Clean up FIPS codes
    patent_df['inventor_fips'] = patent_df['inventor_fips'].apply(clean_fips)
    patent_df['assignee_fips'] = patent_df['assignee_fips'].apply(clean_fips)


    period_df = patent_df[
        (pd.to_datetime(patent_df['patent_date']) >= period_start) &
        (pd.to_datetime(patent_df['patent_date']) < period_end)
    ]

    logger.info(f"Loaded and filtered patent data in {time.time() - loading_start:.2f}s")
    
    grouping_start = time.time()
    grouped = period_df.groupby('inventor_fips')
    logger.info(f"Grouped data in {time.time() - grouping_start:.2f}s")
    
    features = pd.DataFrame()
    features_start = time.time()
    features['num_patents']        = grouped['patent_id'].nunique()
    features['num_inventors']      = grouped['inventor_firstname'].nunique()
    features['num_assignees']      = grouped['assignee'].nunique()
    features['unique_wipo_fields'] = grouped['wipo_field_title'].nunique()

    # Removing significant outliers (more than 3 standard deviations away)
    length_before = len(features)
    num_cols_affected = 0
    for col in features.columns:
        new_length = len(features)
        z_scores = (features[col] - features[col].mean()) / features[col].std()
        features = features[(z_scores < 3) & (z_scores > -3)]
        if len(features) < new_length:
            num_cols_affected += 1
    logger.info(f"Removed {length_before - len(features)} outliers from {num_cols_affected} columns")

    # Scaling features to be between 0 and 1
    for col in features.columns:
        features[col] = (features[col] - features[col].min()) / (features[col].max() - features[col].min())

    logger.info(f"Generated features in {time.time() - features_start:.2f}s")
    
    def shannon_diversity(group):
        """
        Calculate the Shannon diversity index for a given group.

        The Shannon diversity index is a measure of the diversity in a dataset. 
        It accounts for both the abundance and evenness of the categories present.

        Args:
            group (pd.DataFrame): A pandas DataFrame containing a column 'wipo_field_title' 
                              which represents the categories to calculate the diversity for.

        Returns:
            float: The Shannon diversity index. A higher value indicates greater diversity.
        """
        counts = group['wipo_field_title'].value_counts()
        proportions = counts / counts.sum()
        score = -sum(proportions * np.log(proportions)) if not counts.empty else 0
        return abs(score)
    
    div_start = time.time()
    features['diversity_score'] = grouped.apply(lambda group: shannon_diversity(group), include_groups=False)

    # Scaling diversity score to be between 0 and 1
    features['diversity_score'] = (features['diversity_score'] - features['diversity_score'].min()) / (features['diversity_score'].max() - features['diversity_score'].min())
    logger.info(f"Calculated diversity score in {time.time() - div_start:.2f}s")
    
    def exports_score(group):
        """
        Number of patents where assignee isn't the same county as inventor.

        Args:
            group (pd.DataFrame): Group of patents for a given inventor

        Returns:
            int: Number of patents where assignee isn't the same county as inventor
        """
        return group['assignee_fips'].nunique()
    
    exports_start = time.time()
    features['exports_score'] = grouped.apply(lambda group: exports_score(group), include_groups=False)

    # Scaling exports score to be between 0 and 1
    features['exports_score'] = (features['exports_score'] - features['exports_score'].min()) / (features['exports_score'].max() - features['exports_score'].min())
    logger.info(f"Calculated exports score in {time.time() - exports_start:.2f}s")
    
    features.reset_index(inplace=True)
    features = features.rename(columns={'inventor_fips': 'county_fips'})
    
    logger.info(f"Generated predictors for {period_start.year} in {time.time() - st:.2f}s")
    return features

def add_population_count(predictors: List[pd.DataFrame]) -> List[pd.DataFrame]:
    """
    Prior to calculating the innovation_scores, adds the `population_count` 
    column from the BEA data to each year's DataFrame.

    Args:
        predictors: List of DataFrames with the following columns:
            - county_fips
            - num_patents
            - num_inventors
            - num_assignees
            - unique_wipo_fields
            - diversity_score
            - exports_score
            - year

    Returns:
        List[DataFrame] with the following columns:
            - county_fips
            - num_patents
            - num_inventors
            - num_assignees
            - unique_wipo_fields
            - diversity_score
            - exports_score
            - year
            - population_count
    """
    ttt = time.time()
    for i, predictors_df in enumerate(predictors):
        bea_df = pd.read_csv(os.path.join(BEA_FOLDER, f'bea_predictors_{predictors_df["year"].iloc[0]}.tsv'), sep='\t')
        bea_df['county_fips'] = bea_df['GeoFIPS'].astype(str)
        bea_df = bea_df.drop(columns=['GeoFIPS'])
        predictors_df = predictors_df.merge(bea_df[['county_fips', 'population_count']], on='county_fips', how='inner')
        predictors[i] = predictors_df

    logger.info(f"Added population counts in {time.time() - ttt:.2f}s")

    return predictors

def calculate_innovation_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate innovation score using scaled inputs.
    
    Args:
        df: DataFrame with scaled features
        
    Returns:
        DataFrame with innovation_score added
    """
    # Calculate innovation score based on weighted scaled metrics
    df['innovation_score'] = (
        df['per_capita_income_dollars'] * 0.3 + 
        df['earnings_per_capita'] * 0.2 +
        df['employment_per_capita'] * 0.2 +
        df['real_gdp_thousands'] * 0.4 +
        df['average_earnings_per_job_dollars'] * 0.1
    )
    
    # Scale innovation score to 0-1
    df['innovation_score'] = (df['innovation_score'] - df['innovation_score'].min()) / (
        df['innovation_score'].max() - df['innovation_score'].min()
    )
    
    return df

def add_temporal_metrics(patent_features: pd.DataFrame, bea_features: pd.DataFrame) -> pd.DataFrame:
    """
    Because time exists, unfortunately.
    Run before scaling and before adding innovation scores.
    
    Args:
        patent_features: DataFrame with patent-based metrics
        bea_features: DataFrame with BEA data for this year
    
    Returns:
        DataFrame with all features combined including temporal metrics
    """
    temp_start = time.time()

    # Clean up before merge
    patent_features['county_fips'] = patent_features['county_fips'].apply(clean_fips)
    bea_features['county_fips'] = bea_features['county_fips'].astype(str).str.zfill(5)

    # Add diagnostics
    # logger.info(f"Patent features FIPS sample: {patent_features['county_fips'].head()}")
    # logger.info(f"BEA features FIPS sample: {bea_features['county_fips'].head()}")
    # logger.info(f"Number of unique FIPS in patent_features: {len(patent_features['county_fips'].unique())}")
    # logger.info(f"Number of unique FIPS in bea_features: {len(bea_features['county_fips'].unique())}")
    
    # Merge patent and BEA features
    df = patent_features.merge(bea_features, on='county_fips', how='inner')
    
    # More diagnostics
    # logger.info(f"Number of rows after merge: {len(df)}")
    # if len(df) == 0:
    #     logger.error("Merge resulted in empty DataFrame!")
    #     logger.error(f"Sample of patent_features FIPS: {sorted(patent_features['county_fips'].unique())[:5]}")
    #     logger.error(f"Sample of bea_features FIPS: {sorted(bea_features['county_fips'].unique())[:5]}")


    # Merge patent and BEA features
    #df = patent_features.merge(bea_features, on='county_fips', how='inner')
    
    # Calculate temporal metrics if previous year exists
    prev_year_path = os.path.join(BEA_FOLDER, f'bea_predictors_{df["year"].iloc[0]-1}.tsv')
    if os.path.exists(prev_year_path):
        prev_bea = pd.read_csv(prev_year_path, sep='\t')
        prev_bea['county_fips'] = prev_bea['GeoFIPS'].astype(str)
        
        # Cast numeric columns (this is ugly - sue me)
        prev_bea['per_capita_income_dollars'] = prev_bea['per_capita_income_dollars'].replace('(NA)', np.nan).astype(float)
        prev_bea['earnings_per_capita'] = prev_bea['earnings_per_capita'].replace('(NA)', np.nan).astype(float)
        prev_bea['employment_per_capita'] = prev_bea['employment_per_capita'].replace('(NA)', np.nan).astype(float)
        prev_bea['real_gdp_thousands'] = prev_bea['real_gdp_thousands'].replace('(NA)', np.nan).astype(float)
        prev_bea['average_earnings_per_job_dollars'] = prev_bea['average_earnings_per_job_dollars'].replace('(NA)', np.nan).astype(float)
        prev_bea['population_count'] = prev_bea['population_count'].replace('(NA)', np.nan).astype(float)
        df['per_capita_income_dollars'] = df['per_capita_income_dollars'].replace('(NA)', np.nan).astype(float)
        df['earnings_per_capita'] = df['earnings_per_capita'].replace('(NA)', np.nan).astype(float)
        df['employment_per_capita'] = df['employment_per_capita'].replace('(NA)', np.nan).astype(float)
        df['real_gdp_thousands'] = df['real_gdp_thousands'].replace('(NA)', np.nan).astype(float)
        df['average_earnings_per_job_dollars'] = df['average_earnings_per_job_dollars'].replace('(NA)', np.nan).astype(float)
        df['population_count'] = df['population_count'].replace('(NA)', np.nan).astype(float)

        # Population changes
        df['population_pct_change'] = df.merge(
            prev_bea[['county_fips', 'population_count']], 
            on='county_fips', 
            suffixes=('', '_prev')
        ).apply(lambda x: ((x['population_count'] - x['population_count_prev']) / 
                         x['population_count_prev'] * 100) if x['population_count_prev'] != 0 else 0, axis=1)

        # Economic momentum (relative change in GDP per capita)
        df['gdp_per_capita'] = df['real_gdp_thousands'] / df['population_count']
        # logger.info("Prev BEA:")
        # print(prev_bea.columns)
        # print(prev_bea.head())
        # logger.info("Current BEA (post-merge):")
        # print(df.columns)
        # print(df.head())
        prev_bea['gdp_per_capita'] = prev_bea['real_gdp_thousands'] / prev_bea['population_count']
        
        df['economic_momentum'] = df.merge(
            prev_bea[['county_fips', 'gdp_per_capita']], 
            on='county_fips',
            suffixes=('', '_prev')
        ).apply(lambda x: ((x['gdp_per_capita'] - x['gdp_per_capita_prev']) / 
                         x['gdp_per_capita_prev'] * 100) if x['gdp_per_capita_prev'] != 0 else 0, axis=1)
    else:
        # First year, set changes to 0
        df['population_pct_change'] = 0
        df['economic_momentum'] = 0
    
    # Clean up intermediate columns
    # print(df.columns)
    # print(df.head())
    #df = df.drop(columns=['gdp_per_capita'])
    df = df.drop(columns=['population_count'])

    logger.info(f"Added temporal metrics for {df['year'].iloc[0]} in {time.time() - temp_start:.2f}s")
    
    return df

def generate_predictors():
    """
    Generate all predictors for each year.
    """
    patent_df = pd.read_csv(PATENTS_PATH, sep='\t')
    patent_df['patent_date'] = pd.to_datetime(patent_df['patent_date'])
    
    # Generate predictors for each year
    # These are fully un-scaled and un-normalized so we can use them for additional temporal metrics
    predictors = []
    for year in tqdm(range(2001, 2022+1), desc="Processing years"):
        period_start = datetime(year, 1, 1)
        period_end   = datetime(year + 1, 1, 1)
        
        # Calculate base features using patent data
        patent_features = calculate_patent_features(patent_df, period_start, period_end)
        patent_features['year'] = year
        
        # Load BEA features and combine with temporal metrics
        bea_features = load_bea_features(year)

        # Clean numeric columns in both DataFrames
        numeric_cols = ['per_capita_income_dollars', 'earnings_per_capita', 
                       'employment_per_capita', 'real_gdp_thousands', 
                       'average_earnings_per_job_dollars']
                       
        for col in numeric_cols:
            if col in bea_features.columns:
                bea_features[col] = bea_features[col].replace('(NA)', np.nan)
                bea_features[col] = pd.to_numeric(bea_features[col], errors='coerce')
                logger.info(f"Column {col} unique values: {bea_features[col].unique()[:5]}")

        features_df = add_temporal_metrics(patent_features, bea_features)

        # Check for any remaining '(NA)' values
        for col in features_df.columns:
            if features_df[col].astype(str).eq('(NA)').any():
                logger.warning(f"Found '(NA)' values in column {col}")
                features_df[col] = features_df[col].replace('(NA)', np.nan)
                features_df[col] = pd.to_numeric(features_df[col], errors='coerce')

        predictors.append(features_df)

    # Now, however, we do want to scale them as the innovation_score is based on these
    # scaled features
    
    # Combine all years for global scaling
    all_data = pd.concat(predictors)
    
    # Define which features to scale
    features_to_scale = [
        'num_patents', 'num_inventors', 'num_assignees', 'unique_wipo_fields',
        'diversity_score', 'exports_score', 'population_pct_change',
        'economic_momentum', 'per_capita_income_dollars', 'earnings_per_capita',
        'employment_per_capita', 'real_gdp_thousands', 'average_earnings_per_job_dollars'
    ]

    # Convert all '(NA)' strings to NaN
    for col in features_to_scale:
        if col in all_data.columns:
            all_data[col] = all_data[col].replace('(NA)', np.nan)
            all_data[col] = pd.to_numeric(all_data[col], errors='coerce')
    
    # Print all the rows containing '(NA)' values
    logger.info("Rows with NA values:")
    logger.info(all_data[all_data.isin(['(NA)']).any(axis=1)])
    
    # Remove rows where scaling feature NaN
    all_data = all_data.dropna(subset=features_to_scale, how='all')

    logger.info("Fitting global scaler...")
    scaler = RobustScaler()
    
    # Fit and save scaler
    non_nan_mask = ~all_data[features_to_scale].isna().any(axis=1)
    scaler.fit(all_data[features_to_scale][non_nan_mask])
    joblib.dump(scaler, os.path.join(MODEL_FOLDER, 'feature_scaler.joblib'))
    
    # Scale and calculate innovation scores
    logger.info("Applying scaling and calculating innovation scores...")
    scaled_predictors = []
    for df in tqdm(predictors, desc="Scaling and calculating innovation scores"):
        df[features_to_scale] = scaler.transform(df[features_to_scale])
        df = calculate_innovation_score(df)

        # Drop the columns we only wanted to use for innovation_score
        no_no_cols = ['per_capita_income_dollars', 'earnings_per_capita', 
                      'employment_per_capita', 'real_gdp_thousands', 
                      'average_earnings_per_job_dollars']
    
        df = df.drop(columns=no_no_cols)

        scaled_predictors.append(df)
        
        # Save each year's data
        year = df['year'].iloc[0]
        output_path = os.path.join(MODEL_FOLDER, f'traindata_{year}.tsv')
        df.to_csv(output_path, sep='\t', index=False)
        #logger.info(f"Saved predictors for {year}")
    
    logger.info("Finished generating all predictors")
    return scaled_predictors

def calculate_county_modifiers(yearly_predictors: List[pd.DataFrame]) -> pd.Series:
    """
    Calculate a modifier for each county based on its historical innovation performance
    relative to the global average. This captures the inherent "innovation potential"
    of each county while accounting for temporal trends.

    This value does a lot of the heavy lifting in the model, as it allows us to remove
    the county_fips column and use the modifier as a proxy for the county's innovation
    potential. I do worry, however, that it might be accounting for too much of the
    model's performance and causing us to lose some of the patent-specific predictors'
    predictive power.
    
    The modifier is calculated as a combination of:
    1. Historical average innovation score relative to global mean
    2. Innovation score growth rate
    3. Innovation score stability (inverse of variance)
    
    Args:
        yearly_predictors (List[pd.DataFrame]): List of DataFrames containing predictors for each year
        
    Returns:
        pd.Series with county_fips as index and modifier as values
    """
    # Combine all years
    all_data = pd.concat(yearly_predictors)[['county_fips', 'year', 'innovation_score']]

    # Calculate global statistics for normalization
    global_mean = all_data['innovation_score'].mean()
    global_std  = all_data['innovation_score'].std()
    
    # Calculate metrics for each county
    county_metrics = {}
    
    for county in all_data['county_fips'].unique():
        county_data = all_data[all_data['county_fips'] == county].sort_values('year')
        
        if len(county_data) < 2:  # Need at least 2 years for meaningful metrics
            county_metrics[county] = 1.0  # Neutral modifier for counties with insufficient data
            continue

        ###  
        # Relative Performance (z-score)
        ###

        county_mean = county_data['innovation_score'].mean()
        relative_performance = (county_mean - global_mean) / global_std

        # Convert to a 0.5-1.5 range
        relative_performance = 1 + (relative_performance / 5)  # dampen extreme values
        relative_performance = np.clip(relative_performance, 0.5, 1.5)
        
        ###
        # Growth Rate
        ###
        
        growth_rates = county_data['innovation_score'].pct_change().dropna()
        if len(growth_rates) > 0:
            # Using median because outliers have been a problem
            avg_growth = np.median(growth_rates)

            # Convert to a 0.8-1.2 range and dampen extreme values
            growth_component = 1 + (avg_growth * 0.2)
            growth_component = np.clip(growth_component, 0.8, 1.2)
        else:
            growth_component = 1.0
        
        ###
        # Stability
        ###

        if len(county_data) >= 3:  # Need at least 3 points for meaningful stability
            # Use coefficient of variation (CV) with protection against zero mean
            mean_score = county_data['innovation_score'].mean()
            std_score  = county_data['innovation_score'].std()
            if mean_score > 0:
                cv = std_score / mean_score
                # Convert to stability score (inverse of CV, normalized)
                stability = 1 / (1 + cv)  # between 0 and 1
                # Convert to 0.9-1.1 range
                stability = 0.9 + (stability * 0.2)
            else:
                stability = 1.0
        else:
            stability = 1.0
        
        # Combine metrics with updated weights
        modifier = (
            0.5 * relative_performance +  # Base performance level
            0.3 * growth_component +      # Growth trajectory
            0.2 * stability               # Consistency
        )
        county_metrics[county] = modifier
    
    # Convert to series
    modifiers = pd.Series(county_metrics)
    
    # Final cleanup: clip extreme values but allow more variation
    modifiers = modifiers.clip(0.5, 2.0)
    
    # Add some diagnostic logging
    logger.info(f"Modifier statistics:")
    logger.info(f"  Mean: {modifiers.mean():.3f}")
    logger.info(f"  Std:  {modifiers.std():.3f}")
    logger.info(f"  Min:  {modifiers.min():.3f}")
    logger.info(f"  Max:  {modifiers.max():.3f}")
    logger.info(f"  Number of counties: {len(modifiers)}")
    
    return modifiers

def add_county_modifiers(training_data: pd.DataFrame, yearly_predictors: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Add county modifiers to the training data.
    This is a measure of each county's historical innovation performance relative
    to the global average, accounting for growth and stability.
    It allows us to remove the county_fips column and use the modifier
    as a proxy for the county's inherent innovation potential.
    
    Args:
        training_data: DataFrame containing the training data
        yearly_predictors: List of DataFrames containing predictors for each year
        
    Returns:
        DataFrame with county_modifier column added and county_fips removed
    """
    # Calculate county modifiers
    modifiers = calculate_county_modifiers(yearly_predictors)
    
    # Add modifiers to training data
    training_data['county_modifier'] = training_data['county_fips'].map(modifiers)
    
    # Fill any missing modifiers with 1 (neutral)
    training_data['county_modifier'] = training_data['county_modifier'].fillna(1.0)

    # Save mapping for future reference
    modifier_path = os.path.join(MODEL_FOLDER, 'county_modifiers.tsv')
    modifiers.reset_index().rename(columns={'index': 'county_fips', 0: 'county_modifier'}).to_csv(
        modifier_path, sep='\t', index=False
    )
    
    return training_data

def prepare_training_data(yearly_predictors: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Prepare training data by matching current year predictors with next year's 
    innovation score. This creates a dataset where X features are from year N
    and y (innovation_score) is from year N+1.
    
    Args:
        yearly_predictors: List of DataFrames containing predictors for each year,
                          sorted by year in ascending order
    
    Returns:
        DataFrame with features from current year and next year's innovation score,
        ready for training
    """
    training_data = []
    
    # Iterate through all years except the last (since we need N+1)
    for i in tqdm(range(len(yearly_predictors) - 1), desc="Preparing training data"):
        current_year = yearly_predictors[i].copy()
        next_year    = yearly_predictors[i + 1].copy()
        
        # Get next year's innovation score
        next_year_scores = next_year[['county_fips', 'innovation_score']].rename(
            columns={'innovation_score': 'next_innovation_score'}
        )
        
        # Merge current year features with next year's score
        merged = current_year.merge(
            next_year_scores,
            on='county_fips',
            how='inner'
        )
        
        # Drop current year's innovation score
        merged = merged.drop(columns=['innovation_score'])
        
        training_data.append(merged)
    
    # We do, however, want to keep the 2022 year as well for prediction
    last_year = yearly_predictors[-1].copy()
    last_year['next_innovation_score'] = np.nan
    last_year = last_year.drop(columns=['innovation_score'])
    training_data.append(last_year)
    
    # Combine all years into one DataFrame
    final_df = pd.concat(training_data, ignore_index=True)

    # Add county metadata
    final_df = add_county_metadata(final_df)

    # Add county modifiers
    final_df = add_county_modifiers(final_df, yearly_predictors)

    try:
        # Had some trouble with gdp_per_capita popping up
        final_df = final_df.drop(columns=['gdp_per_capita'])
    except KeyError:
        pass
    
    # Ensure location columns are preserved
    preserve_cols = ['county_fips', 'state_name', 'county_name', 'latitude', 'longitude']
    other_cols    = [col for col in final_df.columns if col not in preserve_cols and col != 'next_innovation_score']
    
    # Reorder columns with location data first, then features, then target
    final_df      = final_df[preserve_cols + other_cols + ['next_innovation_score']]
    
    return final_df

def save_training_data(predictors: List[pd.DataFrame]):
    """
    Process yearly predictor files into a single training dataset and save it.
    
    Args:
        predictors: List of yearly predictor DataFrames
    """
    # Sort predictors by year to ensure correct sequence
    predictors.sort(key=lambda x: x['year'].iloc[0])
    
    # Prepare training data
    training_df = prepare_training_data(predictors)

    # Reorder so next_innovation_score is last
    cols = training_df.columns.tolist()
    cols.remove('next_innovation_score')
    cols.append('next_innovation_score')
    training_df = training_df[cols]
    
    # Save the combined training data
    output_path = os.path.join(MODEL_FOLDER, 'training_data.tsv')
    training_df.to_csv(output_path, sep='\t', index=False)
    
    logger.info(f"Created training dataset with {len(training_df)} samples")
    logger.info(f"Features included: {[col for col in training_df.columns if col != 'innovation_score']}")

if __name__ == '__main__':
    scaled_predictors = generate_predictors()
    save_training_data(scaled_predictors)
