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
from sklearn.preprocessing import RobustScaler
from typing import List
from datetime import datetime
from src.other.logging import PatentLogger
from src.other.helpers import local_filename

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


###############################################################################
#                               PREDICTOR GEN                                 #
###############################################################################


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
    #patent_df['patent_date'] = pd.to_datetime(patent_df['patent_date'])

    # Fill missing FIPS codes with 0 (there shouldn't be any, at least for inventors)
    patent_df['inventor_fips'] = patent_df['inventor_fips'].fillna(0).astype(int)
    patent_df['assignee_fips'] = patent_df['assignee_fips'].fillna(0).astype(int)

    # Set them to 5-digit strings
    patent_df['inventor_fips'] = patent_df['inventor_fips'].astype(str).str.zfill(5)
    patent_df['assignee_fips'] = patent_df['assignee_fips'].astype(str).str.zfill(5)

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
    
    # Diversity Score: Evenness of WIPO field distribution (Shannon Diversity Index)
    def shannon_diversity(group):
        counts = group['wipo_field_title'].value_counts()
        proportions = counts / counts.sum()
        score = -sum(proportions * np.log(proportions)) if not counts.empty else 0
        return abs(score)
    
    div_start = time.time()
    features['diversity_score'] = grouped.apply(lambda group: shannon_diversity(group), include_groups=False)

    # Scaling diversity score to be between 0 and 1
    features['diversity_score'] = (features['diversity_score'] - features['diversity_score'].min()) / (features['diversity_score'].max() - features['diversity_score'].min())
    logger.info(f"Calculated diversity score in {time.time() - div_start:.2f}s")
    
    # Exports score: Number of patents where the assignee is not in the same county as the inventor
    def exports_score(group):
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

# def add_innovation_scores(predictors: List[pd.DataFrame]) -> List[pd.DataFrame]:
#     """
#     Adds innovation_score for each county for every year's DataFrame based on 
#     a combination of BEA data factors.

#     Args:
#         predictors: List of DataFrames with the following columns:
#             - county_fips
#             - num_patents
#             - num_inventors
#             - num_assignees
#             - unique_wipo_fields
#             - diversity_score
#             - exports_score

#     Returns:
#         List[DataFrame] with the following columns:
#             - county_fips
#             - num_patents
#             - num_inventors
#             - num_assignees
#             - unique_wipo_fields
#             - diversity_score
#             - exports_score
#             - innovation_score
#     """
#     st = time.time()
#     for i, predictors_df in enumerate(predictors):
#         bea_df = pd.read_csv(os.path.join(BEA_FOLDER, f'bea_predictors_{predictors_df["year"].iloc[0]}.tsv'), sep='\t')
#         bea_df['county_fips'] = bea_df['GeoFIPS'].astype(str)
#         bea_df = bea_df.drop(columns=['GeoFIPS'])
#         bea_cols = bea_df.columns.tolist()
#         bea_cols.remove('county_fips')

#         # print(predictors_df.columns)
        
#         # Merge the BEA data with the predictors but only if the county is in both
#         predictors_df = predictors_df.merge(bea_df, on='county_fips', how='inner')
        
#         # Calculate innovation score based on weighted BEA data matching on county FIPS
#         # GeoFIPS	per_capita_income_dollars	population_count	employment_per_capita	earnings_per_capita	real_gdp_thousands	average_earnings_per_job_dollars
#         predictors_df['innovation_score'] = (
#             predictors_df['per_capita_income_dollars'] * 0.3 + 
#             predictors_df['earnings_per_capita'] * 0.2 +
#             predictors_df['employment_per_capita'] * 0.2 +
#             predictors_df['real_gdp_thousands'] * 0.4 +
#             predictors_df['average_earnings_per_job_dollars'] * 0.1
#         )

#         # Scaling innovation score to be between 0 and 1
#         predictors_df['innovation_score'] = (predictors_df['innovation_score'] - predictors_df['innovation_score'].min()) / (predictors_df['innovation_score'].max() - predictors_df['innovation_score'].min())

#         predictors_df = predictors_df.drop(columns=bea_cols)
#         predictors[i] = predictors_df
    
#     logger.info(f"Added innovation scores in {time.time() - st:.2f}s")
#     return predictors

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
    # Merge patent and BEA features
    df = patent_features.merge(bea_features, on='county_fips', how='inner')
    
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
        features_df = add_temporal_metrics(patent_features, bea_features)
        predictors.append(features_df)

    # Now, however, we do want to scale them as the innovation_score is based on these
    # scaled features
    
    # Combine all years for global scaling
    logger.info("Fitting global scaler...")
    all_data = pd.concat(predictors)
    scaler = RobustScaler()
    
    # Define which features to scale
    features_to_scale = [
        'num_patents', 'num_inventors', 'num_assignees', 'unique_wipo_fields',
        'diversity_score', 'exports_score', 'population_pct_change',
        'economic_momentum', 'per_capita_income_dollars', 'earnings_per_capita',
        'employment_per_capita', 'real_gdp_thousands', 'average_earnings_per_job_dollars'
    ]
    
    # Fit and save scaler
    scaler.fit(all_data[features_to_scale])
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
    
    The modifier is calculated as a combination of:
    1. Historical average innovation score relative to global mean
    2. Innovation score growth rate
    3. Innovation score stability (inverse of variance)
    
    Args:
        yearly_predictors: List of DataFrames containing predictors for each year
        
    Returns:
        Series with county_fips as index and modifier as values
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
            
        # 1. Relative Performance (z-score based)
        county_mean = county_data['innovation_score'].mean()
        relative_performance = (county_mean - global_mean) / global_std
        # Convert to a 0.5-1.5 range
        relative_performance = 1 + (relative_performance / 5)  # Divide by 5 to dampen extreme values
        relative_performance = np.clip(relative_performance, 0.5, 1.5)
        
        # 2. Growth Rate (with better handling of edge cases)
        growth_rates = county_data['innovation_score'].pct_change().dropna()
        if len(growth_rates) > 0:
            # Use median instead of mean for robustness
            avg_growth = np.median(growth_rates)
            # Convert to a 0.8-1.2 range
            growth_component = 1 + (avg_growth * 0.2)  # Scale factor of 0.2 to dampen
            growth_component = np.clip(growth_component, 0.8, 1.2)
        else:
            growth_component = 1.0
        
        # 3. Stability (redesigned)
        if len(county_data) >= 3:  # Need at least 3 points for meaningful stability
            # Use coefficient of variation (CV) with protection against zero mean
            mean_score = county_data['innovation_score'].mean()
            std_score  = county_data['innovation_score'].std()
            if mean_score > 0:
                cv = std_score / mean_score
                # Convert to stability score (inverse of CV, normalized)
                stability = 1 / (1 + cv)  # Will give a value between 0 and 1
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
    
    # Log distribution in bins
    bins = pd.cut(modifiers, bins=10)
    logger.info("\nModifier distribution:")
    for bin_label, count in bins.value_counts().sort_index().items():
        logger.info(f"  {bin_label}: {count}")
    
    return modifiers

def add_county_modifiers(training_data: pd.DataFrame, yearly_predictors: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Add county modifiers to the training data.
    This is a measure of each county's historical innovation performance relative
    to the global average, accounting for growth and stability.
    It essentially allows us to remove the county_fips column and use the modifier
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
    
    # Remove county_fips as it's no longer needed
    training_data = training_data.drop(columns=['county_fips'])
    
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
        
        # Drop current year's innovation score and year column
        merged = merged.drop(columns=['innovation_score', 'year'])
        
        training_data.append(merged)
    
    # Combine all years into one DataFrame
    final_df = pd.concat(training_data, ignore_index=True)

    # Add county modifiers and remove county_fips
    final_df = add_county_modifiers(final_df, yearly_predictors)

    try:
        # Had some trouble with gdp_per_capita popping up
        final_df = final_df.drop(columns=['gdp_per_capita'])
    except KeyError:
        pass
    
    #final_df = final_df.rename(columns={'next_innovation_score': 'innovation_score'})
    
    return final_df



def save_training_data(predictors: List[pd.DataFrame]) -> None:
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
