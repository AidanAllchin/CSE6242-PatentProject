#!/usr/bin/env python3
#-*- coding: utf-8 -*-
"""
Created: Tue Nov 12 2024
Author: Aidan Allchin

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

import pandas as pd
import numpy as np
import time
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
#                               PREDICTOR GEN                                 #
###############################################################################


def calculate_patent_features(period_start: datetime, period_end: datetime) -> pd.DataFrame:
    """
    Generate predictors for the given period.

    Args:
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
    patent_df = pd.read_csv(PATENTS_PATH, sep='\t')
    patent_df['patent_date'] = pd.to_datetime(patent_df['patent_date'])

    # Fill missing FIPS codes with 0 (there shouldn't be any, at least for inventors)
    patent_df['inventor_fips'] = patent_df['inventor_fips'].fillna(0).astype(int)
    patent_df['assignee_fips'] = patent_df['assignee_fips'].fillna(0).astype(int)

    # Set them to 5-digit strings
    patent_df['inventor_fips'] = patent_df['inventor_fips'].astype(str).str.zfill(5)
    patent_df['assignee_fips'] = patent_df['assignee_fips'].astype(str).str.zfill(5)

    logger.info(f"Loaded patent data in {time.time() - loading_start:.2f}s")

    period_df = patent_df[
        (pd.to_datetime(patent_df['patent_date']) >= period_start) &
        (pd.to_datetime(patent_df['patent_date']) < period_end)
    ]
    
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

def add_innovation_scores(predictors: List[pd.DataFrame]) -> List[pd.DataFrame]:
    """
    Adds innovation_score for each county for every year's DataFrame based on 
    a combination of BEA data factors.

    Args:
        predictors: List of DataFrames with the following columns:
            - county_fips
            - num_patents
            - num_inventors
            - num_assignees
            - unique_wipo_fields
            - diversity_score
            - exports_score

    Returns:
        List[DataFrame] with the following columns:
            - county_fips
            - num_patents
            - num_inventors
            - num_assignees
            - unique_wipo_fields
            - diversity_score
            - exports_score
            - innovation_score
    """
    st = time.time()
    for i, predictors_df in enumerate(predictors):
        bea_df = pd.read_csv(os.path.join(BEA_FOLDER, f'bea_predictors_{predictors_df["year"].iloc[0]}.tsv'), sep='\t')
        bea_df['county_fips'] = bea_df['GeoFIPS'].astype(str)
        bea_df = bea_df.drop(columns=['GeoFIPS'])
        bea_cols = bea_df.columns.tolist()
        bea_cols.remove('county_fips')
        
        # Merge the BEA data with the predictors but only if the county is in both
        predictors_df = predictors_df.merge(bea_df, on='county_fips', how='inner')
        
        # Calculate innovation score based on weighted BEA data matching on county FIPS
        predictors_df['innovation_score'] = (
            predictors_df['residence_net_earnings_thousands'] * 0.3 +
            predictors_df['per_capita_income_dollars'] * 0.1 +
            predictors_df['personal_income_thousands'] * 0.3 +
            predictors_df['population_count'] * 0.1 +
            predictors_df['total_employment_count'] * 0.1 +
            predictors_df['current_gdp_thousands'] * 0.3 +
            predictors_df['real_gdp_thousands'] * 0.4 +
            predictors_df['average_earnings_per_job_dollars'] * 0.1
        )

        # Scaling innovation score to be between 0 and 1
        predictors_df['innovation_score'] = (predictors_df['innovation_score'] - predictors_df['innovation_score'].min()) / (predictors_df['innovation_score'].max() - predictors_df['innovation_score'].min())

        predictors_df = predictors_df.drop(columns=bea_cols)
        predictors[i] = predictors_df
    
    logger.info(f"Added innovation scores in {time.time() - st:.2f}s")
    return predictors

def generate_predictors():
    """
    Generate all predictors for each year.
    """
    patent_df = pd.read_csv(PATENTS_PATH, sep='\t')
    patent_df['patent_date'] = pd.to_datetime(patent_df['patent_date'])
    
    # Generate predictors for each year
    predictors = []
    for year in range(2001, 2022+1):  # just making it clear 22 is inclusive
        period_start = datetime(year, 1, 1)
        period_end   = datetime(year + 1, 1, 1)
        
        patent_features = calculate_patent_features(period_start, period_end)
        patent_features['year'] = year
        #print(patent_features)
        #sys.exit()
        predictors.append(patent_features)
    predictors = add_innovation_scores(predictors=predictors)

    print(predictors[0].head())
    logger.info("Finished generating predictors")
    
    # Show some stats
    logger.info(f"Number of unique counties: {len(patent_df['inventor_fips'].unique())}")
    
    # Save to TSV
    for i, predictors_df in enumerate(predictors):
        output_path = os.path.join(MODEL_FOLDER, f'traindata_{predictors_df["year"].iloc[0]}.tsv')
        predictors_df.to_csv(output_path, sep='\t', index=False)
        logger.info(f"Saved predictors for {predictors_df['year'].iloc[0]} to {local_filename(output_path)}")

if __name__ == '__main__':
    generate_predictors()
