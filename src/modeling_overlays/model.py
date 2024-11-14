#!/usr/bin/env python3
#-*- coding: utf-8 -*-
"""
Created: Wed Nov 13 2024
Author: Aidan Allchin

Trains the model using the generated data in `generate_predictors.py`. The
model is then saved to disk for later use.

IF YOU RUN THIS ON MACOS AND GET AN ERROR ABOUT `lightgbm` NOT BEING ABLE TO
LOAD, TRY RUNNING THIS COMMAND IN THE TERMINAL:
`brew install libomp`
"""
import os, sys
from pathlib import Path

# ALL FILES IN THE PROJECT SHOULD BE IMPORTED RELATIVE TO THE PROJECT ROOT
# the # of `.parent`s should be adjusted based on the file's location
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
try:
    import lightgbm as lgb
except OSError as e:
    print(e)
    print("Try running `brew install libomp` in the terminal if on MacOS. Then in your correct python environment run the following:")
    print("pip uninstall lightgbm\npip install lightgbm --no-binary lightgbm")
    print("If it's still not working, you might have one of the Apple Silicon Macs, and I have not found a way to get lightgbm to work on those yet.")
    sys.exit(1)
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List, Dict, Tuple
from src.other.logging import PatentLogger
from src.other.helpers import local_filename

###############################################################################
#                               CONFIGURATION                                 #
###############################################################################

# Initialize logger
logger = PatentLogger.get_logger(__name__)

# Directories
DATA_PATH = os.path.join(project_root, 'data')
PREDICTORS_PATH = os.path.join(DATA_PATH, 'model', 'training_data.tsv')

# Model
MODEL_PATH = os.path.join(DATA_PATH, 'model', 'model.pkl')
RANDOM_STATE = 42


###############################################################################
#                              MAIN FUNCTIONS                                 #
###############################################################################



class InnovationPredictor:
    def __init__(self, params: Dict = None):
        """Initialize with same parameters as before"""
        self.default_params = {
            'objective': 'regression',
            'boosting': 'dart',
            'num_leaves': 31,
            'learning_rate': 0.01,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_data_in_leaf': 20,
            'max_depth': 6,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1,
            'metric': 'rmse'
        }
        self.params = params if params else self.default_params
        self.model = None
        self.feature_importances = None
        
    def prepare_data_for_training(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """
        Prepare data for training, separating 2022 data and features.
        
        Args:
            df: Complete DataFrame including 2022
            
        Returns:
            Tuple of (training features, 2022 features, training targets)
        """
        # Separate 2022 data (which has no next_innovation_score)
        data_2022     = df[df['year'] == 2022].copy()
        training_data = df[df['year'] < 2022].copy()
        
        # Define feature columns
        feature_cols  = [
            'num_patents', 'num_inventors', 'num_assignees', 
            'unique_wipo_fields', 'diversity_score', 'exports_score',
            'population_pct_change', 'economic_momentum', 'county_modifier'
        ]
        
        # Extract features and target
        X_train = training_data[feature_cols]
        X_2022  = data_2022[feature_cols]
        y_train = training_data['next_innovation_score']
        
        return X_train, X_2022, y_train
    
    def train_and_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Train model and generate predictions for all years including 2022.
        
        Args:
            df: Complete DataFrame including 2022 data
            
        Returns:
            DataFrame with original data plus predictions
        """
        # Prepare data
        X_train, X_2022, y_train = self.prepare_data_for_training(df)
        
        # Create training dataset
        train_data = lgb.Dataset(X_train, label=y_train)
        
        # Train model
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=1000,
            callbacks=[lgb.log_evaluation(period=100)]
        )
        
        # Store feature importances
        self.feature_importances = dict(zip(
            X_train.columns,
            self.model.feature_importance(importance_type='gain')
        ))

        # Plot feature importances
        fig, ax = plt.subplots()
        ax.barh(list(self.feature_importances.keys()), list(self.feature_importances.values()))
        ax.set_xlabel('Feature Importance')
        ax.set_ylabel('Feature')
        ax.set_title('Feature Importances')

        fig_path = os.path.join(DATA_PATH, 'model', 'feature_importances.png')
        fig.savefig(fig_path)
        logger.info(f"Feature importances saved to {fig_path}")
        
        # Generate predictions for all years
        results_df = df.copy()
        
        # Generate predictions for non-2022 data
        non_2022_mask = results_df['year'] < 2022
        X_all = results_df.loc[non_2022_mask, X_train.columns]
        results_df.loc[non_2022_mask, 'predicted_next_innovation_score'] = self.model.predict(X_all)
        
        # Generate predictions for 2022
        results_df.loc[results_df['year'] == 2022, 'predicted_next_innovation_score'] = self.model.predict(X_2022)
        
        return results_df

    def evaluate_predictions(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Obviously we only want to evaluate model performance on non-2022 data 
        where we have actual values.
        
        Args:
            df: DataFrame with predictions and actual values
            
        Returns:
            Dict of evaluation metrics
        """
        eval_df = df[df['year'] < 2022].copy()
        
        metrics = {
            'rmse': np.sqrt(mean_squared_error(
                eval_df['next_innovation_score'], 
                eval_df['predicted_next_innovation_score']
            )),
            'r2': r2_score(
                eval_df['next_innovation_score'], 
                eval_df['predicted_next_innovation_score']
            )
        }
        
        return metrics
    
def display_prediction_summary(df: pd.DataFrame) -> None:
    """
    Display a summary of predictions with geographic context.
    
    Args:
        df: DataFrame with predictions and location data
    """
    predictions_2022 = df[df['year'] == 2022]
    print("\nTop 10 predicted innovation hubs for 2023 (based on 2022 data):")
    top_predictions = predictions_2022.nlargest(10, 'predicted_next_innovation_score')
    
    # Create a clean summary table
    summary = top_predictions[[
        'state_name', 'county_name', 'predicted_next_innovation_score',
        'latitude', 'longitude'
    ]].copy()
    
    # Round predictions for cleaner display
    summary['predicted_next_innovation_score'] = summary['predicted_next_innovation_score'].round(4)
    
    # Fixing formatting
    pd.set_option('display.max_rows', None)
    print(summary.to_string(index=False))

def generate_predictions_file(input_path: str, output_path: str):
    """
    Generate predictions file from training data.
    
    Args:
        input_path: Path to training_data.tsv
        output_path: Path to save predictions.tsv
    """
    # Load data
    df = pd.read_csv(input_path, sep='\t')
    
    # Initialize and train model
    predictor  = InnovationPredictor()
    results_df = predictor.train_and_predict(df)
    
    # Evaluate model performance
    metrics = predictor.evaluate_predictions(results_df)
    logger.info(f"Model Performance:")
    logger.info(f"RMSE: {metrics['rmse']:.4f}")
    logger.info(f"R^2: {metrics['r2']:.4f}")

    # Drop unnecessary columns
    results_df = results_df.drop(columns=[
        'num_patents', 'num_inventors', 'num_assignees', 
        'unique_wipo_fields', 'diversity_score', 'exports_score',
        'population_pct_change', 'economic_momentum', 'county_modifier'
    ])
    
    # Save predictions
    results_df.to_csv(output_path, sep='\t', index=False)
    logger.info(f"\nPredictions saved to {output_path}")
    logger.info(f"Number of predictions for 2022: {len(results_df[results_df['year'] == 2022])}")


if __name__ == "__main__":
    input_path  = os.path.join(project_root, 'data', 'model', 'training_data.tsv')
    output_path = os.path.join(project_root, 'data', 'model', 'predictions.tsv')
    
    generate_predictions_file(input_path, output_path)
    
    # Additional reporting
    results = pd.read_csv(output_path, sep='\t')
    predictions_2022 = results[results['year'] == 2022]
    
    logger.info("\nTop 10 predicted innovation hubs for 2023 (based on 2022 data):")
    top_predictions = predictions_2022.nlargest(10, 'predicted_next_innovation_score')
    logger.info(top_predictions[['county_fips', 'predicted_next_innovation_score']])


