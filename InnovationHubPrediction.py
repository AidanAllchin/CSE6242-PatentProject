import requests
import pandas as pd
import numpy as np
from census import Census
from us import states
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import datetime

# Constants
BEA_API_KEY = 'F9919422-E7E7-4562-89DB-3051ACF0B6A8'
CENSUS_API_KEY = '29e0fb5c56cb38bb77a95450557344c1e217e72f'
FRED_API_KEY = '51fbd42a3ad7fc528dd32ac27499a7ee'

# 1. Fetch GDP Data from BEA API
def get_msa_gdp(bea_api_key, year, msa_fips):
    """
    Fetch GDP data for a specific MSA and year from BEA API.
    """
    url = "https://apps.bea.gov/api/data/"
    params = {
        'UserID': bea_api_key,
        'method': 'GetData',
        'datasetname': 'Regional',
        'TableName': 'CAGDP2',
        'LineCode': '1',  # Total GDP
        'GeoFips': msa_fips,
        'Year': year,
        'ResultFormat': 'JSON'
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    # Extract GDP value
    try:
        gdp_value = float(data['BEAAPI']['Results']['Data'][0]['DataValue'].replace(',', ''))
    except (KeyError, IndexError, ValueError):
        gdp_value = np.nan
    
    return gdp_value

# 2. Fetch Census Data
def get_msa_census_data(census_api_key, year):
    """
    Get MSA-level census data.
    """
    c = Census(census_api_key, year=year)
    
    # Get data for all MSAs
    msa_data = c.acs5.get(
        variables=[
            'B01003_001E',  # Total population
            'B15003_022E',  # Bachelor's degree
            'B15003_023E',  # Master's degree
            'B15003_024E',  # Professional degree
            'B15003_025E',  # Doctorate degree
            'B19013_001E',  # Median household income
            'B23025_002E',  # Labor force total
        ],
        geo={'for': 'metropolitan statistical area/micropolitan statistical area:*'}
    )
    
    # Convert to DataFrame
    df = pd.DataFrame(msa_data)
    
    # Calculate higher education percentage
    df = df.apply(pd.to_numeric, errors='ignore')
    df['higher_ed_percent'] = (
        df['B15003_022E'] + df['B15003_023E'] + df['B15003_024E'] + df['B15003_025E']
    ) / df['B01003_001E'] * 100
    
    # Drop unnecessary columns
    df = df.rename(columns={
        'B01003_001E': 'total_population',
        'B19013_001E': 'median_household_income',
        'B23025_002E': 'labor_force_total',
        'metropolitan statistical area/micropolitan statistical area': 'msa_fips'
    })
    df = df[['msa_fips', 'total_population', 'median_household_income', 'labor_force_total', 'higher_ed_percent']]
    
    return df

# 3. Map Patent Data to MSAs
def map_patents_to_msa(patent_df, msa_shapefile):
    """
    Map patent data to MSAs based on inventor locations.
    """
    # Load MSA shapefile
    msa_shapes = gpd.read_file(msa_shapefile)
    
    # Convert patent locations to GeoDataFrame
    patent_gdf = gpd.GeoDataFrame(
        patent_df,
        geometry=gpd.points_from_xy(patent_df['inventor_longitude'], patent_df['inventor_latitude']),
        crs='EPSG:4326'
    )
    
    # Spatial join patents with MSAs
    patent_msa = gpd.sjoin(patent_gdf, msa_shapes, how='left', op='within')
    
    # Drop patents without an MSA
    patent_msa = patent_msa.dropna(subset=['msa_fips'])
    
    return patent_msa

# 4. Calculate Patent Features
def calculate_patent_features(patent_msa_df, period_start, period_end):
    """
    Calculate patent-related features for each MSA within a time period.
    """
    period_df = patent_msa_df[
        (patent_msa_df['application_date'] >= period_start) &
        (patent_msa_df['application_date'] < period_end)
    ]
    
    # Group by MSA
    grouped = period_df.groupby('msa_fips')
    
    features = pd.DataFrame()
    features['patent_count'] = grouped['patent_id'].nunique()
    features['inventor_count'] = grouped['inventor_id'].nunique()
    features['assignee_count'] = grouped['assignee_id'].nunique()
    features['unique_ipc_classes'] = grouped['ipc_class'].nunique()
    
    # Diversity Score: Evenness of IPC class distribution (Shannon Diversity Index)
    def shannon_diversity(group):
        counts = group['ipc_class'].value_counts()
        proportions = counts / counts.sum()
        return -sum(proportions * np.log(proportions))
    
    features['diversity_score'] = grouped.apply(shannon_diversity)
    
    # Collaboration Score: Ratio of patents with multiple assignees
    def collaboration_ratio(group):
        multi_assignee = group[group['assignee_count'] > 1]['patent_id'].nunique()
        total_patents = group['patent_id'].nunique()
        return multi_assignee / total_patents if total_patents > 0 else 0
    
    features['collaboration_score'] = grouped.apply(collaboration_ratio)
    
    features.reset_index(inplace=True)
    
    return features

# 5. Calculate Economic Features
def get_economic_features(msa_list, year):
    """
    Fetch economic features for a list of MSAs.
    """
    # Initialize DataFrame
    economic_df = pd.DataFrame()
    economic_df['msa_fips'] = msa_list
    economic_df['year'] = year
    
    # Fetch GDP per capita for each MSA
    gdp_values = []
    for msa in msa_list:
        gdp = get_msa_gdp(BEA_API_KEY, str(year), msa)
        gdp_values.append(gdp)
    economic_df['gdp'] = gdp_values
    
    # Fetch Census data
    census_df = get_msa_census_data(CENSUS_API_KEY, year)
    
    # Merge GDP and Census data
    economic_features = pd.merge(economic_df, census_df, on='msa_fips', how='left')
    
    # Calculate GDP per capita
    economic_features['gdp_per_capita'] = economic_features['gdp'] / economic_features['total_population']
    
    return economic_features

# 6. Combine All Features
def combine_features(patent_features, economic_features):
    """
    Combine patent-related features with economic features.
    """
    combined_df = pd.merge(patent_features, economic_features, on='msa_fips', how='left')
    
    # Fill missing values
    combined_df.fillna(0, inplace=True)
    
    return combined_df

# 7. Calculate Response Variable
def calculate_response_variable(patent_msa_df, period_start, period_end):
    """
    Calculate the next_period_innovation_score for each MSA.
    """
    # Shift period by one (next period)
    next_period_start = period_end
    next_period_end = period_end + (period_end - period_start)
    
    next_period_features = calculate_patent_features(patent_msa_df, next_period_start, next_period_end)
    
    # Here, we'll define the innovation score as a normalized sum of selected features
    scaler = MinMaxScaler()
    next_period_features[['patent_count', 'diversity_score', 'collaboration_score']] = scaler.fit_transform(
        next_period_features[['patent_count', 'diversity_score', 'collaboration_score']]
    )
    
    # Calculate innovation score
    weights = {
        'patent_count': 0.4,
        'diversity_score': 0.3,
        'collaboration_score': 0.3
    }
    next_period_features['next_period_innovation_score'] = (
        next_period_features['patent_count'] * weights['patent_count'] +
        next_period_features['diversity_score'] * weights['diversity_score'] +
        next_period_features['collaboration_score'] * weights['collaboration_score']
    )
    
    return next_period_features[['msa_fips', 'next_period_innovation_score']]

# 8. Prepare Final Dataset
def prepare_dataset(patent_msa_df, period_start, period_end):
    """
    Prepare the final dataset for modeling.
    """
    # Calculate features for the period
    patent_features = calculate_patent_features(patent_msa_df, period_start, period_end)
    
    # Get list of MSAs
    msa_list = patent_features['msa_fips'].tolist()
    
    # Get economic features
    year = period_start.year
    economic_features = get_economic_features(msa_list, year)
    
    # Combine features
    combined_features = combine_features(patent_features, economic_features)
    
    # Calculate response variable
    response_variable = calculate_response_variable(patent_msa_df, period_start, period_end)
    
    # Merge response variable
    final_dataset = pd.merge(combined_features, response_variable, on='msa_fips', how='left')
    
    # Drop rows with missing response variable
    final_dataset.dropna(subset=['next_period_innovation_score'], inplace=True)
    
    return final_dataset

# 9. Model Training
def train_model(final_dataset):
    """
    Train a predictive model using the prepared dataset.
    """
    # Split features and target
    X = final_dataset.drop(columns=['msa_fips', 'next_period_innovation_score'])
    y = final_dataset['next_period_innovation_score']
    
    # Split into training and testing sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
    
    # Initialize and train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predict and evaluate
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f'RMSE: {rmse}')
    
    return model

# 10. Main Execution
def main():
    # Define period
    period_start = pd.Timestamp('2017-01-01')
    period_end = pd.Timestamp('2018-01-01')
    
    # Load patent data (placeholder)
    # patent_df = pd.read_csv('us_patent_data.csv')
    # Assuming patent_df has columns: patent_id, application_date, inventor_id, inventor_latitude, inventor_longitude, assignee_id, ipc_class
    # For this example, we'll create a mock DataFrame
    patent_df = pd.DataFrame({
        'patent_id': np.arange(1, 1001),
        'application_date': pd.date_range(start='2017-01-01', periods=1000, freq='D'),
        'inventor_id': np.random.randint(1, 500, size=1000),
        'inventor_latitude': np.random.uniform(25, 50, size=1000),
        'inventor_longitude': np.random.uniform(-125, -66, size=1000),
        'assignee_id': np.random.randint(1, 300, size=1000),
        'ipc_class': np.random.choice(['A', 'B', 'C', 'D', 'E'], size=1000)
    })
    
    # Map patents to MSAs
    msa_shapefile = 'tl_2024_us_cbsa/tl_2024_us_cbsa.shp'  
    patent_msa_df = map_patents_to_msa(patent_df, msa_shapefile)
    
    # Prepare dataset
    final_dataset = prepare_dataset(patent_msa_df, period_start, period_end)
    
    # Train model
    model = train_model(final_dataset)
    
    # Save model
    # import joblib
    # joblib.dump(model, 'innovation_model.pkl')

if __name__ == '__main__':
    main()
