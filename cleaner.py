import pandas as pd

def clean_dataset(ds, missing_threshold=0.3, retain_columns=None):
    """
    Cleans the dataset by removing columns with more than a specified threshold of missing values,
    but retains specified columns regardless of missing values.

    Parameters:
        ds (pd.DataFrame): The input dataset to clean.
        missing_threshold (float): The threshold for missing values, default is 0.3 (30%).
        retain_columns (list): List of column names to retain in the cleaned dataset, regardless of missing values.

    Returns:
        pd.DataFrame: A new DataFrame with columns containing less than the specified threshold of missing values,
                      plus any retained columns.
    """
    if retain_columns is None:
        retain_columns = []

    # Calculate the percentage of missing values per column
    missing_percentages = (ds.isna() | (ds == 'MISSING')).mean()
    
    # Filter columns where the percentage of missing values is less than the threshold
    columns_to_keep = missing_percentages[missing_percentages < missing_threshold].index.tolist()
    
    # Add the retained columns, ensuring no duplicates
    columns_to_keep = list(set(columns_to_keep + retain_columns))
    
    # Create a new DataFrame with the filtered columns
    cleaned_ds = ds[columns_to_keep].copy()
    
    # Optional: Display the columns retained and removed
    removed_columns = [col for col in ds.columns if col not in columns_to_keep]
    print("Dataset cleaned. Columns removed:", removed_columns)
    
    # Delete all rows that have NaN total_area_sqm
    cleaned_ds = cleaned_ds.dropna(subset=['total_area_sqm', 'latitude', 'longitude'])

    return cleaned_ds


def remove_NaN(ds):
    """
    Replaces NaN values in specific columns with the mean values, calculated based on property type.
    
    Parameters:
        ds (pd.DataFrame): The dataset in which to replace NaN values.
    
    Returns:
        pd.DataFrame: A DataFrame with NaN values in specific columns replaced by property-type-specific mean values.
    """
    # Calculate median terrace size based on property type
    total_area_means = ds.groupby('property_type')['total_area_sqm'].median()
    terrace_means = ds.groupby('property_type')['terrace_sqm'].median()
    garden_means = ds.groupby('property_type')['garden_sqm'].median()
    

    # Fill NaN values in total_area_sqm with the median based on property type
    ds['total_area_sqm'] = ds.apply(
        lambda row: total_area_means[row['property_type']] if pd.isna(row['total_area_sqm']) else row['total_area_sqm'],
        axis=1
    )
    
    # Fill NaN values in terrace_sqm with the median based on property type
    ds['terrace_sqm'] = ds.apply(
        lambda row: terrace_means[row['property_type']] if pd.isna(row['terrace_sqm']) else row['terrace_sqm'],
        axis=1
    )

    # Fill NaN values in garden_sqm with the median based on property type
    ds['garden_sqm'] = ds.apply(
        lambda row: garden_means[row['property_type']] if pd.isna(row['garden_sqm']) else row['garden_sqm'],
        axis=1
    )
    
    print("NaN values replaced with mean values for terrace_sqm by property type and overall mean for garden_sqm.")
    
    return ds
