import pandas as pd

def clean_tax_data(tax_df):
    """
    Clean and transform the tax benefits DataFrame to long format.
    
    Args:
        tax_df (pd.DataFrame): Input tax benefits DataFrame.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame in long format.
    """
    # Remove rows with all NaN values
    tax_df = tax_df.dropna(how='all').reset_index(drop=True)
    
    # Extract header row (assumed to be the first row after dropping NaNs)
    header_row = tax_df.iloc[0]
    tax_df = tax_df[1:]  # Remove the header row from data
    tax_df.columns = header_row  # Set header as column names
    
    # Convert year columns to single years (e.g., '1977-78' -> '1977')
    tax_df.columns = [str(col).split('-')[0] if '-' in str(col) else str(col) for col in tax_df.columns]
    
    # Convert float-like years to integers (e.g., '1977.0' -> 1977)
    def convert_year(col):
        try:
            return int(float(col)) if str(col).replace('.', '').replace('-', '').isdigit() else col
        except (ValueError, TypeError):
            return col
    
    tax_df.columns = [convert_year(col) for col in tax_df.columns]
    
    # Convert to long format
    df_long = tax_df.melt(
        id_vars=['Geography', 'Geography code', 'Quintile', 'AveragesAndPercentiles', 'Income', 'Deflation'],
        var_name='Year',
        value_name='Value'
    )
    
    # Convert Year to integer
    df_long['Year'] = df_long['Year'].astype(int)
    
    # Drop rows with missing values in 'Value' column
    df_long = df_long.dropna(subset=['Value'])
    
    return df_long

def process_monthly_data(df, value_name):
    """
    Process monthly unemployment or inflation data to annual averages.
    
    Args:
        df (pd.DataFrame): Input DataFrame with monthly data.
        value_name (str): Name of the value column (e.g., 'Unemployment', 'Inflation').
    
    Returns:
        pd.DataFrame: Annual averages with 'Year' and value_name columns.
    """
    melted = df.melt(id_vars='Year', var_name='Month', value_name=value_name)
    melted = melted[melted['Month'] != 'Year']  # Remove any residual year rows
    melted[value_name] = pd.to_numeric(melted[value_name], errors='coerce')
    annual = melted.groupby('Year')[value_name].mean().reset_index()
    annual['Year'] = annual['Year'].astype(int)
    return annual

def clean_economic_data(unemp_df, infl_df):
    """
    Process and merge unemployment and inflation data.
    
    Args:
        unemp_df (pd.DataFrame): Unemployment DataFrame.
        infl_df (pd.DataFrame): Inflation DataFrame.
    
    Returns:
        pd.DataFrame: Merged DataFrame with annual averages.
    """
    unemp_annual = process_monthly_data(unemp_df, 'Unemployment')
    infl_annual = process_monthly_data(infl_df, 'Inflation')
    merged = pd.merge(unemp_annual, infl_annual, on='Year', how='inner')
    return merged

def prepare_gini_data(tax_df_long):
    """
    Prepare data for Gini coefficient calculation by pivoting disposable income.
    
    Args:
        tax_df_long (pd.DataFrame): Long-format tax benefits DataFrame.
    
    Returns:
        pd.DataFrame: Pivoted DataFrame with years as index and quintiles as columns.
    """
    gini_data = tax_df_long[
        (tax_df_long['Income'] == 'Disposable') &
        (tax_df_long['Deflation'] == 'Deflated value') &
        (tax_df_long['AveragesAndPercentiles'] == 'Mean')
    ]
    gini_pivot = gini_data.pivot_table(
        values='Value',
        index='Year',
        columns='Quintile'
    ).dropna()
    return gini_pivot

def save_cleaned_data(tax_df, econ_df, tax_output_path, econ_output_path):
    """
    Save the cleaned tax benefits and economic indicators DataFrames to CSV files.
    
    Args:
        tax_df (pd.DataFrame): Cleaned tax benefits DataFrame.
        econ_df (pd.DataFrame): Cleaned economic indicators DataFrame.
        tax_output_path (str): Path to save the tax benefits CSV.
        econ_output_path (str): Path to save the economic indicators CSV.
    """
    tax_df.to_csv(tax_output_path, index=False)
    econ_df.to_csv(econ_output_path, index=False)
    print(f"Cleaned tax benefits data saved to {tax_output_path}")
    print(f"Cleaned economic indicators data saved to {econ_output_path}")
