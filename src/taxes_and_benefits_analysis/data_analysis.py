import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests

def calculate_income_inequality(tax_df_long):
    """
    Calculate the income ratio between the 5th and 1st quintiles.
    
    Args:
        tax_df_long (pd.DataFrame): Long-format tax benefits DataFrame.
    
    Returns:
        pd.DataFrame: DataFrame with income ratio.
    """
    disposable_income = tax_df_long[
        (tax_df_long['Income'] == 'Disposable') &
        (tax_df_long['Deflation'] == 'Deflated value') &
        (tax_df_long['AveragesAndPercentiles'] == 'Mean')
    ]
    
    quintile_1 = disposable_income[disposable_income['Quintile'] == '1st']
    quintile_5 = disposable_income[disposable_income['Quintile'] == '5th']
    
    inequality = pd.merge(quintile_1, quintile_5, on='Year', suffixes=('_1st', '_5th'))
    
    inequality['Income Ratio'] = inequality['Value_5th'] / inequality['Value_1st']
    
    return inequality

def calculate_income_growth_rates(tax_df_long):
    """
    Calculate annual growth rates of deflated disposable income by quintile.
    
    Args:
        tax_df_long (pd.DataFrame): Long-format tax benefits DataFrame.
    
    Returns:
        pd.DataFrame: DataFrame with growth rates by year and quintile.
    """
    disposable_income = tax_df_long[
        (tax_df_long['Income'] == 'Disposable') &
        (tax_df_long['Deflation'] == 'Deflated value') &
        (tax_df_long['AveragesAndPercentiles'] == 'Mean')
    ]
    pivot = disposable_income.pivot_table(values='Value', index='Year', columns='Quintile').dropna()
    growth_rates = pivot.pct_change() * 100  # Convert to percentage
    growth_rates = growth_rates.reset_index().melt(id_vars='Year', var_name='Quintile', value_name='Growth Rate')
    growth_rates = growth_rates.dropna()
    return growth_rates

def calculate_gini_coefficient(gini_data):
    """
    Calculate Gini coefficient for each year based on quintile incomes.
    
    Args:
        gini_data (pd.DataFrame): Pivoted DataFrame with years as index and quintiles as columns.
    
    Returns:
        pd.DataFrame: DataFrame with Year and Gini coefficient.
    """
    gini_coeffs = []
    years = gini_data.index
    
    for year in years:
        incomes = gini_data.loc[year].values
        n = len(incomes)
        if n == 0 or np.any(incomes < 0):
            continue
        
        # Sort incomes
        incomes = np.sort(incomes)
        # Calculate cumulative income and population shares
        cum_income = np.cumsum(incomes) / np.sum(incomes)
        cum_pop = np.arange(1, n + 1) / n
        # Calculate area under Lorenz curve using trapezoidal rule
        area_under_lorenz = np.trapz(cum_income, cum_pop)
        # Gini coefficient = 1 - 2 * area under Lorenz curve
        gini = 1 - 2 * area_under_lorenz
        gini_coeffs.append({'Year': int(year), 'Gini Coefficient': gini})
    
    return pd.DataFrame(gini_coeffs)

def calculate_redistribution_impact(tax_df_long):
    """
    Calculate the redistributive impact of taxes and benefits by quintile.
    
    Args:
        tax_df_long (pd.DataFrame): Long-format tax benefits DataFrame.
    
    Returns:
        pd.DataFrame: DataFrame with redistribution impact by year and quintile.
    """
    # Ensure Value is numeric
    tax_df_long['Value'] = pd.to_numeric(tax_df_long['Value'], errors='coerce')
    
    income_data = tax_df_long[
        (tax_df_long['Income'].isin(['Original', 'Disposable'])) &
        (tax_df_long['Deflation'] == 'Deflated value') &
        (tax_df_long['AveragesAndPercentiles'] == 'Mean')
    ]
    
    # Pivot table to get Original and Disposable incomes
    pivot = income_data.pivot_table(
        values='Value', 
        index=['Year', 'Quintile'], 
        columns='Income'
    ).dropna()
    
    # Calculate Redistribution Impact
    pivot['Redistribution Impact'] = pivot['Disposable'] - pivot['Original']
    
    # Reset index and ensure types
    result = pivot.reset_index()[['Year', 'Quintile', 'Redistribution Impact']]
    result['Year'] = result['Year'].astype(int)
    result['Quintile'] = result['Quintile'].astype(str)
    result['Redistribution Impact'] = pd.to_numeric(result['Redistribution Impact'], errors='coerce')
    
    # Drop any rows with NaN in Redistribution Impact
    result = result.dropna(subset=['Redistribution Impact'])
    
    return result

def analyze_economic_indicators(econ_df):
    """
    Perform descriptive statistics and Granger causality tests for economic indicators.
    
    Args:
        econ_df (pd.DataFrame): Merged DataFrame with 'Year', 'Unemployment', 'Inflation'.
    """
    print("\nDescriptive Statistics for Economic Indicators:")
    print(econ_df.describe())
    
    print("\nGranger Causality Tests (max lag = 2):")
    data = econ_df[['Unemployment', 'Inflation']].dropna()
    try:
        print("Testing if Inflation Granger-causes Unemployment:")
        grangercausalitytests(data[['Unemployment', 'Inflation']], maxlag=2, verbose=True)
        
        print("\nTesting if Unemployment Granger-causes Inflation:")
        grangercausalitytests(data[['Inflation', 'Unemployment']], maxlag=2, verbose=True)
    except Exception as e:
        print(f"Error in Granger causality tests: {e}")
