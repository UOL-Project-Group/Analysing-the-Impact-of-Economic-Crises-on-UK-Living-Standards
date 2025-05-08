import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
import statsmodels.nonparametric.smoothers_lowess as sm_lowess

# Define crisis periods
crisis_periods = {
    (1990, 1991): 'blue',   # 1990-1991 Recession
    (1997, 1998): 'green',  # 1997-1998 Asian Financial Crisis
    (2007, 2009): 'purple', # 2008 Global Financial Crisis
}

def plot_disposable_income_trends(tax_df_long):
    disposable_income = tax_df_long[
        (tax_df_long['Income'] == 'Disposable') &
        (tax_df_long['Deflation'] == 'Deflated value') &
        (tax_df_long['AveragesAndPercentiles'] == 'Mean')
    ]
    disposable_income = disposable_income.sort_values(by='Year')
    fig = plt.figure(figsize=(14, 7))
    sns.lineplot(data=disposable_income, x='Year', y='Value', hue='Quintile', palette='viridis')
    plt.title('Trends in Mean Disposable Income (Deflated) by Quintile (1977-2021)', fontsize=16)
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Income (£)', fontsize=14)
    for (start, end), color in crisis_periods.items():
        plt.axvspan(start, end, color=color, alpha=0.2, label=f'Crisis {start}-{end}')
    plt.legend(title='Quintile', title_fontsize=12, fontsize=12)
    plt.grid(True)
    plt.xticks(ticks=disposable_income['Year'].unique(), rotation=45)
    plt.tight_layout()
    return fig

def plot_income_inequality(inequality):
    fig = plt.figure(figsize=(12, 6))
    sns.lineplot(data=inequality, x='Year', y='Income Ratio', color='red')
    plt.title('Income Inequality (5th Quintile / 1st Quintile) Over Time (1977-2021)', fontsize=16)
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Income Ratio', fontsize=14)
    for (start, end), color in crisis_periods.items():
        plt.axvspan(start, end, color=color, alpha=0.2, label=f'Crisis {start}-{end}')
    plt.grid(True)
    plt.xticks(ticks=inequality['Year'].unique(), rotation=45)
    plt.tight_layout()
    return fig

def plot_deflated_vs_undeflated(tax_df_long):
    income = tax_df_long[
        (tax_df_long['Income'] == 'Disposable') &
        (tax_df_long['AveragesAndPercentiles'] == 'Mean') &
        (tax_df_long['Quintile'] == '1st')
    ]
    income_pivot = income.pivot_table(values='Value', index='Year', columns='Deflation').reset_index()
    fig = plt.figure(figsize=(12, 6))
    plt.plot(income_pivot['Year'], income_pivot['Deflated value'], label='Deflated', color='blue')
    plt.plot(income_pivot['Year'], income_pivot['Undeflated value'], label='Undeflated', color='orange')
    plt.title('Deflated vs. Undeflated Disposable Income Over Time (1977-2021)', fontsize=16)
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Income (£)', fontsize=14)
    for (start, end), color in crisis_periods.items():
        plt.axvspan(start, end, color=color, alpha=0.2, label=f'Crisis {start}-{end}')
    plt.legend(title='Deflation Status', title_fontsize=12, fontsize=12)
    plt.grid(True)
    plt.xticks(ticks=income_pivot['Year'].unique(), rotation=45)
    plt.tight_layout()
    return fig

def plot_residuals(tax_df_long):
    disposable_income = tax_df_long[
        (tax_df_long['Income'] == 'Disposable') &
        (tax_df_long['Deflation'] == 'Deflated value') &
        (tax_df_long['AveragesAndPercentiles'] == 'Mean') &
        (tax_df_long['Quintile'] == '1st')
    ]
    X = disposable_income['Year'].values.reshape(-1, 1)
    y = disposable_income['Value'].values
    degree = 2
    polyreg = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    polyreg.fit(X, y)
    y_pred = polyreg.predict(X)
    residuals = y - y_pred
    sorted_indices = np.argsort(y_pred)
    y_pred_sorted = y_pred[sorted_indices]
    residuals_sorted = residuals[sorted_indices]
    lowess_result = sm_lowess.lowess(residuals_sorted, y_pred_sorted, frac=0.2)
    smoothed_x = lowess_result[:, 0]
    smoothed_y = lowess_result[:, 1]
    fig = plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5, label='Residuals')
    plt.plot(smoothed_x, smoothed_y, color='red', linewidth=2, label='LOWESS Curve')
    plt.title('Residual Plot with LOWESS Curve (Polynomial Regression)', fontsize=16)
    plt.xlabel('Predicted Income (£)', fontsize=14)
    plt.ylabel('Residuals', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    return fig

def plot_economic_time_series(econ_df):
    fig = plt.figure(figsize=(14, 7))
    ax1 = plt.gca()
    sns.lineplot(data=econ_df, x='Year', y='Unemployment', color='green', ax=ax1, label='Unemployment')
    ax2 = ax1.twinx()
    sns.lineplot(data=econ_df, x='Year', y='Inflation', color='red', ax=ax2, label='Inflation')
    for (start, end), color in crisis_periods.items():
        plt.axvspan(start, end, color=color, alpha=0.2, label=f'Crisis {start}-{end}')
    ax1.set_xlabel('Year', fontsize=14)
    ax1.set_ylabel('Unemployment (%)', color='green', fontsize=14)
    ax2.set_ylabel('Inflation (%)', color='red', fontsize=14)
    plt.title('Economic Indicators Over Time', fontsize=16)
    ax1.legend(loc='upper left', fontsize=12)
    ax2.legend(loc='upper right', fontsize=12)
    plt.grid(True)
    plt.xticks(ticks=econ_df['Year'].unique(), rotation=90)
    plt.tight_layout()
    return fig

def plot_economic_heatmap(econ_df):
    corr_matrix = econ_df[['Unemployment', 'Inflation']].corr()
    fig = plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Between Economic Indicators', fontsize=16)
    plt.tight_layout()
    return fig

def plot_economic_rolling_averages(econ_df, window=5):
    econ_df = econ_df.copy()
    econ_df.set_index('Year', inplace=True)
    rolling = econ_df.rolling(window=window).mean()
    fig = plt.figure(figsize=(14, 8))
    rolling.plot(subplots=True, layout=(2, 1), title='5-Year Rolling Averages')
    plt.tight_layout()
    econ_df.reset_index(inplace=True)
    return fig

def plot_income_growth_rates(growth_rates):
    fig = plt.figure(figsize=(14, 7))
    sns.lineplot(data=growth_rates, x='Year', y='Growth Rate', hue='Quintile', palette='viridis')
    plt.title('Annual Growth Rates of Disposable Income by Quintile (1977-2021)', fontsize=16)
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Growth Rate (%)', fontsize=14)
    for (start, end), color in crisis_periods.items():
        plt.axvspan(start, end, color=color, alpha=0.2, label=f'Crisis {start}-{end}')
    plt.legend(title='Quintile', title_fontsize=12, fontsize=12)
    plt.grid(True)
    plt.xticks(ticks=growth_rates['Year'].unique(), rotation=45)
    plt.tight_layout()
    return fig

def plot_gini_coefficient(gini_df):
    fig = plt.figure(figsize=(12, 6))
    sns.lineplot(data=gini_df, x='Year', y='Gini Coefficient', color='teal')
    plt.title('Gini Coefficient of Disposable Income Over Time (1977-2021)', fontsize=16)
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Gini Coefficient', fontsize=14)
    for (start, end), color in crisis_periods.items():
        plt.axvspan(start, end, color=color, alpha=0.2, label=f'Crisis {start}-{end}')
    plt.grid(True)
    plt.xticks(ticks=gini_df['Year'].unique(), rotation=45)
    plt.tight_layout()
    return fig

def plot_redistribution_impact(redistribution_df):
    redistribution_df = redistribution_df.copy()
    redistribution_df['Redistribution Impact'] = pd.to_numeric(redistribution_df['Redistribution Impact'], errors='coerce')
    selected_years = redistribution_df[redistribution_df['Year'].isin(range(1977, 2022, 5))]
    pivot = selected_years.pivot_table(
        values='Redistribution Impact',
        index='Year',
        columns='Quintile'
    )
    pivot = pivot.dropna(axis=1, how='any')
    if pivot.empty:
        return None
    fig = plt.figure(figsize=(12, 8))
    pivot.plot(kind='bar', stacked=False, width=0.8)
    plt.title('Redistribution Impact of Taxes and Benefits by Quintile', fontsize=16)
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Redistribution Impact (£)', fontsize=14)
    plt.legend(title='Quintile', title_fontsize=12, fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    return fig

def plot_income_composition(tax_df_long):
    income_data = tax_df_long[
        (tax_df_long['Income'].isin(['Original', 'Gross', 'Disposable', 'Final'])) &
        (tax_df_long['Deflation'] == 'Deflated value') &
        (tax_df_long['AveragesAndPercentiles'] == 'Mean') &
        (tax_df_long['Quintile'].isin(['1st', '5th']))
    ]
    pivot = income_data.pivot_table(values='Value', index=['Year', 'Quintile'], columns='Income').reset_index()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    q1_data = pivot[pivot['Quintile'] == '1st']
    ax1.stackplot(q1_data['Year'], 
                  q1_data['Original'], q1_data['Gross'], q1_data['Disposable'], q1_data['Final'],
                  labels=['Original', 'Gross', 'Disposable', 'Final'], alpha=0.8)
    ax1.set_title('Income Composition for 1st Quintile (1977-2021)', fontsize=14)
    ax1.set_ylabel('Income (£)', fontsize=12)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True)
    q5_data = pivot[pivot['Quintile'] == '5th']
    ax2.stackplot(q5_data['Year'], 
                  q5_data['Original'], q5_data['Gross'], q5_data['Disposable'], q5_data['Final'],
                  labels=['Original', 'Gross', 'Disposable', 'Final'], alpha=0.8)
    ax2.set_title('Income Composition for 5th Quintile (1977-2021)', fontsize=14)
    ax2.set_xlabel('Year', fontsize=12)
    ax2.set_ylabel('Income (£)', fontsize=12)
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True)
    plt.xticks(ticks=q1_data['Year'].unique(), rotation=45)
    plt.tight_layout()
    return fig
