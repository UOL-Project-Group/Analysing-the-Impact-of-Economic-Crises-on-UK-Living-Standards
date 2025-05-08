from flask import Flask, render_template
import matplotlib.pyplot as plt
import pandas as pd
import os
from data_management import load_data
from data_cleaning import clean_tax_data, clean_economic_data, prepare_gini_data
from data_analysis import (
    calculate_income_inequality,
    calculate_income_growth_rates,
    calculate_gini_coefficient,
    calculate_redistribution_impact,
    analyze_economic_indicators
)
from data_visualization import (
    plot_disposable_income_trends,
    plot_income_inequality,
    plot_deflated_vs_undeflated,
    plot_residuals,
    plot_economic_time_series,
    plot_economic_heatmap,
    plot_economic_rolling_averages,
    plot_income_growth_rates,
    plot_gini_coefficient,
    plot_redistribution_impact,
    plot_income_composition
)

app = Flask(__name__)

# Data file paths
DATA_DIR = 'data'
TAX_FILE = os.path.join(DATA_DIR, 'csv/tax-benefits-statistics-time-series-v3-filtered-2025-03-17T15-44-25Z_data.csv')  # Updated to CSV
UNEMP_FILE = os.path.join(DATA_DIR, 'csv/Unemployment_rate_monthly.csv')
INFL_FILE = os.path.join(DATA_DIR, 'csv/Inflation_monthly.csv')

# Plot directory
PLOT_DIR = os.path.join('static', 'plots')
if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)

# Modified load_data function to handle CSV for tax file
def custom_load_data(tax_file, unemp_file, infl_file):
    try:
        if tax_file.endswith('.csv'):
            tax_df = pd.read_csv(tax_file)
        else:
            tax_df = pd.read_excel(tax_file, sheet_name=0)
        unemp_df = pd.read_csv(unemp_file)
        infl_df = pd.read_csv(infl_file)
        print("All data loaded successfully.")
        return tax_df, unemp_df, infl_df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None

# Load and process data
tax_df, unemp_df, infl_df = custom_load_data(TAX_FILE, UNEMP_FILE, INFL_FILE)
if tax_df is None or unemp_df is None or infl_df is None:
    raise Exception("Failed to load data files")

tax_df_long = clean_tax_data(tax_df)
econ_df = clean_economic_data(unemp_df, infl_df)
gini_data = prepare_gini_data(tax_df_long)
inequality = calculate_income_inequality(tax_df_long)
growth_rates = calculate_income_growth_rates(tax_df_long)
gini_df = calculate_gini_coefficient(gini_data)
redistribution_df = calculate_redistribution_impact(tax_df_long)

# Route to display all plots and analysis
@app.route('/')
def home():
    plot_urls = {}
    plots = [
        ('disposable_income_trends', plot_disposable_income_trends, [tax_df_long]),
        ('income_inequality', plot_income_inequality, [inequality]),
        ('deflated_vs_undeflated', plot_deflated_vs_undeflated, [tax_df_long]),
        ('residuals', plot_residuals, [tax_df_long]),
        ('economic_time_series', plot_economic_time_series, [econ_df]),
        ('economic_heatmap', plot_economic_heatmap, [econ_df]),
        ('economic_rolling_averages', plot_economic_rolling_averages, [econ_df]),
        ('income_growth_rates', plot_income_growth_rates, [growth_rates]),
        ('gini_coefficient', plot_gini_coefficient, [gini_df]),
        ('redistribution_impact', plot_redistribution_impact, [redistribution_df]),
        ('income_composition', plot_income_composition, [tax_df_long])
    ]

    for plot_name, plot_func, args in plots:
        try:
            fig = plot_func(*args)
            if fig is None:
                print(f"Skipping {plot_name} due to empty data")
                continue
            plot_path = os.path.join(PLOT_DIR, f'{plot_name}.png')
            fig.savefig(plot_path, format='png', bbox_inches='tight')
            plt.close(fig)
            plot_urls[plot_name] = f'plots/{plot_name}.png'
        except Exception as e:
            print(f"Error generating {plot_name}: {e}")

    # Capture analysis output
    from io import StringIO
    import sys
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    analyze_economic_indicators(econ_df)
    analysis_output = sys.stdout.getvalue()
    sys.stdout = old_stdout

    return render_template('index.html', plot_urls=plot_urls, analysis_output=analysis_output)

if __name__ == '__main__':
    app.run(debug=True)
