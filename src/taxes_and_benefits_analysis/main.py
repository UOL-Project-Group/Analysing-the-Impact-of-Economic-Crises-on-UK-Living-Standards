from data_management import load_data
from data_cleaning import clean_tax_data, clean_economic_data, prepare_gini_data, save_cleaned_data
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

def main():
    # File paths
    tax_input_file = 'csv/tax-benefits-statistics-time-series-v3-filtered-2025-03-17T15-44-25Z_data.xlsx
    unemp_input_file = 'csv/Inflation_monthly_data.csv'
    infl_input_file = 'csv/Inflation_monthly_data.csv'
    tax_output_csv = 'cleaned_tax_benefits_data.csv'
    econ_output_csv = 'cleaned_economic_indicators_data.csv'
    
    # Load data
    print("Loading data...")
    tax_df, unemp_df, infl_df = load_data(tax_input_file, unemp_input_file, infl_input_file)
    if tax_df is None or unemp_df is None or infl_df is None:
        return
    
    # Clean data
    print("\nCleaning data...")
    tax_df_long = clean_tax_data(tax_df)
    econ_df = clean_economic_data(unemp_df, infl_df)
    gini_data = prepare_gini_data(tax_df_long)
    
    # Save cleaned data
    save_cleaned_data(tax_df_long, econ_df, tax_output_csv, econ_output_csv)
    
    # Display cleaned data previews
    print("\nCleaned tax benefits dataset preview:")
    print(tax_df_long.head())
    print(f"Tax benefits dataset shape: {tax_df_long.shape}")
    
    print("\nCleaned economic indicators dataset preview:")
    print(econ_df.head())
    print(f"Economic indicators dataset shape: {econ_df.shape}")
    
    # Perform analysis
    print("\nPerforming analysis...")
    inequality = calculate_income_inequality(tax_df_long)
    growth_rates = calculate_income_growth_rates(tax_df_long)
    gini_df = calculate_gini_coefficient(gini_data)
    redistribution_df = calculate_redistribution_impact(tax_df_long)
    analyze_economic_indicators(econ_df)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_disposable_income_trends(tax_df_long)
    plot_income_inequality(inequality)
    plot_deflated_vs_undeflated(tax_df_long)
    plot_residuals(tax_df_long)
    plot_income_growth_rates(growth_rates)
    plot_gini_coefficient(gini_df)
    plot_redistribution_impact(redistribution_df)
    plot_income_composition(tax_df_long)
    plot_economic_time_series(econ_df)
    plot_economic_heatmap(econ_df)
    plot_economic_rolling_averages(econ_df)

if __name__ == "__main__":
    main()
