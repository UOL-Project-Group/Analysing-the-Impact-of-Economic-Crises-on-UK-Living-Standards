import pandas as pd

def load_data(tax_file_path, unemp_file_path, infl_file_path):
    """
    Load the tax benefits Excel file and unemployment/inflation CSV files.
    
    Args:
        tax_file_path (str): Path to the tax benefits Excel file.
        unemp_file_path (str): Path to the unemployment CSV file.
        infl_file_path (str): Path to the inflation CSV file.
    
    Returns:
        tuple: (tax_df, unemp_df, infl_df) or (None, None, None) if loading fails.
    """
    try:
        tax_df = pd.read_excel(tax_file_path, sheet_name=0)
        unemp_df = pd.read_csv(unemp_file_path)
        infl_df = pd.read_csv(infl_file_path)
        print("All data loaded successfully.")
        return tax_df, unemp_df, infl_df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None
