# Analysing the Impact of Economic Crises on UK Living Standards

<!-- ABOUT THE PROJECT -->
### :beginner: About The Project
The project investigates the impact of economic crises on the United Kingdom (UK), with a particular focus on how such crises affect key macroeconomic indicators and broader societal well-being. Leveraging data from the Office for National Statistics (ONS), the study examines major UK economic crises between 1973 and 2022, with a deeper focus on the 2008 Global Financial Crisis and the 2022 Energy & Inflation Crisis. An economic crisis is defined as two consecutive quarters of negative GDP growth, with supporting indicators such as consumer confidence, interest rates, and house prices used to explore underlying causes and concurrent effects. Our aim is not to draw authoritative economic conclusions, but to apply data-driven methods—including statistical analysis, machine learning, and natural language processing (NLP) - to understand patterns in business activity, public health outcomes, cost of living, and demographic changes during times of crisis. Due to limitations in time and data availability, the scope was narrowed to selected variables for in-depth analysis. In addition to achieving core objectives like data collection, preprocessing, time-series and correlation analysis, and causal inference, we placed a strong emphasis on building modular and reusable code. This culminated in the development of an interactive dashboard in Streamlit, enabling users to upload their own datasets and generate dynamic visual insights. The project ultimately serves as a practical application of data science methods to complex, real-world economic challenges, offering tools for both historical understanding and future exploration.

<!-- ABOUT THE REPOSITORY -->
### :zap: About The Repository
This repository contains a Python-based analysis of key macroeconomic indicators, with a focus on income distribution, inequality, interest rates, exchange rates, house prices, and the redistributive impact of taxes and benefits. The project includes data processing, analytical methods (e.g., Gini coefficient, regression), and visualisations. Analyses on the ONS dataset were conducted in Jupyter Notebooks, while modularised code for data processing and analysis is provided in standalone Python files. The codebase follows a modular structure, with separate components for data management, cleaning, analysis, visualisation, and execution.

## :file_folder: Repository Structure

```
.
├── src
│   ├── app.py
│   ├── data.py
│   ├── data_analysis.py
│   ├── data_cleaning.py
│   ├── data_management.py
│   ├── data_visualization.py
│   ├── main.py
│   ├── stat.py
│   └── streamlit_version.py
│   └── requirements.txt
├── jupyter_notebooks
│   └── 2008_Global_Financial_Crises.ipynb
│   └── 2022_Energy_and_Inflation_Crises.ipynb
│   └── tax_analysis.ipynb
└── README.md
```

## :wrench: Getting Started

### :notebook: Pre-Requisites
- Python: Version 3.8 or higher
- Dependencies: Listed in requirements.txt.
- Data files: Datasets are not included due to potential sensitivity. Please use your own datasets.

### Installation
1. Clone the Repository:
    ```sh
    git clone https://github.com/your-username/uk-tax-benefits-analysis.git
    cd uk-tax-benefits-analysis
    ```
2. Install Dependencies:
    ```sh
    pip install -r requirements.txt
    ```
3. Place Data Files:
   - Copy the required Excel and CSV files to the data/ directory
   - Update file paths in main.py if necessary (default: ./data/)
   - Alternatively, use placeholder data as described in data/README.md.
4. Run the Analysis:
   ```sh
   python main.py 
   ```
6. [If you want to run Streamlit] Google Gemini API Key at https://ai.google.dev/gemini-api/docs/api-key
7. [If you want to run Streamlit] Enter your API key in 'streamlit_version.py'
    ```sh
    genai.configure(api_key="INSERT API KEY HERE")
    ```
8. Change git remote url to avoid accidental pushes to base project
   ```sh
   git remote set-url origin github_username/repo_name
   git remote -v # confirm the changes
   ```

## 🔥 Outputs
### Analysis
- Cleaned CSVs
- PNG plots in the working directory
- Console output includes data previews, statistics, and Granger causality tests
### Website
- Interactive dashboard where users can upload data and view the resulting charts

## 🖊️ Technical Notes
A TypeError in the redistribution impact plot was resolved by enforcing numeric types and handling NaNs (see tax_benefits_report.md).

The codebase is modular and extensible, with debugging outputs for transparency.
Data files are not included; users must provide their own or use placeholders.


