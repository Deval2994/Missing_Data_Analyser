👻 Phantom Spectrum - analyse the missing and compare CCA vs imputation side by side

A user-friendly web app to analyze, visualize, and clean missing data from CSV files. This tool helps data scientists, analysts, and ML engineers quickly identify missing data patterns and apply different imputation techniques — all without writing a single line of code.

🔗 **Live App:** [Try it here on Streamlit 🚀](https://phantomspectrum-visualize-data-cleaning-techniques.streamlit.app/) 

🔍 Features
📊 Upload your own dataset (CSV format)

📉 Analyze the percentage and pattern of missing values

🔎 Detect MCAR (Missing Completely At Random) using statistical testing

🧮 Choose from multiple missing data handling techniques:

Complete Case Analysis (CCA)

Mean / Median / Mode Imputation

KNN Imputation

🖼 Visualize missing data before and after cleaning

💾 Download the cleaned dataset

🛠 Tech Stack
Python

Streamlit (for UI)

Pandas, NumPy, Scikit-learn (for data handling & imputation)

Seaborn & Matplotlib (for visualizations)

🤖 Use Cases
Preprocessing step in your data science or ML pipeline

Teaching tool for understanding the impact of missing data

Exploratory analysis of real-world messy datasets

app/
├── app.py                # Streamlit app entry point
├── cca.py                # Complete Case Analysis logic
├── imputation.py         # Imputation techniques (mean, median, mode, KNN)
├── mcar_test.py          # MCAR detection using Little's test
├── data_loader_information.py  # Data insights and stats
├── utils.py              # Helper functions
├── visualizations.py     # Visualization logic
data/
└── data.csv              # Sample dataset
