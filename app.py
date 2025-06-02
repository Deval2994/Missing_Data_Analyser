"""
Description :   Main entry point of the app. Manages the user interface (likely with Streamlit), handles user inputs
                (like file uploads, cleaning options), and coordinates calling functions from other modules
            *** ADD OPTION TO DROP UNWANTED COLUMNS ***
"""
from pathlib import Path

import streamlit as st
import pandas as pd
import os

from mcar_test import is_mcar
import cca
import visualizations as VG
import imputation as I
import nan_mapping

import streamlit as st

st.set_page_config(
    page_title="Phantom Spectrum",
    page_icon="👻",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<div style="display: flex; align-items: baseline;">
    <h1 style="margin-right: 10px;">👻 Phantom Spectrum</h1>
    <span style="color: gray; font-size: 16px;"> Mapping the missing. Restoring the full spectrum here</span>
</div>
""", unsafe_allow_html=True)

st.info("I built a web app to show my understanding of Complete Case Analysis (CCA) and imputation techniques."
             "  CCA - where the data doesn't impute missing by itself but delete that entire row (data entry or  "
             "information) which is best in some cases where data loss is minimum and data is MCAR. The app lets users "
             "upload a dataset, perform CCA or various imputations, and visualize how the data "
             "distributions change before and after cleaning by comparing both CCA and imputation side by side.")
# Title
st.title("🧹 Missing Data Analysis")
# Sidebar for dataset selection                                                                                         Sidebar for dataset selection
st.sidebar.header("📂 Dataset Options")
use_default = st.sidebar.radio("Choose dataset source:", ("sample dataset (healthcare)", "Upload your own"))

# Load the dataset based on user selection
df = None

if use_default == "sample dataset (healthcare)":
    current_dir = Path(__file__).parent
    default_path = current_dir / "healthcare_dataset.csv"

    # Load the dataset
    df = pd.read_csv(default_path)
    st.success("Using default dataset - healthcare_dataset.csv")

else:
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("Uploaded custom dataset.")
    else:
        st.warning("Please upload a CSV file to proceed.")






# Store app state using session_state                                                                                   Store app state using session_state
if "step" not in st.session_state:
    st.session_state.step = "preview"

# Step 1: Dataset loaded and showing preview
if df is not None and st.session_state.step == "preview":

    st.header("📄 Dataset Preview")
    st.dataframe(df.head())

    if st.button("Continue with dropping columns"):
        st.session_state.step = "drop_columns"
        st.session_state.df = df  # Save to session


if df is not None:
    try:
        st.header('❗Missing values in column, drop if 30-40%')
        st.markdown("**recommendation:** *drop if >30%, because the data loss will be more than 30%, which is not okay for CCA*")
        df = nan_mapping.nan_decoding(df)
        nan_percentages = cca.nan_percentage_per_column(df)
        st.dataframe(nan_percentages.to_frame(name="Missing (%)"))
    except:
        pass


# Step 2: Drop Columns UI                                                                                               Step 2: Drop Columns UI
if st.session_state.step == "drop_columns":
    st.subheader(" 🗑️ Drop Unnecessary Columns")

    df = st.session_state.df  # Load the saved dataframe

    columns_to_drop = st.multiselect("Select columns to drop:", df.columns.tolist())

    if st.button("Drop Selected Columns"):
        df = df.drop(columns=columns_to_drop)
        st.session_state.df = df  # Update session state
        st.success(f"Dropped columns: {', '.join(columns_to_drop)}")

        st.write("🔍 Updated Data Preview:")
        st.dataframe(df.head())

        # ✅ Run MCAR test AFTER drop
        try:
            is_data_mcar = is_mcar(df)
        except Exception as e:
            is_data_mcar = f"MCAR test failed. {e}"

        try:
            data_loss = cca.data_loss_percentage(df)
        except:
            data_loss = "text failed..."

        st.caption("If the MCAR test is true and the Data loss percent is < 5-10%, CCA is a best option")
        st.info(f"MCAR test: {is_data_mcar}")
        st.info(f"Data loss percent: {data_loss}%")

    if st.button("Continue with handling missing values"):
        st.session_state.step = "handle_missing_value"
        st.session_state.df = df  # Save to session

# Step 3: graph representation                                                                                          Step 3: graph representation
if st.session_state.step == "handle_missing_value":

    col1, col2 = st.columns(2)
    df = st.session_state.df
    cca_fig = None
    imp_fig = None

    with col1:
        st.subheader("Complete Case Analysis (CCA)")
        st.write("No missing data handling options here.")
        cca_df = cca.complete_case_analysis(df)

        # Select column to plot for CCA
        cca_cols = [c for c in cca_df.columns if cca_df[c].dtype in ['int64', 'float64', 'object']]
        cca_col = st.selectbox("Select CCA column to visualize:", cca_cols, key="cca_col")

        # Numeric or categorical graph choices
        if cca_df[cca_col].dtype in ['int64', 'float64']:
            graph_type = st.selectbox("Select graph type:", ['box', 'hist', 'qq'], key="cca_graph")
            # call your numeric visualization here
            fig = VG.visualize_numeric_distribution(cca_df, cca_col, graph_type)
        else:
            graph_type = st.selectbox("Select graph type:", ['pie', 'bar', 'heatmap', 'treemap'], key="cca_graph_cat")
            fig = VG.visualize_categorical_distribution(cca_df, cca_col, graph_type)

        cca_fig = fig

    with col2:
        st.subheader("Imputation")

        df = st.session_state.df.copy()

        # 1. Select any column
        all_cols = [col for col in df.columns if df[col].dtype in ['int64', 'float64', 'object']]
        imp_col = st.selectbox("Select column to work on:", all_cols, key="imp_col")

        # 2. If column has missing values, show imputation method options
        if df[imp_col].isnull().sum() > 0:
            st.warning(f"Column '{imp_col}' has {df[imp_col].isnull().sum()} missing values.")

            if df[imp_col].dtype in ['int64', 'float64']:
                imp_method = st.selectbox(
                    "Select imputation method:",
                    ["mean", "median", "mode", "KNN", "Random"],
                    key="imp_method_num"
                )
                if imp_method == "KNN":
                    imputed_df = I.fill_numeric_with_knn_imputer(df.copy(),imp_col)
                elif imp_method == "Random":
                    imputed_df = I.fill_with_random_values(df.copy(),imp_col)
                else:
                    imputed_df = I.fill_numeric_columns(df.copy(), imp_col, imp_method)
            else:
                imp_method = st.selectbox(
                    "Select imputation method:",
                    ["mode", "new_category", "Random"],
                    key="imp_method_cat"
                )
                if imp_method == "Random":
                    imputed_df = I.fill_with_random_values(df.copy(), imp_col)
                else:
                    imputed_df = I.fill_missing_object_columns(df.copy(),imp_col,imp_method)

        else:
            st.success("No missing values in this column.")
            imputed_df = df.copy()

        # 4. Visualization
        if imputed_df[imp_col].dtype in ['int64', 'float64']:
            graph_type = st.selectbox("Select graph type:", ['box', 'hist', 'qq'], key="imp_graph")
            fig = VG.visualize_numeric_distribution(imputed_df, imp_col, graph_type)
        else:
            graph_type = st.selectbox("Select graph type:", ['pie', 'bar', 'heatmap', 'treemap'], key="imp_graph_cat")
            fig = VG.visualize_categorical_distribution(imputed_df, imp_col, graph_type)

        imp_fig = fig

    # NEW ROW FOR FIGURES
    fig_col1, fig_col2 = st.columns(2)
    with fig_col1:
        st.pyplot(cca_fig)
    with fig_col2:
        st.pyplot(imp_fig)



