
import streamlit as st
import pandas as pd
from hybrid_module import hybrid_forecast_and_classify

st.set_page_config(page_title="📡 Antenna Forecast & Classification", layout="wide")
st.title("📡 Antenna Forecast & Classification")

steps = st.slider("Forecast steps (hours)", 1, 168, 24)
uploaded_file = st.file_uploader("Upload last 5 rows of antenna data (CSV)", type=["csv"])

if st.button("Run Forecast"):
    if uploaded_file is not None:
        df_input = pd.read_csv(uploaded_file, index_col=0, parse_dates=True)
        forecast_df = hybrid_forecast_and_classify(df_input, steps)
        st.subheader("📊 Forecast Results")
        st.dataframe(forecast_df)
    else:
        st.warning("Please upload a CSV file with the last 5 rows of data.")
