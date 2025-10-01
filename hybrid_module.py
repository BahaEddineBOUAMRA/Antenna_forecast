
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

# --- Numeric & categorical column lists ---
numeric_cols = [
    "Signal_RSRP_dBm", "SINR_dB", "Throughput_DL_Mbps",
    "Latency_ms", "Drop_Rate_pct", "Temperature_C",
    "Voltage_V", "Power_W", "Battery_pct",
    "Active_Users", "Traffic_GBph", "Humidity_pct"
]
categorical_cols = [
    "Firmware_Version", "Technology_Type", "Antenna_Type",
    "Backhaul_Type", "Power_Source"
]

# --- Load models ---
d = "."  # models are in same folder
xgb_models = joblib.load(f"{d}/xgb_forecast_models.joblib")
clf_status = joblib.load(f"{d}/clf_status.joblib")
try:
    clf_warning = joblib.load(f"{d}/clf_warning.joblib")
except:
    clf_warning = None
try:
    clf_fail = joblib.load(f"{d}/clf_fail.joblib")
except:
    clf_fail = None

def hybrid_forecast_and_classify(last_history: pd.DataFrame, forecast_steps: int = 24, original_df: pd.DataFrame = None):
    history = last_history.copy()
    forecast_records = []

    # Handle categorical encoders
    cat_encoders = {}
    if original_df is not None:
        for col in categorical_cols:
            if col in original_df.columns:
                le = LabelEncoder()
                le.fit(pd.concat([original_df[col]]).astype(str).unique())
                cat_encoders[col] = le

    for step in range(forecast_steps):
        # Build lagged features
        features = {}
        for col in numeric_cols:
            for lag in range(1, 6):
                features[f"{col}_lag{lag}"] = history[col].iloc[-lag]
        for cat in categorical_cols:
            features[cat] = history[cat].iloc[-1]

        features_df = pd.DataFrame([features])[list(xgb_models.values())[0].get_booster().feature_names]
        preds = {col: xgb_models[col].predict(features_df)[0] for col in numeric_cols}

        # Append new row to history
        new_row = preds.copy()
        for cat in categorical_cols:
            new_row[cat] = history[cat].iloc[-1]
        new_row_df = pd.DataFrame([new_row], index=[history.index[-1] + pd.Timedelta(hours=1)])
        history = pd.concat([history, new_row_df])

        # Prepare classification input
        clf_input = pd.DataFrame([preds])
        for cat in categorical_cols:
            if cat in cat_encoders:
                clf_input[cat] = history[cat].iloc[-2]  # last encoded value

        # Status
        status_pred_encoded = clf_status.predict(clf_input)[0]
        if original_df is not None:
            status_pred = LabelEncoder().fit(original_df['Status']).inverse_transform([status_pred_encoded])[0]
        else:
            status_pred = status_pred_encoded

        # Warning/Fail subtypes
        warning_type, fail_type = None, None
        if status_pred == 'Warning' and clf_warning is not None:
            warning_type_pred = clf_warning.predict(clf_input)[0]
            if original_df is not None:
                warning_type = LabelEncoder().fit(original_df[original_df['Status']=='Warning']['Warning_Type']).inverse_transform([warning_type_pred])[0]
            else:
                warning_type = warning_type_pred
        elif status_pred == 'Fail' and clf_fail is not None:
            fail_type_pred = clf_fail.predict(clf_input)[0]
            if original_df is not None:
                fail_type = LabelEncoder().fit(original_df[original_df['Status']=='Fail']['Fail_Type'].dropna()).inverse_transform([fail_type_pred])[0]
            else:
                fail_type = fail_type_pred

        forecast_records.append({
            "Datetime": new_row_df.index[0],
            **preds,
            "Status": status_pred,
            "Warning_Type": warning_type,
            "Fail_Type": fail_type
        })

    return pd.DataFrame(forecast_records).set_index("Datetime")
