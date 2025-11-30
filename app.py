# -------------------------------
# Energy Demand Predictor (Low-Memory Version)
# -------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from urllib.parse import quote
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import altair as alt

st.set_page_config(page_title="Energy Demand Predictor", layout="wide")

# ---------- GitHub Repo Settings ----------
GITHUB_USER = "asingh1me25-cell"
GITHUB_REPO = "Energy-Demand-Prediction"
FILE_LOAD = "Hourly_Load_India_Final_Panama_Format colab.csv"


def gh_raw(filename):
    return f"https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/main/{quote(filename)}"


# ---------- CSV Loader ----------
@st.cache_data
def read_from_url(url):
    return pd.read_csv(url)


df_load = read_from_url(gh_raw(FILE_LOAD))


# ---------- Helpers ----------
def parse_timestamp(df):
    df = df.copy()
    for c in df.columns:
        if "date" in c.lower() or "time" in c.lower():
            try:
                df["timestamp"] = pd.to_datetime(df[c])
                return df
            except:
                pass
    df["timestamp"] = pd.to_datetime(df.iloc[:, 0])
    return df


# ---------- Training Function ----------
@st.cache_resource
def train(df_load_local):
    df = parse_timestamp(df_load_local).sort_values("timestamp")

    candidates = ["National_Demand", "National_Demand_MW", "Demand", "Total_Demand"]
    value_col = next((c for c in candidates if c in df.columns),
                     df.select_dtypes(include=[np.number]).columns[0])

    df["hour"] = df["timestamp"].dt.hour
    df["dow"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["lag1"] = df[value_col].shift(1)
    df["lag24"] = df[value_col].shift(24)
    df["roll24"] = df[value_col].rolling(24, min_periods=1).mean()

    dfm = df.dropna().reset_index(drop=True)

    X = dfm[["hour","dow","month","lag1","lag24","roll24"]]
    y = dfm[value_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = RandomForestRegressor(n_estimators=120, random_state=42)
    model.fit(X_train, y_train)

    return model, dfm, value_col


# ---------- UI ----------
st.title("⚡ Energy Demand Predictor")

pred_horizon = st.number_input(
    "Select Prediction Horizon (1–24 hours):",
    min_value=1, max_value=24, value=12, step=1
)

predict_button = st.button("Predict")


# ---------- Prediction ----------
if predict_button:

    st.info("Training model (cached)…")
    model, dfm, value_col = train(df_load)
    st.success("Model ready!")

    # -------------------------------------
    # MEMORY-SAFE PREDICTION LOGIC
    # -------------------------------------
    last_ts = dfm["timestamp"].iloc[-1]
    future_ts = [last_ts + pd.Timedelta(hours=i+1) for i in range(pred_horizon)]

    # Keep only last 24 values for lightweight lag/rolling
    last_values = dfm[value_col].iloc[-24:].tolist()

    preds = []

    for t in future_ts:
        lag1 = last_values[-1]
        lag24 = last_values[-24] if len(last_values) >= 24 else lag1
        roll24 = np.mean(last_values)

        row = {
            "hour": t.hour,
            "dow": t.dayofweek,
            "month": t.month,
            "lag1": lag1,
            "lag24": lag24,
            "roll24": roll24
        }

        pred = model.predict(pd.DataFrame([row]))[0]
        preds.append(pred)

        # update small in-memory buffer (always max 24 entries)
        last_values.append(pred)
        if len(last_values) > 24:
            last_values.pop(0)

    # Build output minimal DataFrame
    out = pd.DataFrame({"timestamp": future_ts, "prediction": preds})

    # ---------- Plot using Altair (very memory-efficient) ----------
    chart = (
        alt.Chart(out)
        .mark_line(point=True)
        .encode(
            x="timestamp:T",
            y="prediction:Q",
            tooltip=["timestamp:T", "prediction:Q"]
        )
        .properties(title=f"Next {pred_horizon}-Hour Forecast", width=700)
    )

    st.altair_chart(chart, use_container_width=True)

    # ---------- Download ----------
    st.download_button(
        label=f"Download {pred_horizon}h Predictions CSV",
        data=out.to_csv(index=False).encode("utf-8"),
        file_name=f"predictions_{pred_horizon}h.csv",
        mime="text/csv"
    )
