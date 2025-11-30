# -------------------------------
# Energy Demand Predictor 
# -------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from urllib.parse import quote
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Energy Demand Predictor", layout="wide")

# ---------- GitHub Repo Settings ----------
GITHUB_USER = "asingh1me25-cell"
GITHUB_REPO = "Energy-Demand-Prediction"
FILE_LOAD = "Hourly_Load_India_Final_Panama_Format colab.csv"


def gh_raw(filename):
    return f"https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/main/{quote(filename)}"


# ---------- CSV Loader ----------
def read_from_url(url):
    try:
        return pd.read_csv(url)
    except Exception as e:
        st.error(f"Failed to load {url}: {e}")
        return None


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

    y_pred = model.predict(X_test)

    metrics = {
        "MAE": float(mean_absolute_error(y_test, y_pred)),
        "RMSE": float(mean_squared_error(y_test, y_pred)**0.5),
        "R2": float(r2_score(y_test, y_pred))
    }

    return model, dfm, value_col, metrics


# ---------- UI ----------
st.title("⚡ Energy Demand Predictor")

pred_horizon = st.number_input(
    "Select Prediction Horizon (1–24 hours):",
    min_value=1, max_value=24, value=12, step=1
)

predict_button = st.button("Predict")


# ---------- Auto-Train on First Attempt ----------
if "model" not in st.session_state and predict_button:
    st.info("Training model for the first time…")
    model, dfm, valcol, mets = train(df_load)
    st.session_state["model"] = model
    st.session_state["df"] = dfm
    st.session_state["valcol"] = valcol
    st.session_state["metrics"] = mets

    # Save model to local file
    with open("trained_model.pkl", "wb") as f:
        pickle.dump(model, f)

    st.success("Training complete!")
    st.write(mets)

    st.download_button(
        "Download Trained Model (PKL)",
        open("trained_model.pkl", "rb"),
        "trained_model.pkl"
    )


# ---------- Prediction ----------
if predict_button and "model" in st.session_state:
    st.header(f"Predictions (Next {pred_horizon} Hours)")

    model = st.session_state["model"]
    dfm = st.session_state["df"]
    value_col = st.session_state["valcol"]

    last_ts = dfm["timestamp"].iloc[-1]
    future_ts = [last_ts + pd.Timedelta(hours=i+1) for i in range(int(pred_horizon))]

    preds = []
    temp_df = dfm.copy()

    for t in future_ts:
        row = {
            "hour": t.hour,
            "dow": t.dayofweek,
            "month": t.month,
            "lag1": temp_df[value_col].iloc[-1],
            "lag24": temp_df[value_col].iloc[-24] if len(temp_df)>=24 else temp_df[value_col].iloc[-1],
            "roll24": temp_df[value_col].rolling(24, min_periods=1).mean().iloc[-1]
        }
        X_new = pd.DataFrame([row])
        pred = model.predict(X_new)[0]
        preds.append(pred)
        temp_df.loc[len(temp_df)] = {**row, value_col: pred, "timestamp": t}

    out = pd.DataFrame({"timestamp": future_ts, "prediction": preds})

    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(out["timestamp"], out["prediction"], marker="o")
    plt.xticks(rotation=45)
    ax.set_title(f"Next {pred_horizon}-Hour Forecast")
    st.pyplot(fig)

    st.download_button(
        f"Download {pred_horizon}h Predictions CSV",
        out.to_csv(index=False),
        f"predictions_{pred_horizon}h.csv",
        "text/csv"
    )
