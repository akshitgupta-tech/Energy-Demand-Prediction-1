import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Energy Demand Explorer", layout="wide")


# -----------------------------------------------------------
# CSV Reader
# -----------------------------------------------------------
def read_csv(file):
    if file is None:
        return None
    try:
        return pd.read_csv(file)
    except:
        file.seek(0)
        return pd.read_csv(file, encoding="latin-1")


# -----------------------------------------------------------
# Sidebar
# -----------------------------------------------------------
st.sidebar.title("Controls")

load_file = st.sidebar.file_uploader("Upload Hourly Load CSV", type=["csv"])
gen_file = st.sidebar.file_uploader("Upload Generation CSV", type=["csv"])
temp_file = st.sidebar.file_uploader("Upload Temperature CSV", type=["csv"])

run_eda = st.sidebar.button("Run EDA")
run_vmd = st.sidebar.button("Run VMD")
train_model = st.sidebar.button("Train Model")
predict_button = st.sidebar.button("Predict")

horizon = st.sidebar.number_input("Prediction Horizon (hours)", min_value=1, max_value=500, value=24)


# -----------------------------------------------------------
# Load datasets
# -----------------------------------------------------------
df_load = read_csv(load_file)
df_gen = read_csv(gen_file)
df_temp = read_csv(temp_file)


# -----------------------------------------------------------
# UI – Dataset Preview
# -----------------------------------------------------------
st.header("Energy Demand Explorer — Clean UI")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Load Data")
    st.write(df_load.head() if df_load is not None else "Upload load CSV")

with col2:
    st.subheader("Generation Data")
    st.write(df_gen.head() if df_gen is not None else "Upload generation CSV")

with col3:
    st.subheader("Temperature Data")
    st.write(df_temp.head() if df_temp is not None else "Upload temperature CSV")


# -----------------------------------------------------------
# Utility: Timestamp parser
# -----------------------------------------------------------
def parse_timestamp(df):
    df = df.copy()
    if "datetime" in df.columns:
        df["timestamp"] = pd.to_datetime(df["datetime"])
    else:
        df["timestamp"] = pd.to_datetime(df.iloc[:, 0])
    return df


# -----------------------------------------------------------
# EDA
# -----------------------------------------------------------
if run_eda:
    st.subheader("Exploratory Data Analysis (EDA)")

    if df_load is None:
        st.error("Upload load CSV first.")
    else:
        df = parse_timestamp(df_load)

        target = (
            "National_Demand" 
            if "National_Demand" in df.columns 
            else df.select_dtypes(include=[np.number]).columns[0]
        )

        st.info(f"Using target column: **{target}**")

        # Summary
        st.write(df[[target]].describe())

        # Hourly pattern
        df["hour"] = df["timestamp"].dt.hour
        hourly = df.groupby("hour")[target].agg(["mean", "std"]).reset_index()

        fig, ax = plt.subplots()
        ax.plot(hourly["hour"], hourly["mean"], marker="o")
        ax.fill_between(hourly["hour"], hourly["mean"]-hourly["std"], hourly["mean"]+hourly["std"], alpha=0.3)
        ax.set_title("Hourly Demand Pattern")
        ax.set_xlabel("Hour")
        st.pyplot(fig)

        # Monthly
        df["month"] = df["timestamp"].dt.month
        monthly = df.groupby("month")[target].mean().reset_index()

        fig, ax = plt.subplots()
        ax.bar(monthly["month"], monthly[target])
        ax.set_title("Monthly Demand")
        ax.set_xlabel("Month")
        st.pyplot(fig)


# -----------------------------------------------------------
# VMD (fallback simple moving average)
# -----------------------------------------------------------
if run_vmd:
    st.subheader("VMD (Fallback Decomposition)")

    if df_load is None:
        st.error("Upload load CSV first.")
    else:
        df = parse_timestamp(df_load)

        target = (
            "National_Demand" 
            if "National_Demand" in df.columns 
            else df.select_dtypes(include=[np.number]).columns[0]
        )

        series = df[target].fillna(method="ffill").values

        st.info("Using fallback moving-average based IMFs.")

        windows = [3, 24, 168]
        for i, w in enumerate(windows):
            imf = pd.Series(series).rolling(w, min_periods=1).mean().values

            fig, ax = plt.subplots(figsize=(9, 2))
            ax.plot(imf)
            ax.set_title(f"IMF {i+1}  (window={w})")
            st.pyplot(fig)


# -----------------------------------------------------------
# Model Training
# -----------------------------------------------------------
def train_rf(df):
    df = parse_timestamp(df)
    df = df.sort_values("timestamp")

    target = (
        "National_Demand"
        if "National_Demand" in df.columns
        else df.select_dtypes(include=[np.number]).columns[0]
    )

    # Feature engineering
    df["hour"] = df["timestamp"].dt.hour
    df["dow"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["lag1"] = df[target].shift(1)
    df["lag24"] = df[target].shift(24)
    df["roll24"] = df[target].rolling(24, min_periods=1).mean()

    df = df.dropna()

    X = df[["hour", "dow", "month", "lag1", "lag24", "roll24"]]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = RandomForestRegressor(n_estimators=150, random_state=42)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    return {
        "model": model,
        "df": df,
        "target": target,
        "mae": mean_absolute_error(y_test, pred),
        "rmse": mean_squared_error(y_test, pred, squared=False),
        "r2": r2_score(y_test, pred)
    }


if train_model:
    st.subheader("Model Training")

    if df_load is None:
        st.error("Upload load CSV first.")
    else:
        results = train_rf(df_load)
        st.session_state["model"] = results

        st.success("Training Completed!")
        st.write(f"MAE: {results['mae']:.3f}")
        st.write(f"RMSE: {results['rmse']:.3f}")
        st.write(f"R²: {results['r2']:.3f}")


# -----------------------------------------------------------
# Predictions
# -----------------------------------------------------------
if predict_button:
    st.subheader("Predictions")

    if "model" not in st.session_state:
        st.error("Train model first.")
    else:
        res = st.session_state["model"]
        model = res["model"]
        df = res["df"]
        target = res["target"]

        # Future timestamps
        future_ts = [
            df["timestamp"].iloc[-1] + pd.Timedelta(hours=i+1)
            for i in range(horizon)
        ]

        # Recursive forecasting
        preds = []
        temp_df = df.copy()

        for t in future_ts:
            row = {
                "hour": t.hour,
                "dow": t.dayofweek,
                "month": t.month,
                "lag1": temp_df[target].iloc[-1],
                "lag24": temp_df[target].iloc[-24] if len(temp_df) >= 24 else temp_df[target].iloc[-1],
                "roll24": temp_df[target].rolling(24, min_periods=1).mean().iloc[-1],
            }

            X_new = pd.DataFrame([row])
            prediction = model.predict(X_new)[0]
            preds.append(prediction)

            # Add to temp DF
            temp_df.loc[len(temp_df)] = {"timestamp": t, target: prediction, **row}

        out = pd.DataFrame({"timestamp": future_ts, "prediction": preds})

        st.write(out)

        # ------------ CLEAN X-AXIS GRAPH --------------
        fig, ax = plt.subplots(figsize=(12, 4))

        out["ts_str"] = out["timestamp"].dt.strftime("%H:%M")

        ax.plot(out["ts_str"], out["prediction"], marker="o")

        # Display only ~6 ticks
        step = max(1, len(out) // 6)
        ax.set_xticks(out["ts_str"][::step])

        plt.xticks(rotation=45)
        ax.set_title("Future Predictions")
        ax.set_xlabel("Time")
        ax.set_ylabel("Predicted Demand")

        plt.tight_layout()
        st.pyplot(fig)

        # Download
        st.download_button(
            "Download Predictions CSV",
            out.to_csv(index=False),
            "predictions.csv",
            "text/csv"
        )
