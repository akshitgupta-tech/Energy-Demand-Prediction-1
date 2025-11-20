# app.py — Final deployment-ready Streamlit app
# Requirements: streamlit, pandas, numpy, scikit-learn, matplotlib

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import importlib.util
import os
from urllib.parse import quote
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Energy Demand Explorer (Deploy)", layout="wide")

# ---------- USER / REPO SETTINGS ----------
GITHUB_USER = "asingh1me25-cell"
GITHUB_REPO = "Energy-Demand-Prediction"
# Filenames in your repo (exact)
FILE_LOAD = "Hourly_Load_India_Final_Panama_Format colab.csv"
FILE_GEN = "Daily_Power_Gen_Source_Final_Panama_Format colab.csv"
FILE_TEMP = "Temperatures_Final_Panama_Format colab.csv"

# Local VMD file path (you uploaded this earlier)
LOCAL_VMD_PATH = "/mnt/data/vmd_new_final_02.py"   # <-- developer-provided path

# Helper to build raw GitHub URL (handles spaces)
def gh_raw(filename):
    # percent-encode filename
    return f"https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/main/{quote(filename)}"

# ---------- Data loader ----------
def read_csv_from_file_or_url(uploaded_file, url):
    # priority: uploaded_file (Streamlit uploader) > local file path > URL
    if uploaded_file is not None:
        try:
            return pd.read_csv(uploaded_file)
        except Exception:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, encoding="latin-1")
    # try local path (useful when testing locally)
    local_path = os.path.join(os.getcwd(), os.path.basename(url))
    if os.path.exists(local_path):
        try:
            return pd.read_csv(local_path)
        except:
            pass
    # try URL
    try:
        return pd.read_csv(url)
    except Exception as e:
        st.warning(f"Could not load URL {url}: {e}")
        return None

# ---------- Try to import vmd module ----------
def import_vmd_module():
    # 1) Try local uploaded file path (exists on your machine per conversation)
    if os.path.exists(LOCAL_VMD_PATH):
        try:
            spec = importlib.util.spec_from_file_location("vmd_module_local", LOCAL_VMD_PATH)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            st.info("Loaded VMD from local path.")
            return module
        except Exception as e:
            st.info(f"Local VMD import failed: {e}")
    # 2) Try to fetch vmd.py from GitHub repo raw URL (if present)
    vmd_url = gh_raw("vmd.py")
    try:
        import requests, tempfile
        r = requests.get(vmd_url, timeout=10)
        if r.status_code == 200 and "def" in r.text:
            tmp = tempfile.NamedTemporaryFile(suffix=".py", delete=False)
            tmp.write(r.content)
            tmp.flush()
            tmp.close()
            spec = importlib.util.spec_from_file_location("vmd_module_remote", tmp.name)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            st.info("Loaded VMD from GitHub raw vmd.py.")
            return module
    except Exception:
        pass
    # 3) no module
    return None

# ---------- Streamlit UI: Sidebar ----------
st.sidebar.title("Controls")
st.sidebar.markdown("**Use default GitHub datasets** or upload your own CSVs to override.")

uploaded_load = st.sidebar.file_uploader("Hourly Load CSV", type=["csv"])
uploaded_gen = st.sidebar.file_uploader("Generation CSV", type=["csv"])
uploaded_temp = st.sidebar.file_uploader("Temperature CSV", type=["csv"])

use_github_defaults = st.sidebar.checkbox("Use default GitHub datasets", value=True)

run_eda = st.sidebar.button("Run EDA")
run_vmd = st.sidebar.button("Run VMD")
train_models = st.sidebar.button("Train Models")
predict_button = st.sidebar.button("Predict")

horizon = st.sidebar.number_input("Prediction horizon (hours)", min_value=1, max_value=500, value=24)

# ---------- Load datasets (either uploaded or GitHub raw) ----------
default_load_url = gh_raw(FILE_LOAD)
default_gen_url = gh_raw(FILE_GEN)
default_temp_url = gh_raw(FILE_TEMP)

load_url = default_load_url if use_github_defaults else None
gen_url = default_gen_url if use_github_defaults else None
temp_url = default_temp_url if use_github_defaults else None

df_load = read_csv_from_file_or_url(uploaded_load, load_url) if (uploaded_load or load_url) else None
df_gen = read_csv_from_file_or_url(uploaded_gen, gen_url) if (uploaded_gen or gen_url) else None
df_temp = read_csv_from_file_or_url(uploaded_temp, temp_url) if (uploaded_temp or temp_url) else None

# ---------- Page header and previews ----------
st.title("Energy Demand Explorer — Deploy Ready")
st.markdown("Upload your CSVs, or use the default GitHub-hosted datasets (checkbox).")

c1, c2, c3 = st.columns(3)
with c1:
    st.subheader("Hourly Load")
    if df_load is not None:
        st.dataframe(df_load.head(5))
        st.write("Columns:", list(df_load.columns))
    else:
        st.warning("Load dataset not available.")

with c2:
    st.subheader("Generation")
    if df_gen is not None:
        st.dataframe(df_gen.head(4))
        st.write("Columns:", list(df_gen.columns))
    else:
        st.warning("Generation dataset not available.")

with c3:
    st.subheader("Temperatures")
    if df_temp is not None:
        st.dataframe(df_temp.head(4))
        st.write("Columns:", list(df_temp.columns))
    else:
        st.warning("Temperature dataset not available.")

# ---------- Utilities ----------
def parse_timestamp(df):
    df = df.copy()
    if "datetime" in df.columns:
        df["timestamp"] = pd.to_datetime(df["datetime"])
    else:
        # try common names or first column
        for c in df.columns:
            if "date" in c.lower() or "time" in c.lower():
                try:
                    df["timestamp"] = pd.to_datetime(df[c])
                    return df
                except:
                    pass
        df["timestamp"] = pd.to_datetime(df.iloc[:, 0])
    return df

# ---------- EDA ----------
if run_eda:
    st.header("Exploratory Data Analysis (EDA)")
    if df_load is None:
        st.error("Please provide a load CSV (upload or enable GitHub defaults).")
    else:
        df = parse_timestamp(df_load)
        # choose sensible target
        candidates = ["National_Demand", "National_Demand_MW", "Demand", "Total_Demand"]
        value_col = next((c for c in candidates if c in df.columns), None)
        if value_col is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            value_col = numeric_cols[0] if len(numeric_cols) else df.columns[1]

        st.subheader(f"Using target: {value_col}")
        st.write(df[[value_col]].describe())

        # hourly plot
        df["hour"] = df["timestamp"].dt.hour
        hourly = df.groupby("hour")[value_col].agg(["mean", "std"]).reset_index()
        fig, ax = plt.subplots()
        ax.plot(hourly["hour"], hourly["mean"], marker="o")
        ax.fill_between(hourly["hour"], hourly["mean"] - hourly["std"], hourly["mean"] + hourly["std"], alpha=0.25)
        ax.set_xlabel("Hour of day")
        ax.set_ylabel(value_col)
        ax.set_title("Hourly average demand")
        st.pyplot(fig)

        # monthly
        df["month"] = df["timestamp"].dt.month
        monthly = df.groupby("month")[value_col].mean().reset_index()
        fig, ax = plt.subplots()
        ax.bar(monthly["month"], monthly[value_col])
        ax.set_xlabel("Month")
        ax.set_ylabel(value_col)
        ax.set_title("Monthly average demand")
        st.pyplot(fig)

# ---------- VMD ----------
if run_vmd:
    st.header("VMD / Decomposition")
    if df_load is None:
        st.error("Provide load CSV first.")
    else:
        df = parse_timestamp(df_load)
        # pick numeric series
        num_cols = df.select_dtypes(include=[np.number]).columns
        if len(num_cols) == 0:
            st.error("No numeric columns to decompose.")
        else:
            value_col = "National_Demand" if "National_Demand" in df.columns else num_cols[0]
            series = df[value_col].fillna(method="ffill").values

            vmod = import_vmd_module()
            if vmod is not None:
                # try to locate a plausible function
                fn_names = ["VMD", "vmd", "perform_vmd", "VMD_decompose", "vmd_decompose"]
                imfs = None
                for fn in fn_names:
                    if hasattr(vmod, fn):
                        try:
                            func = getattr(vmod, fn)
                            imfs = func(np.asarray(series).copy())
                            break
                        except Exception as e:
                            st.info(f"Found {fn} but call failed: {e}")
                if imfs is not None:
                    st.success(f"Decomposed into {len(imfs)} components (via vmd).")
                    for i, imf in enumerate(imfs[:6]):
                        fig, ax = plt.subplots(figsize=(9, 2))
                        ax.plot(imf)
                        ax.set_title(f"IMF {i+1}")
                        st.pyplot(fig)
                else:
                    st.info("vmd.py present but no callable function found; falling back to moving averages.")
                    imfs = [pd.Series(series).rolling(w, min_periods=1).mean().values for w in (3,24,168)]
                    for i, imf in enumerate(imfs):
                        fig, ax = plt.subplots(figsize=(9,2))
                        ax.plot(imf); ax.set_title(f"Pseudo-IMF {i+1}"); st.pyplot(fig)
            else:
                st.info("No VMD module found. Using fallback moving-average decomposition.")
                imfs = [pd.Series(series).rolling(w, min_periods=1).mean().values for w in (3,24,168)]
                for i, imf in enumerate(imfs):
                    fig, ax = plt.subplots(figsize=(9,2))
                    ax.plot(imf); ax.set_title(f"Pseudo-IMF {i+1}"); st.pyplot(fig)

# ---------- Model training ----------
def train_and_return(df_load_local):
    df = parse_timestamp(df_load_local).sort_values("timestamp")
    candidates = ["National_Demand", "National_Demand_MW", "Demand", "Total_Demand"]
    value_col = next((c for c in candidates if c in df.columns), None)
    if value_col is None:
        numeric = df.select_dtypes(include=[np.number]).columns
        if len(numeric) == 0:
            raise ValueError("No numeric column found in load data.")
        value_col = numeric[0]

    df["hour"] = df["timestamp"].dt.hour
    df["dow"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["lag1"] = df[value_col].shift(1)
    df["lag24"] = df[value_col].shift(24)
    df["roll24"] = df[value_col].rolling(24, min_periods=1).mean()

    df_model = df.dropna().reset_index(drop=True)
    if len(df_model) < 30:
        raise ValueError("Insufficient rows after feature creation. Need >=30 rows.")

    X = df_model[["hour","dow","month","lag1","lag24","roll24"]]
    y = df_model[value_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    if len(X_train) == 0:
        raise ValueError("Training set empty after split.")

    rf = RandomForestRegressor(n_estimators=120, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    metrics = {"mae": float(mean_absolute_error(y_test, y_pred)),
               "rmse": float(mean_squared_error(y_test, y_pred, squared=False)),
               "r2": float(r2_score(y_test, y_pred))}

    return {"model": rf, "metrics": metrics, "last_df": df_model, "value_col": value_col}

if train_models:
    st.header("Model Training")
    if df_load is None:
        st.error("Provide load CSV first.")
    else:
        try:
            with st.spinner("Training RandomForest..."):
                trained = train_and_return(df_load)
                st.session_state["trained"] = trained
            st.success("Training completed.")
            st.write(trained["metrics"])
        except Exception as e:
            st.error(f"Training failed: {e}")

# ---------- Prediction ----------
if predict_button:
    st.header("Predictions")
    if "trained" not in st.session_state:
        st.error("Train models first.")
    else:
        trained = st.session_state["trained"]
        model = trained["model"]
        df_for_feat = trained["last_df"]
        value_col = trained["value_col"]

        # create future timestamps
        last_ts = df_for_feat["timestamp"].iloc[-1]
        future_ts = [last_ts + pd.Timedelta(hours=i+1) for i in range(horizon)]

        preds = []
        temp_df = df_for_feat.copy()

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
            p = model.predict(X_new)[0]
            preds.append(p)
            # append synthetic record
            temp_df.loc[len(temp_df)] = {**row, value_col: p, "timestamp": t}

        out = pd.DataFrame({"timestamp": future_ts, "prediction": preds})
        st.dataframe(out)

        # clean x-axis display
        fig, ax = plt.subplots(figsize=(11,4))
        out["ts_str"] = out["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
        ax.plot(out["ts_str"], out["prediction"], marker="o")
        step = max(1, len(out) // 6)
        ax.set_xticks(out["ts_str"][::step])
        plt.xticks(rotation=45)
        ax.set_title("Predicted demand (recursive)")
        ax.set_xlabel("Time")
        ax.set_ylabel("Prediction")
        plt.tight_layout()
        st.pyplot(fig)

        st.download_button("Download Predictions CSV", out.to_csv(index=False), "predictions.csv", "text/csv")

# ---------- footer ----------
st.markdown("---")
st.markdown("**Note:** This app attempts to load default CSV files from your GitHub repository. If you deploy on Streamlit Cloud, ensure the CSV files and `app.py` are in the repo root (or disable the 'Use default GitHub datasets' checkbox and upload files manually).")

# End of app.py
