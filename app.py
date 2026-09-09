
from __future__ import annotations

import hashlib
import io
import joblib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt   
import numpy as np
import pandas as pd   
import streamlit as st   
from sklearn.base import BaseEstimator, TransformerMixin   
from sklearn.compose import ColumnTransformer   
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor   
from sklearn.impute import SimpleImputer   
from sklearn.linear_model import LinearRegression, Ridge   
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score   
from sklearn.model_selection import GridSearchCV, train_test_split   
from sklearn.pipeline import Pipeline   
from sklearn.preprocessing import OneHotEncoder, StandardScaler   
from sklearn.tree import DecisionTreeRegressor   

#  
# Constants
#  
BASE_DIR = Path('household_power_consumption.csv').resolve().parent
DATA_PATH = BASE_DIR / "household_power_consumption.csv"
MODEL_PATH = BASE_DIR / "model.pkl"

TARGET = "Global_active_power"  # y (regression target, in kW)
NUMERIC_FEATURES = [
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]
CATEGORICAL_FEATURES = ["hour", "dayofweek", "month"]  # engineered, step 3
DATE_FORMAT = "%d/%m/%Y %H:%M:%S"
IQR_K = 1.5  # outlier capping factor (step 4)


#  #
# Step 1 / data reading - read the file CORRECTLY
#  #
def _read_csv(source) -> dict:
    """Read the dataset: ';' separator, '?' = missing, dd/mm/yyyy dates.

    Returns {"df": cleaned (but unsampled) dataframe, "stats": {..}} where the
    stats dict records what the cleaning step actually did (for the UI).
    """
    df = pd.read_csv(source, sep=";", na_values=["?"], low_memory=False)
    stats = {"rows_raw": int(len(df)), "columns": list(df.columns)}

    #  step 1a: fix formats -
    # Parse Date + Time into one datetime column (dd/mm/yyyy + HH:MM:SS).
    df["datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"], format=DATE_FORMAT, errors="coerce"
    )
    # Coerce the 7 measurement columns to float (defensive: files may arrive
    # as text or with non-numeric junk).
    for col in NUMERIC_FEATURES + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    #  step 1b: remove duplicates 
    n_before = int(len(df))
    df = df.drop_duplicates().copy()
    stats["duplicates_removed"] = n_before - int(len(df))
    stats["rows_after_clean"] = int(len(df))

    bad_dates = int(df["datetime"].isna().sum())
    stats["bad_datetime_rows"] = bad_dates
    return {"df": df, "stats": stats}


@st.cache_data(show_spinner="Reading raw file ...")
def load_bundled_data() -> dict:
    return _read_csv(DATA_PATH)


@st.cache_data(show_spinner="Reading uploaded file ...")
def load_uploaded_data(file_bytes: bytes) -> dict:
    return _read_csv(io.BytesIO(file_bytes))


#  #
# Steps 2-7: feature engineering + preprocessing transformers
#  #
class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Derive modelling features from the raw (cleaned) dataframe.

    Input : DataFrame with Date, Time and the 7 measurement columns.
    Output: DataFrame of numeric measurement columns + categorical time
            features (hour, day of week, month).

    Kept inside the final sklearn.Pipeline (step 14) so training and
    inference always apply the exact same transformation.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if "datetime" in X.columns:
            dt = pd.to_datetime(X["datetime"], errors="coerce")
        else:
            dt = pd.to_datetime(
                X["Date"] + " " + X["Time"], format=DATE_FORMAT, errors="coerce"
            )
        out = X[NUMERIC_FEATURES].copy()  # numeric measurements
        out["hour"] = dt.dt.hour.fillna(0).astype(int)
        out["dayofweek"] = dt.dt.dayofweek.fillna(0).astype(int)
        out["month"] = dt.dt.month.fillna(1).astype(int)
        return out


class IQRCapper(BaseEstimator, TransformerMixin):
    """Step 4 - cap outliers to [Q1 - k*IQR, Q3 + k*IQR].

    Bounds are fitted on the training data only, so the transform never
    leaks validation/test information (same discipline as the scaler).
    """

    def __init__(self, k: float = 1.5):
        self.k = k

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        q1 = np.nanpercentile(X, 25, axis=0)
        q3 = np.nanpercentile(X, 75, axis=0)
        self.lo_ = q1 - self.k * (q3 - q1)
        self.hi_ = q3 + self.k * (q3 - q1)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float).copy()
        for j in range(X.shape[1]):
            lo, hi = self.lo_[j], self.hi_[j]
            if np.isnan(lo) and np.isnan(hi):
                continue  # column unusable -> leave untouched
            if not np.isnan(lo):
                X[:, j] = np.fmax(X[:, j], lo)
            if not np.isnan(hi):
                X[:, j] = np.fmin(X[:, j], hi)
        return X


def build_preprocessing() -> Pipeline:
    """Steps 2 + 4 + 7 (+3) as one transformer pipeline.

    imputer (median) -> IQR capper -> StandardScaler  for numeric columns,
    OneHotEncoder(handle_unknown='ignore')            for categorical columns.
    Scaling is StandardScaler fit on train only (see run_experiment).
    """
    numeric_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),  # step 2
            ("outlier_iqr", IQRCapper(k=IQR_K)),            # step 4
            ("scaler", StandardScaler()),                   # step 7
        ]
    )
    return Pipeline(
        [
            ("features", FeatureEngineer()),
            (
                "columns",
                ColumnTransformer(
                    transformers=[
                        ("num", numeric_pipe, NUMERIC_FEATURES),
                        (  # step 3: categorical -> one-hot
                            "cat",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                            CATEGORICAL_FEATURES,
                        ),
                    ]
                ),
            ),
        ]
    )


#  #
# Step 5 + 6: define X / y and split into train / validation / test
#  #
def prepare_model_data(df: pd.DataFrame, sample_size: int | None = None,
                       random_state: int = 42) -> tuple:
    """Step 5 - split dataframe into X (features) and y (target).

    Optionally subsample first (the app uses this to keep training fast).
    Rows whose target is missing are dropped (can't learn from them).
    Returns (X, y, stats).
    """
    stats = {}
    df = df.copy()
    if sample_size is not None and sample_size < len(df):
        df = df.sample(sample_size, random_state=random_state)
        stats["subsampled"] = sample_size
    else:
        stats["subsampled"] = int(len(df))

    before = int(len(df))
    df = df[df[TARGET].notna()].copy()
    stats["target_missing_dropped"] = before - int(len(df))
    stats["rows_for_model"] = int(len(df))

    X = df[NUMERIC_FEATURES + ["Date", "Time"]].copy()  # features (raw inputs)
    y = df[TARGET].astype(float)                        # target
    return X, y, stats


def split_data(X, y, random_state: int = 0) -> tuple:
    """Step 6 - 60% train / 20% validation / 20% test (chronological order
    preserved could be an option; we keep a random shuffle for simplicity)."""
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.4, random_state=random_state
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


#  #
# Steps 9-13: candidate models, selection, tuning, retrain, final test
#  #
CANDIDATE_MODELS = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=42),
    "Random Forest": RandomForestRegressor(
        n_estimators=100, max_depth=12, n_jobs=-1, random_state=42
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100, max_depth=4, random_state=42
    ),
}

# GridSearchCV param grids per model (step 11). Linear Regression has no
# hyperparameters so it is skipped during the tuning phase.
TUNING_GRIDS = {
    "Ridge Regression": {"alpha": [0.1, 1.0, 10.0]},
    "Decision Tree": {"max_depth": [5, 8, 12, 16], "min_samples_leaf": [1, 5, 10]},
    "Random Forest": {
        "n_estimators": [50, 100],
        "max_depth": [8, 12],
        "min_samples_leaf": [1, 5],
    },
    "Gradient Boosting": {
        "n_estimators": [50, 100],
        "max_depth": [3, 4],
        "learning_rate": [0.05, 0.1],
    },
}


def evaluate(y_true, y_pred) -> dict:
    """Step 9c - MAE, MSE, RMSE, R2."""
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "R2": r2_score(y_true, y_pred),
    }


def run_experiment(df: pd.DataFrame, sample_size: int = 20000,
                   random_state: int = 42, tune: bool = True,
                   progress_cb=None) -> dict:
    """Run the full 15-step pipeline on (a subsample of) the data.

    progress_cb(i, total) is called during the candidate-model loop (step 9).
    Returns a dict with every result the UI needs, including the final
    joblib-saveable Pipeline (step 14).
    """
    # Steps 5-6 -
    X, y, prep_stats = prepare_model_data(df, sample_size=sample_size,
                                          random_state=random_state)
    (X_train, X_val, X_test, y_train, y_val, y_test) = split_data(X, y)

    # Steps 2-7: preprocess - fitted on TRAIN only (transform val/test).
    preprocessor = build_preprocessing()
    X_train_pp = preprocessor.fit_transform(X_train, y_train)
    X_val_pp = preprocessor.transform(X_val)
    X_test_pp = preprocessor.transform(X_test)

    # Step 4 report: how many train cells did IQR capping actually change?
    cap_report = _count_capped_cells(X_train, preprocessor)

    # Step 9: candidate-model loop 
    results = []
    total = len(CANDIDATE_MODELS)
    for i, (name, model) in enumerate(CANDIDATE_MODELS.items()):
        model.fit(X_train_pp, y_train)          # 9a fit
        y_pred_val = model.predict(X_val_pp)    # 9b predict
        metrics = evaluate(y_val, y_pred_val)   # 9c evaluate
        results.append({"Model": name, **metrics})
        if progress_cb:
            progress_cb(i + 1, total)

    models_eval = pd.DataFrame(results).set_index("Model")

    # Step 10: pick the best by highest R2 (lowest RMSE is used as a tie-break).
    models_eval_sorted = models_eval.sort_values(["R2", "RMSE"],
                                                 ascending=[False, True])
    best_model_name = models_eval_sorted.index[0]

    # Step 11: hyperparameter tuning on the best model (GridSearchCV).
    tuning_df = None
    tuned_model = CANDIDATE_MODELS[best_model_name]
    grid = TUNING_GRIDS.get(best_model_name, {})
    if tune and grid:
        gs = GridSearchCV(
            tuned_model, grid, cv=3, scoring="neg_root_mean_squared_error",
            n_jobs=-1, refit=True,
        )
        gs.fit(X_train_pp, y_train)
        cv = gs.cv_results_
        params_df = pd.DataFrame(cv["params"])
        tuning_df = pd.DataFrame({
            **{f"param_{c}": params_df[c].values for c in params_df.columns},
            "mean_cv_RMSE": -cv["mean_test_score"],
        }).sort_values("mean_cv_RMSE")
        tuned_model = gs.best_estimator_

    # Step 12: retrain the tuned model on train + validation combined.
    X_trainval = pd.concat([X_train, X_val])
    y_trainval = pd.concat([y_train, y_val])
    X_trainval_pp = preprocessor.transform(X_trainval)  # transform ONLY
    tuned_model.fit(X_trainval_pp, y_trainval)

    # Step 13: final evaluation on the untouched test set.
    y_pred_test = tuned_model.predict(X_test_pp)
    final_test = evaluate(y_test, y_pred_test)
    final_test["n_test"] = int(len(y_test))

    # Step 14 (+15 tested by caller): wrap preprocessing + model in one Pipeline.
    final_pipeline = Pipeline(
        [("preprocessing", preprocessor), ("model", tuned_model)]
    )

    return {
        "prep_stats": prep_stats,
        "cap_report": cap_report,
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_test)),
        "models_eval": models_eval,
        "best_model_name": best_model_name,
        "tuning": tuning_df,
        "final_test": final_test,
        "pipeline": final_pipeline,
        "y_test_true": np.asarray(y_test),
        "y_test_pred": np.asarray(y_pred_test),
    }


def _count_capped_cells(X_train: pd.DataFrame, preprocessor: Pipeline) -> dict:
    """Count train cells modified by IQR capping (pure reporting)."""
    try:
        fe = preprocessor.named_steps["features"]
        col_sel = preprocessor.named_steps["columns"]
        num_pipe = col_sel.named_transformers_["num"]

        imputer = Pipeline([("imputer", SimpleImputer(strategy="median"))])
        capper = num_pipe.named_steps["outlier_iqr"]

        engineered = fe.transform(X_train)
        num_train = engineered[NUMERIC_FEATURES].to_numpy(dtype=float)
        imputed = np.nan_to_num(imputer.fit_transform(num_train))
        capped = capper.fit_transform(imputed)
        changed = ~np.isclose(imputed, capped)
        return {
            col: int(changed[:, i].sum())
            for i, col in enumerate(NUMERIC_FEATURES)
        }
    except Exception:  # pragma: no cover - reporting must never break the app
        return {}


#  #
# Streamlit UI (default design - no custom CSS, no emojis, no custom colors)
#  #
st.set_page_config(
    page_title="Household Power Consumption - DS Pipeline",
    layout="wide",
)

st.title("Household Electric Power Consumption")
st.caption(
    "UCI dataset: a house in Sceaux, France, 1-minute readings, "
    "16/12/2006 to 26/11/2010 (~2.08M rows). Full 15-step ML pipeline, "
    "default Streamlit theme (no custom styling)."
)


def sidebar_controls():
    """Data source + training options."""
    with st.sidebar:
        st.header("Options")
        source = st.radio(
            "Data source",
            ["Bundled file", "Upload your own"],
            help="Bundled file = household_power_consumption.txt next to app.py",
            key="data_source",
        )
        uploaded = None
        if source == "Upload your own":
            uploaded = st.file_uploader("Choose a CSV", type=["txt", "csv", "data"])
            if uploaded is None:
                st.markdown("Using the bundled file for the demo.")
        sample_size = st.slider(
            "Rows used for modelling (subsample)",
            min_value=5_000, max_value=50_000, value=20_000, step=5_000,
            help="The app reads the full file, but trains on a random subsample "
                 "so the demo stays fast.",
        )
        tune = st.checkbox("Hyperparameter tuning (GridSearchCV)", value=True)
        run_clicked = st.button(
            "Run full pipeline (steps 1-15)",
            type="primary", use_container_width=True,
        )
        return source, uploaded, sample_size, tune, run_clicked


def show_pipeline_steps_tab():
    st.subheader("How the 15 required steps are implemented")
    steps = [
        ("1. Clean data (remove duplicates, fix formats)",
         "`_read_csv()` - dedupes rows, parses `Date`+`Time` to datetime "
         "(dd/mm/yyyy), coerces all 7 measurement columns to float."),
        ("2. Handle missing values (impute or drop)",
         "Rows with a missing target are dropped (`prepare_model_data`); "
         "missing numeric features are median-imputed inside the pipeline "
         "(`SimpleImputer(strategy='median')`, fitted on train only)."),
        ("3. Handle categorical data (one-hot encoding)",
         "`FeatureEngineer` derives hour / day-of-week / month from the "
         "timestamp; `OneHotEncoder(handle_unknown='ignore')` encodes them "
         "(43 new columns)."),
        ("4. Handle outliers (cap)",
         "`IQRCapper` clips every numeric column to "
         "[Q1 - 1.5*IQR, Q3 + 1.5*IQR]; bounds fitted on train only."),
        ("5. Define X and y",
         "`prepare_model_data()` - X = 6 measurements + date/time components, "
         "y = `Global_active_power` (kW)."),
        ("6. Split train / validation / test",
         "`split_data()` - 60 / 20 / 20 random split."),
        ("7. Scaling - fit_transform on train, transform only on val/test",
         "`preprocessor.fit_transform(X_train)` then "
         "`preprocessor.transform(X_val/X_test)`; the scaler never sees "
         "validation/test data."),
        ("8. Handle imbalance",
         "Not applicable - this is a regression task, not a classification one."),
        ("9. Loop through candidate models",
         "`run_experiment()` fits 5 regressors (Linear, Ridge, Decision Tree, "
         "Random Forest, Gradient Boosting) and evaluates MAE / MSE / RMSE / R2 "
         "on validation."),
        ("10. Compare and pick the best",
         "Sorted by highest R2 (lowest RMSE as tie-break). Model chosen is "
         "typically Random Forest on this dataset."),
        ("11. Hyperparameter tuning",
         "`GridSearchCV` (3-fold, neg RMSE) on the best model; grid defined in "
         "`TUNING_GRIDS`."),
        ("12. Retrain tuned model on train + validation",
         "`tuned_model.fit(X_trainval_pp, y_trainval)` - preprocessing "
         "transform only (already fitted on train)."),
        ("13. Final evaluation on the untouched test set",
         "MAE / MSE / RMSE / R2 computed on the 20% test fold that was never "
         "used for cleaning, scaling, tuning or training."),
        ("14. Wrap preprocessing + model into one sklearn Pipeline",
         "`Pipeline([('preprocessing', preprocessor), ('model', tuned_model)])` - "
         "training and inference always match."),
        ("15. Save the model and pipeline",
         "`joblib.dump(pipeline, 'model.pkl')` - written next to app.py after "
         "every successful run."),
    ]
    for title, body in steps:
        with st.expander(title):
            st.markdown(body)


def main():
    source, uploaded, sample_size, tune, run_clicked = sidebar_controls()

    tab_data, tab_clean, tab_steps, tab_model, tab_predict = st.tabs(
        ["Data", "Cleaning", "Pipeline Steps", "Modelling", "Predict"]
    )

    #  load data --
    data = None
    if uploaded is not None:
        data = load_uploaded_data(uploaded.getvalue())
    else:
        data = load_bundled_data()
    df, read_stats = data["df"], data["stats"]

    #  tab 1: data --
    with tab_data:
        st.subheader("Read the data correctly - sanity checks")
        checks = [
            (read_stats["rows_raw"] == 2_075_259, "2,075,259 measurements"),
            (len(read_stats["columns"]) == 9, "9 columns read"),
            (";".join(read_stats["columns"]) ==
             "Date;Time;Global_active_power;Global_reactive_power;Voltage;"
             "Global_intensity;Sub_metering_1;Sub_metering_2;Sub_metering_3",
             "Column names match (semicolon-separated)"),
            (df["datetime"].min() <= pd.Timestamp("2006-12-16"),
             "Starts 16/12/2006"),
            (df["datetime"].max() >= pd.Timestamp("2010-11-26"),
             "Ends 26/11/2010"),
            (df[NUMERIC_FEATURES + [TARGET]].isna().any(axis=1).sum() > 0,
             "Missing values ('?') correctly read as NaN"),
        ]
        check_data = pd.DataFrame(
            [(label, "OK" if ok else "Missing") for ok, label in checks],
            columns=["Check", "Status"],
        )
        st.dataframe(check_data)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{read_stats['rows_raw']:,}")
        c2.metric("Columns", len(read_stats["columns"]))
        c3.metric("Date range", f"{df['datetime'].min():%d/%m/%Y} to "
                                f"{df['datetime'].max():%d/%m/%Y}")
        c4.metric("Rows with '?' (NaN)", f"{df.isna().any(axis=1).sum():,}")

        st.subheader("Raw data (first 10 rows)")
        st.dataframe(df.drop(columns=["datetime"]).head(10))

        st.subheader("Descriptive statistics")
        st.dataframe(df[NUMERIC_FEATURES + [TARGET]].describe().T)

        # EDA plots (on the training subsample for speed)
        sample = df.sample(min(sample_size, len(df)), random_state=42)
        eda = sample.set_index("datetime").sort_index()

        st.subheader("Global active power over time (subsample)")
        st.line_chart(eda[TARGET])

        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("**Histogram of power (kW)**")
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.hist(eda[TARGET], bins=60, edgecolor="white")
            ax.set_xlabel("Global_active_power (kW)")
            ax.set_ylabel("count")
            st.pyplot(fig)
        with col_r:
            st.markdown("**Average power per month of year**")
            monthly = eda.groupby(eda.index.month)[TARGET].mean()
            st.bar_chart(monthly)

        st.subheader("Correlation with the target")
        corr = (df[NUMERIC_FEATURES + [TARGET]].corr()[TARGET]).sort_values(
            ascending=False)
        st.dataframe(corr.rename("corr with target"))

    #  tab 2: cleaning 
    with tab_clean:
        st.subheader("Step 1 - Cleaning report")
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows read", f"{read_stats['rows_raw']:,}")
        c2.metric("Duplicates removed", f"{read_stats['duplicates_removed']:,}")
        c3.metric("Rows after cleaning", f"{read_stats['rows_after_clean']:,}")
        st.caption(
            "Formats fixed: Date+Time to one datetime column; all 7 "
            "measurement columns coerced to float; '?' to NaN."
        )

        st.subheader("Step 2 - Missing values (before imputation)")
        missing = df[NUMERIC_FEATURES + [TARGET]].isna().sum()
        st.dataframe(
            pd.DataFrame({"missing": missing, "pct": 100 * missing / len(df)})
        )
        st.markdown(
            "Strategy: rows with a missing **target** are dropped; missing "
            "**features** are median-imputed inside the pipeline (train only)."
        )

        st.subheader("Step 3 - Categorical to one-hot")
        st.markdown(
            "From the timestamp we derive `hour` (24), `dayofweek` (7) and "
            "`month` (12) then one-hot encode, giving 43 binary columns. "
            "`handle_unknown='ignore'` means unseen category values are safe."
        )

        st.subheader("Step 4 - Outliers (IQR capping)")
        if "outlier_cap_stats" not in st.session_state:
            st.markdown(
                "Run the pipeline (Modelling tab) to see how many cells "
                "were capped on the training data."
            )
        else:
            st.dataframe(
                pd.DataFrame(st.session_state["outlier_cap_stats"],
                             index=["cells capped (train)"])
            )

    #  tab 3: pipeline steps (explanatory) -
    with tab_steps:
        show_pipeline_steps_tab()

    #  tab 4: modelling 
    with tab_model:
        st.subheader("Pipeline run (steps 5-15)")

        if run_clicked:
            with st.spinner(
                "Running steps 1-15... First run takes about a minute; "
                "results are cached. Subsequent runs are instant."
            ):
                progress_bar = st.progress(0.0, text="Fitting candidate models")
                def _cb(done, total):
                    progress_bar.progress(done / total,
                                          text=f"Fitting model {done} of {total}")

                result = _cached_experiment(
                    sample_size, tune, hash_data(df)
                ) if uploaded is None else _cached_uploaded_experiment(
                    sample_size, tune, hash_data(df), uploaded.getvalue()
                )
                progress_bar.progress(1.0, text="Done.")
                st.session_state["result"] = result
                st.session_state["outlier_cap_stats"] = result["cap_report"]

                # Step 15: save
                joblib.dump(result["pipeline"], MODEL_PATH)
                st.session_state["model_saved"] = True

        result = st.session_state.get("result")

        if result is None:
            st.markdown(
                "Click **Run full pipeline (steps 1-15)** in the sidebar. "
                "The run does steps 5 to 15 (split, scale, 5 candidate models, "
                "GridSearchCV, retrain, final test) and saves `model.pkl`."
            )
        else:
            st.markdown(
                f"**Best model:** {result['best_model_name']} "
                f"(by validation R2)"
            )

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Train rows", f"{result['n_train']:,}")
            m2.metric("Validation rows", f"{result['n_val']:,}")
            m3.metric("Test rows (unseen)", f"{result['n_test']:,}")
            m4.metric("Features", "49")

            st.subheader("Step 9 - Candidate model comparison (validation)")
            st.dataframe(result["models_eval"])
            st.bar_chart(result["models_eval"][["RMSE", "MAE"]])

            if result["tuning"] is not None:
                st.subheader(
                    f"Step 11 - GridSearchCV on {result['best_model_name']}"
                )
                st.dataframe(result["tuning"])
            else:
                st.markdown(
                    "Tuning skipped - Linear Regression has no "
                    "hyperparameters to search."
                )

            st.subheader(
                "Steps 12/13 - Retrained on train+val, "
                "final score on the untouched test set"
            )
            ft = result["final_test"]
            t1, t2, t3, t4 = st.columns(4)
            t1.metric("MAE", f"{ft['MAE']:.4f}")
            t2.metric("MSE", f"{ft['MSE']:.4f}")
            t3.metric("RMSE", f"{ft['RMSE']:.4f}")
            t4.metric("R2", f"{ft['R2']:.4f}")
            st.caption(
                "This is the true unbiased estimate: the test fold was never "
                "used for scaling, imputation, capping, tuning or training."
            )

            st.subheader("Predicted vs actual (test set)")
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(
                result["y_test_true"], result["y_test_pred"],
                s=6, alpha=0.4,
            )
            lim = [0, max(result["y_test_true"].max(),
                          result["y_test_pred"].max()) + 0.5]
            ax.plot(lim, lim, "--", label="perfect")
            ax.set_xlabel("actual (kW)")
            ax.set_ylabel("predicted (kW)")
            ax.legend()
            st.pyplot(fig)

            st.subheader("Step 14 + 15 - Pipeline and save")
            st.code(
                "Pipeline([('preprocessing', preprocessor), ('model', "
                "<best tuned model>)])"
            )
            size_mb = MODEL_PATH.stat().st_size / 1e6
            if st.session_state.get("model_saved"):
                st.markdown(
                    f"Saved: `{MODEL_PATH.name}` "
                    f"({size_mb:.1f} MB)"
                )
            else:
                st.markdown("Saved after the run. Reload it in the Predict tab.")

    #  tab 5: predict 
    with tab_predict:
        st.subheader("Predict global active power for a new minute")

        pipeline = None
        if "result" in st.session_state:
            pipeline = st.session_state["result"]["pipeline"]
        elif MODEL_PATH.exists():
            pipeline = joblib.load(MODEL_PATH)
            st.caption("Loaded saved pipeline from model.pkl")

        if pipeline is None:
            st.markdown("No model yet - run the pipeline first (Modelling tab).")
            return

        col_a, col_b = st.columns(2)
        with col_a:
            day = st.date_input("Date", value=pd.Timestamp("2007-03-15"))
            t_of_day = st.time_input(
                "Time", value=pd.Timestamp("12:00:00").time()
            )
        with col_b:
            inputs = {}
            inputs["Global_reactive_power"] = st.number_input(
                "Global reactive power (kVarh)", value=0.4, step=0.1
            )
            inputs["Voltage"] = st.number_input(
                "Voltage (V)", value=240.0, step=0.5
            )
            inputs["Global_intensity"] = st.number_input(
                "Global intensity (A)", value=12.0, step=0.5
            )
            inputs["Sub_metering_1"] = st.number_input(
                "Sub-metering 1 (Wh)", value=1.0, step=1.0
            )
            inputs["Sub_metering_2"] = st.number_input(
                "Sub-metering 2 (Wh)", value=1.0, step=1.0
            )
            inputs["Sub_metering_3"] = st.number_input(
                "Sub-metering 3 (Wh)", value=8.0, step=1.0
            )

        if st.button("Predict", type="primary"):
            row = {"Date": day.strftime("%d/%m/%Y"),
                   "Time": t_of_day.strftime("%H:%M:%S"), **inputs}
            X_new = pd.DataFrame([row])
            pred = pipeline.predict(X_new)[0]
            st.metric("Predicted Global_active_power", f"{pred:.3f} kW")
            st.caption(
                "Uses the exact same pipeline that was trained: "
                "feature engineering, then median imputation, then "
                "IQR capping, then scaling, then model."
            )


@st.cache_resource(show_spinner="Training models ...")
def _cached_experiment(sample_size: int, tune: bool, data_hash: str) -> dict:
    del data_hash  # used only as a cache key
    return run_experiment(load_bundled_data()["df"], sample_size=sample_size,
                          tune=tune)


@st.cache_resource(show_spinner="Training models ...")
def _cached_uploaded_experiment(sample_size: int, tune: bool,
                                data_hash: str, file_bytes: bytes) -> dict:
    del data_hash  # used only as a cache key
    return run_experiment(load_uploaded_data(file_bytes)["df"],
                          sample_size=sample_size, tune=tune)


def hash_data(df: pd.DataFrame) -> str:
    """Cheap cache key so a changed file reruns the experiment."""
    return hashlib.sha1(df["datetime"].max().isoformat().encode()).hexdigest()


if __name__ == "__main__":
    main()
