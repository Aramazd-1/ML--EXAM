import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor  # or RandomForestClassifier


# -----------------------------
# 0) Cleaning + feature engineering
# -----------------------------
def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # strip whitespace in column names (you have many trailing spaces)
    df.columns = [c.strip() for c in df.columns]
    # normalize empty strings to NaN
    df = df.replace(r"^\s*$", np.nan, regex=True)
    return df


def _to_num(s: pd.Series) -> pd.Series:
    """Convert messy numeric strings to float."""
    if s is None:
        return s
    cleaned = s.astype(str).str.replace(r"[^0-9,\.\-]", "", regex=True)
    cleaned = cleaned.str.replace(",", ".", regex=False)
    return pd.to_numeric(cleaned, errors="coerce")


def _coalesce(df: pd.DataFrame, cols: list[str], new_name: str) -> pd.Series:
    out = None
    for c in cols:
        if c in df.columns:
            s = df[c].copy()
            # normalize blanks and common placeholders (e.g. '-' , 'NA', 'N/A', 'none') to NaN
            s = s.replace(r'^\s*$', np.nan, regex=True)
            s = s.replace(r'^\s*[-–—]\s*$', np.nan, regex=True)
            s = s.replace(r'^(?i:\s*na|n/a|none)\s*$', np.nan, regex=True)
            # strip surrounding whitespace for string values
            s = s.apply(lambda v: v.strip() if isinstance(v, str) else v)
            out = s if out is None else out.combine_first(s)
    if out is None:
        out = pd.Series([np.nan] * len(df), index=df.index)
    out.name = new_name
    return out


def _bin_firm_size(emp: pd.Series) -> pd.Series:
    """
    Your colleague rule was ambiguous at 6 ("da 6 in giù small, 6-20 medium").
    I implement: <=6 small, 7-20 medium, >20 large.
    """
    e = _to_num(emp)
    return pd.cut(e, [-np.inf, 6, 20, np.inf], labels=["small", "medium", "large"])


def build_model_df(df_raw: pd.DataFrame, damage_missing_to_zero: bool = True) -> pd.DataFrame:
    df = _standardize_columns(df_raw)

    # Merge Sector columns: "Sector" (first) + "Sector.1" (third)
    df["Sector_merged"] = _coalesce(df, ["Sector", "Sector.1"], "Sector_merged")
    df["Sector_merged"] = df["Sector_merged"].apply(lambda v: v.lower() if isinstance(v, str) else v)
    df.drop(columns=["Sector", "Sector.1"], inplace=True)

    # Employees: use Employees, fallback to N. of employees
    df["Employees_merged"] = _coalesce(df, ["Employees", "N. of employees"], "Employees_merged")
    df["Employees_merged"] = _to_num(df["Employees_merged"])
    df["Firm_size"] = _bin_firm_size(df["Employees_merged"]).astype("object")

    # Surface = Width x Length (keep raw Notes.2 as additional text feature if you want)
    if "Width" in df.columns:
        df["Width_num"] = _to_num(df["Width"])
    elif "Width " in df.columns:  # just in case stripping didn’t catch
        df["Width_num"] = _to_num(df["Width "])
    else:
        df["Width_num"] = np.nan

    if "Length" in df.columns:
        df["Length_num"] = _to_num(df["Length"])
    elif "Length " in df.columns:
        df["Length_num"] = _to_num(df["Length "])
    else:
        df["Length_num"] = np.nan

    df["Surface_calc"] = df["Width_num"] * df["Length_num"]

    # Elevation variables: keep deltaQ/hg/h1; (don’t hard-impute here, pipeline will handle)
    delta_candidates = ["∆Q", "ΔQ", "deltaQ", "DeltaQ"]
    delta_col = next((c for c in delta_candidates if c in df.columns), None)
    df["deltaQ"] = _to_num(df[delta_col]) if delta_col else np.nan
    df["hg"] = _to_num(df["hg"]) if "hg" in df.columns else np.nan
    df["h1"] = _to_num(df["h1"]) if "h1" in df.columns else np.nan

    # Water height outside: hw
    df["hw"] = _to_num(df["hw"]) if "hw" in df.columns else np.nan

    # Observed damages: optionally treat missing as zero (common if blank truly means “no damage recorded”)
    damage_cols = [
        "Building structure and plants",
        "machinery, production plants, equipment and furniture",
        "store and archives",
        "movable goods",
        "recovery and mitigation costs",
        "indirect damage: usability, activity disruption",
    ]
    for c in damage_cols:
        if c in df.columns:
            df[c] = _to_num(df[c])
            if damage_missing_to_zero:
                df[c] = df[c].fillna(0.0)

    # Drop “duration/time” block + water surface elevation + h2 + sediments/contaminants blocks
    drop_cols = [
        "Starting time", "Starting date", "End time", "End date",
        "Peak of water depth time", "Peak of water depth date",
        "Water surface elevation", "h2",
        "Presence", "Fine", "Coarse", "Vegetation/wood", "Garbage",
        "Other.1", "Other,specify",
        "Presence of contaminants", "Type",
        "Damage to employees", "Date of the survey",
        "Hamlet", "Specify.1", "Specify.2", "Specify.3", "Specify.5", "Cause",
        "Seveso (Chemical ) Authorization",
        "Environmental Authorization", "Sanitary Authorization (biomedical products)",
        "Sanitary Authorization (agro-food products)", "Attached buildings N.", "Specify.4", "Notes",
        "Other, specify", "Notes.1", "Notes.2", "Notes.3", "Notes.4", "Notes.5", "N. of employees",
        "Inside the premise", "Elsewhere", "Causes, specify", "Form B", "Form C", "Form D", "Form E",
        "Specify", "Employees",
    ]
    for c in drop_cols:
        if c in df.columns:
            df = df.drop(columns=c)

    return df


# -----------------------------
# 1) Preprocess + RF pipeline (sklearn RF cannot take NaNs -> we impute)
# -----------------------------
def make_rf_pipeline(X: pd.DataFrame):
    # Identify feature types (after feature engineering)
    numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    # OneHotEncoder API differs slightly across sklearn versions
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    num_pipe = Pipeline(
        steps=[
            # median impute + add missing indicators (lets RF learn missingness)
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ]
    )

    cat_pipe = Pipeline(
        steps=[
            # keep missingness as a category explicitly
            ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
            ("ohe", ohe),
        ]
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_features),
            ("cat", cat_pipe, categorical_features),
        ],
        remainder="drop",
    )

    model = RandomForestRegressor(
        n_estimators=500,
        random_state=0,
        n_jobs=-1,
    )

    pipe = Pipeline(steps=[("preprocess", preprocess), ("model", model)])
    return pipe


# -----------------------------
# 2) Example usage
# -----------------------------
df_raw = pd.read_excel("Data-Marche/Industrial activities dataset/Form_A.xlsx", header=2)
df_clean = build_model_df(df_raw, damage_missing_to_zero=True)
df_clean.to_excel("Data-Marche/survey_clean.xlsx", index=False)







