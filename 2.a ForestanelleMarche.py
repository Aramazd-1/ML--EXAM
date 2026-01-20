import os
import re
import time as tm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import LeaveOneGroupOut, ShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score,
    log_loss
)

# ============================================================
# 0) CONFIG
# ============================================================
DATA_PATH = "Data-Marche/survey_clean.xlsx"
OUT_DIR = "rf_output_Marche_clean"
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_STATE = 42

# RF defaults (same spirit as your NFIP script)
N_ESTIMATORS = 300
MIN_SAMPLES_LEAF = 20
MAX_FEATURES_DEFAULT = 0.30  # fraction of features per split (after preprocessing it's applied as float)

# toggles
DO_FULL_OOB = True
DO_LOEO_MUNICIPALITY = True
DO_LOSO_SECTOR = True
DO_RANDOM_80_20 = True
DO_DIAG = True
DO_CLASSIFY_NONZERO = True

# ============================================================
# 1) COLUMNS (features + outcomes)
# ============================================================
OUTCOME_COLS = [
    "Building structure and plants",
    "machinery, production plants, equipment and furniture",
    "store and archives",
    "movable goods",
    "recovery and mitigation costs",
    "indirect damage: usability, activity disruption",
]

FEATURE_COLS = [
    "Municipality", "Latitude", "Longitude", "Type of activity",
    "Seasonal criticalities", "Authorization",
    "Building typology", "Period of construction", "Building structure",
    "Width", "Length",
    "External areas", "Yard", "Service area", "Other",
    "Level of maintenance",
    "∆Q", "hg", "h1", "hw",
    "Sector_merged", "Employees_merged", "Firm_size",
    "Width_num", "Length_num", "Surface_calc", "deltaQ",
]

# numeric-ish (force coercion)
NUM_LIKE = [
    "Latitude", "Longitude",
    "Employees_merged",
    "Width", "Length",
    "Width_num", "Length_num", "Surface_calc",
    "∆Q", "deltaQ", "hg", "h1", "hw",
]

# ============================================================
# 2) UTILS
# ============================================================
def slugify(s: str) -> str:
    s = str(s)
    s = s.strip().lower()
    s = re.sub(r"[^\w\s\-]+", "", s)
    s = re.sub(r"[\s\-]+", "_", s)
    return s[:120]


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def pearson_corr(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[m]
    y_pred = y_pred[m]
    if len(y_true) < 2:
        return np.nan
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def metrics_reg(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = float(mean_absolute_error(y_true, y_pred))
    mbe = float(np.mean(y_pred - y_true))
    r2 = float(r2_score(y_true, y_pred))
    corr = pearson_corr(y_true, y_pred)
    _rmse = rmse(y_true, y_pred)

    mean_y = float(np.mean(y_true))
    mae_norm = float(mae / mean_y) if mean_y != 0 else np.nan
    mbe_norm = float(mbe / mean_y) if mean_y != 0 else np.nan

    return {
        "rmse": _rmse,
        "mae": mae,
        "mbe": mbe,
        "r2": r2,
        "corr": corr,
        "mae_norm": mae_norm,
        "mbe_norm": mbe_norm,
    }


def metrics_clf(y_true, y_proba):
    """
    y_proba: probability of class 1.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_proba = np.asarray(y_proba, dtype=float)
    y_pred = (y_proba >= 0.5).astype(int)

    out = {
        "acc": float(accuracy_score(y_true, y_pred)),
        "bal_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "prec": float(precision_score(y_true, y_pred, zero_division=0)),
        "rec": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    # log loss needs both classes in y_true sometimes; keep it simple:
    out["logloss"] = float(log_loss(y_true, np.c_[1 - y_proba, y_proba], labels=[0, 1]))
    return out


def pooled_from_folds_reg(df_folds):
    N = float(df_folds["n_test"].sum())
    SSE = float(df_folds["SSE"].sum())
    SST = float(df_folds["SST"].sum())
    SAE = float(df_folds["SAE"].sum())
    SBE = float(df_folds["SBE"].sum())

    rmse_p = float(np.sqrt(SSE / N))
    mae_p = float(SAE / N)
    mbe_p = float(SBE / N)
    r2_p = float(1.0 - SSE / SST) if SST > 0 else np.nan

    sum_y = float(df_folds["SUM_Y"].sum())
    sum_p = float(df_folds["SUM_P"].sum())
    sum_yy = float(df_folds["SUM_YY"].sum())
    sum_pp = float(df_folds["SUM_PP"].sum())
    sum_yp = float(df_folds["SUM_YP"].sum())

    cov = sum_yp - (sum_y * sum_p) / N
    var_y = sum_yy - (sum_y * sum_y) / N
    var_p = sum_pp - (sum_p * sum_p) / N
    corr = float(cov / np.sqrt(var_y * var_p)) if (var_y > 0 and var_p > 0) else np.nan

    mean_y = sum_y / N
    mae_norm = float(mae_p / mean_y) if mean_y != 0 else np.nan
    mbe_norm = float(mbe_p / mean_y) if mean_y != 0 else np.nan

    return {
        "rmse": rmse_p,
        "mae": mae_p,
        "mbe": mbe_p,
        "r2": r2_p,
        "corr": corr,
        "mae_norm": mae_norm,
        "mbe_norm": mbe_norm,
    }


def upsert_summary_csv(path: str, new_rows: pd.DataFrame, key_cols=("target", "task", "spec")):
    for c in key_cols:
        if c not in new_rows.columns:
            raise ValueError(f"new_rows missing key col: {c}")

    if os.path.exists(path):
        old = pd.read_csv(path)
        combo = pd.concat([old, new_rows], ignore_index=True, sort=False)
        combo = combo.drop_duplicates(subset=list(key_cols), keep="last")
    else:
        combo = new_rows.copy()

    combo.to_csv(path, index=False)
    return combo


# ============================================================
# 3) PREPROCESS + MODELS (correct missing handling)
# ============================================================
def make_preprocess(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = [c for c in X.columns if (c in NUM_LIKE) or pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]

    numeric_pipe = Pipeline([
        # KEY: add_indicator=True -> missingness becomes explicit features
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
    ])

    categorical_pipe = Pipeline([
        # KEY: missing becomes its own category
        ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
        ("to_str", FunctionTransformer(lambda a: a.astype(str), feature_names_out="one-to-one")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("num", numeric_pipe, num_cols),
        ("cat", categorical_pipe, cat_cols),
    ])


def make_regressor(prep: ColumnTransformer, max_features=MAX_FEATURES_DEFAULT) -> Pipeline:
    rf = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        max_features=max_features,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        bootstrap=True,
        oob_score=True,
    )
    return Pipeline([("prep", prep), ("rf", rf)])


def make_classifier(prep: ColumnTransformer, max_features=MAX_FEATURES_DEFAULT) -> Pipeline:
    rf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        max_features=max_features,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        bootstrap=True,
        oob_score=True,
    )
    return Pipeline([("prep", prep), ("rf", rf)])


# ============================================================
# 4) OOB metrics (reg + clf)
# ============================================================
def oob_metrics_reg(model: Pipeline, y_train):
    rf = model.named_steps["rf"]
    oob = rf.oob_prediction_
    m = ~np.isnan(oob)
    y_true = np.asarray(y_train, dtype=float)[m]
    y_pred = np.asarray(oob, dtype=float)[m]
    return metrics_reg(y_true, y_pred)


def oob_metrics_clf(model: Pipeline, y_train):
    rf = model.named_steps["rf"]
    proba = rf.oob_decision_function_[:, 1]
    y_true = np.asarray(y_train, dtype=int)
    return metrics_clf(y_true, proba)


# ============================================================
# 5) Permutation importance (raw-feature level)
# ============================================================
def perm_importance_inc_mse(model: Pipeline, X_test: pd.DataFrame, y_test, n_repeats=5):
    y_test = np.asarray(y_test, dtype=float)
    base_pred = model.predict(X_test)
    base_mse = mean_squared_error(y_test, base_pred)

    rng = np.random.default_rng(RANDOM_STATE)
    out = []
    for col in X_test.columns:
        mses = []
        for _ in range(n_repeats):
            Xp = X_test.copy()
            Xp[col] = rng.permutation(Xp[col].values)
            pred_p = model.predict(Xp)
            mses.append(mean_squared_error(y_test, pred_p))
        mse_p = float(np.mean(mses))
        inc = 100.0 * (mse_p - base_mse) / base_mse
        out.append((col, max(0.0, inc)))

    imp = pd.DataFrame(out, columns=["feature", "pct_inc_mse"]).sort_values("pct_inc_mse", ascending=False)
    return imp


def perm_importance_inc_logloss(model: Pipeline, X_test: pd.DataFrame, y_test, n_repeats=5):
    y_test = np.asarray(y_test, dtype=int)
    base_proba = model.predict_proba(X_test)[:, 1]
    base_ll = log_loss(y_test, np.c_[1 - base_proba, base_proba], labels=[0, 1])

    rng = np.random.default_rng(RANDOM_STATE)
    out = []
    for col in X_test.columns:
        lls = []
        for _ in range(n_repeats):
            Xp = X_test.copy()
            Xp[col] = rng.permutation(Xp[col].values)
            proba_p = model.predict_proba(Xp)[:, 1]
            lls.append(log_loss(y_test, np.c_[1 - proba_p, proba_p], labels=[0, 1]))
        ll_p = float(np.mean(lls))
        inc = 100.0 * (ll_p - base_ll) / base_ll
        out.append((col, max(0.0, inc)))

    imp = pd.DataFrame(out, columns=["feature", "pct_inc_logloss"]).sort_values("pct_inc_logloss", ascending=False)
    return imp


def collapse_mdi_by_variable(model: Pipeline, top_n=30):
    prep = model.named_steps["prep"]
    rf = model.named_steps["rf"]
    feat_names = prep.get_feature_names_out()
    imp = rf.feature_importances_
    df_imp = pd.DataFrame({"feat": feat_names, "imp": imp})

    # "num__Latitude" or "cat__Sector_merged_xxx"
    def base_var(s):
        s = s.split("__", 1)[-1]
        return s.split("_", 1)[0]  # collapse one-hot categories

    df_imp["var"] = df_imp["feat"].apply(base_var)
    out = df_imp.groupby("var", as_index=False)["imp"].sum().sort_values("imp", ascending=False)
    return out.head(top_n)


# ============================================================
# 6) Curves: error vs trees, effect of mtry
# ============================================================
def plot_error_vs_trees_reg(prep, X_train, y_train, X_test, y_test, out_png,
                            n_estimators_max=600, step=50, max_features=MAX_FEATURES_DEFAULT):
    Xt = prep.transform(X_train)
    Xte = prep.transform(X_test)
    ytr = np.asarray(y_train, dtype=float)
    yte = np.asarray(y_test, dtype=float)

    rf = RandomForestRegressor(
        n_estimators=step,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        max_features=max_features,
        bootstrap=True,
        oob_score=True,
        warm_start=True,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    ns, oob_mse, te_mse = [], [], []
    for n in range(step, n_estimators_max + 1, step):
        rf.set_params(n_estimators=n)
        rf.fit(Xt, ytr)

        oob = rf.oob_prediction_
        m = ~np.isnan(oob)
        oob_mse.append(mean_squared_error(ytr[m], oob[m]))

        te_pred = rf.predict(Xte)
        te_mse.append(mean_squared_error(yte, te_pred))
        ns.append(n)

    plt.figure()
    plt.plot(ns, oob_mse, marker="o", linestyle="-", label="OOB MSE")
    plt.plot(ns, te_mse, marker="o", linestyle="--", label="Holdout MSE")
    plt.xlabel("trees")
    plt.ylabel("MSE")
    plt.title("Regression: Error vs number of trees")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_effect_mtry_reg(prep, X_train, y_train, X_test, y_test, out_png,
                         n_points=22, frac_min=0.02, frac_max=1.0, n_estimators=400):
    Xt = prep.transform(X_train)
    Xte = prep.transform(X_test)
    ytr = np.asarray(y_train, dtype=float)
    yte = np.asarray(y_test, dtype=float)

    p = Xt.shape[1]
    fracs = np.linspace(frac_min, frac_max, n_points)
    k_grid = sorted(set(max(1, min(p, int(f * p))) for f in fracs))

    oob_mse, te_mse = [], []
    for k in k_grid:
        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            max_features=k,
            bootstrap=True,
            oob_score=True,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
        rf.fit(Xt, ytr)

        oob = rf.oob_prediction_
        m = ~np.isnan(oob)
        oob_mse.append(mean_squared_error(ytr[m], oob[m]))

        te_pred = rf.predict(Xte)
        te_mse.append(mean_squared_error(yte, te_pred))

    plt.figure()
    plt.plot(k_grid, oob_mse, marker="o", linestyle="-", label="OOB MSE")
    plt.plot(k_grid, te_mse, marker="o", linestyle="--", label="Holdout MSE")
    plt.xlabel(f"max_features=k (after one-hot), p={p}")
    plt.ylabel("MSE")
    plt.title("Regression: Error vs max_features (mtry)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_error_vs_trees_clf(prep, X_train, y_train, X_test, y_test, out_png,
                            n_estimators_max=600, step=50, max_features=MAX_FEATURES_DEFAULT):
    Xt = prep.transform(X_train)
    Xte = prep.transform(X_test)
    ytr = np.asarray(y_train, dtype=int)
    yte = np.asarray(y_test, dtype=int)

    rf = RandomForestClassifier(
        n_estimators=step,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        max_features=max_features,
        bootstrap=True,
        oob_score=True,
        warm_start=True,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    ns, oob_err, te_err = [], [], []
    for n in range(step, n_estimators_max + 1, step):
        rf.set_params(n_estimators=n)
        rf.fit(Xt, ytr)

        # oob_score_ is accuracy on OOB
        oob_err.append(1.0 - float(rf.oob_score_))

        te_pred = rf.predict(Xte)
        te_err.append(1.0 - float(accuracy_score(yte, te_pred)))
        ns.append(n)

    plt.figure()
    plt.plot(ns, oob_err, marker="o", linestyle="-", label="OOB error (1-acc)")
    plt.plot(ns, te_err, marker="o", linestyle="--", label="Holdout error (1-acc)")
    plt.xlabel("trees")
    plt.ylabel("error")
    plt.title("Classification: Error vs number of trees")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_effect_mtry_clf(prep, X_train, y_train, X_test, y_test, out_png,
                         n_points=22, frac_min=0.02, frac_max=1.0, n_estimators=400):
    Xt = prep.transform(X_train)
    Xte = prep.transform(X_test)
    ytr = np.asarray(y_train, dtype=int)
    yte = np.asarray(y_test, dtype=int)

    p = Xt.shape[1]
    fracs = np.linspace(frac_min, frac_max, n_points)
    k_grid = sorted(set(max(1, min(p, int(f * p))) for f in fracs))

    oob_err, te_err = [], []
    for k in k_grid:
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            max_features=k,
            bootstrap=True,
            oob_score=True,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
        rf.fit(Xt, ytr)

        oob_err.append(1.0 - float(rf.oob_score_))
        te_pred = rf.predict(Xte)
        te_err.append(1.0 - float(accuracy_score(yte, te_pred)))

    plt.figure()
    plt.plot(k_grid, oob_err, marker="o", linestyle="-", label="OOB error (1-acc)")
    plt.plot(k_grid, te_err, marker="o", linestyle="--", label="Holdout error (1-acc)")
    plt.xlabel(f"max_features=k (after one-hot), p={p}")
    plt.ylabel("error")
    plt.title("Classification: Error vs max_features (mtry)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


# ============================================================
# 7) CV runners
# ============================================================
def run_group_cv_reg(spec_name, X, y, groups):
    splitter = LeaveOneGroupOut()
    rows = []

    for fold, (tr, te) in enumerate(splitter.split(X, y, groups=groups), start=1):
        prep = make_preprocess(X.iloc[tr])
        model = make_regressor(prep, max_features=MAX_FEATURES_DEFAULT)
        model.fit(X.iloc[tr], y.iloc[tr])

        pred = model.predict(X.iloc[te])
        y_te = np.asarray(y.iloc[te], dtype=float)

        m_test = metrics_reg(y_te, pred)

        err = y_te - pred
        SSE = float(np.sum(err ** 2))
        SAE = float(np.sum(np.abs(err)))
        SBE = float(np.sum(pred - y_te))

        ybar = float(np.mean(y_te))
        SST = float(np.sum((y_te - ybar) ** 2))

        SUM_Y = float(np.sum(y_te))
        SUM_P = float(np.sum(pred))
        SUM_YY = float(np.sum(y_te ** 2))
        SUM_PP = float(np.sum(pred ** 2))
        SUM_YP = float(np.sum(y_te * pred))

        m_oob = oob_metrics_reg(model, y.iloc[tr])

        group_id = str(pd.Series(groups).iloc[te].astype("string").unique()[0])

        rows.append({
            "spec": spec_name,
            "fold": fold,
            "group_id": group_id,
            "n_test": len(te),
            "SSE": SSE, "SST": SST, "SAE": SAE, "SBE": SBE,
            "SUM_Y": SUM_Y, "SUM_P": SUM_P, "SUM_YY": SUM_YY, "SUM_PP": SUM_PP, "SUM_YP": SUM_YP,
            **{f"test_{k}": v for k, v in m_test.items()},
            **{f"oob_{k}": v for k, v in m_oob.items()},
        })

    return pd.DataFrame(rows)


def run_group_cv_clf(spec_name, X, y_bin, groups):
    splitter = LeaveOneGroupOut()
    rows = []

    for fold, (tr, te) in enumerate(splitter.split(X, y_bin, groups=groups), start=1):
        prep = make_preprocess(X.iloc[tr])
        model = make_classifier(prep, max_features=MAX_FEATURES_DEFAULT)
        model.fit(X.iloc[tr], y_bin.iloc[tr])

        proba = model.predict_proba(X.iloc[te])[:, 1]
        y_te = np.asarray(y_bin.iloc[te], dtype=int)

        m_test = metrics_clf(y_te, proba)
        m_oob = oob_metrics_clf(model, y_bin.iloc[tr])

        group_id = str(pd.Series(groups).iloc[te].astype("string").unique()[0])

        rows.append({
            "spec": spec_name,
            "fold": fold,
            "group_id": group_id,
            "n_test": len(te),
            **{f"test_{k}": v for k, v in m_test.items()},
            **{f"oob_{k}": v for k, v in m_oob.items()},
        })

    return pd.DataFrame(rows)


def run_random_cv_reg(spec_name, X, y, n_splits=5, test_size=0.2):
    ss = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=RANDOM_STATE)
    rows = []

    for fold, (tr, te) in enumerate(ss.split(X, y), start=1):
        prep = make_preprocess(X.iloc[tr])
        model = make_regressor(prep, max_features=MAX_FEATURES_DEFAULT)
        model.fit(X.iloc[tr], y.iloc[tr])

        pred = model.predict(X.iloc[te])
        y_te = np.asarray(y.iloc[te], dtype=float)

        m_test = metrics_reg(y_te, pred)
        m_oob = oob_metrics_reg(model, y.iloc[tr])

        err = y_te - pred
        SSE = float(np.sum(err ** 2))
        SAE = float(np.sum(np.abs(err)))
        SBE = float(np.sum(pred - y_te))

        ybar = float(np.mean(y_te))
        SST = float(np.sum((y_te - ybar) ** 2))

        SUM_Y = float(np.sum(y_te))
        SUM_P = float(np.sum(pred))
        SUM_YY = float(np.sum(y_te ** 2))
        SUM_PP = float(np.sum(pred ** 2))
        SUM_YP = float(np.sum(y_te * pred))

        rows.append({
            "spec": spec_name,
            "fold": fold,
            "n_test": len(te),
            "SSE": SSE, "SST": SST, "SAE": SAE, "SBE": SBE,
            "SUM_Y": SUM_Y, "SUM_P": SUM_P, "SUM_YY": SUM_YY, "SUM_PP": SUM_PP, "SUM_YP": SUM_YP,
            **{f"test_{k}": v for k, v in m_test.items()},
            **{f"oob_{k}": v for k, v in m_oob.items()},
        })

    return pd.DataFrame(rows), ss


def run_random_cv_clf(spec_name, X, y_bin, n_splits=5, test_size=0.2):
    ss = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=RANDOM_STATE)
    rows = []

    for fold, (tr, te) in enumerate(ss.split(X, y_bin), start=1):
        prep = make_preprocess(X.iloc[tr])
        model = make_classifier(prep, max_features=MAX_FEATURES_DEFAULT)
        model.fit(X.iloc[tr], y_bin.iloc[tr])

        proba = model.predict_proba(X.iloc[te])[:, 1]
        y_te = np.asarray(y_bin.iloc[te], dtype=int)

        m_test = metrics_clf(y_te, proba)
        m_oob = oob_metrics_clf(model, y_bin.iloc[tr])

        rows.append({
            "spec": spec_name,
            "fold": fold,
            "n_test": len(te),
            **{f"test_{k}": v for k, v in m_test.items()},
            **{f"oob_{k}": v for k, v in m_oob.items()},
        })

    return pd.DataFrame(rows), ss


def representative_holdout(groups):
    counts = pd.Series(groups).value_counts()
    holdout = counts.index[0]
    te_mask = (pd.Series(groups) == holdout).to_numpy()
    tr_idx = np.where(~te_mask)[0]
    te_idx = np.where(te_mask)[0]
    return tr_idx, te_idx, str(holdout)


# ============================================================
# 8) MAIN
# ============================================================
def main():
    df = pd.read_csv(DATA_PATH, low_memory=False)

    # Normalize column names: strip spaces (you had trailing spaces in the raw survey)
    df.columns = [c.strip() for c in df.columns]

    # Keep only required columns (features + outcomes + group cols)
    needed = set(FEATURE_COLS + OUTCOME_COLS + ["Sector_merged"])
    present = [c for c in df.columns if c in needed]
    df = df[present].copy()

    # Coerce numeric-like feature cols
    for c in NUM_LIKE:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Group vars
    if "Municipality" not in df.columns:
        raise ValueError("Municipality column missing (needed for LOEO_municipality).")
    groups_muni = df["Municipality"].astype("string").fillna("MISSING")

    groups_sector = None
    if "Sector_merged" in df.columns:
        groups_sector = df["Sector_merged"].astype("string").fillna("MISSING")

    summary_path = os.path.join(OUT_DIR, "summary_targets_tasks_specs.csv")
    all_summary_rows = []

    for outcome in OUTCOME_COLS:
        if outcome not in df.columns:
            print(f"Skipping missing outcome column: {outcome}")
            continue

        outcome_slug = slugify(outcome)

        # X is ONLY features (never any other outcome)
        X = df[[c for c in FEATURE_COLS if c in df.columns]].copy()

        # y regression: treat missing observed damages as 0 (as per your cleaning convention)
        y = pd.to_numeric(df[outcome], errors="coerce").fillna(0.0)

        # per-outcome dirs
        out_dir_reg = os.path.join(OUT_DIR, "regression", outcome_slug)
        os.makedirs(out_dir_reg, exist_ok=True)

        # =========================
        # REGRESSION
        # =========================
        if DO_FULL_OOB:
            prep = make_preprocess(X)
            model = make_regressor(prep, max_features=MAX_FEATURES_DEFAULT)
            model.fit(X, y)
            m_oob = oob_metrics_reg(model, y)

            all_summary_rows.append({
                "target": outcome,
                "task": "regression",
                "spec": "FULL_OOB",
                **m_oob
            })
            print(f"[REG | {outcome}] FULL_OOB:", m_oob)

        if DO_LOEO_MUNICIPALITY and (pd.Series(groups_muni).nunique() >= 2):
            df_loeo = run_group_cv_reg("LOEO_municipality", X, y, groups_muni)
            df_loeo.to_csv(os.path.join(out_dir_reg, "folds_LOEO_municipality.csv"), index=False)

            pooled = pooled_from_folds_reg(df_loeo)
            all_summary_rows.append({
                "target": outcome,
                "task": "regression",
                "spec": "LOEO_municipality_pooled",
                **pooled
            })

            if DO_DIAG:
                tr, te, holdout_id = representative_holdout(groups_muni)
                prep = make_preprocess(X.iloc[tr])
                model = make_regressor(prep, max_features=MAX_FEATURES_DEFAULT)
                model.fit(X.iloc[tr], y.iloc[tr])

                # MDI collapsed
                mdi = collapse_mdi_by_variable(model, top_n=30)
                mdi.to_csv(os.path.join(out_dir_reg, f"imp_LOEO_municipality_holdout_{slugify(holdout_id)}_MDI.csv"),
                           index=False)

                plt.figure()
                plt.barh(mdi["var"][::-1], mdi["imp"][::-1])
                plt.title("RF importance (MDI), collapsed (Top 30)")
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir_reg, f"imp_LOEO_municipality_holdout_{slugify(holdout_id)}_MDI.png"),
                            dpi=200)
                plt.close()

                # permutation importance
                pim = perm_importance_inc_mse(model, X.iloc[te], y.iloc[te], n_repeats=5).head(25)
                pim.to_csv(os.path.join(out_dir_reg, f"imp_LOEO_municipality_holdout_{slugify(holdout_id)}_pctIncMSE.csv"),
                           index=False)

                plt.figure()
                plt.barh(pim["feature"][::-1], pim["pct_inc_mse"][::-1])
                plt.xlabel("%IncMSE (neg -> 0)")
                plt.title("Permutation importance (holdout)")
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir_reg,
                                         f"imp_LOEO_municipality_holdout_{slugify(holdout_id)}_pctIncMSE.png"),
                            dpi=200)
                plt.close()

                # curves
                plot_error_vs_trees_reg(
                    prep=model.named_steps["prep"],
                    X_train=X.iloc[tr], y_train=y.iloc[tr],
                    X_test=X.iloc[te], y_test=y.iloc[te],
                    out_png=os.path.join(out_dir_reg,
                                         f"curve_LOEO_municipality_holdout_{slugify(holdout_id)}_error_vs_trees.png"),
                    n_estimators_max=600, step=50
                )
                plot_effect_mtry_reg(
                    prep=model.named_steps["prep"],
                    X_train=X.iloc[tr], y_train=y.iloc[tr],
                    X_test=X.iloc[te], y_test=y.iloc[te],
                    out_png=os.path.join(out_dir_reg,
                                         f"curve_LOEO_municipality_holdout_{slugify(holdout_id)}_effect_mtry.png"),
                    n_points=22, frac_min=0.02, frac_max=1.0, n_estimators=400
                )

        if DO_LOSO_SECTOR and (groups_sector is not None) and (pd.Series(groups_sector).nunique() >= 2):
            df_loso = run_group_cv_reg("LOSO_sector", X, y, groups_sector)
            df_loso.to_csv(os.path.join(out_dir_reg, "folds_LOSO_sector.csv"), index=False)

            pooled = pooled_from_folds_reg(df_loso)
            all_summary_rows.append({
                "target": outcome,
                "task": "regression",
                "spec": "LOSO_sector_pooled",
                **pooled
            })

            if DO_DIAG:
                tr, te, holdout_id = representative_holdout(groups_sector)
                prep = make_preprocess(X.iloc[tr])
                model = make_regressor(prep, max_features=MAX_FEATURES_DEFAULT)
                model.fit(X.iloc[tr], y.iloc[tr])

                mdi = collapse_mdi_by_variable(model, top_n=30)
                mdi.to_csv(os.path.join(out_dir_reg, f"imp_LOSO_sector_holdout_{slugify(holdout_id)}_MDI.csv"),
                           index=False)

                plot_error_vs_trees_reg(
                    prep=model.named_steps["prep"],
                    X_train=X.iloc[tr], y_train=y.iloc[tr],
                    X_test=X.iloc[te], y_test=y.iloc[te],
                    out_png=os.path.join(out_dir_reg,
                                         f"curve_LOSO_sector_holdout_{slugify(holdout_id)}_error_vs_trees.png"),
                    n_estimators_max=600, step=50
                )
                plot_effect_mtry_reg(
                    prep=model.named_steps["prep"],
                    X_train=X.iloc[tr], y_train=y.iloc[tr],
                    X_test=X.iloc[te], y_test=y.iloc[te],
                    out_png=os.path.join(out_dir_reg,
                                         f"curve_LOSO_sector_holdout_{slugify(holdout_id)}_effect_mtry.png"),
                    n_points=22, frac_min=0.02, frac_max=1.0, n_estimators=400
                )

        if DO_RANDOM_80_20:
            df_rand, ss = run_random_cv_reg("Random_80_20", X, y, n_splits=5, test_size=0.2)
            df_rand.to_csv(os.path.join(out_dir_reg, "folds_Random_80_20.csv"), index=False)

            pooled = pooled_from_folds_reg(df_rand)
            all_summary_rows.append({
                "target": outcome,
                "task": "regression",
                "spec": "Random_80_20_pooled",
                **pooled
            })

            if DO_DIAG:
                tr, te = next(ss.split(X, y))
                prep = make_preprocess(X.iloc[tr])
                model = make_regressor(prep, max_features=MAX_FEATURES_DEFAULT)
                model.fit(X.iloc[tr], y.iloc[tr])

                pim = perm_importance_inc_mse(model, X.iloc[te], y.iloc[te], n_repeats=5).head(25)
                pim.to_csv(os.path.join(out_dir_reg, "imp_Random_80_20_pctIncMSE.csv"), index=False)

                plot_error_vs_trees_reg(
                    prep=model.named_steps["prep"],
                    X_train=X.iloc[tr], y_train=y.iloc[tr],
                    X_test=X.iloc[te], y_test=y.iloc[te],
                    out_png=os.path.join(out_dir_reg, "curve_Random_80_20_error_vs_trees.png"),
                    n_estimators_max=600, step=50
                )
                plot_effect_mtry_reg(
                    prep=model.named_steps["prep"],
                    X_train=X.iloc[tr], y_train=y.iloc[tr],
                    X_test=X.iloc[te], y_test=y.iloc[te],
                    out_png=os.path.join(out_dir_reg, "curve_Random_80_20_effect_mtry.png"),
                    n_points=22, frac_min=0.02, frac_max=1.0, n_estimators=400
                )

        # =========================
        # CLASSIFICATION: 1{damage>0}
        # =========================
        if DO_CLASSIFY_NONZERO:
            out_dir_clf = os.path.join(OUT_DIR, "classification_nonzero", outcome_slug)
            os.makedirs(out_dir_clf, exist_ok=True)

            y_bin = (y > 0).astype(int)

            if DO_FULL_OOB:
                prep = make_preprocess(X)
                model = make_classifier(prep, max_features=MAX_FEATURES_DEFAULT)
                model.fit(X, y_bin)
                m_oob = oob_metrics_clf(model, y_bin)

                all_summary_rows.append({
                    "target": outcome,
                    "task": "classification_nonzero",
                    "spec": "FULL_OOB",
                    **m_oob
                })
                print(f"[CLF | {outcome}] FULL_OOB:", m_oob)

            if DO_LOEO_MUNICIPALITY and (pd.Series(groups_muni).nunique() >= 2):
                df_loeo = run_group_cv_clf("LOEO_municipality", X, y_bin, groups_muni)
                df_loeo.to_csv(os.path.join(out_dir_clf, "folds_LOEO_municipality.csv"), index=False)

                # diagnostics on representative holdout
                if DO_DIAG:
                    tr, te, holdout_id = representative_holdout(groups_muni)
                    prep = make_preprocess(X.iloc[tr])
                    model = make_classifier(prep, max_features=MAX_FEATURES_DEFAULT)
                    model.fit(X.iloc[tr], y_bin.iloc[tr])

                    mdi = collapse_mdi_by_variable(model, top_n=30)
                    mdi.to_csv(os.path.join(out_dir_clf, f"imp_LOEO_municipality_holdout_{slugify(holdout_id)}_MDI.csv"),
                               index=False)

                    pim = perm_importance_inc_logloss(model, X.iloc[te], y_bin.iloc[te], n_repeats=5).head(25)
                    pim.to_csv(os.path.join(out_dir_clf,
                                            f"imp_LOEO_municipality_holdout_{slugify(holdout_id)}_pctIncLogLoss.csv"),
                               index=False)

                    plot_error_vs_trees_clf(
                        prep=model.named_steps["prep"],
                        X_train=X.iloc[tr], y_train=y_bin.iloc[tr],
                        X_test=X.iloc[te], y_test=y_bin.iloc[te],
                        out_png=os.path.join(out_dir_clf,
                                             f"curve_LOEO_municipality_holdout_{slugify(holdout_id)}_error_vs_trees.png"),
                        n_estimators_max=600, step=50
                    )
                    plot_effect_mtry_clf(
                        prep=model.named_steps["prep"],
                        X_train=X.iloc[tr], y_train=y_bin.iloc[tr],
                        X_test=X.iloc[te], y_test=y_bin.iloc[te],
                        out_png=os.path.join(out_dir_clf,
                                             f"curve_LOEO_municipality_holdout_{slugify(holdout_id)}_effect_mtry.png"),
                        n_points=22, frac_min=0.02, frac_max=1.0, n_estimators=400
                    )

            if DO_LOSO_SECTOR and (groups_sector is not None) and (pd.Series(groups_sector).nunique() >= 2):
                df_loso = run_group_cv_clf("LOSO_sector", X, y_bin, groups_sector)
                df_loso.to_csv(os.path.join(out_dir_clf, "folds_LOSO_sector.csv"), index=False)

            if DO_RANDOM_80_20:
                df_rand, ss = run_random_cv_clf("Random_80_20", X, y_bin, n_splits=5, test_size=0.2)
                df_rand.to_csv(os.path.join(out_dir_clf, "folds_Random_80_20.csv"), index=False)

                if DO_DIAG:
                    tr, te = next(ss.split(X, y_bin))
                    prep = make_preprocess(X.iloc[tr])
                    model = make_classifier(prep, max_features=MAX_FEATURES_DEFAULT)
                    model.fit(X.iloc[tr], y_bin.iloc[tr])

                    plot_error_vs_trees_clf(
                        prep=model.named_steps["prep"],
                        X_train=X.iloc[tr], y_train=y_bin.iloc[tr],
                        X_test=X.iloc[te], y_test=y_bin.iloc[te],
                        out_png=os.path.join(out_dir_clf, "curve_Random_80_20_error_vs_trees.png"),
                        n_estimators_max=600, step=50
                    )
                    plot_effect_mtry_clf(
                        prep=model.named_steps["prep"],
                        X_train=X.iloc[tr], y_train=y_bin.iloc[tr],
                        X_test=X.iloc[te], y_test=y_bin.iloc[te],
                        out_png=os.path.join(out_dir_clf, "curve_Random_80_20_effect_mtry.png"),
                        n_points=22, frac_min=0.02, frac_max=1.0, n_estimators=400
                    )

    # write / upsert summary
    if len(all_summary_rows) > 0:
        new_rows = pd.DataFrame(all_summary_rows)
        summary_all = upsert_summary_csv(summary_path, new_rows, key_cols=("target", "task", "spec"))
        print("\nUpdated summary:", summary_path)
        print(summary_all.sort_values(["task", "target", "spec"]).to_string(index=False))


if __name__ == "__main__":
    main()
