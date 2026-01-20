import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneGroupOut, ShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time as tm

# ============================================================
# 0) CONFIG
# ============================================================
DATA_PATH = "nfip_claims_ALL_STATES_2020.csv"  # <-- set this
OUT_DIR = "rf_outputs_nfip_clean"
os.makedirs(OUT_DIR, exist_ok=True)
RANDOM_STATE = 42

# Extended labels (covers ALL FIELDS; you can shorten/rename anytime)
LABELS = {
    "agricultureStructureIndicator": "Agriculture structure (indicator)",
    "amountPaidOnBuildingClaim": "Paid on building claim (LEAKAGE)",
    "amountPaidOnContentsClaim": "Paid on contents claim (LEAKAGE)",
    "amountPaidOnIncreasedCostOfComplianceClaim": "Paid on ICC claim (LEAKAGE)",
    "asOfDate": "As-of date",
    "baseFloodElevation": "Base flood elevation (BFE)",
    "basementEnclosureCrawlspaceType": "Basement/enclosure/crawlspace type",
    "buildingDamageAmount": "Building damage amount",
    "buildingDeductibleCode": "Building deductible code",
    "buildingDescriptionCode": "Building description code",
    "buildingPropertyValue": "Building property value",
    "buildingReplacementCost": "Building replacement cost",
    "causeOfDamage": "Cause of damage",
    "censusBlockGroupFips": "Census block group (FIPS)",
    "censusTract": "Census tract",
    "condominiumCoverageTypeCode": "Condo coverage type code",
    "contentsDamageAmount": "Contents damage amount",
    "contentsDeductibleCode": "Contents deductible code",
    "contentsPropertyValue": "Contents property value",
    "contentsReplacementCost": "Contents replacement cost",
    "countyCode": "County code",
    "crsClassificationCode": "CRS classification",
    "dateOfLoss": "Date of loss",
    "disasterAssistanceCoverageRequired": "Disaster assistance coverage required",
    "elevatedBuildingIndicator": "Elevated building (indicator)",
    "elevationCertificateIndicator": "Elevation certificate (indicator)",
    "elevationDifference": "Elevation difference",
    "eventDesignationNumber": "Event designation number",
    "ficoNumber": "FICO number",
    "floodCharacteristicsIndicator": "Flood characteristics (indicator)",
    "floodEvent": "Flood event",
    "floodWaterDuration": "Floodwater duration",
    "floodZoneCurrent": "Flood zone (current)",
    "floodproofedIndicator": "Floodproofed (indicator)",
    "houseWorship": "House of worship (indicator)",
    "iccCoverage": "ICC coverage",
    "id": "Record ID",
    "latitude": "Latitude",
    "locationOfContents": "Location of contents",
    "longitude": "Longitude",
    "lowestAdjacentGrade": "Lowest adjacent grade",
    "lowestFloorElevation": "Lowest floor elevation",
    "netBuildingPaymentAmount": "Net building payment (LEAKAGE)",
    "netContentsPaymentAmount": "Net contents payment (LEAKAGE)",
    "netIccPaymentAmount": "Net ICC payment (LEAKAGE)",
    "nfipCommunityName": "NFIP community name",
    "nfipCommunityNumberCurrent": "NFIP community number (current)",
    "nfipRatedCommunityNumber": "NFIP rated community number",
    "nonPaymentReasonBuilding": "Non-payment reason (building)",
    "nonPaymentReasonContents": "Non-payment reason (contents)",
    "nonProfitIndicator": "Non-profit (indicator)",
    "numberOfFloorsInTheInsuredBuilding": "Number of floors",
    "numberOfUnits": "Number of units",
    "obstructionType": "Obstruction type",
    "occupancyType": "Occupancy type",
    "originalConstructionDate": "Original construction date",
    "originalNBDate": "Original policy inception date (NB)",
    "policyCount": "Policy count",
    "postFIRMConstructionIndicator": "Post-FIRM construction (indicator)",
    "primaryResidenceIndicator": "Primary residence (indicator)",
    "rateMethod": "Rate method",
    "ratedFloodZone": "Flood zone (rated)",
    "rentalPropertyIndicator": "Rental property (indicator)",
    "replacementCostBasis": "Replacement cost basis",
    "reportedCity": "Reported city",
    "reportedZipCode": "Reported ZIP",
    "smallBusinessIndicatorBuilding": "Small business (building) indicator",
    "state": "State",
    "stateOwnedIndicator": "State-owned (indicator)",
    "totalBuildingInsuranceCoverage": "Total building coverage",
    "totalContentsInsuranceCoverage": "Total contents coverage",
    "waterDepth": "Water depth",
    "yearOfLoss": "Year of loss",
}

FIELDS = [
    'agricultureStructureIndicator', 'amountPaidOnBuildingClaim',
    'amountPaidOnContentsClaim', 'amountPaidOnIncreasedCostOfComplianceClaim',
    'asOfDate', 'baseFloodElevation', 'basementEnclosureCrawlspaceType',
    'buildingDamageAmount', 'buildingDeductibleCode', 'buildingDescriptionCode',
    'buildingPropertyValue', 'buildingReplacementCost', 'causeOfDamage',
    'censusBlockGroupFips', 'censusTract', 'condominiumCoverageTypeCode',
    'contentsDamageAmount', 'contentsDeductibleCode', 'contentsPropertyValue',
    'contentsReplacementCost', 'countyCode', 'crsClassificationCode',
    'dateOfLoss', 'disasterAssistanceCoverageRequired', 'elevatedBuildingIndicator',
    'elevationCertificateIndicator', 'elevationDifference', 'eventDesignationNumber',
    'ficoNumber', 'floodCharacteristicsIndicator', 'floodEvent', 'floodWaterDuration',
    'floodZoneCurrent', 'floodproofedIndicator', 'houseWorship', 'iccCoverage',
    'id', 'latitude', 'locationOfContents', 'longitude', 'lowestAdjacentGrade',
    'lowestFloorElevation', 'netBuildingPaymentAmount', 'netContentsPaymentAmount',
    'netIccPaymentAmount', 'nfipCommunityName', 'nfipCommunityNumberCurrent',
    'nfipRatedCommunityNumber', 'nonPaymentReasonBuilding', 'nonPaymentReasonContents',
    'nonProfitIndicator', 'numberOfFloorsInTheInsuredBuilding', 'numberOfUnits',
    'obstructionType', 'occupancyType', 'originalConstructionDate', 'originalNBDate',
    'policyCount', 'postFIRMConstructionIndicator', 'primaryResidenceIndicator',
    'rateMethod', 'ratedFloodZone', 'rentalPropertyIndicator', 'replacementCostBasis',
    'reportedCity', 'reportedZipCode', 'smallBusinessIndicatorBuilding', 'state',
    'stateOwnedIndicator', 'totalBuildingInsuranceCoverage',
    'totalContentsInsuranceCoverage', 'waterDepth', 'yearOfLoss'
]

DROP = {"asOfDate", "dateOfLoss", "yearOfLoss"}
FIELDS_MINUS = [f for f in FIELDS if f not in DROP]

NUM_LIKE = [
    "buildingDamageAmount", "contentsDamageAmount",
    "buildingReplacementCost", "contentsReplacementCost", "contentsPropertyValue",
    "totalBuildingInsuranceCoverage", "totalContentsInsuranceCoverage",
    "waterDepth", "floodWaterDuration",
    "baseFloodElevation", "elevationDifference", "lowestAdjacentGrade", "lowestFloorElevation",
    "latitude", "longitude",
    "numberOfFloorsInTheInsuredBuilding", "numberOfUnits", "policyCount",
]

LEAKAGE_COLS = [
    "amountPaidOnBuildingClaim",
    "amountPaidOnContentsClaim",
    "amountPaidOnIncreasedCostOfComplianceClaim",
    "netBuildingPaymentAmount",
    "netContentsPaymentAmount",
    "netIccPaymentAmount",
    "nonPaymentReasonBuilding",
    "nonPaymentReasonContents",
    "buildingPropertyValue",
]

ID_COLS = ["id"]

# RF defaults
N_ESTIMATORS = 200  # Trees used in the estimation
MIN_SAMPLES_LEAF = 20   # Minimum samples per leaf
MAX_FEATURES_DEFAULT = 0.3  # fraction of features to consider at each split

# ============================================================
# 0b) WHICH ESTIMATES TO RUN (you said you already have these)
# ============================================================
DO_FULL_OOB = True
DO_LOEO_EVENT = True
DO_LOSO_STATE = True
DO_RANDOM_80_20 = True

# ============================================================
# 0c) WHICH FEATURE-SPECS TO RUN (NEW)
# ============================================================
RUN_RATIO_BASE = True
RUN_LOG_BASE = True
RUN_LOG_NO_REPL_COST = True  # log_damage but drop repl costs from X

# ============================================================
# 0d) DIAGNOSTICS PER FEATURE-SPEC (NEW)
# (master switch still replicate_whole)
# ============================================================
DIAG_RATIO_BASE = True
DIAG_LOG_BASE = True
DIAG_LOG_NO_REPL_COST = True


# ============================================================
# 1) METRICS
# ============================================================
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


def metrics_paper(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mae = float(mean_absolute_error(y_true, y_pred))
    mbe = float(np.mean(y_pred - y_true))  # positive = overprediction
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


# ============================================================
# 2) PREPROCESS + MODEL
# ============================================================
def make_preprocess(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = [c for c in X.columns if (c in NUM_LIKE) or pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("to_str", FunctionTransformer(lambda a: a.astype(str), feature_names_out="one-to-one")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("num", numeric_pipe, num_cols),
        ("cat", categorical_pipe, cat_cols),
    ])


def make_model(prep: ColumnTransformer, max_features=MAX_FEATURES_DEFAULT) -> Pipeline:
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


def postprocess_fn(target_kind: str):
    if target_kind == "ratio":
        return lambda p: np.clip(p, 0, None)  # keep >=0, do NOT cap at 1 unless you want
    else:
        return lambda p: np.maximum(p, 0)


# ============================================================
# 3) DATASET BUILDER
# ============================================================
def build_dataset(df: pd.DataFrame, target_kind: str, extra_drop_X=None):
    extra_drop_X = extra_drop_X or []
    use_cols = [c for c in FIELDS_MINUS if c in df.columns]
    df_use = df[use_cols].copy()

    # coerce numeric-like
    for c in NUM_LIKE:
        if c in df_use.columns:
            df_use[c] = pd.to_numeric(df_use[c], errors="coerce")

    dmg_b = df_use.get("buildingDamageAmount", 0).fillna(0)
    dmg_c = df_use.get("contentsDamageAmount", 0).fillna(0)
    damage_total = dmg_b + dmg_c

    repl_b = df_use.get("buildingReplacementCost", 0).fillna(0)
    repl_c = df_use.get("contentsReplacementCost", 0).fillna(0)
    repl_total = repl_b + repl_c

    if target_kind == "ratio":
        MIN_REPL = 1000
        mask = (repl_total >= MIN_REPL)

        mask &= ~((dmg_c > 0) & (repl_c <= 0))
        mask &= ~((dmg_b > 0) & (repl_b <= 0))

        y_raw = (damage_total[mask] / repl_total[mask]).astype(float)  # pandas Series
        CAP_Q = 99.5
        cap = np.nanpercentile(y_raw.to_numpy(), CAP_Q)
        y = pd.Series(np.clip(y_raw.to_numpy(), 0, cap), index=y_raw.index)

        target_drop = [
            "buildingDamageAmount", "contentsDamageAmount",
            "buildingReplacementCost", "contentsReplacementCost"
        ]

    elif target_kind == "log_damage":
        mask = np.ones(len(df_use), dtype=bool)
        y = np.log1p(damage_total).astype(float)
        target_drop = ["buildingDamageAmount", "contentsDamageAmount"]

    else:
        raise ValueError("target_kind must be 'ratio' or 'log_damage'")

    groups_event = None
    if "eventDesignationNumber" in df_use.columns:
        groups_event = df_use.loc[mask, "eventDesignationNumber"].astype("string").fillna("MISSING")

    groups_state = None
    if "state" in df_use.columns:
        groups_state = df_use.loc[mask, "state"].astype("string").fillna("MISSING")

    drop_cols = [c for c in (LEAKAGE_COLS + ID_COLS + target_drop + list(extra_drop_X)) if c in df_use.columns]
    X = df_use.loc[mask].drop(columns=drop_cols).copy()

    # remove group cols from X
    for gc in ["eventDesignationNumber", "state"]:
        if gc in X.columns:
            X = X.drop(columns=[gc])

    # drop all-missing cols
    all_missing = [c for c in X.columns if X[c].isna().all()]
    if all_missing:
        X = X.drop(columns=all_missing)

    # Align y index to X (important for ratio case)
    if isinstance(y, pd.Series):
        y = y.loc[X.index]
    else:
        y = pd.Series(y, index=X.index)

    return X, y, groups_event, groups_state


# ============================================================
# 4) OOB METRICS FROM FITTED MODEL
# ============================================================
def oob_metrics_from_model(model: Pipeline, y_train, postprocess):
    rf = model.named_steps["rf"]
    oob = rf.oob_prediction_
    mask = ~np.isnan(oob)
    y_true = np.asarray(y_train)[mask]
    y_pred = np.asarray(oob)[mask]
    y_pred = postprocess(y_pred)
    return metrics_paper(y_true, y_pred)


# ============================================================
# 5) PERMUTATION IMPORTANCE as %IncMSE (neg -> 0)
# ============================================================
def perm_importance_percent_inc_mse(model: Pipeline, X_test: pd.DataFrame, y_test,
                                    postprocess=lambda p: p, n_repeats=5):
    y_test = np.asarray(y_test)
    base_pred = postprocess(model.predict(X_test))
    base_mse = mean_squared_error(y_test, base_pred)

    rng = np.random.default_rng(RANDOM_STATE)
    out = []
    for col in X_test.columns:
        mses = []
        for _ in range(n_repeats):
            Xp = X_test.copy()
            Xp[col] = rng.permutation(Xp[col].values)
            pred_p = postprocess(model.predict(Xp))
            mses.append(mean_squared_error(y_test, pred_p))
        mse_p = float(np.mean(mses))
        inc = 100.0 * (mse_p - base_mse) / base_mse
        out.append((col, inc))

    imp_df = pd.DataFrame(out, columns=["feature", "pct_inc_mse"])
    imp_df["pct_inc_mse"] = imp_df["pct_inc_mse"].clip(lower=0)
    return imp_df.sort_values("pct_inc_mse", ascending=False)


def plot_varimp_percent_inc_mse(model, X_test, y_test, postprocess, out_png, top_n=20, n_repeats=5):
    imp_df = perm_importance_percent_inc_mse(
        model, X_test, y_test, postprocess=postprocess, n_repeats=n_repeats
    ).head(top_n)

    imp_df["label"] = imp_df["feature"].map(LABELS).fillna(imp_df["feature"])

    plt.figure()
    plt.barh(imp_df["label"][::-1], imp_df["pct_inc_mse"][::-1])
    plt.xlabel("%IncMSE (neg -> 0)")
    plt.title("Permutation importance (holdout)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def collapse_mdi_by_variable(model, top_n=30):
    prep = model.named_steps["prep"]
    rf = model.named_steps["rf"]
    feat_names = prep.get_feature_names_out()
    imp = rf.feature_importances_

    df_imp = pd.DataFrame({"feat": feat_names, "imp": imp})

    # formats like "num__waterDepth" or "cat__occupancyType_3"
    def base_var(s):
        s = s.split("__", 1)[-1]
        return s.split("_", 1)[0]  # before first "_" in one-hot

    df_imp["var"] = df_imp["feat"].apply(base_var)
    out = df_imp.groupby("var", as_index=False)["imp"].sum().sort_values("imp", ascending=False)
    out["label"] = out["var"].map(LABELS).fillna(out["var"])
    return out.head(top_n)


def plot_mdi_importance(model, out_png, top_n=30):
    imp_df = collapse_mdi_by_variable(model, top_n=top_n)
    plt.figure()
    plt.barh(imp_df["label"][::-1], imp_df["imp"][::-1])
    plt.title(f"RF importance (MDI), collapsed (Top {top_n})")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


# ============================================================
# 6) CURVES: error vs trees, effect of mtry
# ============================================================
def plot_error_vs_trees(prep, X_train, y_train, X_test, y_test, postprocess, out_png,
                        n_estimators_max=500, step=25, max_features=MAX_FEATURES_DEFAULT):
    Xt = prep.transform(X_train)
    Xte = prep.transform(X_test)
    ytr = np.asarray(y_train)
    yte = np.asarray(y_test)

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
        oob_pred = postprocess(oob[m])
        ytr_m = ytr[m]
        oob_mse.append(mean_squared_error(ytr_m, oob_pred))

        pred_te = postprocess(rf.predict(Xte))
        te_mse.append(mean_squared_error(yte, pred_te))
        ns.append(n)

    plt.figure()
    plt.plot(ns, oob_mse, marker="o", linestyle="-", label="Out-of-bag MSE")
    plt.plot(ns, te_mse, marker="o", linestyle="--", label="Holdout MSE")
    plt.xlabel("trees")
    plt.ylabel("MSE")
    plt.title("Error vs number of trees")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_effect_mtry_auto(prep, X_train, y_train, X_test, y_test, postprocess, out_png,
                          n_points=22, frac_min=0.02, frac_max=1.0, n_estimators=300):
    """
    Plots OOB and holdout MSE vs max_features = k, where k is the *actual number*
    of features considered at each split (after one-hot expansion).

    n_points=22  -> ~2x the density of your old grid (11 points)
    frac_max=1.0 -> extends beyond 0.3 all the way to 100% to see stabilization
    """
    Xt = prep.transform(X_train)
    Xte = prep.transform(X_test)
    ytr = np.asarray(y_train)
    yte = np.asarray(y_test)

    p = Xt.shape[1]  # number of features AFTER preprocessing (incl. one-hot)

    # Build a denser grid in terms of fractions, then convert to integer k
    fracs = np.linspace(frac_min, frac_max, n_points)
    k_grid = [max(1, int(f * p)) for f in fracs]

    # Also force-in some classic reference points
    k_grid += [
        1, 2, 5, 10,
        max(1, int(np.log2(p))),  # "log2"
        max(1, int(np.sqrt(p))),  # "sqrt"
        p  # all features
    ]

    # Deduplicate + clamp
    k_grid = sorted(set(min(max(1, k), p) for k in k_grid))

    oob_mse, te_mse = [], []
    for k in k_grid:
        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            max_features=k,  # <-- ACTUAL NUMBER OF FEATURES
            bootstrap=True,
            oob_score=True,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
        rf.fit(Xt, ytr)

        oob = rf.oob_prediction_
        m = ~np.isnan(oob)
        oob_pred = postprocess(oob[m])
        ytr_m = ytr[m]
        oob_mse.append(mean_squared_error(ytr_m, oob_pred))

        pred_te = postprocess(rf.predict(Xte))
        te_mse.append(mean_squared_error(yte, pred_te))

    plt.figure()
    plt.plot(k_grid, oob_mse, marker="o", linestyle="-", label="Out-of-bag MSE")
    plt.plot(k_grid, te_mse, marker="o", linestyle="--", label="Holdout MSE")
    plt.xlabel(f"max_features = k (actual # features tried per split), p={p} after one-hot")
    plt.ylabel("MSE")
    plt.title("Error Vs maximum number of features (integer k-grid)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


# ============================================================
# 7) CV runner (metrics across folds; OOB metrics on each fold train)
# ============================================================
def run_group_cv(spec_name, X, y, groups, postprocess):
    splitter = LeaveOneGroupOut()
    rows = []

    for fold, (tr, te) in enumerate(splitter.split(X, y, groups=groups), start=1):
        prep = make_preprocess(X.iloc[tr])
        model = make_model(prep, max_features=MAX_FEATURES_DEFAULT)
        model.fit(X.iloc[tr], y.iloc[tr])

        pred = postprocess(model.predict(X.iloc[te]))
        y_te = np.asarray(y.iloc[te])
        y_te_mean = float(np.mean(y_te))
        y_te_std = float(np.std(y_te))
        group_id = str(pd.Series(groups).iloc[te].astype("string").unique()[0])

        m_test = metrics_paper(y_te, pred)
        err = y_te - pred

        SSE = float(np.sum(err ** 2))
        SAE = float(np.sum(np.abs(err)))
        SBE = float(np.sum(pred - y_te))

        # Use fold mean baseline for SST (consistent with fold R2)
        ybar = float(np.mean(y_te))
        SST = float(np.sum((y_te - ybar) ** 2))

        # Sufficient stats for pooled correlation
        SUM_Y = float(np.sum(y_te))
        SUM_P = float(np.sum(pred))
        SUM_YY = float(np.sum(y_te ** 2))
        SUM_PP = float(np.sum(pred ** 2))
        SUM_YP = float(np.sum(y_te * pred))
        base_pred = np.repeat(np.mean(y.iloc[tr]), len(te))
        m_base = metrics_paper(y_te, base_pred)

        m_oob = oob_metrics_from_model(model, y_train=y.iloc[tr], postprocess=postprocess)

        rows.append({
            "spec": spec_name,
            "fold": fold,
            "group_id": group_id,
            "n_test": len(te),
            "y_test_mean": y_te_mean,
            "y_test_std": y_te_std,
            "SSE": SSE, "SST": SST, "SAE": SAE, "SBE": SBE,
            "SUM_Y": SUM_Y, "SUM_P": SUM_P, "SUM_YY": SUM_YY, "SUM_PP": SUM_PP, "SUM_YP": SUM_YP,
            **{f"test_{k}": v for k, v in m_test.items()},
            **{f"test_{k}": v for k, v in m_test.items()},
            **{f"base_{k}": v for k, v in m_base.items()},
            **{f"oob_{k}": v for k, v in m_oob.items()},
        })

    return pd.DataFrame(rows)


def run_random_cv(spec_name, X, y, postprocess, n_splits=5, test_size=0.2):
    ss = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=RANDOM_STATE)
    rows = []

    for fold, (tr, te) in enumerate(ss.split(X, y), start=1):
        prep = make_preprocess(X.iloc[tr])
        model = make_model(prep, max_features=MAX_FEATURES_DEFAULT)
        model.fit(X.iloc[tr], y.iloc[tr])

        pred = postprocess(model.predict(X.iloc[te]))
        y_te = np.asarray(y.iloc[te])
        m_test = metrics_paper(y_te, pred)
        y_te = np.asarray(y.iloc[te], dtype=float)
        pred = np.asarray(postprocess(model.predict(X.iloc[te])), dtype=float)

        err = y_te - pred

        SSE = float(np.sum(err ** 2))
        SAE = float(np.sum(np.abs(err)))
        SBE = float(np.sum(pred - y_te))

        # Use fold mean baseline for SST (consistent with fold R2)
        ybar = float(np.mean(y_te))
        SST = float(np.sum((y_te - ybar) ** 2))

        # Sufficient stats for pooled correlation
        SUM_Y = float(np.sum(y_te))
        SUM_P = float(np.sum(pred))
        SUM_YY = float(np.sum(y_te ** 2))
        SUM_PP = float(np.sum(pred ** 2))
        SUM_YP = float(np.sum(y_te * pred))

        base_pred = np.repeat(np.mean(y.iloc[tr]), len(te))
        m_base = metrics_paper(y_te, base_pred)

        m_oob = oob_metrics_from_model(model, y_train=y.iloc[tr], postprocess=postprocess)

        rows.append({
            "spec": spec_name,
            "fold": fold,
            "n_test": len(te),

            "SSE": SSE, "SST": SST, "SAE": SAE, "SBE": SBE,
            "SUM_Y": SUM_Y, "SUM_P": SUM_P, "SUM_YY": SUM_YY, "SUM_PP": SUM_PP, "SUM_YP": SUM_YP,
            **{f"test_{k}": v for k, v in m_test.items()},
            **{f"base_{k}": v for k, v in m_base.items()},
            **{f"oob_{k}": v for k, v in m_oob.items()},
        })

    return pd.DataFrame(rows), ss


def pooled_from_folds(df_folds):
    N = float(df_folds["n_test"].sum())

    SSE = float(df_folds["SSE"].sum())
    SST = float(df_folds["SST"].sum())
    SAE = float(df_folds["SAE"].sum())
    SBE = float(df_folds["SBE"].sum())

    # pooled point metrics
    rmse = float(np.sqrt(SSE / N))
    mae = float(SAE / N)
    mbe = float(SBE / N)
    r2 = float(1.0 - SSE / SST) if SST > 0 else np.nan

    # pooled correlation from sufficient stats
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
    mae_norm = float(mae / mean_y) if mean_y != 0 else np.nan
    mbe_norm = float(mbe / mean_y) if mean_y != 0 else np.nan

    return {
        "rmse": rmse,
        "mae": mae,
        "mbe": mbe,
        "r2": r2,
        "corr": corr,
        "mae_norm": mae_norm,
        "mbe_norm": mbe_norm,
    }


# ============================================================
# 8) Representative split selection for importance + curves
# ============================================================
def representative_holdout_event(groups_event):
    counts = pd.Series(groups_event).value_counts()
    holdout = counts.index[0]
    te_mask = (groups_event == holdout).to_numpy()
    tr_idx = np.where(~te_mask)[0]
    te_idx = np.where(te_mask)[0]
    return tr_idx, te_idx, str(holdout)


def representative_holdout_state(groups_state):
    counts = pd.Series(groups_state).value_counts()
    holdout = counts.index[0]
    te_mask = (groups_state == holdout).to_numpy()
    tr_idx = np.where(~te_mask)[0]
    te_idx = np.where(te_mask)[0]
    return tr_idx, te_idx, str(holdout)

SUMMARY_PATH = os.path.join(OUT_DIR, "summary_specs_targets.csv")

def upsert_summary_csv(path: str, new_rows: pd.DataFrame, key_cols=("target", "feat_spec", "spec")):
    # new_rows must include key cols
    for c in key_cols:
        if c not in new_rows.columns:
            raise ValueError(f"new_rows is missing required key column: {c}")

    if os.path.exists(path):
        old = pd.read_csv(path)
        if "feat_spec" not in old.columns:
            old["feat_spec"] = "BASE"
        # backward compat if you previously wrote without feat_spec
        if "feat_spec" not in old.columns:
            old["feat_spec"] = "BASE"

        combo = pd.concat([old, new_rows], ignore_index=True, sort=False)
        combo = combo.drop_duplicates(subset=list(key_cols), keep="last")
    else:
        combo = new_rows.copy()

    combo.to_csv(path, index=False)
    return combo


# ============================================================
# 9) MAIN
# ============================================================
def main():
    df = pd.read_csv(DATA_PATH, low_memory=False)
    SPECS = [
        ("ratio", "BASE", RUN_RATIO_BASE, DIAG_RATIO_BASE, []),
        ("log_damage", "BASE", RUN_LOG_BASE, DIAG_LOG_BASE, []),
        ("log_damage", "NO_REPL_COST", RUN_LOG_NO_REPL_COST, DIAG_LOG_NO_REPL_COST,
         ["buildingReplacementCost", "contentsReplacementCost"]),
    ]

    # Summary file path
    summary_path = os.path.join(OUT_DIR, "summary_specs_targets.csv")
    os.makedirs(OUT_DIR, exist_ok=True)

    for target_kind, feat_spec, RUN_THIS, RUN_DIAG, extra_drop_X in SPECS:
        if not RUN_THIS:
            continue

        # per-spec collectors
        summary_rows_spec = []
        times = {}

        start_0 = tm.time()
        X, y, groups_event, groups_state = build_dataset(df, target_kind, extra_drop_X=extra_drop_X)
        postprocess = postprocess_fn(target_kind)

        # separate folder for each target+feat_spec
        target_dir = os.path.join(OUT_DIR, target_kind, feat_spec)
        os.makedirs(target_dir, exist_ok=True)

        # ============================================================
        # FULL-DATA OOB baseline
        # ============================================================
        if DO_FULL_OOB:
            prep_full = make_preprocess(X)
            model_full = make_model(prep_full, max_features=MAX_FEATURES_DEFAULT)
            model_full.fit(X, y)
            m_full_oob = oob_metrics_from_model(model_full, y_train=y, postprocess=postprocess)

            summary_rows_spec.append({
                "target": target_kind,
                "feat_spec": feat_spec,
                "spec": "FULL_OOB",
                **m_full_oob
            })
            print(f"[{target_kind} | {feat_spec}] FULL_OOB:", m_full_oob)

            end = tm.time()
            times[f"{target_kind}__{feat_spec}__full_oob"] = end - start_0
            print(f"[{target_kind} | {feat_spec}] Baseline: {end - start_0:.1f} sec, n={len(y)}, p={X.shape[1]}")

        # ============================================================
        # LOEO_event CV (metrics)
        # ============================================================
        if DO_LOEO_EVENT and (groups_event is not None) and (pd.Series(groups_event).nunique() >= 2):
            start = tm.time()
            df_loeo = run_group_cv("LOEO_event", X, y, groups_event, postprocess)
            df_loeo.to_csv(os.path.join(target_dir, "folds_LOEO_event.csv"), index=False)

            pooled = pooled_from_folds(df_loeo)
            summary_rows_spec.append({
                "target": target_kind,
                "feat_spec": feat_spec,
                "spec": "LOEO_event_pooled",
                **pooled
            })

            tr, te, holdout_id = representative_holdout_event(groups_event)
            prep = make_preprocess(X.iloc[tr])
            model = make_model(prep, max_features=MAX_FEATURES_DEFAULT)
            model.fit(X.iloc[tr], y.iloc[tr])

            end = tm.time()
            times[f"{target_kind}__{feat_spec}__loeo_event"] = end - start
            print(f"[{target_kind} | {feat_spec}] LOEO_event: {end - start:.1f} sec")

            if RUN_DIAG:
                start = tm.time()
                plot_varimp_percent_inc_mse(
                    model, X.iloc[te], y.iloc[te],
                    postprocess=postprocess,
                    out_png=os.path.join(target_dir, f"imp_LOEO_event_holdout_{holdout_id}_pctIncMSE.png"),
                    top_n=20, n_repeats=5
                )
                plot_mdi_importance(
                    model,
                    out_png=os.path.join(target_dir, f"imp_LOEO_event_holdout_{holdout_id}_MDI.png"),
                    top_n=30
                )
                plot_error_vs_trees(
                    prep=model.named_steps["prep"],
                    X_train=X.iloc[tr], y_train=y.iloc[tr],
                    X_test=X.iloc[te], y_test=y.iloc[te],
                    postprocess=postprocess,
                    out_png=os.path.join(target_dir,
                                         f"curve_LOEO_event_holdout_{holdout_id}_error_vs_trees.png"),
                    n_estimators_max=500, step=25
                )
                plot_effect_mtry_auto(
                    prep=model.named_steps["prep"],
                    X_train=X.iloc[tr], y_train=y.iloc[tr],
                    X_test=X.iloc[te], y_test=y.iloc[te],
                    postprocess=postprocess,
                    out_png=os.path.join(target_dir, f"curve_LOEO_event_holdout_{holdout_id}_effect_mtry.png"),
                    n_points=22,
                    frac_min=0.02,
                    frac_max=1.0,
                    n_estimators=300
                )
                end = tm.time()
                times[f"{target_kind}__{feat_spec}__loeo_event_diag"] = end - start
                print(f"[{target_kind} | {feat_spec}] LOEO_event diagnostics: {end - start:.1f} sec")

        # ============================================================
        # LOSO_state CV (metrics)
        # ============================================================
        if DO_LOSO_STATE and (groups_state is not None) and (pd.Series(groups_state).nunique() >= 2):
            start = tm.time()
            df_loso = run_group_cv("LOSO_state", X, y, groups_state, postprocess)

            # optional prints you had
            SSE = df_loso["SSE"].sum()
            SST = df_loso["SST"].sum()
            r2_pooled = 1 - SSE / SST if SST != 0 else np.nan
            rmse_pooled = np.sqrt(SSE / df_loso["n_test"].sum())
            print(f"[{target_kind} | {feat_spec}] POOLED LOSO R2:", r2_pooled, "POOLED RMSE:", rmse_pooled)
            print(df_loso.sort_values("test_r2").head(10)[
                ["group_id", "n_test", "y_test_mean", "y_test_std", "test_r2", "test_mae", "test_mbe"]
            ])

            df_loso.to_csv(os.path.join(target_dir, "folds_LOSO_state.csv"), index=False)

            pooled = pooled_from_folds(df_loso)
            summary_rows_spec.append({
                "target": target_kind,
                "feat_spec": feat_spec,
                "spec": "LOSO_state_pooled",
                **pooled
            })

            tr, te, holdout_id = representative_holdout_state(groups_state)
            prep = make_preprocess(X.iloc[tr])
            model = make_model(prep, max_features=MAX_FEATURES_DEFAULT)
            model.fit(X.iloc[tr], y.iloc[tr])

            end = tm.time()
            times[f"{target_kind}__{feat_spec}__loso_state"] = end - start
            print(f"[{target_kind} | {feat_spec}] LOSO_state: {end - start:.1f} sec")

            if RUN_DIAG:
                start = tm.time()
                plot_varimp_percent_inc_mse(
                    model, X.iloc[te], y.iloc[te],
                    postprocess=postprocess,
                    out_png=os.path.join(target_dir, f"imp_LOSO_state_holdout_{holdout_id}_pctIncMSE.png"),
                    top_n=20, n_repeats=5
                )
                plot_mdi_importance(
                    model,
                    out_png=os.path.join(target_dir, f"imp_LOSO_state_holdout_{holdout_id}_MDI.png"),
                    top_n=30
                )
                plot_error_vs_trees(
                    prep=model.named_steps["prep"],
                    X_train=X.iloc[tr], y_train=y.iloc[tr],
                    X_test=X.iloc[te], y_test=y.iloc[te],
                    postprocess=postprocess,
                    out_png=os.path.join(target_dir,
                                         f"curve_LOSO_state_holdout_{holdout_id}_error_vs_trees.png"),
                    n_estimators_max=500, step=25
                )
                plot_effect_mtry_auto(
                    prep=model.named_steps["prep"],
                    X_train=X.iloc[tr], y_train=y.iloc[tr],
                    X_test=X.iloc[te], y_test=y.iloc[te],
                    postprocess=postprocess,
                    out_png=os.path.join(target_dir, f"curve_LOSO_state_holdout_{holdout_id}_effect_mtry.png"),
                    n_points=22,
                    frac_min=0.02,
                    frac_max=1.0,
                    n_estimators=300
                )
                end = tm.time()
                times[f"{target_kind}__{feat_spec}__loso_state_diag"] = end - start
                print(f"[{target_kind} | {feat_spec}] LOSO_state diagnostics: {end - start:.1f} sec")

        # ============================================================
        # Random 80/20 CV (metrics + first split importance)
        # ============================================================
        if DO_RANDOM_80_20:
            start = tm.time()
            df_rand, ss = run_random_cv("Random_80_20", X, y, postprocess, n_splits=5, test_size=0.2)
            df_rand.to_csv(os.path.join(target_dir, "folds_Random_80_20.csv"), index=False)

            pooled = pooled_from_folds(df_rand)
            summary_rows_spec.append({
                "target": target_kind,
                "feat_spec": feat_spec,
                "spec": "Random_80_20_pooled",
                **pooled
            })

            tr, te = next(ss.split(X, y))
            prep = make_preprocess(X.iloc[tr])
            model = make_model(prep, max_features=MAX_FEATURES_DEFAULT)
            model.fit(X.iloc[tr], y.iloc[tr])

            end = tm.time()
            times[f"{target_kind}__{feat_spec}__random_80_20"] = end - start
            print(f"[{target_kind} | {feat_spec}] Random_80_20: {end - start:.1f} sec")

            if  RUN_DIAG:
                start = tm.time()
                plot_varimp_percent_inc_mse(
                    model, X.iloc[te], y.iloc[te],
                    postprocess=postprocess,
                    out_png=os.path.join(target_dir, "imp_Random_80_20_pctIncMSE.png"),
                    top_n=20, n_repeats=5
                )
                plot_mdi_importance(
                    model,
                    out_png=os.path.join(target_dir, "imp_Random_80_20_MDI.png"),
                    top_n=30
                )
                plot_error_vs_trees(
                    prep=model.named_steps["prep"],
                    X_train=X.iloc[tr], y_train=y.iloc[tr],
                    X_test=X.iloc[te], y_test=y.iloc[te],
                    postprocess=postprocess,
                    out_png=os.path.join(target_dir, "curve_Random_80_20_error_vs_trees.png"),
                    n_estimators_max=500, step=25
                )
                plot_effect_mtry_auto(
                    prep=model.named_steps["prep"],
                    X_train=X.iloc[tr], y_train=y.iloc[tr],
                    X_test=X.iloc[te], y_test=y.iloc[te],
                    postprocess=postprocess,
                    out_png=os.path.join(target_dir, "curve_Random_80_20_effect_mtry.png"),
                    n_points=22,
                    frac_min=0.02,
                    frac_max=1.0,
                    n_estimators=300
                )
                end = tm.time()
                times[f"{target_kind}__{feat_spec}__random_80_20_diag"] = end - start
                print(f"[{target_kind} | {feat_spec}] Random_80_20 diagnostics: {end - start:.1f} sec")

        # ============================================================
        # WRITE / UPSERT SUMMARY FOR THIS SPEC (MODULAR RUNS)
        # ============================================================
        times[f"{target_kind}__{feat_spec}__total"] = tm.time() - start_0
        times_summary = pd.DataFrame([
            {"step": k, "time_sec": v, "time_min": v / 60.0, "time_hr": v / 3600.0}
            for k, v in times.items()
        ])
        print(times_summary)

        new_rows = pd.DataFrame(summary_rows_spec)

        # If nothing ran, skip upsert
        if len(new_rows) == 0:
            print(f"[{target_kind} | {feat_spec}] Nothing to write (all DO_* toggles off).")
            continue

        # This function must exist in your script (you said you added it)
        summary_all = upsert_summary_csv(
            summary_path,
            new_rows,
            key_cols=("target", "feat_spec", "spec")
        )

        print("\nUpdated summary:", summary_path)
        print(summary_all.sort_values(["target", "feat_spec", "spec"]).to_string(index=False))

    if os.path.exists(summary_path):
        final = pd.read_csv(summary_path)
        print("\nFINAL summary:", summary_path)
        print(final.sort_values(["target", "feat_spec", "spec"]).to_string(index=False))


if __name__ == "__main__":
    main()
