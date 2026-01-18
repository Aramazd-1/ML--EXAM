import numpy as np
import pandas as pd

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# =========================
# 1) LOAD DATA
# =========================
DATA_PATH = "nfip_claims_FL_2020.csv"   # <- swap to multi-state/multi-year file when ready
df = pd.read_csv(DATA_PATH, low_memory=False)

# =========================
# 2) NUMERIC CASTS (only for columns we may use)
# =========================
num_like = [
    # Damage + replacement costs (for target construction)
    "buildingDamageAmount", "contentsDamageAmount",
    "buildingReplacementCost", "contentsReplacementCost",

    # Exposure/value proxies (features)
    "buildingPropertyValue", "contentsPropertyValue",
    "totalBuildingInsuranceCoverage", "totalContentsInsuranceCoverage",

    # Hazard / intensity proxies (features)
    "waterDepth", "floodWaterDuration",
    "baseFloodElevation", "elevationDifference", "lowestAdjacentGrade", "lowestFloorElevation",

    # Location / structure (features)
    "latitude", "longitude",
    "numberOfFloorsInTheInsuredBuilding", "numberOfUnits", "policyCount",

    # Time
    "yearOfLoss"
]
for c in num_like:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# =========================
# 3) TARGET: DAMAGE RATIO (building + contents)
# =========================
needed = {
    "buildingDamageAmount", "contentsDamageAmount",
    "buildingReplacementCost", "contentsReplacementCost"
}
missing = [c for c in needed if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns for damage ratio: {missing}")

damage_total = df["buildingDamageAmount"].fillna(0) + df["contentsDamageAmount"].fillna(0)
repl_total = df["buildingReplacementCost"].fillna(0) + df["contentsReplacementCost"].fillna(0)

valid = repl_total > 0
df = df.loc[valid].copy()
damage_total = damage_total.loc[valid]
repl_total = repl_total.loc[valid]

y = (damage_total / repl_total).clip(lower=0, upper=1)

# =========================
# 4) FEATURES: DROP LEAKAGE / POST-OUTCOME VARS + TARGET-CONSTRUCTION VARS
# =========================
leakage_cols = [
    # payments / payouts (post-claim)
    "amountPaidOnBuildingClaim",
    "amountPaidOnContentsClaim",
    "amountPaidOnIncreasedCostOfComplianceClaim",
    "netBuildingPaymentAmount",
    "netContentsPaymentAmount",
    "netIccPaymentAmount",
    "paid_total_net",
]

# Drop ALL 4 variables used to construct y (so X doesn't contain numerator or denominator components)
target_components_to_drop = [
    "buildingDamageAmount",
    "contentsDamageAmount",
    "buildingReplacementCost",
    "contentsReplacementCost",
]

id_cols = ["id"]

drop_cols = [c for c in (leakage_cols + target_components_to_drop + id_cols) if c in df.columns]
X = df.drop(columns=drop_cols)

# Drop columns entirely missing
all_missing = [c for c in X.columns if X[c].isna().all()]
if all_missing:
    X = X.drop(columns=all_missing)

# =========================
# 5) CHOOSE GROUPING (event > state) else time split
# =========================
event_candidates = ["eventDesignationNumber", "femaDisasterNumber", "disasterNumber", "eventId"]
state_candidates = ["state", "stateCode", "reportedState", "propertyState"]
year_col = "yearOfLoss" if "yearOfLoss" in df.columns else None

group_col = None
group_kind = None
groups = None

for c in event_candidates:
    if c in df.columns:
        g = df.loc[X.index, c].astype("string").fillna("MISSING")
        if g.nunique() >= 2:
            group_col, group_kind, groups = c, "event", g
            break

if group_col is None:
    for c in state_candidates:
        if c in df.columns:
            g = df.loc[X.index, c].astype("string").fillna("MISSING")
            if g.nunique() >= 2:
                group_col, group_kind, groups = c, "state", g
                break

use_time_split = (group_col is None)

# IMPORTANT: remove grouping col from X if present (avoid memorization)
if group_col is not None and group_col in X.columns:
    X = X.drop(columns=[group_col])

# =========================
# 6) PREPROCESS + MODEL
# =========================
num_cols = [c for c in num_like if c in X.columns]
cat_cols = [c for c in X.columns if c not in num_cols]

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
])

categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("to_str", FunctionTransformer(lambda a: a.astype(str), feature_names_out="one-to-one")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocess = ColumnTransformer([
    ("num", numeric_pipe, num_cols),
    ("cat", categorical_pipe, cat_cols),
])

rf = RandomForestRegressor(
    n_estimators=500,
    min_samples_leaf=5,
    max_features="sqrt",   # << makes it "true RF" rather than near-bagging in many sklearn setups
    n_jobs=-1,
    random_state=42
)

model = Pipeline([("prep", preprocess), ("rf", rf)])

# =========================
# 7) EVALUATION
# =========================
def clip01(a):
    return np.clip(a, 0, 1)

def metrics_ratio(y_true, y_pred):
    y_pred = clip01(y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return rmse, mae, r2

if not use_time_split:
    logo = LeaveOneGroupOut()
    n_groups = groups.nunique()

    print(f"Split type: Leave-One-{group_kind}-Out | group_col={group_col} | #groups={n_groups}")

    fold_rows = []
    y_true_all = []
    y_pred_all = []

    for fold, (tr, te) in enumerate(logo.split(X, y, groups=groups), start=1):
        held_out = str(pd.Series(groups.iloc[te]).iloc[0])
        model.fit(X.iloc[tr], y.iloc[tr])

        pred = clip01(model.predict(X.iloc[te]))
        y_te = y.iloc[te].to_numpy()

        rmse, mae, r2 = metrics_ratio(y_te, pred)
        fold_rows.append({
            group_kind: held_out,
            "n_test_claims": len(te),
            "rmse": rmse,
            "mae": mae,
            "r2": r2
        })

        y_true_all.append(y_te)
        y_pred_all.append(pred)

    fold_df = pd.DataFrame(fold_rows)

    # Macro (each event weight=1) — the clean “generalize across events” summary
    rmse_macro = fold_df["rmse"].mean()
    mae_macro = fold_df["mae"].mean()
    r2_macro = fold_df["r2"].mean()

    # Pooled (claim-weighted) — dominated by large events
    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)
    rmse_pooled, mae_pooled, r2_pooled = metrics_ratio(y_true_all, y_pred_all)

    print("\n=== Per-group results (sorted by RMSE, worst first) ===")
    print(fold_df.sort_values("rmse", ascending=False).to_string(index=False))

    print("\n=== Summary (damage ratio) ===")
    print("Macro avg (each group weight=1):")
    print(f"  RMSE: {rmse_macro:.4f}   MAE: {mae_macro:.4f}   R^2: {r2_macro:.4f}")
    print("Pooled over all held-out predictions (claim-weighted):")
    print(f"  RMSE: {rmse_pooled:.4f}  MAE: {mae_pooled:.4f}  R^2: {r2_pooled:.4f}")

    print("\nRMSE distribution across held-out groups:")
    print(f"  mean={fold_df['rmse'].mean():.4f}  sd={fold_df['rmse'].std(ddof=1):.4f}  "
          f"p10/p50/p90={np.quantile(fold_df['rmse'], [0.1, 0.5, 0.9])}")

else:
    # --- time split evaluation (rolling-ish) ---
    if year_col is None:
        raise ValueError("No feasible event/state grouping and no yearOfLoss for a time split.")

    years = df.loc[X.index, year_col].astype(int)
    uniq_years = np.sort(years.unique())
    if len(uniq_years) < 3:
        raise ValueError("Not enough years for meaningful time evaluation (need >= 3).")

    start_test_pos = max(1, int(np.floor(0.8 * len(uniq_years))))
    test_years = uniq_years[start_test_pos:]

    rmse_list, mae_list, r2_list = [], [], []

    print(f"Split type: TIME (train on past, test on year) | year_col={year_col}")
    print(f"Test years: {list(test_years)}")

    for ty in test_years:
        tr = np.where(years < ty)[0]
        te = np.where(years == ty)[0]
        if len(te) == 0 or len(tr) == 0:
            continue

        model.fit(X.iloc[tr], y.iloc[tr])
        pred = clip01(model.predict(X.iloc[te]))

        rmse, mae, r2 = metrics_ratio(y.iloc[te].to_numpy(), pred)
        rmse_list.append(rmse)
        mae_list.append(mae)
        r2_list.append(r2)

    rmse_list = np.array(rmse_list)
    mae_list = np.array(mae_list)
    r2_list = np.array(r2_list)

    print("\n=== Year-by-year performance (damage ratio) ===")
    print(f"RMSE: mean={rmse_list.mean():.4f}  sd={rmse_list.std(ddof=1):.4f}")
    print(f"MAE : mean={mae_list.mean():.4f}  sd={mae_list.std(ddof=1):.4f}")
    print(f"R^2 : mean={r2_list.mean():.4f}  sd={r2_list.std(ddof=1):.4f}")
