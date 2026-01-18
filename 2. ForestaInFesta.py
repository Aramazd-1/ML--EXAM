import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# =========================
# 1) LOAD DATA
# =========================
DATA_PATH = "nfip_claims_FL_2020.csv"   # <- swap to multi-state file when ready
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

    # Time (feature or splitting)
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

# Keep only rows where denominator is positive and target is observed in a meaningful way
valid = repl_total > 0
df = df.loc[valid].copy()
damage_total = damage_total.loc[valid]
repl_total = repl_total.loc[valid]

y = (damage_total / repl_total).clip(lower=0, upper=1)  # ratio in [0,1]

# =========================
# 4) FEATURES: DROP LEAKAGE / POST-OUTCOME PAYMENT VARS
# =========================
# Anything that is an outcome/payment realization should not be in X.
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

# Also drop the components used to *construct the target* only if you consider them "label-side"
# Damage components MUST be dropped (they are literally the numerator of y).
target_components_to_drop = [
    "buildingDamageAmount",
    "contentsDamageAmount",
]

# Identifiers you don't want as predictors
id_cols = ["id"]

drop_cols = [c for c in (leakage_cols + target_components_to_drop + id_cols) if c in df.columns]

X = df.drop(columns=drop_cols)

# Drop columns that are entirely missing
all_missing = [c for c in X.columns if X[c].isna().all()]
if all_missing:
    X = X.drop(columns=all_missing)

# =========================
# 5) SPLIT DESIGN + REPEATED EVALUATION
# =========================
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

event_candidates = ["eventDesignationNumber", "femaDisasterNumber", "disasterNumber", "eventId"]
state_candidates = ["state", "stateCode", "reportedState", "propertyState"]
year_col = "yearOfLoss" if "yearOfLoss" in df.columns else None

# ---- choose grouping column (event > state) ----
group_col = None
group_kind = None
groups = None

for c in event_candidates:
    if c in df.columns:
        g = df.loc[X.index, c].astype("string").fillna("MISSING")
        if g.nunique() >= 5:
            group_col = c
            group_kind = "event"
            groups = g
            break

if group_col is None:
    for c in state_candidates:
        if c in df.columns:
            g = df.loc[X.index, c].astype("string").fillna("MISSING")
            if g.nunique() >= 5:
                group_col = c
                group_kind = "state"
                groups = g
                break

use_time_split = (group_col is None)

# IMPORTANT: remove grouping col from X if present (avoid memorization)
if group_col is not None and group_col in X.columns:
    X = X.drop(columns=[group_col])

# =========================
# 6) PREPROCESS + MODEL (define ONCE, fit many times)
# =========================
# Numeric columns = those in num_like that survived AND are in X
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
    n_jobs=-1,
    random_state=42
)

model = Pipeline([("prep", preprocess), ("rf", rf)])

# =========================
# 7) EVALUATION
# =========================
def metrics_ratio(y_true, y_pred):
    y_pred = np.clip(y_pred, 0, 1)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return rmse, mae, r2

def group_weighted_rmse(y_true, y_pred, g):
    # each group has weight 1 (average of per-group RMSE)
    y_true = np.asarray(y_true)
    y_pred = np.clip(np.asarray(y_pred), 0, 1)
    g = pd.Series(g).astype("string")
    per = []
    for key in g.unique():
        m = (g == key).to_numpy()
        per.append(np.sqrt(np.mean((y_true[m] - y_pred[m]) ** 2)))
    return float(np.mean(per))

if not use_time_split:
    # --- repeated grouped holdouts ---
    R = 50
    test_size = 0.20
    base_seed = 42

    rmse_claim, mae_claim, r2_claim = [], [], []
    rmse_group = []

    for r in range(R):
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=base_seed + r)
        tr, te = next(gss.split(X, y, groups=groups))

        model.fit(X.iloc[tr], y.iloc[tr])
        pred = model.predict(X.iloc[te])

        rmse, mae, r2 = metrics_ratio(y.iloc[te], pred)
        rmse_claim.append(rmse)
        mae_claim.append(mae)
        r2_claim.append(r2)

        rmse_group.append(group_weighted_rmse(y.iloc[te].to_numpy(), pred, groups.iloc[te]))

    rmse_claim = np.array(rmse_claim)
    mae_claim = np.array(mae_claim)
    r2_claim = np.array(r2_claim)
    rmse_group = np.array(rmse_group)

    print(f"Split type: REPEATED GROUPED holdout by {group_kind} ({group_col}) | #groups={groups.nunique()}")
    print(f"Repetitions: R={R}, test_size={test_size}")

    print("\n=== Claim-weighted performance (each claim weight=1) ===")
    print(f"RMSE: mean={rmse_claim.mean():.4f}  sd={rmse_claim.std(ddof=1):.4f}  p10/p50/p90={np.quantile(rmse_claim,[.1,.5,.9])}")
    print(f"MAE : mean={mae_claim.mean():.4f}  sd={mae_claim.std(ddof=1):.4f}  p10/p50/p90={np.quantile(mae_claim,[.1,.5,.9])}")
    print(f"R^2 : mean={r2_claim.mean():.4f}  sd={r2_claim.std(ddof=1):.4f}  p10/p50/p90={np.quantile(r2_claim,[.1,.5,.9])}")

    print(f"\n=== {group_kind}-weighted RMSE (each {group_kind} weight=1) ===")
    print(f"RMSE: mean={rmse_group.mean():.4f}  sd={rmse_group.std(ddof=1):.4f}  p10/p50/p90={np.quantile(rmse_group,[.1,.5,.9])}")

else:
    # --- time split evaluation (rolling-ish) ---
    if year_col is None:
        raise ValueError("No feasible event/state grouping and no yearOfLoss for a time split.")

    years = df.loc[X.index, year_col].astype(int)
    uniq_years = np.sort(years.unique())
    if len(uniq_years) < 3:
        raise ValueError("Not enough years for meaningful time evaluation (need >= 3).")

    # Evaluate on last ~20% of years one-by-one (train on all prior years)
    start_test_pos = max(1, int(np.floor(0.8 * len(uniq_years))))
    test_years = uniq_years[start_test_pos:]

    rmse_list, mae_list, r2_list = [], [], []

    for ty in test_years:
        tr = np.where(years < ty)[0]
        te = np.where(years == ty)[0]
        if len(te) == 0 or len(tr) == 0:
            continue

        model.fit(X.iloc[tr], y.iloc[tr])
        pred = model.predict(X.iloc[te])
        rmse, mae, r2 = metrics_ratio(y.iloc[te], pred)

        rmse_list.append(rmse)
        mae_list.append(mae)
        r2_list.append(r2)

    rmse_list = np.array(rmse_list)
    mae_list = np.array(mae_list)
    r2_list = np.array(r2_list)

    print(f"Split type: TIME (train on past, test on year) by {year_col}")
    print(f"Test years: {list(test_years)}")

    print("\n=== Year-by-year performance (damage ratio) ===")
    print(f"RMSE: mean={rmse_list.mean():.4f}  sd={rmse_list.std(ddof=1):.4f}")
    print(f"MAE : mean={mae_list.mean():.4f}  sd={mae_list.std(ddof=1):.4f}")
    print(f"R^2 : mean={r2_list.mean():.4f}  sd={r2_list.std(ddof=1):.4f}")

