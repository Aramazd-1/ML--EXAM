import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from pathlib import Path
import textwrap

# ============================================================
# 0) USER CONFIG
# ============================================================
DATA_PATH = "nfip_claims_ALL_STATES_2020.csv"
OUTDIR = Path("Figures/Correlation")
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
    "originalNBDate": "Original new/renewal date (NB)",
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

def pretty(col: str) -> str:
    return LABELS.get(col, col)

def wrap(s: str, width: int = 22) -> str:
    # helps long labels fit in the heatmap
    return "\n".join(textwrap.wrap(s, width=width)) if len(s) > width else s
OUTDIR.mkdir(parents=True, exist_ok=True)
def pretty(col: str) -> str:
    return LABELS.get(col, col)

def wrap(s: str, width: int = 22) -> str:
    # helps long labels fit in the heatmap
    return "\n".join(textwrap.wrap(s, width=width)) if len(s) > width else s
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
FIELDS_MINUS_leak = [f for f in FIELDS_MINUS if f not in LEAKAGE_COLS]

# Cap very-high-cardinality categoricals to keep computations tractable
MAX_LEVELS_PER_CATEGORICAL = 40

# Optional: patterns that often mean "codes/indicators" -> treat as categorical
FORCE_CATEGORICAL_PATTERNS = (
    "Indicator", "Code", "Type", "Zone", "Reason", "Method", "Basis", "occupancy",
    "obstruction", "rateMethod", "floodEvent", "causeOfDamage", "reportedCity",
    "reportedZipCode", "state", "countyCode", "census", "Community", "Number"
)

# ============================================================
# 1) HELPERS: typing + association metrics
# ============================================================
def reduce_levels(s: pd.Series, max_levels: int) -> pd.Series:
    """Keep top-N frequent levels; collapse the rest into '__OTHER__'."""
    s = s.astype("string")
    vc = s.value_counts(dropna=True)
    keep = set(vc.head(max_levels).index)
    return s.where(s.isna() | s.isin(keep), "__OTHER__")

def infer_types(df: pd.DataFrame, cols: list[str]) -> tuple[list[str], list[str]]:
    """
    Infer numeric vs categorical using a simple rule:
    - if name matches FORCE_CATEGORICAL_PATTERNS -> categorical
    - else if >=90% of non-missing parses as numeric -> numeric
    - else categorical
    """
    numeric_cols, cat_cols = [], []
    for c in cols:
        name = str(c)
        if any(p.lower() in name.lower() for p in FORCE_CATEGORICAL_PATTERNS):
            cat_cols.append(c)
            continue
        x = pd.to_numeric(df[c], errors="coerce")
        nonmiss = df[c].notna().sum()
        numeric_share = (x.notna().sum() / nonmiss) if nonmiss > 0 else 0.0
        if numeric_share >= 0.90:
            numeric_cols.append(c)
        else:
            cat_cols.append(c)
    return numeric_cols, cat_cols

def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Bias-corrected Cramér's V (0..1) for categorical-categorical association."""
    x = x.astype("string")
    y = y.astype("string")
    mask = x.notna() & y.notna()
    x = x[mask]
    y = y[mask]
    if len(x) == 0:
        return np.nan

    conf = pd.crosstab(x, y)
    if conf.shape[0] < 2 or conf.shape[1] < 2:
        return 0.0

    chi2 = chi2_contingency(conf, correction=False)[0]
    n = conf.to_numpy().sum()
    if n <= 1:
        return np.nan

    phi2 = chi2 / n
    r, k = conf.shape
    # bias correction
    phi2corr = max(0.0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    denom = min(kcorr - 1, rcorr - 1)
    if denom <= 0:
        return 0.0
    return float(np.sqrt(phi2corr / denom))

def correlation_ratio(categories: pd.Series, measurements: pd.Series) -> float:
    """Correlation ratio eta (0..1) for categorical->numeric association."""
    categories = categories.astype("string")
    measurements = pd.to_numeric(measurements, errors="coerce")
    mask = categories.notna() & measurements.notna()
    categories = categories[mask]
    measurements = measurements[mask]
    if len(measurements) == 0:
        return np.nan

    overall_mean = measurements.mean()
    # group stats
    grp = measurements.groupby(categories)
    n_k = grp.size()
    mean_k = grp.mean()

    numerator = (n_k * (mean_k - overall_mean) ** 2).sum()
    denominator = ((measurements - overall_mean) ** 2).sum()
    if denominator == 0:
        return 0.0
    return float(np.sqrt(numerator / denominator))

# ============================================================
# 2) LOAD + SELECT COLUMNS
# ============================================================
df = pd.read_csv(DATA_PATH, low_memory=False)

cols = [c for c in FIELDS_MINUS if c in df.columns]
df = df[cols].copy()

numeric_cols, cat_cols = infer_types(df, cols)

# standardize categoricals (and cap levels)
for c in cat_cols:
    df[c] = reduce_levels(df[c], MAX_LEVELS_PER_CATEGORICAL)

# cast numeric columns
for c in numeric_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

print("N rows:", len(df))
print("Numeric cols:", len(numeric_cols))
print("Categorical cols:", len(cat_cols))

# ============================================================
# 3) BUILD MIXED ASSOCIATION MATRIX
# ============================================================
all_cols = cols
p = len(all_cols)
A = pd.DataFrame(np.nan, index=all_cols, columns=all_cols, dtype=float)

# diagonal
np.fill_diagonal(A.values, 1.0)

# numeric-numeric Spearman (signed)
if len(numeric_cols) > 0:
    spearman = df[numeric_cols].corr(method="spearman")
    for i in numeric_cols:
        for j in numeric_cols:
            A.loc[i, j] = spearman.loc[i, j]

# categorical-categorical Cramér's V (0..1)
for i, ci in enumerate(cat_cols):
    for cj in cat_cols[i:]:
        v = cramers_v(df[ci], df[cj])
        A.loc[ci, cj] = v
        A.loc[cj, ci] = v

# categorical-numeric eta (0..1)
for c in cat_cols:
    for n in numeric_cols:
        eta = correlation_ratio(df[c], df[n])
        A.loc[c, n] = eta
        A.loc[n, c] = eta

# ============================================================
# 4) VISUALIZATION HELPERS
# ============================================================
def plot_heatmap(mat: pd.DataFrame, title: str, fname: str, clustered: bool = True,
                 vmin=None, vmax=None):
    M = mat.copy()

    # clustering order based on absolute association
    order = list(range(M.shape[0]))
    if clustered and M.shape[0] >= 3:
        X = M.to_numpy()
        X = np.nan_to_num(X, nan=0.0)
        D = 1.0 - np.abs(X)
        np.fill_diagonal(D, 0.0)
        Z = linkage(squareform(D, checks=False), method="average")
        order = leaves_list(Z).tolist()
        M = M.iloc[order, order]

    fig_w = max(10, 0.26 * M.shape[1])
    fig_h = max(8, 0.26 * M.shape[0])
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(M.to_numpy(), aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_title(title)

    ax.set_xticks(np.arange(M.shape[1]))
    ax.set_yticks(np.arange(M.shape[0]))

    xlab = [wrap(pretty(c)) for c in M.columns]
    ylab = [wrap(pretty(r)) for r in M.index]
    ax.set_xticklabels(xlab, rotation=90, fontsize=7)
    ax.set_yticklabels(ylab, fontsize=7)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(fname, dpi=200)
    plt.close(fig)


# Mixed heatmap (keep default scaling or force [-1,1] if you want a "correlation-like" common scale)
plot_heatmap(A, "Mixed association (Spearman / Cramér's V / η) — clustered",
             OUTDIR / "corr_mixed_clustered.png",
             clustered=True, vmin=-1, vmax=1)

plot_heatmap(A, "Mixed association (original order)",
             OUTDIR / "corr_mixed.png",
             clustered=False, vmin=-1, vmax=1)

# Numeric-only: true correlation, always use [-1,1]
if len(numeric_cols) > 0:
    num_mat = A.loc[numeric_cols, numeric_cols]
    plot_heatmap(num_mat, "Numeric-only Spearman correlation — clustered",
                 OUTDIR / "corr_numeric_spearman.png",
                 clustered=True, vmin=-1, vmax=1)

# Categorical-only: association strength in [0,1]
if len(cat_cols) > 0:
    cat_mat = A.loc[cat_cols, cat_cols]
    plot_heatmap(cat_mat, "Categorical-only Cramér's V — clustered",
                 OUTDIR / "assoc_categorical_cramersV.png",
                 clustered=True, vmin=0, vmax=1)

print("Saved figures:")
print(" - assoc_mixed_clustered.png")
print(" - assoc_mixed.png")
print(" - assoc_numeric_spearman.png (if numeric cols)")
print(" - assoc_categorical_cramersV.png (if categorical cols)")

# ============================================================
# 5) TOP ASSOCIATIONS (RANKED)
# ============================================================
pairs = []
for i in range(p):
    for j in range(i + 1, p):
        a = A.iat[i, j]
        if np.isnan(a):
            continue
        ci, cj = A.index[i], A.columns[j]
        # numeric-numeric can be negative -> rank by abs
        score = abs(a) if (ci in numeric_cols and cj in numeric_cols) else a
        pairs.append((score, a, ci, cj))

pairs.sort(reverse=True, key=lambda t: t[0])
top = pd.DataFrame(pairs[:40], columns=["rank_score", "raw_value", "var1", "var2"])
print("\nTop 40 associations (rank_score = abs for num-num, else raw):")
print(top.to_string(index=False))
top.to_csv("top_associations.csv", index=False)
print("\nSaved: top_associations.csv")
