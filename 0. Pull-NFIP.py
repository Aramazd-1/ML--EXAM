import time
import requests
import pandas as pd

BASE = "https://www.fema.gov/api/open/v2/FimaNfipClaims"

# NFIP fields
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

DROP = {
    "asOfDate", "dateOfLoss", "yearOfLoss"
}

FIELDS_MINUS = [f for f in FIELDS if f not in DROP]

def build_filter(year: int,
                 states=None,
                 county_codes=None,
                 flooded_only: bool = False) -> str:
    """
    OpenFEMA uses OData-style filters. See FEMA API docs. :contentReference[oaicite:2]{index=2}
    """
    parts = [f"(yearOfLoss eq {int(year)})"]

    if states:
        s = " or ".join([f"(state eq '{st}')" for st in states])
        parts.append(f"({s})")

    # countyCode is often string-like; quote it to be safe
    if county_codes:
        c = " or ".join([f"(countyCode eq '{cc}')" for cc in county_codes])
        parts.append(f"({c})")

    if flooded_only:
        # “Flooded/intensity observed” proxy: any of these present/positive
        parts.append("((waterDepth ne null) or (floodEvent ne null) or "
                     "(buildingDamageAmount gt 0) or (contentsDamageAmount gt 0))")

    return " and ".join(parts)

def fetch_nfip_claims(year: int,
                      states=None,
                      county_codes=None,
                      flooded_only: bool = False,
                      page_size: int = 1000,
                      max_records: int = 50_000,
                      sleep_s: float = 0.1) -> pd.DataFrame:
    flt = build_filter(year, states=states, county_codes=county_codes, flooded_only=flooded_only)

    rows = []
    skip = 0

    while skip < max_records:
        params = {
            "$top": min(page_size, max_records - skip),
            "$skip": skip,
            "$select": ",".join(FIELDS),
            "$filter": flt,
        }
        r = requests.get(BASE, params=params, timeout=120)
        r.raise_for_status()
        js = r.json()

        batch = js.get("FimaNfipClaims", [])
        if not batch:
            break

        rows.extend(batch)
        skip += len(batch)

        if len(batch) < params["$top"]:
            break

        time.sleep(sleep_s)

    df = pd.DataFrame(rows)

    # Simple numeric casting for common numeric columns (optional but useful)
    num_cols = [
        "amountPaidOnBuildingClaim","amountPaidOnContentsClaim",
        "netBuildingPaymentAmount","netContentsPaymentAmount","netIccPaymentAmount",
        "buildingDamageAmount","contentsDamageAmount",
        "waterDepth","latitude","longitude",
        "buildingPropertyValue","buildingReplacementCost",
        "totalBuildingInsuranceCoverage","totalContentsInsuranceCoverage"
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Convenient totals
    if {"netBuildingPaymentAmount","netContentsPaymentAmount","netIccPaymentAmount"}.issubset(df.columns):
        df["paid_total_net"] = (
            df["netBuildingPaymentAmount"].fillna(0)
            + df["netContentsPaymentAmount"].fillna(0)
            + df["netIccPaymentAmount"].fillna(0)
        )

    if {"buildingDamageAmount","contentsDamageAmount"}.issubset(df.columns):
        df["damage_total"] = df["buildingDamageAmount"].fillna(0) + df["contentsDamageAmount"].fillna(0)

    return df

if __name__ == "__main__":
    # Example: one-year cross-section, Florida only, keep records with some flood/intensity signal
    df = fetch_nfip_claims(
        year=2020,
        states=[
            "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL", "GA", "HI", "ID", "IL", "IN", "IA",
            "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM",
            "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA",
            "WV", "WI", "WY"
        ],
    flooded_only=True,
        max_records=250_000
    )
    print(df.shape)
    print(df.head(3))
    df.to_csv("nfip_claims_FL_2020.csv", index=False)
    print("Saved nfip_claims_FL_2020.csv")
