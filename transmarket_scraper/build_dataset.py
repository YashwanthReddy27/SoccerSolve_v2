from __future__ import annotations
import json
import logging
from pathlib import Path

import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("build_dataset")

DATA_DIR    = Path("../data/raw")
PROC_DIR    = Path("../data/processed")
INPUT_FILE  = DATA_DIR / "matched_players.json"
OUTPUT_CSV  = PROC_DIR / "unified_dataset.csv"
REPORT_FILE = PROC_DIR / "dataset_quality_report.json"

# All SoFIFA skill attributes used as features
SKILL_FEATURES = [
    "crossing", "finishing", "heading_accuracy", "short_passing", "volleys",
    "dribbling", "curve", "fk_accuracy", "long_passing", "ball_control",
    "acceleration", "sprint_speed", "agility", "reactions", "balance",
    "shot_power", "jumping", "stamina", "strength", "long_shots",
    "aggression", "interceptions", "att_position", "vision", "penalties",
    "composure", "defensive_awareness", "standing_tackle", "sliding_tackle",
    "gk_diving", "gk_handling", "gk_kicking", "gk_positioning", "gk_reflexes",
]

# Additional numeric features
NUMERIC_FEATURES = [
    "overall_rating", "potential", "age", "height_cm", "weight_kg",
    "weak_foot", "skill_moves", "intl_reputation",
    "weekly_wage_eur", "release_clause_eur",
] + SKILL_FEATURES

CATEGORICAL_FEATURES = ["position_group", "preferred_foot"]

TARGET = "tm_market_value_eur"


def build_dataset() -> pd.DataFrame:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"{INPUT_FILE} not found. Run tm_scraper.py and matcher.py first."
        )

    with open(INPUT_FILE, encoding="utf-8") as f:
        records = json.load(f)
    log.info("Loaded %d matched records", len(records))

    df = pd.DataFrame(records)
    quality = {}

    # 1. Filter: must have a TM market value (our regression target)
    before = len(df)
    df = df[df[TARGET].notna() & (df[TARGET] > 0)].copy()
    quality["dropped_no_tm_value"] = before - len(df)
    quality["kept_with_tm_value"]  = len(df)
    log.info("Kept %d / %d records with valid TM market value", len(df), before)

    # 2. Flag data quality issues
    issues = []

    # Check for suspicious values (>€200M or <€10k are almost certainly wrong)
    suspicious_high = df[df[TARGET] > 200_000_000]
    suspicious_low  = df[df[TARGET] < 10_000]
    if len(suspicious_high):
        log.warning("  %d players with TM value > €200M, verify:", len(suspicious_high))
        for _, row in suspicious_high.iterrows():
            log.warning("    %s (ID %s): €%s", row.get("sofifa_name"), row.get("sofifa_id"),
                        f"{int(row[TARGET]):,}")
        issues.append(f"{len(suspicious_high)} players with value > €200M")
    if len(suspicious_low):
        log.warning("  %d players with TM value < €10k, likely parse error", len(suspicious_low))
        issues.append(f"{len(suspicious_low)} players with value < €10k")

    # Match quality distribution
    if "match_score" in df.columns:
        low_confidence = df[df["match_score"] < 0.75]
        quality["low_confidence_matches"] = len(low_confidence)
        if len(low_confidence):
            log.warning("  %d matches below 0.75 confidence, inspect match_report.json",
                        len(low_confidence))
            issues.append(f"{len(low_confidence)} low-confidence matches")

    # 3. Numeric feature fill rates
    fill_rates = {}
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            rate = df[col].notna().mean()
            fill_rates[col] = round(rate, 3)
            if rate < 0.80:
                log.warning("  Low fill rate: %s = %.1f%%", col, rate * 100)
                issues.append(f"Low fill rate: {col} = {rate*100:.1f}%")
    quality["fill_rates"] = fill_rates

    # 4. Imputation
    # Strategy:
    #   Skill attributes median by position_group (GKs have different stat distributions)
    #   Age, height, weight median overall
    #   Weekly wage, release clause median by position_group (highly position-dependent)
    #   Do NOT impute the target (already filtered above)

    log.info("Imputing missing values…")

    # Per-group medians for skill attributes + economics
    group_medians = df.groupby("position_group")[
        SKILL_FEATURES + ["weekly_wage_eur", "release_clause_eur"]
    ].transform("median")

    for col in SKILL_FEATURES + ["weekly_wage_eur", "release_clause_eur"]:
        if col in df.columns:
            mask = df[col].isna()
            if mask.any():
                df.loc[mask, col] = group_medians.loc[mask, col]
                # Fallback to global median if group median also null
                df[col] = df[col].fillna(df[col].median())

    # Simple overall median for age/physical stats
    for col in ["age", "height_cm", "weight_kg", "overall_rating", "potential",
                "weak_foot", "skill_moves", "intl_reputation"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    quality["issues"] = issues

    # 5. Encode categoricals
    # position_group: GK, CB, FB, MF, WA, FW already a clean string, keep as-is
    # preferred_foot: Left / Right binary 1/0
    if "preferred_foot" in df.columns:
        df["foot_right"] = (df["preferred_foot"].str.lower() == "right").astype(int)

    # work_rate: "High / Medium" two separate int features
    ## IMPPPPPPPPP o work rate for now
    # if "work_rate" in df.columns:
    #     wr = df["work_rate"].str.split(" / ", expand=True)
    #     wr_map = {"low": 0, "medium": 1, "high": 2}
    #     df["work_rate_att"] = wr[0].str.lower().map(wr_map).fillna(1).astype(int)
    #     df["work_rate_def"] = wr[1].str.lower().map(wr_map).fillna(1).astype(int)

    # season_key ordinal (FC24=0, FC25=1, FC26=2)
    season_map = {"FC24": 0, "FC25": 1, "FC26": 2}
    if "season_key" in df.columns:
        df["season_ord"] = df["season_key"].map(season_map).fillna(1).astype(int)

    # 6. Log target distribution
    log.info("Target variable (tm_market_value_eur) distribution:")
    log.info("  Count:  %d", len(df))
    log.info("  Median: €%s", f"{df[TARGET].median():,.0f}")
    log.info("  Mean:   €%s", f"{df[TARGET].mean():,.0f}")
    log.info("  Min:    €%s", f"{df[TARGET].min():,.0f}")
    log.info("  Max:    €%s", f"{df[TARGET].max():,.0f}")
    log.info("  <€1M:   %d players", (df[TARGET] < 1_000_000).sum())
    log.info("  €1-10M: %d players", ((df[TARGET] >= 1_000_000) & (df[TARGET] < 10_000_000)).sum())
    log.info("  >€10M:  %d players", (df[TARGET] >= 10_000_000).sum())

    # Position group distribution
    log.info("Position group distribution:")
    for grp, cnt in df["position_group"].value_counts().items():
        log.info("  %s: %d", grp, cnt)

    # 7. Save
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    quality["final_rows"]    = len(df)
    quality["final_columns"] = len(df.columns)

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(quality, f, indent=2)

    log.info("Saved %d rows x %d cols to %s", len(df), len(df.columns), OUTPUT_CSV)
    log.info("Quality report: %s", REPORT_FILE)
    return df


if __name__ == "__main__":
    df = build_dataset()
    print(f"\nDataset ready: {len(df)} players × {len(df.columns)} features")
    print(f"Output: {OUTPUT_CSV}")
    print(f"Report: {REPORT_FILE}")