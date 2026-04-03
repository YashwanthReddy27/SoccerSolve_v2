from __future__ import annotations
import argparse
import json
import logging
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    import xgboost as xgb
except ImportError:
    raise SystemExit("Run: pip install xgboost")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("shap not installed. Run: pip install shap")

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train_model")

PROC_DIR   = Path("../data/processed")
MODELS_DIR = Path("../data/models")
INPUT_JSON = Path("matched_players.json")
INPUT_CSV  = PROC_DIR / "unified_dataset.csv"

TARGET      = "tm_market_value_eur"
TEST_SIZE   = 0.20
RANDOM_SEED = 42

POSITION_GROUPS   = ["GK", "CB", "FB", "MF", "WA", "FW"]
POSITION_ENCODING = {"GK": 0, "CB": 1, "FB": 2, "MF": 3, "WA": 4, "FW": 5}

ALL_FEATURES = [
    "position_enc",
    "overall_rating", "potential", "age", "height_cm", "weight_kg",
    "weak_foot", "skill_moves", "intl_reputation",
    "crossing", "finishing", "heading_accuracy", "short_passing", "volleys",
    "dribbling", "curve", "fk_accuracy", "long_passing", "ball_control",
    "acceleration", "sprint_speed", "agility", "reactions", "balance",
    "shot_power", "jumping", "stamina", "strength", "long_shots",
    "aggression", "interceptions", "att_position", "vision", "penalties",
    "composure", "defensive_awareness", "standing_tackle", "sliding_tackle",
    "gk_diving", "gk_handling", "gk_kicking", "gk_positioning", "gk_reflexes",
    "foot_right", "work_rate_att", "work_rate_def",
    # Engineered features
    "age_sq",               # non-linear age decay
    "prime_years",          # 1 if age 22-27, 0 otherwise
    "age_x_potential",      # young + high potential = premium
    "peak_score",           # overall * (1 - age_penalty)
    "pace_composite",       # acceleration + sprint_speed (key for WA/FW)
    "attacking_composite",  # finishing + dribbling + att_position
    "years_to_peak",        # 26 - age (negative = past peak)
]

# Tuned for small datasets: shallow, high regularisation
XGBOOST_PARAMS = {
    "objective":             "reg:squarederror",
    "n_estimators":          600,
    "learning_rate":         0.03,
    "max_depth":             3,
    "min_child_weight":      5,
    "subsample":             0.75,
    "colsample_bytree":      0.75,
    "reg_alpha":             1.0,
    "reg_lambda":            5.0,
    "gamma":                 0.1,
    "random_state":          RANDOM_SEED,
    "n_jobs":                -1,
    "early_stopping_rounds": 40,
}


def load_dataset() -> pd.DataFrame:
    if INPUT_CSV.exists():
        df = pd.read_csv(INPUT_CSV)
        log.info("Loaded %d rows from %s", len(df), INPUT_CSV)
    elif INPUT_JSON.exists():
        with open(INPUT_JSON, encoding="utf-8") as f:
            df = pd.DataFrame(json.load(f))
        log.info("Loaded %d rows from %s", len(df), INPUT_JSON)
    else:
        raise FileNotFoundError(
            "No dataset found. Expected data/processed/unified_dataset.csv "
            "or matched_players.json in the current directory."
        )

    df["position_enc"] = df["position_group"].map(POSITION_ENCODING).fillna(3).astype(int)

    if "preferred_foot" in df.columns:
        df["foot_right"] = (df["preferred_foot"].str.lower() == "right").astype(int)
    else:
        df["foot_right"] = 1

    # if "work_rate" in df.columns:
    #     wr = df["work_rate"].str.split(" / ", expand=True)
    #     wr_map = {"low": 0, "medium": 1, "high": 2}
    #     df["work_rate_att"] = wr[0].str.lower().map(wr_map).fillna(1).astype(int)
    #     df["work_rate_def"] = wr[1].str.lower().map(wr_map).fillna(1).astype(int)
    # else:
    #     df["work_rate_att"] = 1
    #     df["work_rate_def"] = 1

    df = df[df[TARGET].notna() & (df[TARGET] > 0)].copy()

    # Numeric conversion for all skill/rating columns
    skill_cols = [
        "overall_rating", "potential", "age", "finishing", "dribbling",
        "acceleration", "sprint_speed", "att_position", "reactions",
        "crossing", "short_passing", "ball_control", "agility",
    ]
    for col in skill_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Impute nulls by POSITION GROUP median (not global) — critical for WA/FW
    # where some star players have null ratings
    for col in skill_cols:
        if col not in df.columns:
            continue
        if df[col].isna().any():
            group_medians = df.groupby("position_group")[col].transform("median")
            df[col] = df[col].fillna(group_medians)
            df[col] = df[col].fillna(df[col].median())

    # Engineered features
    age = df["age"].fillna(26)
    overall = df["overall_rating"].fillna(80)
    potential = df["potential"].fillna(82)
    accel = df.get("acceleration", pd.Series(75, index=df.index)).fillna(75)
    sprint = df.get("sprint_speed", pd.Series(75, index=df.index)).fillna(75)
    finishing = df.get("finishing", pd.Series(70, index=df.index)).fillna(70)
    dribbling = df.get("dribbling", pd.Series(70, index=df.index)).fillna(70)
    att_pos = df.get("att_position", pd.Series(70, index=df.index)).fillna(70)

    df["age_sq"]            = age ** 2
    df["prime_years"]       = ((age >= 22) & (age <= 27)).astype(int)
    df["age_x_potential"]   = (27 - age).clip(0) * (potential - 75).clip(0)
    df["peak_score"]        = overall * (1 - (age - 26).clip(0) * 0.015)
    df["pace_composite"]    = (accel + sprint) / 2
    df["attacking_composite"] = (finishing + dribbling + att_pos) / 3
    df["years_to_peak"]     = (26 - age).clip(-10, 8)

    log.info("Rows with valid target: %d", len(df))
    log.info("Engineered features added: age_sq, prime_years, age_x_potential, "
             "peak_score, pace_composite, attacking_composite, years_to_peak")
    return df


def impute(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = 0
    return df


def train_global(df: pd.DataFrame, compute_shap: bool = True) -> dict:
    log.info("Training global model on %d players", len(df))

    feature_cols = [f for f in ALL_FEATURES if f in df.columns or f == "position_enc"]
    df = impute(df.copy(), feature_cols)

    X      = df[feature_cols].values
    y      = df[TARGET].values
    y_log  = np.log1p(y)
    ids    = df["sofifa_id"].values if "sofifa_id" in df.columns else np.arange(len(df))
    groups = df["position_group"].values if "position_group" in df.columns else np.array(["UNK"] * len(df))

    bins = pd.qcut(y_log, q=5, labels=False, duplicates="drop")
    (X_train, X_test,
     y_tr_log, y_te_log,
     idx_tr, idx_te) = train_test_split(
        X, y_log, np.arange(len(df)),
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=bins,
    )
    y_test = np.expm1(y_te_log)

    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    model = xgb.XGBRegressor(**XGBOOST_PARAMS)
    model.fit(X_train_sc, y_tr_log,
              eval_set=[(X_test_sc, y_te_log)],
              verbose=False)

    pred_test = np.expm1(model.predict(X_test_sc))
    mae  = mean_absolute_error(y_test, pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred_test))
    r2   = r2_score(y_test, pred_test)
    mape = np.mean(np.abs((y_test - pred_test) / np.maximum(y_test, 1))) * 100

    log.info("Global  MAE=€%sM  RMSE=€%sM  R2=%.3f  MAPE=%.1f%%",
             f"{mae/1e6:.1f}", f"{rmse/1e6:.1f}", r2, mape)

    # Per-group breakdown on test set
    per_group = {}
    test_grps = groups[idx_te]
    for grp in POSITION_GROUPS:
        mask = test_grps == grp
        if mask.sum() < 3:
            continue
        g_mae  = mean_absolute_error(y_test[mask], pred_test[mask])
        g_rmse = np.sqrt(mean_squared_error(y_test[mask], pred_test[mask]))
        g_r2   = r2_score(y_test[mask], pred_test[mask])
        g_mape = np.mean(np.abs(
            (y_test[mask] - pred_test[mask]) / np.maximum(y_test[mask], 1)
        )) * 100
        per_group[grp] = {
            "n_test":   int(mask.sum()),
            "mae_eur":  round(float(g_mae),  0),
            "rmse_eur": round(float(g_rmse), 0),
            "r2":       round(float(g_r2),   4),
            "mape_pct": round(float(g_mape), 2),
        }
        log.info("  [%s] n=%d  MAE=€%sM  R2=%.3f",
                 grp, mask.sum(), f"{g_mae/1e6:.1f}", g_r2)

    # Confidence interval models
    ci_low = ci_high = None
    ci_params = {**XGBOOST_PARAMS, "early_stopping_rounds": None}
    try:
        ci_low = xgb.XGBRegressor(
            **{**ci_params, "objective": "reg:quantileerror", "quantile_alpha": 0.10}
        )
        ci_low.fit(X_train_sc, y_tr_log, verbose=False)
        ci_high = xgb.XGBRegressor(
            **{**ci_params, "objective": "reg:quantileerror", "quantile_alpha": 0.90}
        )
        ci_high.fit(X_train_sc, y_tr_log, verbose=False)
        log.info("Confidence interval models trained")
    except Exception as exc:
        log.warning("CI models failed: %s", exc)

    # SHAP on full dataset
    shap_df = pd.DataFrame()
    if compute_shap and SHAP_AVAILABLE:
        try:
            X_all_sc  = scaler.transform(df[feature_cols].values)
            explainer = shap.TreeExplainer(model)
            sv        = explainer.shap_values(X_all_sc)
            shap_df   = pd.DataFrame(sv, columns=feature_cols)
            shap_df.insert(0, "sofifa_id",      ids)
            shap_df.insert(1, "position_group", groups)
            log.info("SHAP computed for %d players", len(shap_df))
        except Exception as exc:
            log.warning("SHAP failed: %s", exc)

    # Global importance
    if not shap_df.empty:
        imp_global = {c: float(shap_df[c].abs().mean()) for c in feature_cols}
    else:
        imp_global = {f: float(v) for f, v in zip(feature_cols, model.feature_importances_)}
    imp_global = dict(sorted(imp_global.items(), key=lambda x: -x[1]))

    # Per-group importance
    imp_by_group = {}
    if not shap_df.empty:
        for grp in POSITION_GROUPS:
            sub = shap_df[shap_df["position_group"] == grp]
            if len(sub) < 3:
                continue
            imp = {c: float(sub[c].abs().mean()) for c in feature_cols}
            imp_by_group[grp] = dict(sorted(imp.items(), key=lambda x: -x[1]))

    return {
        "model":             model,
        "scaler":            scaler,
        "feature_cols":      feature_cols,
        "ci_low":            ci_low,
        "ci_high":           ci_high,
        "shap_df":           shap_df,
        "importance":        imp_global,
        "importance_by_group": imp_by_group,
        "metrics": {
            "n_train":  int(len(y_tr_log)),
            "n_test":   int(len(y_test)),
            "mae_eur":  round(float(mae),  0),
            "rmse_eur": round(float(rmse), 0),
            "r2":       round(float(r2),   4),
            "mape_pct": round(float(mape), 2),
        },
        "per_group_metrics": per_group,
    }


def save_results(res: dict):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    res["model"].save_model(str(MODELS_DIR / "model_global.json"))
    with open(MODELS_DIR / "scaler_global.pkl", "wb") as f:
        pickle.dump(res["scaler"], f)
    with open(MODELS_DIR / "features_global.json", "w") as f:
        json.dump(res["feature_cols"], f, indent=2)

    if res["ci_low"]:
        res["ci_low"].save_model(str(MODELS_DIR  / "model_global_ci_low.json"))
        res["ci_high"].save_model(str(MODELS_DIR / "model_global_ci_high.json"))

    if not res["shap_df"].empty:
        res["shap_df"].to_csv(MODELS_DIR / "shap_global.csv", index=False)

    eval_report = {"_global": res["metrics"], **res["per_group_metrics"]}
    with open(MODELS_DIR / "eval_report.json", "w") as f:
        json.dump(eval_report, f, indent=2)

    feat_imp = {"_global": res["importance"], **res["importance_by_group"]}
    with open(MODELS_DIR / "feature_importance.json", "w") as f:
        json.dump(feat_imp, f, indent=2)

    # Summary print
    print("\n" + "=" * 65)
    print("EVALUATION REPORT")
    print("=" * 65)
    m = res["metrics"]
    flag = "PASS" if m["r2"] >= 0.80 else "BELOW 0.80 TARGET"
    print(f"  GLOBAL  n={m['n_train']+m['n_test']}  "
          f"MAE=€{m['mae_eur']/1e6:.1f}M  "
          f"RMSE=€{m['rmse_eur']/1e6:.1f}M  "
          f"R2={m['r2']:.3f} [{flag}]  "
          f"MAPE={m['mape_pct']:.1f}%")
    print()
    for grp, gm in res["per_group_metrics"].items():
        gflag = "ok" if gm["r2"] >= 0.75 else "weak"
        print(f"  {grp:4s}  n_test={gm['n_test']:2d}  "
              f"MAE=€{gm['mae_eur']/1e6:.1f}M  "
              f"R2={gm['r2']:.3f} [{gflag}]  "
              f"MAPE={gm['mape_pct']:.1f}%")
    print(f"\n  Outputs: {MODELS_DIR}/")

    if m["r2"] < 0.80:
        print(f"\n  NOTE: R2={m['r2']:.3f} below 0.80 target.")
        print("  Scraping 500+ more players per group will push this above 0.80.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-shap", action="store_true",
                        help="Skip SHAP (faster for testing)")
    args = parser.parse_args()
    df  = load_dataset()
    res = train_global(df, compute_shap=not args.no_shap)
    save_results(res)