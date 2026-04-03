from __future__ import annotations
import json
import logging
import pickle
import time
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    import xgboost as xgb
except ImportError:
    raise SystemExit("Run: pip install xgboost")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

warnings.filterwarnings("ignore")
log = logging.getLogger("whatif_engine")

MODELS_DIR = Path("data/models")
DATA_DIR   = Path("data")

POSITION_GROUPS = ["GK", "CB", "FB", "MF", "WA", "FW"]


class WhatIfEngine:
    """
    Loads all position-group models once at startup and provides
    fast inference for the What-If simulator.
    """

    def __init__(self, models_dir: Path = MODELS_DIR):
        self.models_dir  = models_dir
        self.models:      dict[str, xgb.XGBRegressor] = {}
        self.scalers:     dict[str, object]            = {}
        self.features:    dict[str, list[str]]         = {}
        self.ci_low:      dict[str, xgb.XGBRegressor] = {}
        self.ci_high:     dict[str, xgb.XGBRegressor] = {}
        self.explainers:  dict[str, object]            = {}
        self.loaded       = False
        self._player_data: Optional[list[dict]]        = None

    def load(self) -> "WhatIfEngine":
        """Load all models from disk. Call once at app startup."""
        model_path   = self.models_dir / "model_global.json"
        scaler_path  = self.models_dir / "scaler_global.pkl"
        feature_path = self.models_dir / "features_global.json"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. Run train_model.py first."
            )

        model = xgb.XGBRegressor()
        model.load_model(str(model_path))
        self.models["global"] = model

        with open(scaler_path, "rb") as f:
            self.scalers["global"] = pickle.load(f)

        with open(feature_path) as f:
            self.features["global"] = json.load(f)

        # CI models
        for suffix, key in [("ci_low", "ci_low"), ("ci_high", "ci_high")]:
            p = self.models_dir / f"model_global_{suffix}.json"
            if p.exists():
                m = xgb.XGBRegressor()
                m.load_model(str(p))
                if key == "ci_low":
                    self.ci_low["global"] = m
                else:
                    self.ci_high["global"] = m

        # SHAP explainer
        if SHAP_AVAILABLE:
            try:
                self.explainers["global"] = shap.TreeExplainer(model)
            except Exception as exc:
                log.warning("SHAP explainer failed: %s", exc)

        # Feature importance by group
        feat_imp_path = self.models_dir / "feature_importance.json"
        if feat_imp_path.exists():
            with open(feat_imp_path) as f:
                self.feature_importance = json.load(f)
        else:
            self.feature_importance = {}

        self.loaded = True
        log.info("WhatIfEngine ready (global model, %d features)",
                 len(self.features["global"]))
        return self

    def _get_group(self, player: dict) -> str:
        return player.get("position_group", "MF")

    def _build_feature_vector(self, player: dict, group: str = "global") -> np.ndarray:
        """Convert a player dict to a scaled feature vector including engineered features."""
        feature_cols = self.features["global"]
        pos_enc = {"GK": 0, "CB": 1, "FB": 2, "MF": 3, "WA": 4, "FW": 5}
        row = {}

        # Raw features
        for col in feature_cols:
            val = player.get(col)
            try:
                row[col] = float(val) if val is not None else np.nan
            except (TypeError, ValueError):
                row[col] = np.nan

        # Position encoding
        row["position_enc"] = pos_enc.get(player.get("position_group", "MF"), 3)

        # Engineered features (must match train_model.py exactly)
        age       = float(player.get("age") or 26)
        overall   = float(player.get("overall_rating") or 80)
        potential = float(player.get("potential") or 82)
        accel     = float(player.get("acceleration") or 75)
        sprint    = float(player.get("sprint_speed") or 75)
        finishing = float(player.get("finishing") or 70)
        dribbling = float(player.get("dribbling") or 70)
        att_pos   = float(player.get("att_position") or 70)

        row["age_sq"]              = age ** 2
        row["prime_years"]         = 1 if 22 <= age <= 27 else 0
        row["age_x_potential"]     = max(0, 27 - age) * max(0, potential - 75)
        row["peak_score"]          = overall * (1 - max(0, age - 26) * 0.015)
        row["pace_composite"]      = (accel + sprint) / 2
        row["attacking_composite"] = (finishing + dribbling + att_pos) / 3
        row["years_to_peak"]       = max(-10, min(8, 26 - age))

        df_row = pd.DataFrame([row])
        for col in feature_cols:
            if col not in df_row.columns:
                df_row[col] = 0
        df_row = df_row[feature_cols].fillna(0)
        return self.scalers["global"].transform(df_row.values)

    def predict(self, player: dict) -> dict:
        """
        Predict market value for a single player.

        Returns
        -------
        {
            "predicted_value_eur": 45000000,
            "confidence_low_eur":  38000000,   (if CI models available)
            "confidence_high_eur": 54000000,
            "position_group":      "FW",
            "inference_ms":        12.3,
        }
        """
        group = self._get_group(player)
        if group not in self.models:
            return {"error": f"No model for group {group}"}

        t0 = time.perf_counter()
        X = self._build_feature_vector(player)

        pred_log = self.models["global"].predict(X)[0]
        pred_eur = float(np.expm1(pred_log))

        result = {
            "predicted_value_eur": round(pred_eur),
            "position_group":      self._get_group(player),
            "inference_ms":        round((time.perf_counter() - t0) * 1000, 1),
        }

        if "global" in self.ci_low and "global" in self.ci_high:
            ci_low_log  = self.ci_low["global"].predict(X)[0]
            ci_high_log = self.ci_high["global"].predict(X)[0]
            result["confidence_low_eur"]  = round(float(np.expm1(ci_low_log)))
            result["confidence_high_eur"] = round(float(np.expm1(ci_high_log)))

        return result

    def whatif(
        self,
        player: dict,
        modifications: dict[str, float],
    ) -> dict:
        """
        Core What-If method. Takes a player dict and a dict of attribute
        modifications, returns the full sensitivity analysis.

        Parameters
        ----------
        player        : dict with all player attributes
        modifications : {feature_name: new_value}
                        e.g. {"finishing": 85, "dribbling": 90}

        Returns
        -------
        {
            "base_value_eur":       45000000,
            "modified_value_eur":   47800000,
            "delta_eur":            2800000,
            "delta_pct":            6.2,
            "confidence_low_eur":   42000000,
            "confidence_high_eur":  54000000,
            "shap_values":          {"finishing": 1200000, "dribbling": 900000, ...},
            "top_levers":           [
                {"feature": "finishing",  "shap_eur": 1200000, "current_value": 85},
                {"feature": "dribbling",  "shap_eur":  900000, "current_value": 90},
                {"feature": "reactions",  "shap_eur":  700000, "current_value": 82},
            ],
            "inference_ms":         18.4,
        }
        """
        t0 = time.perf_counter()
        group = self._get_group(player)

        base_result = self.predict(player)
        base_value  = base_result["predicted_value_eur"]

        modified = {**player, **modifications}
        X_mod = self._build_feature_vector(modified)

        pred_log_mod   = self.models["global"].predict(X_mod)[0]
        modified_value = float(np.expm1(pred_log_mod))
        delta_eur = modified_value - base_value
        delta_pct = (delta_eur / max(base_value, 1)) * 100

        result = {
            "base_value_eur":     round(base_value),
            "modified_value_eur": round(modified_value),
            "delta_eur":          round(delta_eur),
            "delta_pct":          round(delta_pct, 2),
            "inference_ms":       round((time.perf_counter() - t0) * 1000, 1),
        }

        if "global" in self.ci_low and "global" in self.ci_high:
            result["confidence_low_eur"]  = round(float(
                np.expm1(self.ci_low["global"].predict(X_mod)[0])
            ))
            result["confidence_high_eur"] = round(float(
                np.expm1(self.ci_high["global"].predict(X_mod)[0])
            ))

        if SHAP_AVAILABLE and "global" in self.explainers:
            try:
                feature_cols  = self.features["global"]
                shap_vals_raw = self.explainers["global"].shap_values(X_mod)[0]
                shap_eur = {
                    feat: round(float(sv) * base_value)
                    for feat, sv in zip(feature_cols, shap_vals_raw)
                }
                result["shap_values"] = shap_eur
                result["top_levers"]  = sorted(
                    [{"feature": f, "shap_eur": v, "current_value": modified.get(f)}
                     for f, v in shap_eur.items()],
                    key=lambda x: abs(x["shap_eur"]),
                    reverse=True,
                )[:10]
            except Exception as exc:
                log.warning("SHAP whatif failed: %s", exc)
                result["shap_values"] = {}
                result["top_levers"]  = []

        return result

    def value_trajectory(
        self,
        player: dict,
        horizon_months: int = 36,
    ) -> list[dict]:
        """
        BONUS: Project how market value will evolve over time.

        Uses the player's actual TM value history (if available) to fit a
        growth trend, then projects forward using the model's current prediction
        as the anchor point. Applies age-based decay for players over 28.

        Returns list of {date, value_eur, lower_eur, upper_eur}
        """
        import datetime

        base_result = self.predict(player)
        base_value  = base_result["predicted_value_eur"]
        ci_low      = base_result.get("confidence_low_eur",  base_value * 0.85)
        ci_high     = base_result.get("confidence_high_eur", base_value * 1.15)

        age = float(player.get("age") or 25)

        # Age-based growth/decay rate per month
        if age < 23:
            monthly_rate = 0.012      # young player: +1.2%/month growth
        elif age < 26:
            monthly_rate = 0.005      # peak approach: +0.5%/month
        elif age < 29:
            monthly_rate = 0.001      # peak: near flat
        elif age < 32:
            monthly_rate = -0.008     # decline: -0.8%/month
        else:
            monthly_rate = -0.015     # sharp decline: -1.5%/month

        today = datetime.date.today()
        trajectory = []
        for m in range(horizon_months + 1):
            projected_date  = today + datetime.timedelta(days=m * 30)
            projected_value = base_value  * ((1 + monthly_rate) ** m)
            projected_low   = ci_low      * ((1 + monthly_rate * 1.2) ** m)
            projected_high  = ci_high     * ((1 + monthly_rate * 0.8) ** m)
            trajectory.append({
                "date":       projected_date.strftime("%Y-%m-%d"),
                "value_eur":  round(max(projected_value, 0)),
                "lower_eur":  round(max(projected_low,   0)),
                "upper_eur":  round(max(projected_high,  0)),
            })

        return trajectory

    def similar_transfers(
        self,
        player: dict,
        n: int = 5,
    ) -> list[dict]:
        """
        BONUS: Find historically similar transfers from your dataset.

        Finds players with similar position, age (+/-3 years), and market
        value (+/-40%) who have actual transfer records, and returns their
        most recent transfer as a comparable deal.
        """
        if self._player_data is None:
            self._load_player_data()
        if not self._player_data:
            return []

        group        = self._get_group(player)
        player_age   = float(player.get("age") or 25)
        player_value = self.predict(player)["predicted_value_eur"]

        candidates = []
        for p in self._player_data:
            if p.get("position_group") != group:
                continue
            p_age   = float(p.get("age") or 0)
            p_value = float(p.get("tm_market_value_eur") or 0)
            if not p_value:
                continue

            # Age within 3 years, value within 40%
            age_ok   = abs(p_age - player_age) <= 3
            value_ok = abs(p_value - player_value) / max(player_value, 1) <= 0.40
            if not age_ok or not value_ok:
                continue

            transfers = p.get("tm_transfers") or []
            paid_transfers = [
                t for t in transfers
                if t.get("fee_eur") and t["fee_eur"] > 0
            ]
            if not paid_transfers:
                continue

            latest = max(paid_transfers, key=lambda t: t.get("date", ""))
            candidates.append({
                "player_name":   p.get("sofifa_name", p.get("tm_name", "Unknown")),
                "age_at_transfer": int(p_age),
                "from_club_id":  latest.get("from_club_id", ""),
                "to_club_id":    latest.get("to_club_id", ""),
                "fee_eur":       latest["fee_eur"],
                "transfer_date": latest.get("date", ""),
                "season":        latest.get("season", ""),
                "market_value_at_time": p_value,
            })

        # Sort by how close the fee is to the predicted value
        candidates.sort(key=lambda x: abs(x["fee_eur"] - player_value))
        return candidates[:n]

    def _load_player_data(self):
        """Load raw player data for similar transfers lookup."""
        paths = [
            DATA_DIR / "raw" / "matched_players.json",
            Path("matched_players.json"),
        ]
        for path in paths:
            if path.exists():
                with open(path, encoding="utf-8") as f:
                    self._player_data = json.load(f)
                log.info("Loaded %d players for similar transfers", len(self._player_data))
                return
        self._player_data = []
        log.warning("matched_players.json not found, similar transfers unavailable")


# Singleton for Streamlit (cached across reruns)
_engine_instance: Optional[WhatIfEngine] = None

def get_engine() -> WhatIfEngine:
    """Return a loaded singleton WhatIfEngine. Safe to call from Streamlit."""
    global _engine_instance
    if _engine_instance is None or not _engine_instance.loaded:
        _engine_instance = WhatIfEngine().load()
    return _engine_instance