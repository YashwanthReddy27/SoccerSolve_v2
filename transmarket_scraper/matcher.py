from __future__ import annotations
import json
import logging
import argparse
from pathlib import Path
from typing import Optional

try:
    from rapidfuzz import fuzz
except ImportError:
    raise SystemExit(
        "rapidfuzz not installed. Run: pip install rapidfuzz"
    )

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("matcher")

# Config
DATA_DIR = Path("../data/raw")

SOFIFA_FILES = {
    "FC26": DATA_DIR / "players_FC26.json",
}
TM_RAW_FILE    = DATA_DIR / "tm_raw.json"
OUTPUT_FILE    = DATA_DIR / "matched_players.json"
REPORT_FILE    = DATA_DIR / "match_report.json"

MATCH_THRESHOLD = 0.70   # composite score to accept a match
AMBIGUOUS_GAP   = 0.05   # flag if best vs 2nd best within this range

# Signal weights (must sum to 1.0)
W_NAME         = 0.45
W_CLUB         = 0.30
W_AGE          = 0.15
W_NATIONALITY  = 0.10

# Normalisation
_ACCENT_MAP = str.maketrans(
    "àáâãäåèéêëìíîïòóôõöùúûüýÿñçæœ",
    "aaaaaaeeeeiiiiooooouuuuyyncao"
)

def _norm(text: str) -> str:
    """Lower, strip accents, remove punctuation for robust comparison."""
    if not text:
        return ""
    return (
        text.lower()
            .translate(_ACCENT_MAP)
            .replace("-", " ")
            .replace(".", " ")
            .replace("'", "")
            .strip()
    )

def _norm_club(name: str) -> str:
    """Strip common TM/SoFIFA suffixes like FC, CF, AFC, United."""
    suffixes = {" fc", " afc", " cf", " sc", " ac", " rc", " fk", " sk",
                " bv", " sv", " vfb", " vfl", " rb", " 1.fc"}
    n = _norm(name)
    for sfx in suffixes:
        if n.endswith(sfx):
            n = n[: -len(sfx)].strip()
    return n

# Scoring
def _score_name(sofifa_name: str, tm_name: str) -> float:
    """
    Uses both token_sort_ratio and partial_ratio and takes the higher score.
    token_sort_ratio handles word order (Son Heung-min vs Heung-Min Son).
    partial_ratio handles TM short names (Alisson vs Alisson Ramses Becker).
    """
    n1 = _norm(sofifa_name)
    n2 = _norm(tm_name)
    return max(
        fuzz.token_sort_ratio(n1, n2),
        fuzz.partial_ratio(n1, n2),
        fuzz.token_set_ratio(n1, n2),
    ) / 100.0


def _score_club(sofifa_club: str, tm_club: str) -> float:
    """partial_ratio handles substrings (e.g. 'Man City' in 'Manchester City')."""
    return fuzz.partial_ratio(_norm_club(sofifa_club), _norm_club(tm_club)) / 100.0


def _score_age(sofifa_age: Optional[int], tm_age: Optional[int]) -> float:
    if sofifa_age is None or tm_age is None:
        return 0.5   # unknown, don't penalise
    diff = abs(sofifa_age - tm_age)
    if diff == 0:
        return 1.0
    if diff == 1:
        return 0.5
    return 0.0


def _score_nationality(sofifa_nat: str, tm_nat: str) -> float:
    if not sofifa_nat or not tm_nat:
        return 0.5   # unknown, don't penalise
    return 1.0 if _norm(sofifa_nat) == _norm(tm_nat) else 0.0


def composite_score(sofifa: dict, candidate: dict) -> float:
    """
    Weighted composite match score (0-1).
    sofifa:    {name, club_name, age, nationality}
    candidate: {tm_name, club, age, nationality}

    Club weight is redistributed to name when TM club is empty.
    TM search results frequently omit the club field, so we cannot
    penalise for a missing club — we treat it as unknown (neutral 0.5)
    and redistribute its weight to the name signal instead.
    """
    s_name = _score_name(sofifa.get("name", ""),
                         candidate.get("tm_name", ""))
    s_age  = _score_age(sofifa.get("age"), candidate.get("age"))
    s_nat  = _score_nationality(sofifa.get("nationality", ""),
                                candidate.get("nationality", ""))

    tm_club = candidate.get("club", "")
    if tm_club:
        s_club = _score_club(sofifa.get("club_name", ""), tm_club)
        return (W_NAME * s_name +
                W_CLUB * s_club +
                W_AGE  * s_age  +
                W_NATIONALITY * s_nat)
    else:
        # Club unavailable: redistribute club weight evenly to name and age
        w_name_adj = W_NAME + W_CLUB * 0.6
        w_age_adj  = W_AGE  + W_CLUB * 0.4
        return (w_name_adj * s_name +
                w_age_adj  * s_age  +
                W_NATIONALITY * s_nat)

# Main matching logic
def match_players(threshold: float = MATCH_THRESHOLD) -> dict:
    """
    Load SoFIFA + TM raw data, run fuzzy matching, save results.
    Returns a summary report dict.
    """
    # Load TM raw data, keyed by sofifa_id for quick lookup
    if not TM_RAW_FILE.exists():
        raise FileNotFoundError(
            f"{TM_RAW_FILE} not found. Run tm_scraper.py first."
        )
    with open(TM_RAW_FILE, encoding="utf-8") as f:
        tm_raw = json.load(f)
    tm_by_sofifa: dict[str, dict] = {str(r["sofifa_id"]): r for r in tm_raw}
    log.info("Loaded %d TM raw records", len(tm_raw))

    # Load SoFIFA data across all seasons
    sofifa_players = []
    for season, path in SOFIFA_FILES.items():
        if not path.exists():
            log.warning("SoFIFA file missing: %s", path)
            continue
        with open(path, encoding="utf-8") as f:
            players = json.load(f)
        for p in players:
            p["_season"] = season
            sofifa_players.append(p)
    log.info("Loaded %d SoFIFA player-season records", len(sofifa_players))

    matched       = []
    unmatched     = []
    ambiguous     = []
    match_quality = []

    for player in sofifa_players:
        sid = str(player.get("sofifa_id", ""))
        tm_record = tm_by_sofifa.get(sid)

        if not tm_record or not tm_record.get("tm_candidates"):
            unmatched.append({
                "sofifa_id":  sid,
                "name":       player.get("name", ""),
                "season":     player.get("_season"),
                "reason":     "no TM candidates found",
            })
            continue

        candidates = tm_record["tm_candidates"]

        # Score all candidates
        scored = []
        for c in candidates:
            score = composite_score(player, c)
            scored.append((score, c))
        scored.sort(key=lambda x: -x[0])

        best_score, best_cand = scored[0]

        # Check ambiguity (two candidates very close)
        is_ambiguous = (
            len(scored) >= 2
            and (best_score - scored[1][0]) < AMBIGUOUS_GAP
        )

        if best_score < threshold:
            unmatched.append({
                "sofifa_id":  sid,
                "name":       player.get("name", ""),
                "season":     player.get("_season"),
                "reason":     f"best_score={best_score:.2f} < threshold={threshold}",
                "candidates": [{"name": c["tm_name"], "score": round(s, 3)}
                               for s, c in scored[:3]],
            })
            continue

        if is_ambiguous:
            ambiguous.append({
                "sofifa_id":  sid,
                "name":       player.get("name", ""),
                "season":     player.get("_season"),
                "top2":       [
                    {"name": scored[0][1]["tm_name"], "score": round(scored[0][0], 3)},
                    {"name": scored[1][1]["tm_name"], "score": round(scored[1][0], 3)},
                ],
            })
            # Still include the match, but flag it
            matched_record = _build_unified_record(
                player, best_cand, tm_record, best_score, ambiguous=True
            )
        else:
            matched_record = _build_unified_record(
                player, best_cand, tm_record, best_score, ambiguous=False
            )

        matched.append(matched_record)
        match_quality.append(best_score)

    # Save unified dataset
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(matched, f, indent=2, ensure_ascii=False)

    # Build report
    avg_score = sum(match_quality) / len(match_quality) if match_quality else 0
    report = {
        "total_sofifa":    len(sofifa_players),
        "total_matched":   len(matched),
        "total_unmatched": len(unmatched),
        "total_ambiguous": len(ambiguous),
        "match_rate_pct":  round(100 * len(matched) / max(len(sofifa_players), 1), 1),
        "avg_match_score": round(avg_score, 3),
        "threshold_used":  threshold,
        "unmatched":       unmatched,
        "ambiguous":       ambiguous,
    }
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    log.info("Matching complete:")
    log.info("  Matched:   %d / %d  (%.1f%%)",
             len(matched), len(sofifa_players),
             100 * len(matched) / max(len(sofifa_players), 1))
    log.info("  Unmatched: %d", len(unmatched))
    log.info("  Ambiguous: %d  (flagged but included)", len(ambiguous))
    log.info("  Avg score: %.3f", avg_score)
    log.info("  Output:    %s", OUTPUT_FILE)
    log.info("  Report:    %s", REPORT_FILE)

    return report


def _build_unified_record(
    sofifa: dict,
    tm_cand: dict,
    tm_record: dict,
    score: float,
    ambiguous: bool,
) -> dict:
    """
    Merge a SoFIFA player record with its matched TM data.
    This is the training row for the regression model.

    Priority rules for overlapping fields:
    market_value_eur  TM wins (real market value, not EA game estimate)
    name              SoFIFA wins (we'll use sofifa_name / tm_name both)
    age, nationality  SoFIFA wins (more standardised)
    """
    tm_data = tm_record.get("tm_data") or {}

    # The critical target variable, TM market value takes priority
    tm_value   = tm_data.get("current_value_eur") or tm_cand.get("value_eur")
    base_value = sofifa.get("market_value_eur")

    return {
        # Identity
        "sofifa_id":       sofifa.get("sofifa_id"),
        "tm_id":           tm_cand.get("tm_id"),
        "season_key":      sofifa.get("_season", sofifa.get("season_key")),
        "sofifa_name":     sofifa.get("name", ""),
        "tm_name":         tm_data.get("tm_name") or tm_cand.get("tm_name", ""),
        "nationality":     sofifa.get("nationality", ""),
        "age":             sofifa.get("age"),
        "dob":             tm_data.get("dob", ""),
        "height_cm":       sofifa.get("height_cm"),
        "weight_kg":       sofifa.get("weight_kg"),
        "preferred_foot":  sofifa.get("preferred_foot", ""),
        "weak_foot":       sofifa.get("weak_foot"),
        "skill_moves":     sofifa.get("skill_moves"),
        "intl_reputation": sofifa.get("intl_reputation"),
        "work_rate":       sofifa.get("work_rate", ""),

        # Position
        "positions":       sofifa.get("positions", []),
        "position_group":  sofifa.get("position_group", ""),
        "tm_position":     tm_data.get("position", ""),

        # Club
        "club_name":       sofifa.get("club_name", ""),
        "club_league":     sofifa.get("club_league", ""),
        "tm_current_club": tm_data.get("current_club", tm_cand.get("club", "")),

        # Economics (TARGET VARIABLE = tm_market_value_eur)
        "tm_market_value_eur":    tm_value,           # regression target
        "sofifa_market_value_eur": base_value,         # SoFIFA estimate for reference
        "weekly_wage_eur":         sofifa.get("weekly_wage_eur"),
        "release_clause_eur":      sofifa.get("release_clause_eur"),

        # Summary ratings
        "overall_rating":  sofifa.get("overall_rating"),
        "potential":       sofifa.get("potential"),

        # All 34 SoFIFA skill attributes
        "crossing":           sofifa.get("crossing"),
        "finishing":          sofifa.get("finishing"),
        "heading_accuracy":   sofifa.get("heading_accuracy"),
        "short_passing":      sofifa.get("short_passing"),
        "volleys":            sofifa.get("volleys"),
        "dribbling":          sofifa.get("dribbling"),
        "curve":              sofifa.get("curve"),
        "fk_accuracy":        sofifa.get("fk_accuracy"),
        "long_passing":       sofifa.get("long_passing"),
        "ball_control":       sofifa.get("ball_control"),
        "acceleration":       sofifa.get("acceleration"),
        "sprint_speed":       sofifa.get("sprint_speed"),
        "agility":            sofifa.get("agility"),
        "reactions":          sofifa.get("reactions"),
        "balance":            sofifa.get("balance"),
        "shot_power":         sofifa.get("shot_power"),
        "jumping":            sofifa.get("jumping"),
        "stamina":            sofifa.get("stamina"),
        "strength":           sofifa.get("strength"),
        "long_shots":         sofifa.get("long_shots"),
        "aggression":         sofifa.get("aggression"),
        "interceptions":      sofifa.get("interceptions"),
        "att_position":       sofifa.get("att_position"),
        "vision":             sofifa.get("vision"),
        "penalties":          sofifa.get("penalties"),
        "composure":          sofifa.get("composure"),
        "defensive_awareness": sofifa.get("defensive_awareness"),
        "standing_tackle":    sofifa.get("standing_tackle"),
        "sliding_tackle":     sofifa.get("sliding_tackle"),
        "gk_diving":          sofifa.get("gk_diving"),
        "gk_handling":        sofifa.get("gk_handling"),
        "gk_kicking":         sofifa.get("gk_kicking"),
        "gk_positioning":     sofifa.get("gk_positioning"),
        "gk_reflexes":        sofifa.get("gk_reflexes"),

        # TM enrichment
        "tm_value_history":   tm_data.get("value_history", []),
        "tm_transfers":       tm_data.get("transfers", []),

        # Data quality metadata
        "match_score":        round(score, 4),
        "match_ambiguous":    ambiguous,
        "has_tm_value":       tm_value is not None,
        "has_value_history":  len(tm_data.get("value_history", [])) > 0,
    }


# CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SoccerSolver TM player matcher")
    parser.add_argument("--threshold", type=float, default=MATCH_THRESHOLD,
                        help=f"Min composite score to accept a match (default {MATCH_THRESHOLD})")
    args = parser.parse_args()

    report = match_players(threshold=args.threshold)

    print("\n" + "=" * 50)
    print("MATCH REPORT SUMMARY")
    print("=" * 50)
    print(f"  Total SoFIFA players: {report['total_sofifa']}")
    print(f"  Matched:              {report['total_matched']} ({report['match_rate_pct']}%)")
    print(f"  Unmatched:            {report['total_unmatched']}")
    print(f"  Ambiguous (flagged):  {report['total_ambiguous']}")
    print(f"  Avg match score:      {report['avg_match_score']:.3f}")
    print(f"\nFull report: {REPORT_FILE}")
    print(f"Unified dataset: {OUTPUT_FILE}")