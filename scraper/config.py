BASE_URL = "https://sofifa.com"

#seasron registry
SEASONS: dict[str, dict] = {
    "FC26":{
        "game_ver": "26",
        "version_id": "260024",
        "label": "2025/26"
    },
    "FC25":
    {
        "game_ver": "25",
        "version_id": "250044",
        "label": "2024/25"
    },
    "FC24":{
        "game_ver": "24",
        "version_id": "240050",
        "label": "2023/24"
    },
}

PLAYERS_PER_PAGE = 60

#Rate limits
DELAY_MIN = 2.5
DELAY_MAX = 5
RETRY_WAIT = 10
MAX_RETRIES = 5

#blocking unnecessary data
BLOCK_RESOURCE_TYPES = {"image", "font", "media"}
BLOCK_URL_PATTERNS   = [
    "*.png", "*.jpg", "*.gif", "*.woff", "*.woff2",
    "google-analytics.com", "doubleclick.net", "googlesyndication.com",
]

STAT_LABEL_MAP: dict[str, str] = {
    # Attacking
    "crossing":            "crossing",
    "finishing":           "finishing",
    "heading accuracy":    "heading_accuracy",
    "short passing":       "short_passing",
    "volleys":             "volleys",
    # Skill
    "dribbling":           "dribbling",
    "curve":               "curve",
    "fk accuracy":         "fk_accuracy",
    "long passing":        "long_passing",
    "ball control":        "ball_control",
    # Movement
    "acceleration":        "acceleration",
    "sprint speed":        "sprint_speed",
    "agility":             "agility",
    "reactions":           "reactions",
    "balance":             "balance",
    # Power
    "shot power":          "shot_power",
    "jumping":             "jumping",
    "stamina":             "stamina",
    "strength":            "strength",
    "long shots":          "long_shots",
    # Mentality
    "aggression":          "aggression",
    "interceptions":       "interceptions",
    "attack position":     "att_position",
    "att. pos.":           "att_position",   # alternate label
    "vision":              "vision",
    "penalties":           "penalties",
    "composure":           "composure",
    # Defending
    "defensive awareness": "defensive_awareness",
    "standing tackle":     "standing_tackle",
    "sliding tackle":      "sliding_tackle",
    # GK
    "gk diving":           "gk_diving",
    "gk handling":         "gk_handling",
    "gk kicking":          "gk_kicking",
    "gk positioning":      "gk_positioning",
    "gk reflexes":         "gk_reflexes",
}

#stat field settings for working with python
ALL_STAT_FIELDS: list[str] = sorted(set(STAT_LABEL_MAP.values()))