from __future__ import annotations
from typing import Optional
import re

#Money parsing
_MONEY_RE = re.compile(r"€?([\d]+\.?[\d]*)([KMBkmb]?)", re.ASCII)
_MULTIPLIERS = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000,
                "k": 1_000, "m": 1_000_000, "b": 1_000_000_000, "": 1}

def parse_money(money_str: str) -> Optional[int]:
    """ Convert a SOFIFA money str into int euros"""
    if not money_str:
        return None
    clean = money_str.strip().replace(",", "").replace(" ", "")
    m = _MONEY_RE.match(clean)
    if not m:
        return None
    return int(float(m.group(1)) *  _MULTIPLIERS.get(m.group(2), 1))

#Weight ad=nd height parsing
_HEIGHT_CM  = re.compile(r"(\d{2,3})\s*cm",  re.I)
_HEIGHT_FT  = re.compile(r"(\d)['\u2032](\d{1,2})")
_WEIGHT_KG  = re.compile(r"(\d{2,3})\s*kg",  re.I)
_WEIGHT_LBS = re.compile(r"(\d{2,3})\s*lbs", re.I)


def parse_height(height_str: str) -> Optional[int]:
    """ Convert a SOFIFA height str into int cm"""
    if not height_str:
        return None
    m = _HEIGHT_CM.search(height_str)
    if m:
        return int(m.group(1))
    m = _HEIGHT_FT.search(height_str)
    if m:
        return round(int(m.group(1)) * 30.48 + int(m.group(2)) * 2.54)
    return None

def parse_weight(weight_str: str) -> Optional[int]:
    """ Convert a SOFIFA weight str into int kg/Lbs"""
    if not weight_str:
        return None
    w = _WEIGHT_KG.search(weight_str)
    if w:
        return int(w.group(1))
    w = _WEIGHT_LBS.search(weight_str)
    if w:
        return round(int(w.group(1)) * 0.453592)
    
#AGE/DOB parsing
_AGE_RE = re.compile(r"(\d{1,2})\s*y\.o\.")
_DOB_RE = re.compile(r"\(([A-Za-z]+\s+\d{1,2},\s+\d{4})\)")


def parse_age(age_str: str) -> Optional[int]:
    """ Convert a SOFIFA age str into int years"""
    if not age_str:
        return None
    age = int(_AGE_RE.search(age_str).group(1)) if _AGE_RE.search(age_str) else None
    dob = _DOB_RE.search(age_str).group(1) if _DOB_RE.search(age_str) else None

    return age, dob

#STAT parsing
_STAT_RE = re.compile(r"^(\d+)")

def parse_stat(stat_str: str) -> Optional[int]:
    """ Convert a SOFIFA stat str into int stat value"""
    if not stat_str:
        return None
    m = _STAT_RE.search(stat_str)
    
    return int(m.group(1)) if m else None

#Player URL parsing
def parse_player_href(href: str) -> tuple[Optional[int], str, str]:
    """
    Decompose a SoFIFA player href into (sofifa_id, slug, version_id)
    """
    parts = href.strip("/").split("/")
    
    if len(parts) < 4 or parts[0] != "player":
        return None, "", ""
    try:
        pid = int(parts[1])
    except ValueError:
        return None, "", ""
    return pid, parts[2], parts[3]

#Money formatting
def extract_money_for_label(text: str, label: str) -> Optional[int]:
    """ Extract a money value from a string like "Release Clause: €120M" """
    fwd = re.search(
        rf"{re.escape(label)}\s+(€[\d.,]+[KMBkmb]?)",
        text, re.I
    )
    if fwd:
        return parse_money(fwd.group(1))

    # "€X … Label"  (within ~20 chars)
    rev = re.search(
        rf"(€[\d.,]+[KMBkmb]?)\s*[·•\-]?\s*{re.escape(label)}",
        text, re.I
    )
    if rev:
        return parse_money(rev.group(1))
    return None

#Position grouping
_POS_GROUP: dict[str, str] = {
    "GK":  "GK",
    "CB":  "CB",  "LCB": "CB",  "RCB": "CB",
    "LB":  "FB",  "RB":  "FB",  "LWB": "FB",  "RWB": "FB",
    "CDM": "MF",  "LDM": "MF",  "RDM": "MF",
    "CM":  "MF",  "LCM": "MF",  "RCM": "MF",
    "CAM": "MF",  "LAM": "MF",  "RAM": "MF",
    "LM":  "WA",  "RM":  "WA",  "LW":  "WA",  "RW":  "WA",
    "CF":  "FW",  "LF":  "FW",  "RF":  "FW",
    "ST":  "FW",  "LS":  "FW",  "RS":  "FW",
}


def position_group(pos: str) -> str:
    """ Map a specific position (e.g. "LCB") to a broader group (e.g. "CB") """
    if not pos:
        return None
    return _POS_GROUP.get(pos[0].upper().strip(), "MF") # default to MF for unknown/empty positions