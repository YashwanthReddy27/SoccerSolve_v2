from __future__ import annotations

import asyncio
import json
import logging 
import random 
import re
import argparse
from typing import Optional
from dataclasses import asdict, dataclass, field 
from pathlib import Path

from playwright.async_api import async_playwright, Page, Browser, BrowserContext

from scraper.config import ( BASE_URL, SEASONS, PLAYERS_PER_PAGE,
    DELAY_MIN, DELAY_MAX, RETRY_WAIT, MAX_RETRIES,
    BLOCK_RESOURCE_TYPES, STAT_LABEL_MAP,
    )

from scraper.parsers import (parse_money, parse_height, parse_weight, parse_age, parse_stat, parse_player_href,
    extract_money_for_label, position_group,)

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("scraper.log", encoding="utf-8"),
    ],
)
 
CHECKPOINT_EVERY = 50

#Data Model
@dataclass
class PlayerRecord:
    """One row per player per season — maps 1-to-1 with player_seasons DB table."""
    sofifa_id:  int
    season_key: str
    version_id: str
    slug:       str = ""
 
    name:             str = ""
    full_name:        str = ""
    nationality:      str = ""
    age:              Optional[int] = None
    dob:              str = ""
    height_cm:        Optional[int] = None
    weight_kg:        Optional[int] = None
    preferred_foot:   str = ""
    weak_foot:        Optional[int] = None
    skill_moves:      Optional[int] = None
    intl_reputation:  Optional[int] = None
    work_rate:        str = ""
    body_type:        str = ""
 
    positions:       list = field(default_factory=list)
    position_group:  str  = ""
 
    club_name:           str = ""
    club_league:         str = ""
    club_position:       str = ""
    club_kit_number:     str = ""
    club_joined:         str = ""
    club_contract_until: str = ""
 
    # Economics — primary focus of the challenge
    weekly_wage_eur:    Optional[int] = None
    release_clause_eur: Optional[int] = None
    market_value_eur:   Optional[int] = None
 
    overall_rating: Optional[int] = None
    potential:      Optional[int] = None
 
    crossing:             Optional[int] = None
    finishing:            Optional[int] = None
    heading_accuracy:     Optional[int] = None
    short_passing:        Optional[int] = None
    volleys:              Optional[int] = None
    dribbling:            Optional[int] = None
    curve:                Optional[int] = None
    fk_accuracy:          Optional[int] = None
    long_passing:         Optional[int] = None
    ball_control:         Optional[int] = None
    acceleration:         Optional[int] = None
    sprint_speed:         Optional[int] = None
    agility:              Optional[int] = None
    reactions:            Optional[int] = None
    balance:              Optional[int] = None
    shot_power:           Optional[int] = None
    jumping:              Optional[int] = None
    stamina:              Optional[int] = None
    strength:             Optional[int] = None
    long_shots:           Optional[int] = None
    aggression:           Optional[int] = None
    interceptions:        Optional[int] = None
    att_position:         Optional[int] = None
    vision:               Optional[int] = None
    penalties:            Optional[int] = None
    composure:            Optional[int] = None
    defensive_awareness:  Optional[int] = None
    standing_tackle:      Optional[int] = None
    sliding_tackle:       Optional[int] = None
    gk_diving:            Optional[int] = None
    gk_handling:          Optional[int] = None
    gk_kicking:           Optional[int] = None
    gk_positioning:       Optional[int] = None
    gk_reflexes:          Optional[int] = None

async def _make_context(browser: Browser) -> BrowserContext:
    ctx = await browser.new_context(
        viewport={"width": 1440, "height": 900},
        user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        locale="en-US",
        timezone_id="America/New_York",
        extra_http_headers={
            "Accept-Language": "en-US,en;q=0.9",
            "Accept":          "text/html,application/xhtml+xml,*/*;q=0.8",
            "DNT":             "1",
        },
    )
    await ctx.route(
        "**/*",
        lambda route: (
            route.abort()
            if route.request.resource_type in BLOCK_RESOURCE_TYPES
            else route.continue_()
        ),
    )
    return ctx
 
 
async def _polite_delay():
    """Responsible scraping: random 2.5–5 second delay between every request."""
    await asyncio.sleep(random.uniform(DELAY_MIN, DELAY_MAX))
 
 
async def _navigate(page: Page, url: str) -> bool:
    """Navigate with retries, rate-limit and Cloudflare handling."""
    wait = RETRY_WAIT
    for attempt in range(1, MAX_RETRIES + 1):
        await _polite_delay()
        try:
            resp = await page.goto(url, wait_until="domcontentloaded", timeout=30_000)
            if resp is None:
                log.warning("No response at %s (attempt %d)", url, attempt)
                continue
            if resp.status == 429:
                log.warning("429 rate-limit — sleeping %ds", wait)
                await asyncio.sleep(wait); wait *= 2; continue
            if resp.status in (403, 503):
                body = await page.content()
                if "cloudflare" in body.lower() or "just a moment" in body.lower():
                    log.warning("Cloudflare challenge (attempt %d) — sleeping %ds", attempt, wait)
                    await asyncio.sleep(wait); wait *= 2; continue
                log.warning("HTTP %d at %s", resp.status, url)
                return False
            if resp.status == 404:
                return False
            return True
        except Exception as exc:
            log.warning("[%s] %s (attempt %d) — sleeping %ds",
                        type(exc).__name__, url, attempt, wait)
            await asyncio.sleep(wait); wait *= 2
 
    log.error("All %d attempts failed: %s", MAX_RETRIES, url)
    return False

POSITION_STRATA: list[dict] = [
    {"pos_code": "GK",  "group": "GK", "target": 83},
    {"pos_code": "CB",  "group": "CB", "target": 83},
    {"pos_code": "LB",  "group": "FB", "target": 42},
    {"pos_code": "RB",  "group": "FB", "target": 41},
    {"pos_code": "CM",  "group": "MF", "target": 28},
    {"pos_code": "CDM", "group": "MF", "target": 28},
    {"pos_code": "CAM", "group": "MF", "target": 27},
    {"pos_code": "LW",  "group": "WA", "target": 42},
    {"pos_code": "RW",  "group": "WA", "target": 41},
    {"pos_code": "ST",  "group": "FW", "target": 83},
]
#Step 1 to get player stubs from list pages
async def scrape_player_list(page: Page,season_key: str,
    max_players: Optional[int] = None,
    stratified: bool = True,) -> list[dict]:
    """
    Collect player stubs for a season.
 
    stratified=True  (default, recommended for ML):
        Scrapes a fixed number of players per position using the SoFIFA
        &pos= filter, giving equal representation across GK/CB/FB/MF/WA/FW.
        max_players is ignored when stratified=True — use POSITION_STRATA targets.
 
    stratified=False:
        Original behaviour — scrapes the top-N players by overall rating.
        Use max_players to cap the count.
    """
    cfg        = SEASONS[season_key]
    game_ver   = cfg["game_ver"]
    version_id = cfg["version_id"]
 
    if stratified:
        return await _scrape_stratified(page, season_key, game_ver, version_id)
    else:
        return await _scrape_top_n(page, season_key, game_ver, version_id, max_players)
 
 
async def _scrape_stratified(    page: Page,
    season_key: str,
    game_ver: str,
    version_id: str,) -> list[dict]:
    """Collect fixed number of players per position group."""
    group_targets: dict[str, int] = {}
    for s in POSITION_STRATA:
        group_targets[s["group"]] = group_targets.get(s["group"], 0) + s["target"]
 
    buckets: dict[str, list[dict]] = {g: [] for g in group_targets}
    seen_ids: set[int] = set()
    offset = 0
 
    log.info("── Phase 1 (stratified): %s ──", season_key)
    log.info("  Targets per group: %s",
             "  ".join(f"{g}:{t}" for g, t in sorted(group_targets.items())))
 
    while True:
        all_full = all(
            len(buckets[g]) >= group_targets[g] for g in group_targets
        )
        if all_full:
            log.info("  All position groups full — stopping list collection")
            break

        url = (
            f"{BASE_URL}/players?col=oa&sort=desc"
            f"&v={game_ver}&e={version_id}&offset={offset}"
        )
        log.info("  offset=%-5d  buckets: %s", offset,
                "  ".join(f"{g}:{len(buckets[g])}/{group_targets[g]}"
                        for g in ["GK","CB","FB","MF","WA","FW"]))

        if not await _navigate(page, url):
            log.error("  Failed at offset %d — stopping", offset)
            break

        try:
            await page.wait_for_selector("tbody tr", timeout=15_000)
        except Exception:
            log.info("  No rows at offset %d — list exhausted", offset)
            break

        rows = await page.query_selector_all("tbody tr")
        log.info("  Got %d rows on this page", len(rows)) 
        if not rows:
            break

        for row in rows:
            stub = await _parse_list_row(row, season_key, version_id)
            if not stub or stub["sofifa_id"] in seen_ids:
                continue

            grp = stub.get("position_group", "")
            if grp in buckets and len(buckets[grp]) < group_targets[grp]:
                buckets[grp].append(stub)
                seen_ids.add(stub["sofifa_id"])

        log.info("  After processing: %s",  # ← ADD THIS LINE
                "  ".join(f"{g}:{len(buckets[g])}" for g in ["GK","CB","FB","MF","WA","FW"]))

        if len(rows) < PLAYERS_PER_PAGE:
            log.info("  Partial page — list exhausted at offset %d", offset)
            break

        offset += PLAYERS_PER_PAGE
 
    # Merge all buckets into a flat list
    all_stubs = [stub for grp in buckets.values() for stub in grp]
    log.info("Phase 1 done: %d stubs for %s", len(all_stubs), season_key)
    log.info("  Final distribution: %s", _count_by_group(all_stubs))
    return all_stubs

async def _scrape_top_n(
    page: Page,season_key: str,
    game_ver: str,
    version_id: str,
    max_players: Optional[int],) -> list[dict]:
    """Original behaviour — top N players by overall rating."""
    stubs: list[dict] = []
    offset = 0
 
    log.info("── Phase 1 (top-N): %s ──", season_key)
 
    while True:
        url = (f"{BASE_URL}/players?col=oa&sort=desc"
               f"&v={game_ver}&e={version_id}&offset={offset}")
        log.info("  offset=%-5d  stubs: %d", offset, len(stubs))
 
        if not await _navigate(page, url):
            break
 
        try:
            await page.wait_for_selector("tbody tr", timeout=15_000)
        except Exception:
            break
 
        rows = await page.query_selector_all("tbody tr")
        if not rows:
            break
 
        for row in rows:
            stub = await _parse_list_row(row, season_key, version_id)
            if stub:
                stubs.append(stub)
 
        if max_players and len(stubs) >= max_players:
            stubs = stubs[:max_players]
            break
 
        next_btn = await page.query_selector("a[rel='next']")
        if not next_btn:
            break
        offset += PLAYERS_PER_PAGE
 
    log.info("Phase 1 done: %d stubs for %s", len(stubs), season_key)
    return stubs
 
 
def _count_by_group(stubs: list[dict]) -> str:
    """Return a summary string like 'GK:80 CB:80 FB:80 MF:100 WA:80 FW:80'"""
    from collections import Counter
    counts = Counter(s.get("position_group", "?") for s in stubs)
    order = ["GK", "CB", "FB", "MF", "WA", "FW"]
    return "  ".join(f"{g}:{counts.get(g, 0)}" for g in order)
 
async def _parse_list_row(row, season_key: str, version_id: str) -> Optional[dict]:
    """
    Extract every field available directly from one list-page <tr> row.
    """
    try:
        # Player ID
        img = await row.query_selector("img[id]")
        if not img:
            return None
        sofifa_id = int(await img.get_attribute("id"))
 
        # Player name + slug
        name_el = await row.query_selector(".col-name [data-tooltip]")
        name    = (await name_el.get_attribute("data-tooltip")) if name_el else ""
 
        url_el = await row.query_selector(".col-name a.tooltip")
        slug   = ""
        if url_el:
            href = await url_el.get_attribute("href") or ""
            _, slug, _ = parse_player_href(href)
 
        # Nationality
        flag        = await row.query_selector("img.flag")
        nationality = (await flag.get_attribute("title")) if flag else ""
 
        # Age
        age_el   = await row.query_selector(".col-ae")
        age_text = (await age_el.inner_text()).strip() if age_el else ""
        age      = int(age_text) if age_text.isdigit() else None
 
        # Positions — all span.pos elements
        pos_els   = await row.query_selector_all("span.pos")
        positions = []
        for p in pos_els:
            t = (await p.inner_text()).strip()
            if t:
                positions.append(t)
 
        # Club name — second anchor inside .col-name (first is player link)
        col_name_links = await row.query_selector_all(".col-name a")
        club_name   = ""
        club_league = ""
        if len(col_name_links) >= 2:
            club_name = (await col_name_links[1].inner_text()).strip()
 
        # League — the anchor with class "sub" inside .col-name
        league_el = await row.query_selector(".col-name a.sub")
        if league_el:
            club_league = (await league_el.inner_text()).strip()
 
        # Overall / potential
        oa_el     = await row.query_selector(".col-oa span")
        pt_el     = await row.query_selector(".col-pt span")
        overall   = parse_stat((await oa_el.inner_text()).strip() if oa_el else None)
        potential = parse_stat((await pt_el.inner_text()).strip() if pt_el else None)
 
        # Market value + weekly wage
        vl_el = await row.query_selector(".col-vl")
        wg_el = await row.query_selector(".col-wg")
        value = parse_money((await vl_el.inner_text()).strip() if vl_el else None)
        wage  = parse_money((await wg_el.inner_text()).strip() if wg_el else None)
 
        # NOTE: release_clause_eur is NOT on the list page.
        # It is extracted by Phase 2 via "Release clause €X" on the profile page.
 
        return {
            "sofifa_id":        sofifa_id,
            "season_key":       season_key,
            "version_id":       version_id,
            "slug":             slug,
            "name":             name,
            "nationality":      nationality,
            "age":              age,
            "positions":        positions,
            "position_group":   position_group(positions),
            "club_name":        club_name,
            "club_league":      club_league,
            "overall_rating":   overall,
            "potential":        potential,
            "market_value_eur": value,
            "weekly_wage_eur":  wage,
        }
    except Exception as exc:
        log.debug("Row parse error: %s", exc)
        return None
    
#Step 2 to get player details from profile pages
async def scrape_player_profile(page: Page, stub: dict) -> Optional[PlayerRecord]:
    sid  = stub["sofifa_id"]
    slug = stub.get("slug", "")
    vid  = stub["version_id"]
    url  = f"{BASE_URL}/player/{sid}/{slug}/{vid}/"
 
    if not await _navigate(page, url):
        log.warning("Skipping player %d (%s) — page unavailable", sid, stub.get("name", "?"))
        return None
 
    try:
        await page.wait_for_selector("article, .player-card, h1", timeout=10_000)
    except Exception:
        pass
 
    rec = PlayerRecord(
        sofifa_id        = sid,
        season_key       = stub["season_key"],
        version_id       = vid,
        slug             = slug,
        # Required fields — pre-fill from Phase 1 stub, profile page will override/enrich
        name             = stub.get("name", ""),
        nationality      = stub.get("nationality", ""),
        age              = stub.get("age"),
        positions        = list(stub.get("positions", [])),
        position_group   = stub.get("position_group", ""),
        club_name        = stub.get("club_name", ""),    # from list page
        club_league      = stub.get("club_league", ""), # from list page
        overall_rating   = stub.get("overall_rating"),
        potential        = stub.get("potential"),
        market_value_eur = stub.get("market_value_eur"),
        weekly_wage_eur  = stub.get("weekly_wage_eur"),
        # release_clause_eur: None here — filled by _fill_economics from profile page
    )
 
    body_text = await page.inner_text("body")
 
    _fill_identity(body_text, rec)
    _fill_economics(body_text, rec)
    _fill_club(body_text, rec)
 
    # Refresh positions from profile page (more reliable)
    pos_spans = await page.query_selector_all("span.pos")
    if pos_spans:
        positions = [(await p.inner_text()).strip() for p in pos_spans]
        positions = [p for p in positions if p]
        if positions:
            rec.positions = list(dict.fromkeys(positions))
            rec.position_group = position_group(rec.positions)
 
    # Full name from H1
    h1 = await page.query_selector("h1")
    if h1:
        rec.full_name = (await h1.inner_text()).strip()

    oa_m = re.search(r"(\d{2,3})\s*[·•\-]?\s*Overall rating", body_text, re.I)
    if oa_m:
        rec.overall_rating = int(oa_m.group(1))
    pt_m = re.search(r"(\d{2,3})\s*[·•\-]?\s*Potential", body_text, re.I)
    if pt_m:
        rec.potential = int(pt_m.group(1))
 
    # Fallback: if name is empty (Phase 1 selector missed it), use full_name
    if not rec.name and rec.full_name:
        rec.name = rec.full_name
 
    # Club / league anchor text
    club_el = await page.query_selector("a[href^='/team/']")
    if club_el:
        rec.club_name = (await club_el.inner_text()).strip()
    league_el = await page.query_selector("a[href^='/league/']")
    if league_el:
        rec.club_league = (await league_el.inner_text()).strip()
 
    await _fill_stats(page, body_text, rec)
    return rec
 
 
def _fill_identity(text: str, rec: PlayerRecord):
    rec.height_cm = parse_height(text)
    rec.weight_kg = parse_weight(text)
    rec.age, rec.dob = parse_age(text)  #dob as well
 
    for pattern, attr in [
        (r"Preferred foot\s+(Left|Right)",        "preferred_foot"),
        (r"(\d)\s+Weak foot",                     "weak_foot"),
        (r"(\d)\s+Skill moves",                   "skill_moves"),
        (r"(\d)\s+International reputation",       "intl_reputation"),
        (r"Work rate\s+([\w]+\s*/\s*[\w]+)",      "work_rate"),
        (r"Body type\s+([^\n·]{3,50})",           "body_type"),
    ]:
        m = re.search(pattern, text, re.I)
        if m:
            val = m.group(1).strip()
            if attr in ("weak_foot", "skill_moves", "intl_reputation"):
                val = int(val)
            setattr(rec, attr, val)
 
 
def _fill_economics(text: str, rec: PlayerRecord):
    rc = extract_money_for_label(text, "Release clause")
    if rc is not None:
        rec.release_clause_eur = rc
 
    wage = extract_money_for_label(text, "Wage")
    if wage is not None:
        rec.weekly_wage_eur = wage
 
    # "€22.5M · Value"  — avoid grabbing release clause
    value_matches = re.findall(
        r"(€[\d.,]+[KMBkmb]?)\s*[·•\-]?\s*Value", text, re.I
    )
    if value_matches:
        # Parse all candidates and take the largest — that's the market value
        parsed = [v for v in (parse_money(m) for m in value_matches) if v]
        if parsed:
            rec.market_value_eur = max(parsed)
    elif rec.market_value_eur is None:
        val = extract_money_for_label(text, "Value")
        if val:
            rec.market_value_eur = val
 
 
def _fill_club(text: str, rec: PlayerRecord):
    for pattern, attr in [
        (r"\bPosition\s+([A-Z]{2,3}|SUB)\b",                   "club_position"),
        (r"Kit number\s+(\d+)",                                  "club_kit_number"),
        (r"Joined\s+([A-Za-z]+\s+\d{1,2},\s+\d{4})",           "club_joined"),
        (r"Contract valid until\s+(\d{4})",                      "club_contract_until"),
    ]:
        m = re.search(pattern, text, re.I)
        if m:
            setattr(rec, attr, m.group(1).strip())
 
 
async def _fill_stats(page: Page, body_text: str, rec: PlayerRecord):
    # Strategy 1: DOM  <span class="label pXX">VALUE</span>  + parent li text
    stat_spans = await page.query_selector_all("span[class*='label p']")
    for span in stat_spans:
        val = parse_stat((await span.inner_text()).strip())
        if val is None:
            continue
        parent = await span.evaluate_handle("el => el.parentElement")
        if parent:
            parent_text = (await parent.inner_text()).lower()
            for label, field_name in STAT_LABEL_MAP.items():
                if label in parent_text and getattr(rec, field_name) is None:
                    setattr(rec, field_name, val)
                    break
 
    # Strategy 2: regex fallback  "61 Crossing"
    for label, field_name in STAT_LABEL_MAP.items():
        if getattr(rec, field_name) is not None:
            continue
        m = re.search(rf"(\d+)\s+{re.escape(label)}", body_text, re.I)
        if m:
            setattr(rec, field_name, int(m.group(1)))

# Zip all phases together
async def _run_season(    season_key: str,    output_dir: Path,
    max_players: Optional[int],
    resume: bool,
    stratified: bool = True,) -> list[dict]:
    stub_file   = output_dir / f"stubs_{season_key}.json"
    output_file = output_dir / f"players_{season_key}.json"
 
    existing: list[dict] = []
    scraped_ids: set[int] = set()
    if resume and output_file.exists():
        with open(output_file, encoding="utf-8") as f:
            existing = json.load(f)
        scraped_ids = {p["sofifa_id"] for p in existing}
        log.info("Resume: %d players already scraped", len(scraped_ids))
 
    async with async_playwright() as pw:
        browser: Browser = await pw.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-blink-features=AutomationControlled",
            ],
        )
        ctx  = await _make_context(browser)
        page = await ctx.new_page()
 
        # Phase 1
        if resume and stub_file.exists():
            with open(stub_file, encoding="utf-8") as f:
                stubs = json.load(f)
            log.info("Loaded %d stubs from cache — distribution: %s",
                     len(stubs), _count_by_group(stubs))
        else:
            stubs = await scrape_player_list(
                page, season_key,
                max_players=max_players,
                stratified=stratified,
            )
            _save_json(stub_file, stubs)
 
        # Phase 2
        results = list(existing)
        errors  = 0
        n_new   = 0
 
        for i, stub in enumerate(stubs):
            sid = stub["sofifa_id"]
            if sid in scraped_ids:
                continue
 
            log.info("[%d/%d] %s (id=%d)", i + 1, len(stubs), stub.get("name", "?"), sid)
            record = await scrape_player_profile(page, stub)
 
            if record is None:
                errors += 1
                continue
 
            results.append(asdict(record))
            scraped_ids.add(sid)
            n_new += 1
 
            if n_new % CHECKPOINT_EVERY == 0:
                _save_json(output_file, results)
                log.info("  checkpoint: %d saved, %d errors", len(results), errors)
 
        _save_json(output_file, results)
        log.info("Season %s done: %d players, %d errors → %s",
                 season_key, len(results), errors, output_file.name)
 
        await browser.close()
 
    return results
 
 
def run(    seasons:     list[str], output_dir:  str = "data/raw",
    max_players: Optional[int] = None,
    resume:      bool = True,
    stratified:  bool = True,) -> dict[str, list[dict]]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    all_results: dict[str, list[dict]] = {}
    for sk in seasons:
        log.info("=" * 60)
        log.info("SEASON: %s  (%s)", sk, SEASONS[sk]["label"])
        log.info("=" * 60)
        all_results[sk] = asyncio.run(
            _run_season(sk, out, max_players, resume, stratified)
        )
    return all_results
 
 
def _save_json(path: Path, data: list[dict]):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SoccerSolver — SoFIFA scraper")
    parser.add_argument("--seasons",       nargs="+", default=list(SEASONS.keys()),
                        choices=list(SEASONS.keys()))
    parser.add_argument("--max-players",   type=int,  default=None,
                        help="Only used with --no-stratify. Cap per season.")
    parser.add_argument("--output-dir",    default="data/raw")
    parser.add_argument("--no-resume",     action="store_true",
                        help="Ignore existing data and start fresh")
    parser.add_argument("--no-stratify",   action="store_true",
                        help="Disable stratified sampling (use top-N by overall instead)")
    parser.add_argument("--scale",         type=float, default=1.0,
                        help="Scale all position targets. 1.0=500/season, 6.0=3000/season")
    args = parser.parse_args()
 
    # Apply scale to position targets if requested
    if args.scale != 1.0:
        for stratum in POSITION_STRATA:
            stratum["target"] = max(10, int(stratum["target"] * args.scale))
        total = sum(s["target"] for s in POSITION_STRATA)
        log.info("Scaled targets (x%.1f): %s  total=%d",
                 args.scale,
                 "  ".join(f"{s['pos_code']}:{s['target']}" for s in POSITION_STRATA),
                 total)
 
    run(
        seasons    = args.seasons,
        output_dir = args.output_dir,
        max_players= args.max_players,
        resume     = not args.no_resume,
        stratified = not args.no_stratify,
    )