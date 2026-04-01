from __future__ import annotations
import asyncio
import json
import logging
import os
import random
import re
import time
import argparse
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from playwright.async_api import async_playwright, Page, Browser, BrowserContext

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("tm_scraper")

# Config
BASE_URL = "https://www.transfermarkt.com"
SEARCH_URL = f"{BASE_URL}/schnellsuche/ergebnis/schnellsuche?query={{name}}&Spieler_page=0"

# Responsible scraping delays (seconds)
DELAY_MIN = 3.0
DELAY_MAX = 6.0
RETRY_WAIT = 15      # base wait on rate-limit / Cloudflare hit
MAX_RETRIES = 4
CHECKPOINT_EVERY = 5

# Input/output paths
DATA_DIR = Path("../data/raw")
OUTPUT_FILE = DATA_DIR / "tm_raw.json"

# Resources to block (speeds up scraping)
BLOCK_RESOURCE_TYPES = {"image", "media", "font"}
BLOCK_URL_PATTERNS = [
    "*.png", "*.jpg", "*.gif", "*.woff", "*.woff2",
    "google-analytics.com", "doubleclick.net", "googlesyndication.com",
    "adsystem.com", "adnxs.com",
]

#  Data classes 
@dataclass
class TmValueHistory:
    date: str              # "2024-01-15"
    value_eur: Optional[int]

@dataclass
class TmTransfer:
    date: str
    season: str
    from_club_id: str
    to_club_id: str
    fee_eur: Optional[int]

@dataclass
class TmPlayer:
    tm_id: str             # Transfermarkt player ID (from URL)
    tm_url: str            # full URL on transfermarkt.com
    tm_name: str           # name as it appears on TM
    nationality: str = ""
    age: Optional[int] = None
    dob: str = ""
    position: str = ""
    current_club: str = ""
    current_value_eur: Optional[int] = None
    value_history: list = field(default_factory=list)  # list of TmValueHistory dicts
    transfers: list = field(default_factory=list)       # list of TmTransfer dicts

#  Helpers 

def _polite_delay():
    time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))

def _parse_value(text: str) -> Optional[int]:
    """
    Parse TM value strings like '€12.50m', '€850k', '€-', '-' into int euros.
    Returns None for undisclosed / unknown.
    """
    if not text:
        return None
    text = text.strip().replace("\u20ac", "").replace(",", ".").lower()
    if text in ("-", "?", "free", "loan", "undisclosed", ""):
        return None
    try:
        if "m" in text:
            return int(float(text.replace("m", "").strip()) * 1_000_000)
        if "k" in text or "th." in text:
            return int(float(text.replace("k", "").replace("th.", "").strip()) * 1_000)
        return int(float(text))
    except (ValueError, TypeError):
        return None

def _parse_date(text: str) -> str:
    """Normalise TM date strings to YYYY-MM-DD. Returns '' on failure.
    Handles formats seen in the wild:
      "02/10/1992 (33)"  actual TM profile format
      "Jan 15, 2024"
      "15.01.2024"
      "2024-01-15"
    """
    if not text:
        return ""
    import datetime
    # Strip trailing age annotation like " (33)" or "(32)"
    clean = re.sub(r'\s*\(\d+\)\s*$', '', text.strip())
    for fmt in ("%d/%m/%Y", "%b %d, %Y", "%d.%m.%Y", "%Y-%m-%d", "%B %d, %Y"):
        try:
            return datetime.datetime.strptime(clean.strip(), fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return clean.strip()

#  Browser setup 

async def _make_context(browser: Browser) -> BrowserContext:
    """Create a browser context with realistic headers to avoid bot detection."""
    ctx = await browser.new_context(
        viewport={"width": 1366, "height": 768},
        locale="en-US",
        timezone_id="Europe/London",
        user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/123.0.0.0 Safari/537.36"
        ),
        extra_http_headers={
            "Accept-Language": "en-US,en;q=0.9",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Referer": "https://www.google.com/",
        },
    )
    # Block heavy resources
    async def block_route(route, request):
        if request.resource_type in BLOCK_RESOURCE_TYPES:
            await route.abort()
            return
        if any(request.url.endswith(p.replace("*", "")) for p in BLOCK_URL_PATTERNS
               if "*" in p):
            await route.abort()
            return
        await route.continue_()

    await ctx.route("**/*", block_route)
    return ctx

async def _safe_goto(page: Page, url: str, retries: int = MAX_RETRIES) -> bool:
    """Navigate with retry logic for Cloudflare / rate-limit hits."""
    wait = RETRY_WAIT
    for attempt in range(1, retries + 1):
        try:
            resp = await page.goto(url, timeout=30_000, wait_until="domcontentloaded")
            if resp and resp.status == 429:
                log.warning("Rate limited (429). Waiting %ds (attempt %d/%d)…",
                            wait, attempt, retries)
                await asyncio.sleep(wait)
                wait *= 2
                continue
            if resp and resp.status in (403, 503):
                log.warning("Bot-blocked (%d). Waiting %ds (attempt %d/%d)…",
                            resp.status, wait, attempt, retries)
                await asyncio.sleep(wait)
                wait *= 2
                continue
            return True
        except Exception as exc:
            log.warning("goto error (%s) — attempt %d/%d", exc, attempt, retries)
            await asyncio.sleep(wait)
            wait *= 2
    return False

#  Phase 1: search 

async def scrape_search(page: Page, name: str) -> list[dict]:
    """
    Search TM for a player name.
    Returns list of candidates: {tm_id, tm_url, tm_name, club, age, value_eur}
    Usually 0–5 results per query.
    """
    url = SEARCH_URL.format(name=name.replace(" ", "+"))
    ok = await _safe_goto(page, url)
    if not ok:
        return []

    candidates = []
    try:
        # TM search results table: each player row has class "odd" or "even"
        rows = await page.query_selector_all(
            "table.items > tbody > tr.odd, table.items > tbody > tr.even"
        )
        for row in rows[:5]:  # top 5 candidates only
            try:
                link_el = await row.query_selector("td.hauptlink a[href*='/profil/spieler/']")
                if not link_el:
                    continue
                href = await link_el.get_attribute("href") or ""
                tm_name = (await link_el.inner_text()).strip()

                # Extract tm_id from URL pattern /profil/spieler/12345
                m = re.search(r"/spieler/(\d+)", href)
                tm_id = m.group(1) if m else ""
                if not tm_id:
                    continue

                # Club
                club_el = await row.query_selector("td.zentriert a[href*='/verein/']")
                club = (await club_el.inner_text()).strip() if club_el else ""

                # Age
                age_els = await row.query_selector_all("td.zentriert")
                age = None
                for el in age_els:
                    txt = (await el.inner_text()).strip()
                    if re.match(r"^\d{2}$", txt):
                        age = int(txt)
                        break

                # Current value
                val_el = await row.query_selector("td.rechts.hauptlink")
                value_eur = _parse_value(
                    (await val_el.inner_text()).strip() if val_el else ""
                )

                candidates.append({
                    "tm_id": tm_id,
                    "tm_url": f"{BASE_URL}/spieler/profil/spieler/{tm_id}",
                    "tm_name": tm_name,
                    "club": club,
                    "age": age,
                    "value_eur": value_eur,
                })
            except Exception as exc:
                log.debug("Row parse error: %s", exc)
                continue
    except Exception as exc:
        log.debug("Search parse error for '%s': %s", name, exc)

    return candidates

#  Phase 2: profile 

async def scrape_profile(page: Page, tm_id: str) -> Optional[TmPlayer]:
    """
    Scrape a player's full TM profile page.
    URL pattern: https://www.transfermarkt.com/x/profil/spieler/{tm_id}
    Returns TmPlayer or None on failure.
    """
    url = f"{BASE_URL}/x/profil/spieler/{tm_id}"
    ok = await _safe_goto(page, url)
    if not ok:
        return None

    player = TmPlayer(tm_id=tm_id, tm_url=url, tm_name="")

    try:
        # Name: TM wraps it in a strong tag inside the header
        name_el = await page.query_selector(".data-header__headline-wrapper strong")
        if not name_el:
            name_el = await page.query_selector("h1.data-header__headline-wrapper--main")
        player.tm_name = (await name_el.inner_text()).strip() if name_el else ""

        # Current market value
        # TM structure: <a class="data-header__market-value-wrapper">
        #   <span class="waehrung">€</span>45.00<span class="waehrung">m</span>
        #   <p class="data-header__last-update">Last update: 16/03/2026</p>
        # </a>
        # inner_text() gives "€45.00m  Last update: ..." so we strip the <p> first
        val_el = await page.query_selector("a.data-header__market-value-wrapper")
        if val_el:
            val_html = await val_el.inner_html()
            val_clean = re.sub(r'<p[^>]*>.*?</p>', '', val_html, flags=re.DOTALL)
            val_text = re.sub(r'<[^>]+>', '', val_clean).strip()
            player.current_value_eur = _parse_value(val_text)

        # Profile info table: label/value pairs
        # TM structure: <span class="info-table__content--regular">Date of birth:</span>
        #               <span class="info-table__content--bold">02/10/1992 (33)</span>
        labels = await page.query_selector_all(".info-table__content--regular")
        values = await page.query_selector_all(".info-table__content--bold")
        for label_el, value_el in zip(labels, values):
            try:
                lbl = (await label_el.inner_text()).strip().lower()
                val = (await value_el.inner_text()).strip()
                if "date of birth" in lbl or "d.o.b" in lbl:
                    # val looks like "02/10/1992 (33)" — parse date and extract age
                    player.dob = _parse_date(val)
                    age_match = re.search(r'\((\d+)\)', val)
                    if age_match:
                        player.age = int(age_match.group(1))
                elif "position" in lbl:
                    player.position = val
                elif "nationality" in lbl or "citizenship" in lbl:
                    # May contain multiple nationalities — take first line
                    player.nationality = val.split("\n")[0].strip()
                elif "current club" in lbl or ("club" in lbl and not player.current_club):
                    player.current_club = val
            except Exception:
                continue

        # Market value history
        # API: tmapi-alpha.transfermarkt.technology/player/{id}/market-value-history
        # Response: {"success":true,"data":{"history":[
        #   {"age":17,"marketValue":{"value":50000,"currency":"EUR","determined":"2009-05-27"}},
        #   ...
        # ]}}
        try:
            mw_resp = await page.request.get(
                f"https://tmapi-alpha.transfermarkt.technology/player/{tm_id}/market-value-history",
                headers={"Referer": url},
                timeout=15_000,
            )
            if mw_resp.ok:
                mw_json = await mw_resp.json()
                entries = mw_json.get("data", {}).get("history", [])
                for entry in entries:
                    mv = entry.get("marketValue", {})
                    raw_val  = mv.get("value")
                    raw_date = mv.get("determined", "")
                    if raw_date and raw_val is not None:
                        player.value_history.append({
                            "date":      raw_date,   # already "YYYY-MM-DD"
                            "value_eur": int(raw_val),
                        })
                log.debug("Value history: %d entries for tm_id=%s", len(player.value_history), tm_id)
        except Exception as exc:
            log.debug("Value history error for tm_id=%s: %s", tm_id, exc)

        # Transfer history
        # API: tmapi-alpha.transfermarkt.technology/transfer/history/player/{id}
        # Response: {"success":true,"data":{"history":{"terminated":[
        #   {"details":{"date":"2018-08-09T00:00:00+02:00","season":{"display":"18/19"},
        #    "fee":{"value":35000000},"marketValue":{"value":65000000}},
        #    "transferSource":{"clubId":"631","competitionId":"ES1"},
        #    "transferDestination":{"clubId":"418"}},
        #   ...
        # ]}}}
        # Note: API returns clubId not clubName. We store the IDs — they're sufficient
        # for the model and can be enriched later if needed.
        try:
            tf_resp = await page.request.get(
                f"https://tmapi-alpha.transfermarkt.technology/transfer/history/player/{tm_id}",
                headers={"Referer": url},
                timeout=15_000,
            )
            if tf_resp.ok:
                tf_json = await tf_resp.json()
                terminated = tf_json.get("data", {}).get("history", {}).get("terminated", [])
                for t in terminated:
                    details = t.get("details", {})
                    raw_date   = details.get("date", "")
                    season_obj = details.get("season", {})
                    fee_obj    = details.get("fee") or {}
                    src        = t.get("transferSource", {})
                    dst        = t.get("transferDestination", {})
                    # ISO date "2018-08-09T00:00:00+02:00" -> strip to date part
                    date_str = raw_date[:10] if raw_date else ""
                    player.transfers.append({
                        "date":          date_str,
                        "season":        season_obj.get("display", ""),
                        "from_club_id":  str(src.get("clubId", "")),
                        "to_club_id":    str(dst.get("clubId", "")),
                        "fee_eur":       int(fee_obj["value"]) if fee_obj.get("value") else None,
                    })
                log.debug("Transfers: %d entries for tm_id=%s", len(player.transfers), tm_id)
        except Exception as exc:
            log.debug("Transfer history error for tm_id=%s: %s", tm_id, exc)

    except Exception as exc:
        log.warning("Profile parse error for tm_id=%s: %s", tm_id, exc)

    return player

#  Main orchestration 

def _load_sofifa_stubs(seasons: list[str]) -> list[dict]:
    """Load player stubs from your existing SoFIFA JSON files."""
    stubs = []
    for season in seasons:
        path = DATA_DIR / f"players_{season}.json"
        if not path.exists():
            log.warning("SoFIFA data not found: %s", path)
            continue
        with open(path, encoding="utf-8") as f:
            players = json.load(f)
        for p in players:
            stubs.append({
                "sofifa_id":   p.get("sofifa_id"),
                "season_key":  p.get("season_key", season),
                "name":        p.get("name", ""),
                "club_name":   p.get("club_name", ""),
                "nationality": p.get("nationality", ""),
                "age":         p.get("age"),
                "overall_rating": p.get("overall_rating"),
                "market_value_eur": p.get("market_value_eur"),
            })
        log.info("Loaded %d players from %s", len(players), path.name)

    # Deduplicate by (sofifa_id, season_key) — same player can appear in multiple seasons
    seen = set()
    unique = []
    for s in stubs:
        key = (s["sofifa_id"], s["season_key"])
        if key not in seen:
            seen.add(key)
            unique.append(s)
    log.info("Total unique player-season stubs: %d", len(unique))
    return unique


def _load_existing_results() -> dict[str, dict]:
    """Load already-scraped TM data for resuming interrupted runs."""
    if not OUTPUT_FILE.exists():
        return {}
    with open(OUTPUT_FILE, encoding="utf-8") as f:
        data = json.load(f)
    return {str(r["sofifa_id"]): r for r in data}


def _save_results(results: list[dict]):
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


async def run_scraping(
    seasons: list[str],
    max_players: Optional[int] = None,
    resume: bool = True,
):
    stubs = _load_sofifa_stubs(seasons)
    if max_players:
        stubs = stubs[:max_players]

    existing = _load_existing_results() if resume else {}
    log.info("Resuming from %d already-scraped players", len(existing))

    results = list(existing.values())
    scraped_ids = set(existing.keys())

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        ctx = await _make_context(browser)
        page = await ctx.new_page()

        for i, stub in enumerate(stubs):
            sid = str(stub["sofifa_id"])
            if sid in scraped_ids:
                continue

            name = stub["name"]
            log.info("[%d/%d] Searching TM for: %s (SoFIFA ID: %s)",
                     i + 1, len(stubs), name, sid)

            _polite_delay()
            candidates = await scrape_search(page, name)

            record = {
                "sofifa_id":    stub["sofifa_id"],
                "season_key":   stub["season_key"],
                "sofifa_name":  name,
                "sofifa_club":  stub["club_name"],
                "sofifa_age":   stub["age"],
                "tm_candidates": candidates,
                "tm_matched":   None,  # filled by matcher.py
                "tm_data":      None,  # filled after profile scrape
            }

            # Auto-select best candidate if there's a clear match
            best = _pick_best_candidate(candidates, stub)
            if best:
                log.info("  → Auto-matched: %s (%s) val=%s",
                         best["tm_name"], best["club"],
                         best.get("value_eur"))
                _polite_delay()
                profile = await scrape_profile(page, best["tm_id"])
                if profile:
                    # If profile scrape didn't get the value, fall back to search result value
                    if profile.current_value_eur is None and best.get("value_eur"):
                        profile.current_value_eur = best["value_eur"]
                    record["tm_matched"] = best
                    record["tm_data"] = asdict(profile)
            else:
                log.info("  → No confident match found (%d candidates)", len(candidates))

            results.append(record)
            scraped_ids.add(sid)

            if len(results) % CHECKPOINT_EVERY == 0:
                _save_results(results)
                log.info("  Checkpoint saved (%d players)", len(results))

        await browser.close()

    _save_results(results)
    log.info("Done. Total records saved: %d → %s", len(results), OUTPUT_FILE)
    return results


def _pick_best_candidate(candidates: list[dict], stub: dict) -> Optional[dict]:
    """
    Simple heuristic pre-filter before the full fuzzy matcher runs.
    Returns a candidate only when confidence is very high (name + club match).
    The real matching is done by matcher.py with full fuzzy logic.
    """
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    stub_name = stub.get("name", "").lower()
    stub_club = stub.get("club_name", "").lower()
    stub_age  = stub.get("age")

    # Score each candidate
    best, best_score = None, 0
    for c in candidates:
        score = 0
        c_name = c.get("tm_name", "").lower()
        c_club = c.get("club", "").lower()
        c_age  = c.get("age")

        # Name similarity (simple token overlap)
        stub_tokens = set(stub_name.split())
        c_tokens    = set(c_name.split())
        overlap = len(stub_tokens & c_tokens) / max(len(stub_tokens), 1)
        score += overlap * 50

        # Club match
        if stub_club and c_club and (stub_club in c_club or c_club in stub_club):
            score += 30

        # Age match
        if stub_age and c_age and abs(stub_age - c_age) <= 1:
            score += 20

        if score > best_score:
            best_score = score
            best = c

    # Only auto-select if we're confident enough
    return best if best_score >= 60 else None


#  CLI 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SoccerSolver — Transfermarkt scraper")
    parser.add_argument("--seasons", nargs="+", default=["FC26", "FC25", "FC24"],
                        choices=["FC26", "FC25", "FC24"])
    parser.add_argument("--max-players", type=int, default=None,
                        help="Cap for testing, e.g. 50")
    parser.add_argument("--no-resume", action="store_true",
                        help="Start fresh even if tm_raw.json exists")
    args = parser.parse_args()

    asyncio.run(run_scraping(
        seasons=args.seasons,
        max_players=args.max_players,
        resume=not args.no_resume,
    ))