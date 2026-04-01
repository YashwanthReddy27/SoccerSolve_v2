"""
test_profile.py
===============
Run this to test scrape_profile() on a single player without scraping
the full dataset. Prints exactly what gets parsed so you can debug
selectors before committing to a full run.

Usage:
  python test_profile.py
  python test_profile.py --tm-id 105470      # Alisson
  python test_profile.py --tm-id 548111      # Alex Baena
"""
import asyncio
import argparse
import json
from dataclasses import asdict
from playwright.async_api import async_playwright
from tm_scraper import _make_context, scrape_profile, scrape_search

async def test_single_profile(tm_id: str):
    print(f"\nTesting profile scrape for tm_id={tm_id}")
    print("=" * 60)
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=False)  # headless=False so you can watch
        ctx = await _make_context(browser)
        page = await ctx.new_page()

        result = await scrape_profile(page, tm_id)
        if result:
            data = asdict(result)
            print(f"tm_name:           {data['tm_name']}")
            print(f"nationality:       {data['nationality']}")
            print(f"age:               {data['age']}")
            print(f"dob:               {data['dob']}")
            print(f"position:          {data['position']}")
            print(f"current_club:      {data['current_club']}")
            print(f"current_value_eur: {data['current_value_eur']}")
            print(f"value_history:     {len(data['value_history'])} entries")
            if data['value_history']:
                print(f"  first: {data['value_history'][0]}")
                print(f"  last:  {data['value_history'][-1]}")
            print(f"transfers:         {len(data['transfers'])} entries")
            if data['transfers']:
                for t in data['transfers'][:3]:
                    print(f"  {t}")
        else:
            print("ERROR: scrape_profile returned None")

        # Also dump the raw page HTML for the market value section so you
        # can inspect what selectors are actually present
        print("\n--- Market value section HTML ---")
        mv_html = await page.evaluate("""
            () => {
                const el = document.querySelector('a.data-header__market-value-wrapper');
                return el ? el.outerHTML : 'NOT FOUND';
            }
        """)
        print(mv_html[:500])

        print("\n--- Value history API (raw JSON first 500 chars) ---")
        profile_url = f"https://www.transfermarkt.com/x/profil/spieler/{tm_id}"
        try:
            mw_resp = await page.request.get(
                f"https://tmapi-alpha.transfermarkt.technology/player/{tm_id}/market-value-history",
                headers={"Referer": profile_url},
                timeout=15_000,
            )
            print(f"Status: {mw_resp.status}")
            raw = await mw_resp.text()
            print(raw[:500])
        except Exception as e:
            print(f"Error: {e}")

        print("\n--- Transfer history API (raw JSON first 500 chars) ---")
        try:
            tf_resp = await page.request.get(
                f"https://tmapi-alpha.transfermarkt.technology/transfer/history/player/{tm_id}",
                headers={"Referer": profile_url},
                timeout=15_000,
            )
            print(f"Status: {tf_resp.status}")
            raw = await tf_resp.text()
            print(raw[:500])
        except Exception as e:
            print(f"Error: {e}")

        await browser.close()

async def test_search(name: str):
    print(f"\nTesting search for: {name}")
    print("=" * 60)
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=False)
        ctx = await _make_context(browser)
        page = await ctx.new_page()
        candidates = await scrape_search(page, name)
        print(f"Found {len(candidates)} candidates:")
        for c in candidates:
            print(f"  {c}")
        await browser.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tm-id", default="548111", help="TM player ID to test")
    parser.add_argument("--search", default="", help="Name to test search for")
    args = parser.parse_args()

    if args.search:
        asyncio.run(test_search(args.search))
    else:
        asyncio.run(test_single_profile(args.tm_id))