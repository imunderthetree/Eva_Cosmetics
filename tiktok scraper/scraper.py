import asyncio
import csv
from datetime import datetime
from TikTokApi import TikTokApi

# ── CONFIG ────────────────────────────────────────────────────────────────────
USERNAME   = "samaalwagih_"   # ← change this
MS_TOKEN   = "VK15Z5IR2lUyH4FWt895GrgtxOy58EePwPYdEQ3x9iQJltzkiBhDveobKq6pQNbKbRzwDKyPs2oiLqDB9v0aC4qeGeBbXWZ51HScQj4qAyCs7-GOY2DKoXsFjOVtsJkEv37xSKazpR42QQ=="     # ← paste your ms_token from browser cookies
POST_COUNT = 100                 # ← how many posts to fetch (max ~2000)
OUTPUT_CSV = f"{USERNAME}_tiktok_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
# ─────────────────────────────────────────────────────────────────────────────


async def scrape(username: str, ms_token: str, count: int, output: str):
    print(f"[+] Scraping @{username} — up to {count} posts...")

    rows = []

    async with TikTokApi() as api:
        await api.create_sessions(
            ms_tokens=[ms_token],
            num_sessions=1,
            sleep_after=3,          # be polite, avoid rate limits
            headless=True,
        )

        user = api.user(username)

        async for video in user.videos(count=count):
            data = video.as_dict
            stats = data.get("stats", {})

            row = {
                "views":       stats.get("playCount", 0),
                "likes":       stats.get("diggCount", 0),
                "created_at":  datetime.utcfromtimestamp(
                                   data.get("createTime", 0)
                               ).strftime("%Y-%m-%d %H:%M:%S"),
            }

            rows.append(row)
            print(f"  • [{len(rows)}] {row['created_at']}  |  "
                f"👁 {row['views']:,}  ❤️ {row['likes']:,}")

    if not rows:
        print("[-] No posts found.")
        return  

    fieldnames = ["views", "likes", "created_at"]

    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[✓] Saved {len(rows)} posts → {output}")


if __name__ == "__main__":
    asyncio.run(scrape(USERNAME, MS_TOKEN, POST_COUNT, OUTPUT_CSV))