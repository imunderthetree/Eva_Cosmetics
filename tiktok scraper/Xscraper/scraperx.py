import asyncio
import csv
import os
from datetime import datetime
from twscrape import API

# ── CONFIG ────────────────────────────────────────────────────────────────────
USERNAME       = "joeundertree"
PASSWORD       = os.environ.get("X_PASSWORD", "Ronty@myheart4ever")  # ← your X password (required for login, but won't be used if cookies are valid)
EMAIL          = "sanoyeager@gmail.com"
EMAIL_PASSWORD = os.environ.get("X_EMAIL_PASSWORD", "Ronty4ever")

# Grab these from browser DevTools → Application → Cookies → https://x.com
AUTH_TOKEN     = os.environ.get("X_AUTH_TOKEN", "41670e04fd4dd9d2cb05efb1b128c14e3b2d1e4d")
CT0            = os.environ.get("X_CT0", "f880c4e57e8ed1e52b4be46dea8e7749a36902afc13f8ca3aa31da15c443a1673c585c277c8c2f29d7ba2929e07e74a408cf2e18f59a0a6c307fffc4793b5db40473dbbf71519c0175538670db4661f5")

QUERY = "evacosmatics"  # ← keep it simple, no operators
LIMIT          = 100              # ← how many tweets to fetch
OUTPUT_CSV     = f"x_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
# ─────────────────────────────────────────────────────────────────────────────


def safe_attr(obj, name, default=""):
    value = getattr(obj, name, default)
    return default if value is None else value


def format_date(value):
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def build_cookie_header(auth_token, ct0):
    if not auth_token or not ct0:
        return ""
    return f"auth_token={auth_token}; ct0={ct0}"


def tweet_to_row(tweet):
    user = safe_attr(tweet, "user", None)
    return {
        "id":         safe_attr(tweet, "id", ""),
        "created_at": format_date(safe_attr(tweet, "date", "")),
        "username":   safe_attr(user, "username", ""),
        "content":    safe_attr(tweet, "rawContent", ""),
        "views":      safe_attr(tweet, "viewCount", 0),
        "likes":      safe_attr(tweet, "likeCount", 0),
        "replies":    safe_attr(tweet, "replyCount", 0),
        "retweets":   safe_attr(tweet, "retweetCount", 0),
        "quotes":     safe_attr(tweet, "quoteCount", 0),
        "lang":       safe_attr(tweet, "lang", ""),
    }


def active_account_count(stats):
    try:
        return int(stats.get("active", 0))
    except (TypeError, ValueError):
        return 0


async def main():
    # Delete stale accounts.db if it exists to avoid the "already exists" warning
    if os.path.exists("accounts.db"):
        os.remove("accounts.db")
        print("[*] Cleared old accounts.db")

    api = API()

    # Add account with cookies (requires a valid logged-in session cookie)
    cookies = build_cookie_header(AUTH_TOKEN, CT0)
    await api.pool.add_account(
        username=USERNAME,
        password=PASSWORD,
        email=EMAIL,
        email_password=EMAIL_PASSWORD,
        cookies=cookies if cookies else None,
    )

    await api.pool.login_all()

    stats = await api.pool.stats()
    active_accounts = active_account_count(stats)
    total_accounts = stats.get("total", "?")
    if active_accounts == 0:
        print(
            "[-] No active X accounts after login "
            f"(active=0, total={total_accounts}). "
            "Fix the account login/cookies, then try again."
        )
        return

    print(f"[+] Active X accounts: {active_accounts}/{total_accounts}")
    print(f"[+] Searching for: {QUERY!r} (limit={LIMIT})")

    rows = []
    async for tweet in api.search(QUERY, limit=LIMIT):
        row = tweet_to_row(tweet)
        rows.append(row)
        snippet = row["content"].replace("\n", " ")[:80]
        print(
            f"  [{len(rows):>3}] @{row['username']:<20} | "
            f"👁 {row['views']:>8,} | ❤️ {row['likes']:>6,} | {snippet}"
        )

    if not rows:
        print("[-] No results found. Check your cookies or query.")
        return

    fieldnames = ["id", "created_at", "username", "content",
                  "views", "likes", "replies", "retweets", "quotes", "lang"]

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[✓] Saved {len(rows)} tweets → {OUTPUT_CSV}")


if __name__ == "__main__":
    asyncio.run(main())
