"""
unsub_agent.py — Brand identification, decision memory, unsubscribe execution.

Claude API is still used here (only) for brand name extraction from an email —
a small, cheap call (~80 tokens) that avoids brittle regex on sender strings.
It is entirely optional: if no API key is set, we fall back to domain parsing.
"""

import json
import re
import urllib.parse
from datetime import datetime
from pathlib import Path

import requests
from bs4 import BeautifulSoup

try:
    import anthropic
    CLAUDE_OK = True
except ImportError:
    CLAUDE_OK = False


# ── brand identification ──────────────────────────────────────────────────────

_BRAND_SYSTEM = """
You extract the brand name from a marketing email.
Reply with ONLY a JSON object, no markdown, no explanation.
Format: {"brand": "<short recognisable brand name, e.g. Decathlon, Netflix, Le Monde>"}
""".strip()


def identify_brand(summary: dict, api_key: str | None = None) -> str:
    """Return a short brand name. Falls back to sender domain if no API key."""
    if api_key and CLAUDE_OK:
        try:
            client = anthropic.Anthropic(api_key=api_key)
            resp = client.messages.create(
                model      = "claude-haiku-4-5-20251001",   # cheapest model — tiny task
                max_tokens = 60,
                system     = _BRAND_SYSTEM,
                messages   = [{"role": "user", "content": json.dumps({
                    "from": summary.get("from", ""),
                    "subject": summary.get("subject", ""),
                })}],
            )
            raw = resp.content[0].text.strip()
            return json.loads(raw).get("brand", _brand_from_domain(summary))
        except Exception:
            pass
    return _brand_from_domain(summary)


def _brand_from_domain(summary: dict) -> str:
    """Heuristic: take the first meaningful part of the sender domain."""
    sender = summary.get("from", "")
    m = re.search(r"@([\w.\-]+)", sender)
    if not m:
        return "Unknown"
    domain = m.group(1).lower()
    # Strip common prefixes like mail., email., noreply., info., newsletter.
    parts = domain.split(".")
    skip = {"mail", "email", "noreply", "no-reply", "info",
            "newsletter", "news", "promo", "offers", "hello", "bonjour"}
    meaningful = [p for p in parts[:-1] if p not in skip]   # drop TLD
    return meaningful[0].capitalize() if meaningful else parts[0].capitalize()


# decisions database

class BrandMemory:
    """
    Persists per-brand decisions to JSON.
    Schema: { "brand_lower": { "action": str, "reclassify_as": str|null, "recorded_at": str } }
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self._db: dict = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            try:
                return json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def _save(self):
        self.path.write_text(json.dumps(self._db, indent=2, ensure_ascii=False), encoding="utf-8")

    def get(self, brand: str) -> dict | None:
        return self._db.get(brand.lower())

    def store(self, brand: str, action: str, reclassify_as: str | None = None):
        self._db[brand.lower()] = {
            "action":        action,
            "reclassify_as": reclassify_as,
            "recorded_at":   datetime.now().isoformat(),
        }
        self._save()

    def forget(self, brand: str):
        self._db.pop(brand.lower(), None)
        self._save()

    def all_brands(self) -> list[tuple[str, dict]]:
        return sorted(self._db.items(), key=lambda x: x[1].get("recorded_at", ""), reverse=True)


# ── unsubscribe execution ─────────────────────────────────────────────────────

def perform_unsub(url: str, live: bool = True) -> tuple[bool, str]:
    """
    Open the unsubscribe URL and attempt to confirm it.

    Returns (success: bool, message: str).

    Strategy:
      1. GET the URL.
      2. If the response contains a <form>, POST it with its default values
         (most one-click unsub pages just need the GET; the form path handles
         the minority that require a second click like "confirm unsubscribe").
      3. Return True if both requests came back < 400.
    """
    if not live:
        return True, f"[DRY RUN] Would open: {url}"

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) "
            "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 "
            "Mobile/15E148 Safari/604.1"
        ),
        "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8",
    }
    try:
        r = requests.get(url, headers=headers, timeout=15, allow_redirects=True)
        if r.status_code >= 400:
            return False, f"HTTP {r.status_code} on GET"

        soup = BeautifulSoup(r.text, "html.parser")
        form = soup.find("form")
        if form:
            action = form.get("action", url)
            if action and not action.startswith("http"):
                base = urllib.parse.urlparse(r.url)
                action = f"{base.scheme}://{base.netloc}{action}"
            data = {}
            for inp in form.find_all("input"):
                name = inp.get("name")
                if name:
                    data[name] = inp.get("value", "")
            for btn in form.find_all(["button", "input"]):
                if btn.get("type") in ("submit", None) and btn.get("name"):
                    data[btn["name"]] = btn.get("value", "")
            r2 = requests.post(action or url, data=data, headers=headers, timeout=15)
            if r2.status_code >= 400:
                return False, f"Form POST returned HTTP {r2.status_code}"

        return True, "Unsubscribed successfully"

    except requests.exceptions.Timeout:
        return False, "Request timed out"
    except Exception as e:
        return False, str(e)


# ── user interaction (CLI / Pythonista) ───────────────────────────────────────

def ask_user(brand: str, subject: str) -> tuple[str, str | None]:
    """
    Ask the user what to do with a new brand.

    Returns (action, reclassify_as):
      action ∈ {"unsubscribe", "keep", "reclassify"}
      reclassify_as ∈ CATEGORIES | None
    """
    try:
        import console  # Pythonista on iPhone
        choice = console.alert(
            f"📢  {brand}",
            f"{subject[:80]}",
            "🚫 Unsubscribe",    # 1
            "✅ Keep in Ads",    # 2
            "📁 Reclassify…",   # 3
        )
        if choice == 1:
            return "unsubscribe", None
        if choice == 2:
            return "keep", None
        # Reclassify sub-prompt
        sub = console.alert(
            "Move to which folder?",
            f"Future emails from {brand} will go here.",
            "✈ Travel",     # 1
            "⚡ Bills",      # 2
            "💼 Jobs",       # 3
            "👤 Personal",   # 4
        )
        cats = {1: "travel", 2: "bills", 3: "jobs", 4: "personal"}
        return "reclassify", cats.get(sub, "personal")

    except ImportError:
        # Standard terminal fallback
        print(f"\n  ┌─ New brand: {brand}")
        print(f"  │  Subject  : {subject[:70]}")
        print(f"  │  [1] Unsubscribe  [2] Keep in Ads  [3] Reclassify")
        c = input("  └▶ Choice: ").strip()
        if c == "1":
            return "unsubscribe", None
        if c == "3":
            print("  Reclassify to: [1] travel  [2] bills  [3] jobs  [4] personal")
            s = input("  ▶ ").strip()
            cats = {"1": "travel", "2": "bills", "3": "jobs", "4": "personal"}
            return "reclassify", cats.get(s, "personal")
        return "keep", None
