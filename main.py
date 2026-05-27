"""
main.py — Email Agent orchestrator

Usage (in a-Shell on iPhone, or any terminal):
    python3 main.py                  # run both agents
    python3 main.py classify         # classifier only
    python3 main.py unsub            # unsubscribe agent only
    python3 main.py status           # print model status
    python3 main.py forget Decathlon # reset a brand decision

Configuration: edit CONFIG below or set the corresponding env vars.
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from classifier  import ActiveLearningClassifier, CATEGORIES
from imap_utils  import (connect, ensure_folders, fetch_unseen,
                          fetch_folder_all, move_email,
                          extract_unsub_link, summarise)
from unsub_agent import identify_brand, BrandMemory, perform_unsub, ask_user

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent.parent  # email-agent

CONFIG = {
    # IMAP
    "imap_host":    os.getenv("IMAP_HOST",    "imap.example.com"),
    "imap_port":    int(os.getenv("IMAP_PORT", "993")),
    "imap_user":    os.getenv("IMAP_USER",    "you@example.com"),
    "imap_pass":    os.getenv("IMAP_PASS",    "yourpassword"),
    "imap_use_ssl": True,

    # IMAP folder names 
    "folders": {
        "travel":   "Agent/Travel",
        "bills":    "Agent/Bills",
        "jobs":     "Agent/Jobs",
        "personal": "Agent/Personal",
        "ads":      "Agent/Ads",
    },

    "fetch_limit":    10,
    "lookback_days":  3,

    # Set True to actually open unsubscribe links.
    # Leave False for a dry run while you're testing.
    "live_unsub": os.getenv("LIVE_UNSUB", "false").lower() == "true",

    # Data directory
    "data_dir": BASE_DIR / "data",
}


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_conf(conf: float) -> str:
    bar_len = 12
    filled  = round(conf * bar_len)
    bar     = "█" * filled + "░" * (bar_len - filled)
    return f"[{bar}] {conf:.0%}"


def _ask_label(summary: dict, prediction: dict) -> str:
    """
    Shows the model's current best guess
    user can confirm or correct with a single keypress.
    """
    cat  = prediction.get("category", "unknown")
    conf = prediction.get("confidence", 0.0)
    zone = prediction.get("zone", 1)

    print(f"\n  ┌─ From   : {summary['from'][:65]}")
    print(f"  │  Subject: {summary['subject'][:65]}")
    if zone > 1:
        print(f"  │  Guess  : {cat} {_fmt_conf(conf)}  ← {prediction.get('reason','')}")
    else:
        print(f"  │  (no model yet — please label this email)")
    print(f"  │")
    print(f"  │  [1] travel  [2] bills  [3] jobs  [4] personal  [5] ads")
    if zone > 1:
        print(f"  │  [↵ Enter] confirm guess ({cat})")
    raw = input("  └▶ ").strip().lower()

    mapping = {"1": "travel", "2": "bills", "3": "jobs", "4": "personal", "5": "ads"}
    if raw == "" and zone > 1:
        return cat          # confirmed guess
    if raw in mapping:
        return mapping[raw]
    # If they typed a word
    for c in CATEGORIES:
        if raw.startswith(c[0]):
            return c
    return cat if zone > 1 else "personal"


# ─────────────────────────────────────────────────────────────────────────────
#  AGENT 1 — CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────

def run_classifier():
    print("\n" + "═" * 55)
    print("  AGENT 1 — CLASSIFIER")
    print("═" * 55)

    clf = ActiveLearningClassifier(CONFIG["data_dir"])

    # Print model status
    report = clf.status_report()
    print(report["summary"])
    print()

    M = connect(CONFIG)
    ensure_folders(M, CONFIG["folders"])
    emails = fetch_unseen(
        M,
        limit        = CONFIG["fetch_limit"],
        lookback_days = CONFIG["lookback_days"],
    )

    if not emails:
        print("  No new emails to process.")
        M.logout()
        return

    print(f"  Found {len(emails)} new email(s).\n")

    asked = 0
    auto  = 0

    for uid, msg in emails:
        summary = summarise(msg)
        prediction = clf.predict(summary)
        zone = prediction["zone"]
        category = prediction["category"]

        if prediction["should_ask"]:
            # ask the user 
            label = _ask_label(summary, prediction)
            was_correction = (zone > 1 and label != category)

            clf.add_label(summary, label, was_correction=was_correction)
            clf.record_asked()
            category = label
            asked += 1

            status = "corrected ✎" if was_correction else "confirmed ✓"
            print(f"  → [{category.upper():8s}] {status}  {summary['subject'][:50]}")

        else:
            # ── act silently ──────────────────────────────────────────────
            clf.record_auto_decision(category)
            auto += 1
            print(f"  → [{category.upper():8s}] auto {_fmt_conf(prediction['confidence'])}  "
                  f"{summary['subject'][:45]}")

        # Move email 
        target = CONFIG["folders"].get(category, CONFIG["folders"]["personal"])
        move_email(M, uid, target)

    M.close()
    M.logout()

    print(f"\n  Done. Auto: {auto}  Asked: {asked}  Total: {len(emails)}")

    # Retrain summary after this batch
    if asked > 0:
        report = clf.status_report()
        print(f"\n  Model update after labelling:")
        print(report["summary"])


# ─────────────────────────────────────────────────────────────────────────────
#  AGENT 2 — UNSUBSCRIBE
# ─────────────────────────────────────────────────────────────────────────────

def run_unsub_agent():
    print("\n" + "═" * 55)
    print("  AGENT 2 — UNSUBSCRIBE AGENT")
    print("═" * 55)

    memory   = BrandMemory(CONFIG["data_dir"] / "brand_decisions.json")
    live     = CONFIG.get("live_unsub", False)

    M       = connect(CONFIG)
    ads_folder = CONFIG["folders"]["ads"]
    emails  = fetch_folder_all(M, ads_folder)

    if not emails:
        print("  Ads folder is empty.")
        M.logout()
        return

    print(f"  {len(emails)} email(s) in Ads folder.\n")

    for uid, msg in emails:
        summary = summarise(msg)
        brand   = identify_brand(summary, api_key or None)

        print(f"  Brand   : {brand}")
        print(f"  Subject : {summary['subject'][:65]}")

        # ── check memory ──────────────────────────────────────────────────
        stored = memory.get(brand)
        if stored:
            action         = stored["action"]
            reclassify_as  = stored.get("reclassify_as")
            print(f"  Memory  : {action}" +
                  (f" → {reclassify_as}" if reclassify_as else "") +
                  "  (remembered)")
        else:
            # ── ask user ──────────────────────────────────────────────────
            action, reclassify_as = ask_user(brand, summary["subject"])
            memory.store(brand, action, reclassify_as)

        # ── execute ───────────────────────────────────────────────────────
        M.select(ads_folder)

        if action == "unsubscribe":
            unsub_url = extract_unsub_link(msg)
            if unsub_url:
                ok, msg_out = perform_unsub(unsub_url, live=live)
                print(f"  Result  : {'✓' if ok else '⚠'}  {msg_out}")
                if ok:
                    # Delete email after successful unsub
                    M.uid("store", uid, "+FLAGS", "\\Deleted")
                    M.expunge()
            else:
                print(f"  ⚠  No unsubscribe link found. "
                      f"Consider reporting to CNIL: signal.cnil.fr")

        elif action == "reclassify":
            target = CONFIG["folders"].get(reclassify_as, CONFIG["folders"]["personal"])
            move_email(M, uid, target)
            print(f"  Moved   : → {target}")

            # Also teach the classifier this decision
            clf = ActiveLearningClassifier(CONFIG["data_dir"])
            clf.add_label(summary, reclassify_as)
            print(f"  Classifier updated with this label.")

        else:  # keep
            print(f"  Kept in Ads.")

        print()
        
    M.expunge()
    M.close()
    M.logout()
    print("  Unsubscribe agent finished.")


# ─────────────────────────────────────────────────────────────────────────────
#  STATUS / ADMIN COMMANDS
# ─────────────────────────────────────────────────────────────────────────────

def run_status():
    clf    = ActiveLearningClassifier(CONFIG["data_dir"])
    memory = BrandMemory(CONFIG["data_dir"] / "brand_decisions.json")

    print("\n" + "═" * 55)
    print("  EMAIL AGENT — STATUS")
    print("═" * 55)
    report = clf.status_report()
    print(report["summary"])

    brands = memory.all_brands()
    if brands:
        print(f"\n  Remembered brand decisions ({len(brands)}):")
        for brand, d in brands[:20]:
            action = d["action"]
            extra  = f" → {d['reclassify_as']}" if d.get("reclassify_as") else ""
            print(f"    {brand.capitalize():25s}  {action}{extra}")
    print()


def run_forget(brand: str):
    memory = BrandMemory(CONFIG["data_dir"] / "brand_decisions.json")
    memory.forget(brand)
    print(f"  Forgot decision for '{brand}'.")


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = sys.argv[1:]
    cmd  = args[0].lower() if args else "all"

    if cmd == "classify":
        run_classifier()
    elif cmd == "unsub":
        run_unsub_agent()
    elif cmd == "status":
        run_status()
    elif cmd == "forget" and len(args) > 1:
        run_forget(" ".join(args[1:]))
    elif cmd == "all":
        run_classifier()
        run_unsub_agent()
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
