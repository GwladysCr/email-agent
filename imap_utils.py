"""
imap_utils.py — IMAP connection, folder management, email parsing.
"""

import email
import email.header
import imaplib
from multiprocessing import context
import re
from email.utils import parseaddr
import socket
import ssl
from time import time

from bs4 import BeautifulSoup

CATEGORIES = ["travel", "bills", "jobs", "personal", "ads"]


# ── connection ────────────────────────────────────────────────────────────────

def connect(cfg: dict):
    
    ip = socket.gethostbyname(cfg["imap_host"])
    sock = socket.create_connection((ip, cfg["imap_port"]), timeout=10)
    context = ssl.create_default_context()
    ssl_sock = context.wrap_socket(sock, server_hostname=cfg["imap_host"])

    if cfg.get("imap_use_ssl", True):
        M = imaplib.IMAP4_SSL(cfg["imap_host"], int(cfg.get("imap_port", 993)))
    else:
        M = imaplib.IMAP4(cfg["imap_host"], int(cfg.get("imap_port", 143)))
    M.sock = ssl_sock
    M.login(cfg["imap_user"], cfg["imap_pass"])
    return M


def ensure_folders(M, folders: dict):
    """Create agent folders if they don't exist yet. Silently ignores existing."""
    for folder in folders.values():
        try:
            M.create(folder)
        except Exception:
            pass


def move_email(M, uid: str, target_folder: str, retries: int = 3):
    for attempt in range(retries):
        try:
            M.uid("copy", uid, target_folder)
            M.uid("store", uid, "+FLAGS", "\\Deleted")
            M.expunge()
            return True

        except (TimeoutError, socket.timeout) as e:
            print(f"  ⚠ IMAP timeout while moving email (attempt {attempt+1}/{retries})")

            if attempt < retries - 1:
                time.sleep(2)
            else:
                print(f"  ✗ Failed to move email to {target_folder}")
                return False

        except Exception as e:
            print(f"  ✗ IMAP move failed: {e}")
            return False


# ── fetching ──────────────────────────────────────────────────────────────────

def fetch_unseen(M, folder: str = "INBOX", limit: int = 30, lookback_days: int = 3) -> list:
    from datetime import datetime, timedelta
    M.select(folder)
    since = (datetime.now() - timedelta(days=lookback_days)).strftime("%d-%b-%Y")
    _, data = M.uid("search", None, f'(UNSEEN SINCE "{since}")')
    uids = data[0].split()[-limit:]
    return _fetch_uids(M, uids)


def fetch_folder_all(M, folder: str, limit: int = 60) -> list:
    try:
        rv, _ = M.select(folder)
        if rv != "OK":
            return []
    except Exception:
        return []
    _, data = M.uid("search", None, "ALL")
    uids = data[0].split()[-limit:]
    return _fetch_uids(M, uids)


def _fetch_uids(M, uids: list) -> list:
    results = []
    for uid in uids:
        try:
            _, raw = M.uid("fetch", uid, "(RFC822)")
            if raw[0] is None:
                continue
            msg = email.message_from_bytes(raw[0][1])
            results.append((uid.decode() if isinstance(uid, bytes) else uid, msg))
        except Exception:
            continue
    return results


# ── parsing ───────────────────────────────────────────────────────────────────

def decode_field(field: str) -> str:
    if not field:
        return ""
    parts = email.header.decode_header(field)
    out = []
    for part, charset in parts:
        if isinstance(part, bytes):
            out.append(part.decode(charset or "utf-8", errors="replace"))
        else:
            out.append(str(part))
    return " ".join(out)


def get_body(msg, max_chars: int = 600) -> str:
    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            ct = part.get_content_type()
            if ct == "text/plain":
                payload = part.get_payload(decode=True)
                if payload:
                    body = payload.decode(part.get_content_charset() or "utf-8", errors="replace")
                    break
            elif ct == "text/html" and not body:
                payload = part.get_payload(decode=True)
                if payload:
                    raw = payload.decode(part.get_content_charset() or "utf-8", errors="replace")
                    body = BeautifulSoup(raw, "html.parser").get_text(" ", strip=True)
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            body = payload.decode(msg.get_content_charset() or "utf-8", errors="replace")
    return body[:max_chars]


def extract_unsub_link(msg) -> str | None:
    """
    Look for unsubscribe URL in:
    1. List-Unsubscribe header (most reliable, RFC 2369)
    2. HTML body links containing unsubscribe keywords (FR + EN)
    """
    lu = msg.get("List-Unsubscribe", "")
    urls = re.findall(r"<(https?://[^>]+)>", lu)
    if urls:
        return urls[0]

    keywords = [
        "unsubscribe", "désabonner", "désinscrire", "desabonner",
        "se desinscrire", "opt-out", "optout", "désinscription",
    ]
    parts = msg.walk() if msg.is_multipart() else [msg]
    for part in parts:
        if part.get_content_type() == "text/html":
            payload = part.get_payload(decode=True)
            if not payload:
                continue
            html = payload.decode(part.get_content_charset() or "utf-8", errors="replace")
            soup = BeautifulSoup(html, "html.parser")
            for a in soup.find_all("a", href=True):
                href = a["href"]
                text = a.get_text().lower()
                if any(k in text for k in keywords) or any(k in href.lower() for k in keywords):
                    return href
    return None


def summarise(msg) -> dict:
    return {
        "from":    decode_field(msg.get("From", "")),
        "subject": decode_field(msg.get("Subject", "(no subject)")),
        "date":    msg.get("Date", ""),
        "body":    get_body(msg),
    }
