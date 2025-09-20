# school_board_poc.py — School email → activities POC
# Features:
# - IMAP fetch with modes (UNSEEN/SINCE/ALL) and PEEK (does not mark as read)
# - Sidebar filters for school domains/senders/keywords
# - Optional Gemini LLM extraction for structured events + urgency scoring
# - Urgency badges (🔴/🟡/🟢), .ics export, and "Add to Google Calendar" links
# - Live monitor: auto-refresh every N seconds

import os
import re
import json
import imaplib
import email
from urllib.parse import urlencode
from email.header import decode_header
from datetime import datetime, timedelta

import streamlit as st
import pandas as pd
import dateparser
from icalendar import Calendar, Event
from dotenv import load_dotenv
from streamlit_autorefresh import st_autorefresh

# --- Optional LLM (Gemini) ---
try:
    import google.generativeai as genai
except Exception:
    genai = None

load_dotenv()

# ---- Config from .env ----
IMAP_HOST = os.getenv("IMAP_HOST", "imap.gmail.com")
IMAP_USER = os.getenv("IMAP_USER", "")
IMAP_PASS = os.getenv("IMAP_PASS", "")
MAILBOX    = os.getenv("MAILBOX", "INBOX")
MAX_EMAILS = int(os.getenv("MAX_EMAILS", "80"))

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
if GEMINI_API_KEY and genai is not None:
    genai.configure(api_key=GEMINI_API_KEY)

# ---- School email filter defaults ----
DEFAULT_SCHOOL_DOMAINS = " "
DEFAULT_SCHOOL_SENDERS = ""  # e.g. "teacher@myschool.org, principal@austinisd.org"
DEFAULT_KEYWORDS       = "school, pta, teacher, classroom, homeroom, field trip, permission slip, bus, assembly, cafeteria, counselor, principal, due, forms"
DEFAULT_NEGATIVE       = "unsubscribe, terms, privacy policy, marketing, promotion, sale, newsletter, invoice, receipt"

# ---------- UI: Title ----------
st.title("Simple School Activity POC")
st.write("Reads emails, finds dates/tasks, and lets you add events to your calendar.")

# ---------- IMAP fetch ----------
@st.cache_data(ttl=5)  # short TTL helps Live Monitor feel responsive
def fetch_emails(search_mode="ALL", since_days=7):
    """
    search_mode: "ALL" | "UNSEEN" | "SINCE"
    since_days: used when search_mode == "SINCE"
    Returns: {"emails": [...], "uids": [...]}
    """
    if not IMAP_USER or not IMAP_PASS:
        return {"error": "Set IMAP_USER and IMAP_PASS in .env"}

    try:
        import datetime as dt
        mail = imaplib.IMAP4_SSL(IMAP_HOST)
        mail.login(IMAP_USER, IMAP_PASS)
        mail.select(MAILBOX)

        if search_mode == "UNSEEN":
            typ, data = mail.search(None, "UNSEEN")
        elif search_mode == "SINCE":
            date_str = (dt.date.today() - dt.timedelta(days=int(since_days))).strftime("%d-%b-%Y")
            typ, data = mail.search(None, f'(SINCE {date_str})')
        else:
            typ, data = mail.search(None, "ALL")

        if typ != "OK":
            mail.logout()
            return {"emails": [], "uids": []}

        ids = data[0].split()
        ids = ids[-MAX_EMAILS:]  # most recent N
        emails, uids = [], []

        for num in reversed(ids):
            # PEEK avoids setting the \Seen flag
            typ, msg_data = mail.fetch(num, '(BODY.PEEK[])')
            if typ != "OK":
                continue
            msg = email.message_from_bytes(msg_data[0][1])

            subject, encoding = decode_header(msg.get("Subject"))[0]
            if isinstance(subject, bytes):
                subject = subject.decode(encoding or "utf-8", errors="ignore")
            from_ = msg.get("From")
            date_ = msg.get("Date")

            body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    ctype = part.get_content_type()
                    cdisp = str(part.get("Content-Disposition"))
                    if ctype == "text/plain" and "attachment" not in cdisp:
                        try:
                            body = part.get_payload(decode=True).decode(part.get_content_charset() or "utf-8", errors="ignore")
                        except Exception:
                            body = str(part.get_payload())
                        break
            else:
                payload = msg.get_payload(decode=True)
                if isinstance(payload, bytes):
                    try:
                        body = payload.decode(msg.get_content_charset() or "utf-8", errors="ignore")
                    except Exception:
                        body = ""
                else:
                    body = str(payload or "")

            emails.append({
                "subject": subject or "",
                "from": from_ or "",
                "date": date_ or "",
                "body": (body or "")[:20000]  # keep generous snippet for parsing
            })
            uids.append(num.decode() if isinstance(num, bytes) else str(num))

        mail.logout()
        return {"emails": emails, "uids": uids}
    except Exception as e:
        return {"error": str(e)}

# ---------- Regex/date parsing helpers ----------
DATE_PATTERNS = [
    r"\b(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b[^\n,]*",
    r"\b(?:tomorrow|today|next week|next month|next|tonight)\b",
    r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b",
    r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[^\n,0-9]{0,6}\d{1,2}(?:st|nd|rd|th)?(?:,?\s*\d{4})?",
    r"\b(?:\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec))\b"
]

def find_possible_dates(text):
    matches = []
    for pat in DATE_PATTERNS:
        for m in re.finditer(pat, text, re.IGNORECASE):
            matches.append(m.group().strip())
    # unique in order
    seen, out = set(), []
    for x in matches:
        xl = x.lower()
        if xl not in seen:
            out.append(x); seen.add(xl)
    return out

def extract_events_from_email(email_item):
    """Heuristic fallback parser (no LLM)."""
    body = email_item.get("body") or ""
    subject = email_item.get("subject") or "School event"
    candidates = []
    for line in body.splitlines():
        line = line.strip()
        if not line:
            continue
        found = find_possible_dates(line)
        if found:
            candidates.append((line, found))
    if not candidates:
        found = find_possible_dates(subject)
        if found:
            candidates.append((subject, found))
    events = []
    for line, found_dates in candidates:
        for fd in found_dates:
            dt = dateparser.parse(fd, settings={"PREFER_DATES_FROM": "future"})
            if not dt:
                dt = dateparser.parse(line, settings={"PREFER_DATES_FROM": "future"})
            if dt:
                title = re.sub(re.escape(fd), "", line, flags=re.IGNORECASE).strip(" -:,.")
                if not title:
                    title = subject
                start = dt
                if start.hour == 0 and start.minute == 0:
                    start = start.replace(hour=10, minute=0)
                end = start + timedelta(hours=1)
                events.append({
                    "title": title,
                    "start": start,
                    "end": end,
                    "type": "event",
                    "location": "",
                    "notes": "",
                    "urgency_score": 0,
                    "source_subject": subject,
                    "source_from": email_item.get("from"),
                    "raw_line": line
                })
    return events

# ---------- School filter helpers ----------
def _norm_split(s):
    return [x.strip().lower() for x in s.split(",") if x.strip()]

def looks_like_school(email_item, domain_str, sender_str, kw_str, neg_str):
    from_raw = (email_item.get("from") or "").lower()
    subject  = (email_item.get("subject") or "").lower()
    body     = (email_item.get("body") or "").lower()

    domains  = _norm_split(domain_str)
    senders  = _norm_split(sender_str)
    must_kws = _norm_split(kw_str)
    neg_kws  = _norm_split(neg_str)

    m = re.search(r'[\w\.-]+@([\w\.-]+)', from_raw)
    dom  = m.group(1) if m else ""
    addr = m.group(0) if m else from_raw

    # 1) Exclude by negative keywords
    for bad in neg_kws:
        if bad and (bad in subject or bad in body):
            return False

    # 2) Keyword test
    text = subject + " " + body
    kw_ok = any(kw in text for kw in must_kws) if must_kws else True

    # 3) Sender tests (domain or exact)
    sender_ok = False
    if addr and addr in senders:
        sender_ok = True
    if dom:
        for d in domains:
            if d and dom.endswith(d):
                sender_ok = True
                break

    # 4) Special: forwarded by me + keywords -> allow
    my_addr = (IMAP_USER or "").lower()
    if my_addr and my_addr in from_raw and kw_ok:
        return True

    if domains or senders:
        return sender_ok and kw_ok
    return kw_ok

# ---------- LLM extractor (Gemini) ----------
def gemini_extract_events(subject: str, body: str):
    """Returns list of events dicts or [] on failure."""
    if not (GEMINI_API_KEY and genai):
        return []

    prompt = f"""
You are an assistant that extracts school-related calendar items from email text.

Return ONLY valid JSON with this shape and no commentary:
{{
  "events":[
    {{
      "title": "short title",
      "start": "YYYY-MM-DD HH:MM",   // 24h local time; if only a date, set 10:00
      "end":   "YYYY-MM-DD HH:MM",   // default start+60m if missing
      "type":  "event" | "deadline" | "reminder",
      "location": "optional",
      "notes": "optional",
      "urgency_score": 0             // integer 0..100 (100 = most urgent)
    }}
  ]
}}

Rules:
- If multiple items exist (bake sale + parent night), include each in "events".
- Prefer FUTURE dates if ambiguous.
- If time missing, use 10:00 and set end to +60 minutes.
- Urgency: due ≤3 days or forms → 80–100; due ≤7 days → 60–79; otherwise ≤59.
- If nothing relevant, return {{"events":[]}} only.

Email Subject: {subject}
Email Body:
{body}
"""

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        resp  = model.generate_content(prompt)
        text  = (resp.text or "").strip()

        # Strip code fences or leading "json:"
        text = text.strip("` \n")
        if text.lower().startswith("json"):
            text = text[4:].lstrip(": \n")

        data = json.loads(text)
        raw_events = data.get("events", [])
        out = []
        for ev in raw_events:
            title  = ev.get("title") or "School event"
            start_s = ev.get("start")
            end_s   = ev.get("end")

            dt_start = dateparser.parse(start_s, settings={"PREFER_DATES_FROM": "future"}) if start_s else None
            if not dt_start:
                now = datetime.now()
                dt_start = now.replace(hour=10, minute=0, second=0, microsecond=0)

            dt_end = dateparser.parse(end_s, settings={"PREFER_DATES_FROM": "future"}) if end_s else None
            if not dt_end or dt_end <= dt_start:
                dt_end = dt_start + timedelta(hours=1)

            out.append({
                "title": title,
                "start": dt_start,
                "end": dt_end,
                "type": ev.get("type") or "event",
                "location": ev.get("location") or "",
                "notes": ev.get("notes") or "",
                "urgency_score": int(ev.get("urgency_score") or 0),
            })
        return out
    except Exception:
        return []

# ---------- Urgency badge ----------
def badge_for_urgency(score: int) -> str:
    try:
        s = int(score)
    except Exception:
        s = 0
    if s >= 80: return "🔴 High"
    if s >= 60: return "🟡 Medium"
    return "🟢 Low"

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Email connection")
    st.write(f"Host: {IMAP_HOST}")
    st.write(f"User: {IMAP_USER or '(not set)'}")

    st.divider()
    st.header("Live monitor")
    live_mode  = st.toggle("Auto-refresh for new emails", value=False)
    interval   = st.select_slider("Refresh every (seconds)", options=[10, 15, 20, 30, 45, 60], value=30)
    fetch_mode = st.selectbox("Fetch scope", ["UNSEEN", "SINCE", "ALL"], help="UNSEEN = unread only; SINCE = last N days; ALL = everything")
    since_days = st.number_input("SINCE: days back", min_value=1, max_value=30, value=7, step=1)

    st.divider()
    st.header("School filters")
    domain_str = st.text_input("Sender domains (comma-separated)", value=DEFAULT_SCHOOL_DOMAINS,
                               help="Domain part of From address. Example: austinisd.org")
    sender_str = st.text_input("Exact sender emails (comma-separated)", value=DEFAULT_SCHOOL_SENDERS,
                               help="teacher@myschool.org, principal@district.k12.tx.us")
    kw_str     = st.text_input("Must contain keywords (comma-separated)", value=DEFAULT_KEYWORDS,
                               help="Searched in subject and body")
    neg_str    = st.text_input("Exclude if contains (comma-separated)", value=DEFAULT_NEGATIVE,
                               help="Use to drop marketing/newsletters")

    st.divider()
    use_gemini = st.toggle("Use Gemini extraction (LLM)", value=bool(GEMINI_API_KEY),
                           help="Requires GEMINI_API_KEY in .env")

# Kick timed refresh if live mode is on
if live_mode:
    st_autorefresh(interval=interval * 1000, key="auto_refresh_tick")

# ---------- Fetch emails ----------
result = fetch_emails(search_mode=fetch_mode, since_days=int(since_days))
if "error" in result:
    st.error("Email read error: " + result["error"])
    st.stop()

emails = result.get("emails", [])
uids   = result.get("uids", [])
st.caption(f"Fetched {len(emails)} email(s) with mode {fetch_mode}.")

# Simple "new mail" toast
if "last_uids" not in st.session_state:
    st.session_state["last_uids"] = set(uids)
else:
    new = [u for u in uids if u not in st.session_state["last_uids"]]
    if new:
        st.toast(f"📬 {len(new)} new email(s) fetched", icon="✉️")
        st.session_state["last_uids"].update(new)

# ---------- Filter emails BEFORE parsing ----------
filtered = [e for e in emails if looks_like_school(e, domain_str, sender_str, kw_str, neg_str)]
st.caption(f"After filtering: {len(filtered)} email(s) remain.")
emails = filtered

# ---------- Optional preview ----------
if st.button("Preview raw emails (subject + snippet)"):
    for e in emails[:10]:
        st.markdown(f"**{e.get('subject','')}** — {e.get('from','')}")
        st.code((e.get('body') or "")[:600])

# ---------- Build events (Gemini first, regex fallback) ----------
all_events = []
for e in emails:
    subj = e.get("subject") or ""
    body = e.get("body") or ""
    evs = []
    if use_gemini and GEMINI_API_KEY and genai:
        evs = gemini_extract_events(subj, body)
    if not evs:
        evs = extract_events_from_email(e)

    for ev in evs:
        # ensure common fields
        ev.setdefault("type", "event")
        ev.setdefault("location", "")
        ev.setdefault("notes", "")
        ev.setdefault("urgency_score", 0)
        ev["source_from"]    = e.get("from")
        ev["source_subject"] = subj
        ev["raw_line"]       = ev.get("raw_line") or subj
        all_events.append(ev)

# ---------- Show table ----------
if not all_events:
    st.info("No events detected with current filters. Try previewing emails or loosening filters.")
else:
    # Sort options
    sort_choice = st.radio("Sort by:", ["Urgency (High→Low)", "Start time (Soonest first)"], horizontal=True)

    rows = []
    for i, ev in enumerate(all_events):
        score = int(ev.get("urgency_score", 0))
        rows.append({
            "id": i + 1,
            "title": ev.get("title", "School event"),
            "type": ev.get("type", "event"),
            "urgency": badge_for_urgency(score),
            "urgency_score": score,  # numeric helper for sorting
            "start": ev["start"].strftime("%Y-%m-%d %H:%M"),
            "end": ev["end"].strftime("%Y-%m-%d %H:%M"),
            "location": ev.get("location", ""),
            "from": ev.get("source_from", ""),
            "note": ev.get("notes") or ev.get("raw_line", "")
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        if sort_choice.startswith("Urgency"):
            df = df.sort_values(by=["urgency_score", "start"], ascending=[False, True]).reset_index(drop=True)
        else:
            df = df.sort_values(by="start", ascending=True).reset_index(drop=True)

    st.subheader("Detected activities")
    st.dataframe(
        df[["id", "title", "type", "urgency", "start", "end", "location", "from", "note"]],
        use_container_width=True
    )

    # ---------- Selection + export ----------
    st.subheader("Select events to export or add")

    options = df["id"].tolist()
    select_all = st.checkbox("Select all visible", value=True if len(options) <= 10 else False)
    default_ids = options if select_all else []

    selected = st.multiselect(
        "Pick event ids",
        options=options,
        default=default_ids,
        format_func=lambda x: f"{x} — {df.loc[df['id']==x,'title'].values[0]}"
    )
    chosen = df[df["id"].isin(selected)]

    if not chosen.empty:
        if st.button("Export selected as .ics"):
            cal = Calendar()
            cal.add("prodid", "-//SchoolBoard POC//")
            cal.add("version", "2.0")
            for _, row in chosen.iterrows():
                ev = Event()
                start = datetime.strptime(row["start"], "%Y-%m-%d %H:%M")
                end = datetime.strptime(row["end"], "%Y-%m-%d %H:%M")
                ev.add("summary", row["title"])
                ev.add("dtstart", start)
                ev.add("dtend", end)
                if row.get("location"):
                    ev.add("location", row["location"])
                ev.add("description", f"From: {row['from']}\nNote: {row['note']}")
                cal.add_component(ev)
            ics_bytes = cal.to_ical()
            st.download_button("Download .ics", data=ics_bytes, file_name="school_events.ics", mime="text/calendar")

        st.write("Quick add to Google Calendar")
        for _, row in chosen.iterrows():
            start = datetime.strptime(row["start"], "%Y-%m-%d %H:%M")
            end = datetime.strptime(row["end"], "%Y-%m-%d %H:%M")
            start_str = start.strftime("%Y%m%dT%H%M00")
            end_str = end.strftime("%Y%m%dT%H%M00")
            title = row["title"]
            details = f"{row['note']} (from {row['from']})"
            params = {
                "action": "TEMPLATE",
                "text": title,
                "dates": f"{start_str}/{end_str}",
                "details": details
            }
            loc = row.get("location")
            if isinstance(loc, str) and loc.strip():
                params["location"] = loc.strip()

            url = "https://www.google.com/calendar/render?" + urlencode(params)
            st.markdown(f"- **{title}** — {row['start']}  [Add to Google Calendar]({url})")
    else:
        st.write("Select events above to see export options.")