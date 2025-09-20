# school_board_poc.py — Gmail OAuth + IMAP fallback + Gemini + Live refresh
# Features:
# - Sign in with Google (Gmail API) so each user uses their own account
# - IMAP fallback (App Password) for local/testing
# - Gemini (optional) to extract structured events + urgency
# - Urgency badges, .ics export, "Add to Google Calendar" links
# - Auto-refresh monitor
# - School email filters

# --- Standard libs
import os
import re
import json
import base64
from datetime import datetime, timedelta
from urllib.parse import urlencode

# --- Email/IMAP
import imaplib
import email as _email
from email.header import decode_header

# --- Streamlit & friends
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import dateparser
from icalendar import Calendar, Event
from dotenv import load_dotenv

# --- Google APIs (OAuth + Gmail/Calendar)
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build

# Optional LLM
try:
    import google.generativeai as genai
except Exception:
    genai = None

# ---------- Page setup ----------
st.set_page_config(
    page_title="School Activity Board",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Env/Secrets helper ----------
load_dotenv()

def env(name: str, default: str = "") -> str:
    """Read from Streamlit secrets first, then environment."""
    return st.secrets.get(name, os.getenv(name, default))

# ---------- Config ----------
# OAuth
GOOGLE_CLIENT_ID     = env("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = env("GOOGLE_CLIENT_SECRET", "")
OAUTH_REDIRECT_URI   = env("OAUTH_REDIRECT_URI", "http://localhost:8501/oauth2callback")
GOOGLE_SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
# If you want to create events via API automatically, also add:
# GOOGLE_SCOPES.append("https://www.googleapis.com/auth/calendar.events")

# Gemini (optional)
GEMINI_API_KEY = env("GEMINI_API_KEY", "")
if GEMINI_API_KEY and genai:
    genai.configure(api_key=GEMINI_API_KEY)

# IMAP fallback
IMAP_HOST = env("IMAP_HOST", "imap.gmail.com")
IMAP_USER = env("IMAP_USER", "")
IMAP_PASS = env("IMAP_PASS", "")
MAILBOX    = env("MAILBOX", "INBOX")
MAX_EMAILS = int(env("MAX_EMAILS", "80"))

# Filter defaults
DEFAULT_SCHOOL_DOMAINS = "austinisd.org, myschool.org, school.edu, k12.tx.us, pta.org"
DEFAULT_SCHOOL_SENDERS = ""  # e.g., "teacher@myschool.org, principal@district.k12.tx.us"
DEFAULT_KEYWORDS       = "school, pta, teacher, classroom, homeroom, field trip, permission slip, bus, assembly, cafeteria, counselor, principal, due, forms"
DEFAULT_NEGATIVE       = "unsubscribe, terms, privacy policy, marketing, promotion, sale, newsletter, invoice, receipt"

# ---------- OAuth helpers ----------
def _oauth_flow() -> Flow:
    cfg = {
        "web": {
            "client_id": GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": [OAUTH_REDIRECT_URI],
        }
    }
    return Flow.from_client_config(cfg, scopes=GOOGLE_SCOPES, redirect_uri=OAUTH_REDIRECT_URI)

def start_login_button():
    flow = _oauth_flow()
    auth_url, _ = flow.authorization_url(
        access_type="offline",           # get refresh_token
        include_granted_scopes=True,
        prompt="consent"                 # ensure refresh_token is issued
    )
    st.link_button("Continue with Google", auth_url, use_container_width=True)

def finish_login_if_callback():
    # On Streamlit Cloud or local, Google redirects with ?code=...
    code = st.query_params.get("code")
    if not code:
        return None
    flow = _oauth_flow()
    flow.fetch_token(code=code)
    creds = flow.credentials
    st.session_state["google_creds"] = {
        "token": creds.token,
        "refresh_token": creds.refresh_token,
        "token_uri": creds.token_uri,
        "client_id": creds.client_id,
        "client_secret": creds.client_secret,
        "scopes": creds.scopes,
    }
    # Clean URL so refreshes don't try to re-complete OAuth
    st.query_params.clear()
    return creds

def current_creds():
    blob = st.session_state.get("google_creds")
    if not blob:
        return None
    return Credentials(**blob)

# ---------- Gmail API fetch (OAuth path) ----------
@st.cache_data(ttl=5)
def fetch_emails_gmailapi(token_blob: dict, query="label:inbox newer_than:14d -category:promotions", max_results=80):
    creds = Credentials(**token_blob)
    service = build("gmail", "v1", credentials=creds)
    resp = service.users().messages().list(userId="me", q=query, maxResults=max_results).execute()
    msgs = resp.get("messages", [])
    emails, uids = [], []
    for m in msgs:
        msg = service.users().messages().get(userId="me", id=m["id"], format="full").execute()
        headers = {h["name"]: h["value"] for h in msg["payload"].get("headers", [])}
        subject = headers.get("Subject", "")
        from_   = headers.get("From", "")
        date_   = headers.get("Date", "")

        # Extract text/plain parts
        body_text = ""
        def walk_parts(p):
            nonlocal body_text
            if "parts" in p:
                for pt in p["parts"]:
                    walk_parts(pt)
            else:
                if p.get("mimeType") == "text/plain" and "data" in p.get("body", {}):
                    body_text += base64.urlsafe_b64decode(p["body"]["data"]).decode("utf-8", "ignore")
        walk_parts(msg["payload"])

        emails.append({"subject": subject, "from": from_, "date": date_, "body": body_text})
        uids.append(m["id"])
    return {"emails": emails, "uids": uids}

# ---------- IMAP fallback fetch ----------
import imaplib
import email as _email
from email.header import decode_header

@st.cache_data(ttl=5)
def fetch_emails(search_mode="ALL", since_days=7):
    """
    search_mode: "ALL" | "UNSEEN" | "SINCE"
    since_days: used when search_mode == "SINCE"
    Returns: {"emails": [...], "uids": [...]}
    """
    if not IMAP_USER or not IMAP_PASS:
        return {"error": "Set IMAP_USER and IMAP_PASS (or use Google Sign-In)"}
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

        ids = data[0].split()[-MAX_EMAILS:]  # most recent N
        emails, uids = [], []
        for num in reversed(ids):
            # PEEK = do not mark as read
            typ, msg_data = mail.fetch(num, '(BODY.PEEK[])')
            if typ != "OK":
                continue
            msg = _email.message_from_bytes(msg_data[0][1])

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

            emails.append({"subject": subject or "", "from": from_ or "", "date": date_ or "", "body": (body or "")[:20000]})
            uids.append(num.decode() if isinstance(num, bytes) else str(num))

        mail.logout()
        return {"emails": emails, "uids": uids}
    except Exception as e:
        return {"error": str(e)}

# ---------- Regex/date parsing ----------
DATE_PATTERNS = [
    r"\b(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b[^\n,]*",
    r"\b(?:tomorrow|today|next week|next month|next|tonight)\b",
    r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b",
    r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[^\n,0-9]{0,6}\d{1,2}(?:st|nd|rd|th)?(?:,?\s*\d{4})?",
    r"\b(?:\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec))\b",
]

def find_possible_dates(text: str):
    matches = []
    for pat in DATE_PATTERNS:
        for m in re.finditer(pat, text, re.IGNORECASE):
            matches.append(m.group().strip())
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

# ---------- School filters ----------
def _norm_split(s: str):
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

    # Exclude by negative keywords
    for bad in neg_kws:
        if bad and (bad in subject or bad in body):
            return False

    # Keyword check
    text = subject + " " + body
    kw_ok = any(kw in text for kw in must_kws) if must_kws else True

    # Sender domain/exact
    sender_ok = False
    if addr and addr in senders:
        sender_ok = True
    if dom:
        for d in domains:
            if d and dom.endswith(d):
                sender_ok = True
                break

    # Forwarded by me + keywords → allow
    my_addr = (IMAP_USER or "").lower()
    if my_addr and my_addr in from_raw and kw_ok:
        return True

    if domains or senders:
        return sender_ok and kw_ok
    return kw_ok

# ---------- Gemini LLM extractor (optional) ----------
def gemini_extract_events(subject: str, body: str):
    if not (GEMINI_API_KEY and genai):
        return []
    prompt = f"""
You are an assistant that extracts school-related calendar items from email text.

Return ONLY valid JSON with this shape and no commentary:
{{
  "events":[
    {{
      "title": "short title",
      "start": "YYYY-MM-DD HH:MM",
      "end":   "YYYY-MM-DD HH:MM",
      "type":  "event" | "deadline" | "reminder",
      "location": "optional",
      "notes": "optional",
      "urgency_score": 0
    }}
  ]
}}
Rules:
- If multiple items exist, include each in "events".
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
        text  = (resp.text or "").strip().strip("` \n")
        if text.lower().startswith("json"):
            text = text[4:].lstrip(": \n")
        data = json.loads(text)
        out = []
        for ev in data.get("events", []):
            title   = ev.get("title") or "School event"
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
    st.header("Sign in with Google")
    creds = current_creds()
    if not creds:
        creds = finish_login_if_callback()
    if creds:
        st.success("Signed in")
        if st.button("Sign out"):
            st.session_state.pop("google_creds", None)
            st.rerun()
    else:
        if GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET:
            start_login_button()
        else:
            st.info("OAuth not configured. Add GOOGLE_CLIENT_ID/SECRET & OAUTH_REDIRECT_URI in Secrets.")

    st.divider()
    st.header("Live monitor")
    live_mode  = st.toggle("Auto-refresh for new emails", value=False)
    interval   = st.select_slider("Refresh every (seconds)", options=[10, 15, 20, 30, 45, 60], value=30)
    fetch_mode = st.selectbox("IMAP fetch scope (fallback)", ["UNSEEN", "SINCE", "ALL"], help="Used only when not signed in with Google")
    since_days = st.number_input("IMAP SINCE: days back", min_value=1, max_value=30, value=7, step=1)

    st.divider()
    st.header("School filters")
    domain_str = st.text_input("Sender domains (comma-separated)", value=DEFAULT_SCHOOL_DOMAINS)
    sender_str = st.text_input("Exact sender emails (comma-separated)", value=DEFAULT_SCHOOL_SENDERS)
    kw_str     = st.text_input("Must contain keywords (comma-separated)", value=DEFAULT_KEYWORDS)
    neg_str    = st.text_input("Exclude if contains (comma-separated)", value=DEFAULT_NEGATIVE)

    st.divider()
    use_gemini = st.toggle("Use Gemini extraction (LLM)", value=bool(GEMINI_API_KEY),
                           help="Requires GEMINI_API_KEY")

# Timed refresh
if live_mode:
    st_autorefresh(interval=interval * 1000, key="auto_refresh_tick")

st.title("School Activity Board")
st.caption("Inbox → activities with filters and optional AI extraction.")

# ---------- Fetch path ----------
if creds:
    result = fetch_emails_gmailapi(st.session_state["google_creds"])
else:
    result = fetch_emails(search_mode=fetch_mode, since_days=int(since_days))

if "error" in result:
    st.error("Email read error: " + result["error"])
    st.stop()

emails = result.get("emails", [])
uids   = result.get("uids", [])
st.caption(f"Fetched {len(emails)} email(s).")

# New mail toast
if "last_uids" not in st.session_state:
    st.session_state["last_uids"] = set(uids)
else:
    new = [u for u in uids if u not in st.session_state["last_uids"]]
    if new:
        st.toast(f"📬 {len(new)} new email(s)", icon="✉️")
        st.session_state["last_uids"].update(new)

# Filter before parsing
filtered = [e for e in emails if looks_like_school(e, domain_str, sender_str, kw_str, neg_str)]
st.caption(f"After filtering: {len(filtered)} email(s).")
emails = filtered

# Optional preview
if st.button("Preview raw emails (subject + snippet)"):
    for e in emails[:10]:
        st.markdown(f"**{e.get('subject','')}** — {e.get('from','')}")
        st.code((e.get('body') or "")[:600])

# Build events (Gemini first, fallback regex)
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
        ev.setdefault("type", "event")
        ev.setdefault("location", "")
        ev.setdefault("notes", "")
        ev.setdefault("urgency_score", 0)
        ev["source_from"]    = e.get("from")
        ev["source_subject"] = subj
        ev["raw_line"]       = ev.get("raw_line") or subj
        all_events.append(ev)

# ---------- UI: two-column layout ----------
if not all_events:
    st.info("No activities detected. Loosen filters or preview emails.")
else:
    sort_choice = st.radio("Sort by:", ["Urgency (High→Low)", "Start time (Soonest first)"], horizontal=True)

    rows = []
    for i, ev in enumerate(all_events):
        score = int(ev.get("urgency_score", 0))
        rows.append({
            "id": i + 1,
            "title": ev.get("title", "School event"),
            "type": ev.get("type", "event"),
            "urgency": badge_for_urgency(score),
            "urgency_score": score,
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

    left, right = st.columns([2, 1], gap="large")

    with left:
        st.subheader("Detected activities")
        st.dataframe(
            df[["id", "title", "type", "urgency", "start", "end", "location", "from", "note"]],
            use_container_width=True,
            hide_index=True,
        )

    with right:
        st.subheader("Select & export")
        options = df["id"].tolist()
        select_all = st.checkbox("Select all", value=min(len(options), 10) > 0)
        default_ids = options if select_all else []
        selected = st.multiselect(
            "Pick event ids",
            options=options,
            default=default_ids,
            format_func=lambda x: f"{x} — {df.loc[df['id']==x,'title'].values[0]}"
        )
        chosen = df[df["id"].isin(selected)]

        if chosen.empty:
            st.caption("Select at least one event.")
        else:
            # .ics export
            if st.button("Export selected as .ics", use_container_width=True):
                cal = Calendar()
                cal.add("prodid", "-//SchoolBoard POC//")
                cal.add("version", "2.0")
                for _, row in chosen.iterrows():
                    ev = Event()
                    start = datetime.strptime(row["start"], "%Y-%m-%d %H:%M")
                    end   = datetime.strptime(row["end"], "%Y-%m-%d %H:%M")
                    ev.add("summary", row["title"])
                    ev.add("dtstart", start)
                    ev.add("dtend", end)
                    if row.get("location"): ev.add("location", row["location"])
                    ev.add("description", f"From: {row['from']}\nNote: {row['note']}")
                    cal.add_component(ev)
                ics_bytes = cal.to_ical()
                st.download_button("Download .ics", data=ics_bytes, file_name="school_events.ics", mime="text/calendar", use_container_width=True)

            # Quick add links to Google Calendar
            st.markdown("**Quick add to Google Calendar**")
            for _, row in chosen.iterrows():
                start = datetime.strptime(row["start"], "%Y-%m-%d %H:%M")
                end   = datetime.strptime(row["end"], "%Y-%m-%d %H:%M")
                start_str = start.strftime("%Y%m%dT%H%M00")
                end_str   = end.strftime("%Y%m%dT%H%M00")
                params = {
                    "action": "TEMPLATE",
                    "text": row["title"],
                    "dates": f"{start_str}/{end_str}",
                    "details": f"{row['note']} (from {row['from']})",
                }
                if isinstance(row.get("location"), str) and row["location"].strip():
                    params["location"] = row["location"].strip()
                url = "https://www.google.com/calendar/render?" + urlencode(params)
                st.markdown(f"- {row['title']} — {row['start']}  [Add to Google Calendar]({url})")
