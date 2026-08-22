"""magicpin AI Challenge — Vera-style merchant assistant bot.

Rule-based 4-context composer (category, merchant, trigger, customer).
Deterministic, no external calls, well under the per-call time budget.
Run: uvicorn bot:app --host 0.0.0.0 --port 8080
"""
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel


def _load_dotenv() -> None:
    """Minimal .env loader (KEY=VALUE lines). Real env vars win."""
    env_file = Path(__file__).resolve().parent / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if key and val:
            os.environ.setdefault(key, val)


_load_dotenv()

app = FastAPI()
START = time.time()

VALID_SCOPES = {"category", "merchant", "customer", "trigger"}

TEAM_NAME = os.environ.get("TEAM_NAME", "Team MagicPin")
TEAM_MEMBERS = [m.strip() for m in os.environ.get("TEAM_MEMBERS", "Harsh").split(",") if m.strip()]
CONTACT_EMAIL = os.environ.get("CONTACT_EMAIL", "team@example.com")

# ── in-memory state ───────────────────────────────────────────────────
contexts: dict[tuple[str, str], dict] = {}      # (scope, id) -> {version, payload}
conversations: dict[str, dict] = {}             # conv_id -> conversation state
suppressed_keys: set[str] = set()               # suppression_keys already acted on
used_conv_ids: set[str] = set()
opted_out: set[str] = set()                     # merchant_ids that asked us to stop
auto_reply_counts: dict[str, int] = {}          # merchant_id -> consecutive auto-replies


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _parse_dt(value) -> Optional[datetime]:
    if not value or not isinstance(value, str):
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return None


# ── pydantic bodies ───────────────────────────────────────────────────
class CtxBody(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: dict[str, Any]
    delivered_at: str = ""


class TickBody(BaseModel):
    now: str = ""
    available_triggers: list[str] = []


class ReplyBody(BaseModel):
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    from_role: str
    message: str
    received_at: str = ""
    turn_number: int = 0


# ── small formatting helpers ──────────────────────────────────────────
def _salutation(merchant: dict) -> str:
    ident = merchant.get("identity", {}) or {}
    owner = (ident.get("owner_first_name") or "").strip()
    name = (ident.get("name") or "Merchant").strip()
    slug = (merchant.get("category_slug") or "").rstrip("s")
    if slug == "dentist":
        base = owner or name
        if not base.lower().startswith("dr"):
            base = "Dr. " + base
        return base
    return owner or (name.split()[0] if name else "Merchant")


def _biz_name(merchant: dict) -> str:
    return (merchant.get("identity", {}) or {}).get("name", "your business")


def _hi_ok(merchant: dict) -> bool:
    langs = ((merchant.get("identity", {}) or {}).get("languages") or [])
    return any(str(l).lower().startswith("hi") for l in langs)


def _cus_hi(customer: Optional[dict], merchant: dict) -> bool:
    """Customer language pref if known; else inherit the merchant's languages."""
    pref = str(((customer or {}).get("identity", {}) or {}).get("language_pref", "")).lower()
    if pref:
        return "hi" in pref or "hindi" in pref
    return _hi_ok(merchant)


def _active_offers(merchant: dict) -> list[str]:
    return [o.get("title", "") for o in merchant.get("offers", []) or []
            if isinstance(o, dict) and o.get("status") == "active" and o.get("title")]


def _catalog_offers(category: dict) -> list[dict]:
    return [o for o in category.get("offer_catalog", []) or [] if isinstance(o, dict)]


def _digest_item(category: dict, item_id: str = "") -> dict:
    digest = [d for d in category.get("digest", []) or [] if isinstance(d, dict)]
    for d in digest:
        if d.get("id") == item_id:
            return d
    return digest[0] if digest else {}


def _peer(category: dict, key: str, default=None):
    return (category.get("peer_stats", {}) or {}).get(key, default)


def _human(text: str) -> str:
    return str(text).replace("_", " ").strip()


def _pct(delta) -> str:
    try:
        return f"{float(delta) * 100:+.0f}%"
    except (TypeError, ValueError):
        return ""


def _abs_pct(delta) -> str:
    try:
        return f"{abs(float(delta)) * 100:.0f}%"
    except (TypeError, ValueError):
        return ""


def _fmt_date(value) -> str:
    dt = _parse_dt(value) if isinstance(value, str) else None
    if not dt:
        return str(value) if value else ""
    return dt.strftime("%d %b %Y").lstrip("0")


def _months_between(start: str, end: str) -> Optional[int]:
    a, b = _parse_dt(start), _parse_dt(end)
    if not a or not b:
        return None
    return max(0, (b.year - a.year) * 12 + (b.month - a.month))


def _clean_body(body: str) -> str:
    body = re.sub(r"https?://\S+", "", body)          # URLs are a hard fail
    body = re.sub(r"\s+", " ", body).strip()
    return body


def _scrub_taboos(body: str, category: dict) -> str:
    """Enforce category.voice.taboos (e.g. 'cure', 'guaranteed') against the composed
    body. Handlers hardcode a plausible tone per category, but the taboo list itself
    is context data that can be updated by the judge at any time (§4.1) — so it must
    be checked against the live pushed value, not assumed from the handler's wording."""
    taboos = ((category or {}).get("voice", {}) or {}).get("taboos", []) or []
    for word in taboos:
        word = str(word).strip()
        if not word:
            continue
        # also swallow a trailing label-colon (e.g. "Deadline:") so removal doesn't
        # leave an orphaned ": 15 Dec 2026" fragment behind
        pattern = r"\b" + re.escape(word) + r"\b\s*:?\s*"
        if re.search(pattern, body, flags=re.IGNORECASE):
            body = re.sub(pattern, " ", body, flags=re.IGNORECASE)
    body = re.sub(r"\s+([.,;:!?])", r"\1", body)   # no space before punctuation
    body = re.sub(r"\s+", " ", body).strip()
    return body


def _trend_signal(category: dict) -> dict:
    """First usable trend_signal from category context, or {} if none pushed."""
    signals = [s for s in (category or {}).get("trend_signals", []) or [] if isinstance(s, dict)]
    return signals[0] if signals else {}


# ── trigger handlers ──────────────────────────────────────────────────
# Each handler returns {body, cta, rationale, template_name, template_params}

def _h_research_digest(cat, mer, trg, cus):
    p = trg.get("payload", {}) or {}
    item = _digest_item(cat, p.get("top_item_id", "") or p.get("digest_item_id", ""))
    title = item.get("title", "")
    source = item.get("source", "")
    sal = _salutation(mer)
    facts = []
    if item.get("trial_n"):
        facts.append(f"{item['trial_n']:,}-patient trial")
    seg = item.get("patient_segment") or ""
    relevance = ""
    signals = mer.get("signals", []) or []
    if "high_risk_adult_cohort" in signals:
        relevance = "directly relevant to your high-risk adult cohort"
    elif seg and seg in str(signals):
        relevance = f"matches your {_human(seg)} segment"
    body = f"{sal}, {title}."
    if facts:
        body += " (" + ", ".join(facts) + ")"
    if relevance:
        body += f" This one is {relevance}."
    cta_line = ("Want me to pull the abstract + draft a short patient post you can share?"
                if not _hi_ok(mer) else
                "Abstract nikal dun + ek chhota patient post draft kar dun?")
    body += f" {cta_line}"
    if source:
        body += f" — {source}"
    rationale = (f"research digest hook ({item.get('id', 'top item')}) with verifiable trial size "
                 f"and source citation; curiosity CTA")
    return {"body": body, "cta": "open_ended",
            "rationale": rationale, "template_name": "vera_research_digest_v1",
            "template_params": [sal, title[:60], source or ""]}


def _h_regulation_change(cat, mer, trg, cus):
    p = trg.get("payload", {}) or {}
    item = _digest_item(cat, p.get("top_item_id", "") or p.get("digest_item_id", ""))
    deadline = p.get("deadline_iso", "")
    sal = _salutation(mer)
    title = item.get("title") or "a compliance update for your category"
    body = f"{sal}, regulatory update: {title}."
    if deadline:
        body += f" Deadline: {_fmt_date(deadline)}."
    body += (" Reply YES and I'll send a short compliance checklist you can run today."
             if not _hi_ok(mer) else
             " Reply YES — main aapko short compliance checklist bhej deti hoon.")
    src = item.get("source", "")
    if src:
        body += f" Source: {src}"
    rationale = "regulation change with concrete deadline; loss-aversion + binary CTA"
    return {"body": body, "cta": "binary",
            "rationale": rationale, "template_name": "vera_regulation_v1",
            "template_params": [sal, title[:60], str(deadline)]}


def _h_cde_opportunity(cat, mer, trg, cus):
    p = trg.get("payload", {}) or {}
    item = _digest_item(cat, p.get("digest_item_id", ""))
    sal = _salutation(mer)
    credits = p.get("credits", "")
    fee = p.get("fee", "")
    title = item.get("title") or "a CDE session relevant to your practice"
    fee_txt = "free for members" if fee == "free_for_members" else str(fee)
    body = f"{sal}, CDE opportunity: {title}."
    if credits:
        body += f" {credits} credit(s), {fee_txt}."
    body += (" Seats are limited — reply YES to block yours."
             if not _hi_ok(mer) else
             " Seats limited hain — reply YES, main aapki seat block kar deti hoon.")
    src = item.get("source", "")
    if src:
        body += f" ({src})"
    rationale = "CDE opportunity from category digest; scarcity framing + binary CTA"
    return {"body": body, "cta": "binary",
            "rationale": rationale, "template_name": "vera_cde_invite_v1",
            "template_params": [sal, title[:60], str(credits)]}


def _h_recall_due(cat, mer, trg, cus):
    p = trg.get("payload", {}) or {}
    ident = (cus or {}).get("identity", {}) or {}
    name = ident.get("name", "there")
    lang_pref = str(ident.get("language_pref", ""))
    hi = _cus_hi(cus, mer)
    svc_raw = str(p.get("service_due", "checkup"))
    m = re.search(r"(\d+)", svc_raw)
    n_months = int(m.group(1)) if m else None
    svc = _human(re.sub(r"\d+_?month_?", "", svc_raw).replace("_month", "")) or "checkup"
    last_date = p.get("last_service_date", "")
    due_date = p.get("due_date", "")
    if n_months is None and last_date and due_date:
        n_months = _months_between(last_date, due_date)
    slots = p.get("available_slots", []) or []
    labels = [s.get("label", "") for s in slots[:2] if isinstance(s, dict) and s.get("label")]
    clinic = _biz_name(mer)
    joiner = "ya" if hi else "or"

    body = f"Hi {name}, {clinic} here."
    if n_months:
        body += f" It's been about {n_months} months since your last visit"
    elif last_date:
        body += f" Your last visit was {_fmt_date(last_date)}"
    if due_date:
        body += f" — your {svc} recall is due ({_fmt_date(due_date)})."
    else:
        body += " — time for your next visit."

    offers = _active_offers(mer)
    offer_match = next((o for o in offers if svc.split()[0].lower() in o.lower()), None)
    price_line = offer_match or (offers[0] if offers else "")
    if price_line:
        body += f" {price_line}."

    if len(labels) >= 2:
        body += f" Apke liye 2 slots ready hain: **{labels[0]}** {joiner} **{labels[1]}**." if hi \
            else f" Two slots open: {labels[0]} {joiner} {labels[1]}."
        body += (" Reply 1 for the first, 2 for the second, ya apna convenient time batayein."
                 if hi else " Reply 1 or 2, or tell us a time that works.")
        cta = "multi_choice_slot"
    elif labels:
        body += f" Next available slot: {labels[0]}. Shall we hold it for you?"
        cta = "binary"
    else:
        body += " Reply with a day/time that suits you and we'll book it."
        cta = "open_ended"

    rationale = (f"customer recall ({svc}, due {due_date or 'soon'}) sent on behalf of merchant; "
                 f"language pref '{lang_pref or 'default'}' honored; real slots + catalog price")
    return {"body": body, "cta": cta,
            "rationale": rationale,
            "send_as": "merchant_on_behalf",
            "template_name": "merchant_recall_reminder_v1",
            "template_params": [name, clinic, f"{n_months or ''} month {svc} recall",
                                " + ".join(labels) or "", price_line]}


def _h_wedding_followup(cat, mer, trg, cus):
    p = trg.get("payload", {}) or {}
    ident = (cus or {}).get("identity", {}) or {}
    name = ident.get("name", "there")
    days = p.get("days_to_wedding", "")
    wdate = _fmt_date(p.get("wedding_date", ""))
    step = _human(p.get("next_step_window_open", "")).replace("program", "program").strip()
    body = (f"Hi {name}, {_biz_name(mer)} here! Your wedding functions are around the corner"
            + (f" ({wdate}, that's {days} days out)" if days else "") + ".")
    if step:
        body += f" The {step} window opens now — starting early makes the biggest difference."
    hi = _cus_hi(cus, mer)
    body += (" Reply YES and we'll plan it around your trial notes."
             if not hi else
             " Reply YES — aapke trial notes ke hisaab se plan taiyaar kar denge.")
    rationale = "bridal follow-up anchored on real wedding date + next-step window"
    return {"body": body, "cta": "binary",
            "rationale": rationale, "send_as": "merchant_on_behalf",
            "template_name": "merchant_bridal_followup_v1",
            "template_params": [name, str(days), step[:40]]}


def _h_trial_followup(cat, mer, trg, cus):
    p = trg.get("payload", {}) or {}
    ident = (cus or {}).get("identity", {}) or {}
    name = ident.get("name", "there")
    trial = _fmt_date(p.get("trial_date", ""))
    opts = p.get("next_session_options", []) or []
    labels = [o.get("label", "") for o in opts[:2] if isinstance(o, dict) and o.get("label")]
    body = f"Hi {name}, {_biz_name(mer)} here. Loved having you at the trial session"
    if trial:
        body += f" on {trial}"
    body += "."
    if labels:
        body += f" Next session: {labels[0]}"
        if len(labels) > 1:
            body += f" (backup: {labels[1]})"
        body += (" Should we confirm your spot?" if not _cus_hi(cus, mer)
                 else " Reply YES — agli session aapke liye confirm kar doon?")
    else:
        body += (" Want us to reserve your next session?" if not _cus_hi(cus, mer)
                 else " Agli session reserve kar dein?")
    rationale = "trial follow-up converting to membership; single slot confirmation ask"
    return {"body": body, "cta": "binary" if labels else "open_ended",
            "rationale": rationale, "send_as": "merchant_on_behalf",
            "template_name": "merchant_trial_followup_v1",
            "template_params": [name, trial, " + ".join(labels)]}


def _h_chronic_refill(cat, mer, trg, cus):
    p = trg.get("payload", {}) or {}
    ident = (cus or {}).get("identity", {}) or {}
    name = ident.get("name", "there")
    molecules = [_human(m) for m in p.get("molecule_list", []) or []]
    runs_out = _fmt_date(p.get("stock_runs_out_iso", ""))
    delivery = p.get("delivery_address_saved", False)
    body = f"Hi {name}, {_biz_name(mer)} here."
    if molecules:
        body += f" Your regular medicines ({', '.join(molecules)}) are due for refill"
    if runs_out:
        body += f" — stock runs out around {runs_out}"
    body += "."
    hi = _cus_hi(cus, mer)
    if delivery:
        body += (" Your delivery address is saved — say YES and we'll pack it for home delivery."
                 if not hi else
                 " Aapka delivery address saved hai — YES boliye, home delivery ke liye pack karwa deta hoon.")
    else:
        body += (" Reply YES and we'll keep a pack ready for pickup."
                 if not hi else
                 " Reply YES — pickup ke liye pack ready rakhwa deta hoon.")
    rationale = "chronic refill reminder using actual molecule list + expiry date; zero-effort binary CTA"
    return {"body": body, "cta": "binary",
            "rationale": rationale, "send_as": "merchant_on_behalf",
            "template_name": "merchant_refill_reminder_v1",
            "template_params": [name, ", ".join(molecules)[:50], str(runs_out)]}


def _h_customer_lapsed(cat, mer, trg, cus, hard=True):
    p = trg.get("payload", {}) or {}
    ident = (cus or {}).get("identity", {}) or {}
    name = ident.get("name", "there")
    days = p.get("days_since_last_visit", "")
    focus = _human(p.get("previous_focus", ""))
    if not focus:
        services = ((cus or {}).get("relationship", {}) or {}).get("services_received") or []
        if services:
            focus = _human(services[-1])
    months = p.get("previous_membership_months", "")
    offers = _active_offers(mer)
    catalog = _catalog_offers(cat)
    offer = offers[0] if offers else (catalog[0].get("title") if catalog else "")
    body = f"Hi {name}, {_biz_name(mer)} here!"
    if hard and days:
        body += f" It's been {days} days since your last session"
        if focus:
            body += f" — you were doing great with your {focus} focus"
        body += "."
    elif days:
        body += f" It's been a while ({days} days) since your last visit."
    if months:
        body += f" You were with us {months} months — your progress is still saved."
    if offer:
        body += f" Restart with {offer}."
    body += (" Reply YES and we'll set up your comeback session."
             if not _cus_hi(cus, mer) else
             " Reply YES — aapka comeback session set kar dete hain.")
    rationale = "win-back message personalized with lapse duration + prior focus; restart offer from catalog"
    return {"body": body, "cta": "binary",
            "rationale": rationale, "send_as": "merchant_on_behalf",
            "template_name": "merchant_winback_v1",
            "template_params": [name, str(days), focus[:30]]}


def _h_perf_spike(cat, mer, trg, cus):
    p = trg.get("payload", {}) or {}
    metric = _human(p.get("metric", "views"))
    delta = p.get("delta_pct", 0)
    driver = _human(p.get("likely_driver", ""))
    sal = _salutation(mer)
    body = f"{sal}, good news — your {metric} are up {_abs_pct(delta)} this {'week' if p.get('window') == '7d' else _human(p.get('window', 'period'))}."
    if driver:
        body += f" Looks like the {driver} is working."
    body += (" What did you change? I want to log it so we can repeat it."
             if not _hi_ok(mer) else
             " Kya naya try kiya aapne? Log kar dun taaki hum isko repeat kar sakein.")
    rationale = f"perf spike (+{_abs_pct(delta)} {metric}); reciprocity + curiosity ask drives reply"
    return {"body": body, "cta": "open_ended",
            "rationale": rationale, "template_name": "vera_perf_spike_v1",
            "template_params": [sal, metric, _pct(abs(float(delta or 0)))]}


def _h_perf_dip(cat, mer, trg, cus):
    p = trg.get("payload", {}) or {}
    metric = _human(p.get("metric", "calls"))
    delta = p.get("delta_pct", 0)
    baseline = p.get("vs_baseline", "")
    peer_calls = _peer(cat, "avg_calls_30d")
    peer_ctr = _peer(cat, "avg_ctr")
    ctr = (mer.get("performance", {}) or {}).get("ctr")
    sal = _salutation(mer)
    body = f"{sal}, quick flag: your {metric} dropped {_abs_pct(delta)} this week"
    if baseline:
        body += f" (vs ~{baseline} usual)"
    body += "."
    comparisons = []
    if metric.lower() == "calls" and peer_calls:
        comparisons.append(f"peers average {peer_calls}/mo")
    if metric.lower() == "views" and peer_ctr and ctr:
        comparisons.append(f"peer CTR is {float(peer_ctr):.1%} vs your {float(ctr):.1%}")
    if comparisons:
        body += " " + "; ".join(comparisons).capitalize() + "."
    body += (" Reply YES and I'll diagnose the top 3 fixable causes."
             if not _hi_ok(mer) else
             " Reply YES — main top 3 fixable reasons check karke batati hoon.")
    rationale = f"perf dip (-{_abs_pct(delta)} {metric}) benchmarked against real peer stats; diagnostic binary CTA"
    return {"body": body, "cta": "binary",
            "rationale": rationale, "template_name": "vera_perf_dip_v1",
            "template_params": [sal, metric, _pct(delta)]}


def _h_seasonal_perf_dip(cat, mer, trg, cus):
    p = trg.get("payload", {}) or {}
    metric = _human(p.get("metric", "views"))
    delta = p.get("delta_pct", 0)
    note = _human(p.get("season_note", ""))
    seasonal = (cat.get("seasonal_beats", []) or [])
    beat = next((s.get("note", "") for s in seasonal
                 if isinstance(s, dict) and any(w in (s.get("note", "").lower()) for w in ("lowest", "post_resolution"))), "")
    sal = _salutation(mer)
    body = f"{sal}, your {metric} are down {_abs_pct(delta)} — don't panic, this is the expected seasonal slide"
    if note:
        body += f" ({_human(note.replace('post_resolution_window_apr_jun', 'post-New-Year resolution window, Apr-Jun'))})"
    body += "."
    if beat:
        body += f" Category pattern: {beat}."
    body += (" Smart move right now is retention, not acquisition. Want 2 low-effort plays for this window?"
             if not _hi_ok(mer) else
             " Is window mein retention smartest move hai. 2 low-effort plays bataun?")
    rationale = "seasonal dip normalized with cited category pattern; pivots to retention play (no false alarm)"
    return {"body": body, "cta": "open_ended",
            "rationale": rationale, "template_name": "vera_seasonal_note_v1",
            "template_params": [sal, metric, _pct(delta)]}


def _h_milestone(cat, mer, trg, cus):
    p = trg.get("payload", {}) or {}
    metric = _human(p.get("metric", "reviews")).replace("review count", "reviews")
    now_v = p.get("value_now", 0)
    target = p.get("milestone_value", 0)
    remaining = max(0, (target or 0) - (now_v or 0))
    lib = cat.get("patient_content_library", []) or []
    asset = next((c.get("title", "") for c in lib if isinstance(c, dict)), "")
    sal = _salutation(mer)
    body = f"{sal}, {now_v} {metric} and counting!"
    if remaining:
        body += f" Just {remaining} away from {target}."
    body += f" Crossing {target} makes new patients trust your listing noticeably more."
    if asset:
        body += f" I've got \"{asset}\" ready to reshare — want it?"
    else:
        body += " Want a ready-to-share post to speed this up?"
    rationale = f"milestone momentum ({now_v}->{target} {metric}); social proof + effort externalization"
    return {"body": body, "cta": "binary",
            "rationale": rationale, "template_name": "vera_milestone_v1",
            "template_params": [sal, str(now_v), str(target)]}


def _h_review_theme(cat, mer, trg, cus):
    p = trg.get("payload", {}) or {}
    theme = _human(p.get("theme", "service"))
    n = p.get("occurrences_30d", 0)
    trend = p.get("trend", "")
    quote = p.get("common_quote", "")
    sal = _salutation(mer)
    body = f"{sal}, {n} reviews in the last 30 days mention \"{theme}\""
    if trend == "rising":
        body += " and the trend is rising"
    body += "."
    if quote:
        body += f" Sample: \"{quote}\"."
    body += (" Left alone this hurts ratings. Reply YES for a 3-point fix checklist + draft replies."
             if not _hi_ok(mer) else
             " Ignore kiya to rating giregi. Reply YES — 3-point fix + draft replies bhejti hoon.")
    rationale = f"review theme ({theme} x{n}) with verbatim quote; loss aversion + packaged help"
    return {"body": body, "cta": "binary",
            "rationale": rationale, "template_name": "vera_review_theme_v1",
            "template_params": [sal, theme, str(n)]}


def _h_renewal_due(cat, mer, trg, cus):
    p = trg.get("payload", {}) or {}
    days = p.get("days_remaining", (mer.get("subscription", {}) or {}).get("days_remaining"))
    plan = p.get("plan", (mer.get("subscription", {}) or {}).get("plan", ""))
    amount = p.get("renewal_amount", "")
    perf = mer.get("performance", {}) or {}
    views, calls, leads = perf.get("views"), perf.get("calls"), perf.get("leads")
    sal = _salutation(mer)
    body = f"{sal}, your {plan} plan renews in {days} days"
    if amount:
        body += f" (₹{amount})"
    body += "."
    value_bits = []
    if views:
        value_bits.append(f"{views:,} views")
    if calls:
        value_bits.append(f"{calls} calls")
    if leads:
        value_bits.append(f"{leads} leads")
    if value_bits:
        body += f" Last 30 days via magicpin: {', '.join(value_bits)}."
    body += (" Reply YES to renew seamlessly, or tell me what's missing and I'll fix it first."
             if not _hi_ok(mer) else
             " Reply YES — renewal smooth karwa deta hoon, ya koi dikkat ho to pehle woh theek kara doon.")
    rationale = f"renewal due in {days} days with concrete ROI recap; honest two-way binary CTA"
    return {"body": body, "cta": "binary",
            "rationale": rationale, "template_name": "vera_renewal_v1",
            "template_params": [sal, str(days), str(amount)]}


def _h_winback(cat, mer, trg, cus):
    p = trg.get("payload", {}) or {}
    days = p.get("days_since_expiry", "")
    dip = p.get("perf_dip_pct", 0)
    lapsed_added = p.get("lapsed_customers_added_since_expiry", 0)
    catalog = _catalog_offers(cat)
    offers = _active_offers(mer)
    relaunch = offers[0] if offers else (catalog[0].get("title") if catalog else "")
    sal = _salutation(mer)
    body = f"{sal}, it's been {days} days since your offer expired"
    if dip:
        body += f" and performance has slipped {_abs_pct(dip)}"
    body += "."
    if lapsed_added:
        body += f" Meanwhile {lapsed_added} of your customers went lapsed — winnable back with one good offer."
    if relaunch:
        body += f" Suggestion: relaunch \"{relaunch}\", I'll draft everything."
    body += " Reply YES to reactivate."
    rationale = "expired-subscription win-back quantifying the cost of waiting; effort externalization"
    return {"body": body, "cta": "binary",
            "rationale": rationale, "template_name": "vera_winback_v1",
            "template_params": [sal, str(days), relaunch[:40]]}


def _h_festival(cat, mer, trg, cus):
    p = trg.get("payload", {}) or {}
    fest = p.get("festival", "the festival")
    days = p.get("days_until", "")
    date = _fmt_date(p.get("date", ""))
    relevance = p.get("category_relevance", []) or []
    slug = mer.get("category_slug", "")
    relevant = (not relevance) or slug in relevance
    sal = _salutation(mer)
    body = f"{sal}, {fest}" + (f" is {days} days away ({date})" if days else f" ({date})" if date else " is coming") + "."
    offers = _active_offers(mer)
    if relevant:
        if offers:
            body += f" High-intent period for your category — a festival angle on \"{offers[0]}\" would land well."
        else:
            body += " High-intent period for your category — worth a festival special."
    body += (" Reply YES and I'll draft the offer + GBP post today."
             if not _hi_ok(mer) else
             " Reply YES — offer + Google post ka draft aaj bana deti hoon.")
    rationale = f"festival prep ({fest}, {days} days out) tied to category relevance; drafting done for merchant"
    return {"body": body, "cta": "binary",
            "rationale": rationale, "template_name": "vera_festival_prep_v1",
            "template_params": [sal, str(fest), str(days)]}


def _h_ipl_match(cat, mer, trg, cus):
    p = trg.get("payload", {}) or {}
    match = p.get("match", "today's match")
    venue = p.get("venue", "")
    kickoff = _parse_dt(p.get("match_time_iso", ""))
    tstr = kickoff.strftime("%I:%M %p").lstrip("0") if kickoff else ""
    weeknight = p.get("is_weeknight")
    sal = _salutation(mer)
    body = f"{sal}, {match}"
    if venue:
        body += f" at {venue}"
    if tstr:
        body += f", {tstr} today"
    body += ". Match-night crowd = full tables."
    if weeknight is False:
        body += " And it's a weekend — expect walk-ins without reservations."
    body += (" Reply YES and I'll put up a match-night offer + post right now."
             if not _hi_ok(mer) else
             " Reply YES — match-night special + post abhi live kar deti hoon.")
    rationale = "same-day local event (IPL) with concrete time/venue; urgency + instant-draft CTA"
    return {"body": body, "cta": "binary",
            "rationale": rationale, "template_name": "vera_match_day_v1",
            "template_params": [sal, match, tstr]}


def _h_curious_ask(cat, mer, trg, cus):
    sal = _salutation(mer)
    q = ("Quick question: what's been your most-asked service or dish this week?"
         if not _hi_ok(mer) else
         "Ek sawaal: is week sabse zyada kaunsi service/order maangi gayi aapke yahan?")
    trend = _trend_signal(cat)
    if trend.get("query") and trend.get("delta_yoy"):
        hook = (f"I aggregate answers across similar businesses — searches for "
                f"\"{trend['query']}\" are up {_abs_pct(trend['delta_yoy'])} YoY"
                + (f" among {trend['segment_age']}" if trend.get("segment_age") else "")
                + ", so this week's answer could flag if that's hitting your walk-ins too.")
    else:
        hook = ("I aggregate answers across similar businesses — last round surfaced a demand shift "
                "most owners caught late. Your answer keeps your benchmark sharp.")
    body = f"{sal}, {q} {hook}"
    rationale = "curiosity-cadence ask (lever: asking the merchant); anchored on live category trend_signal" \
        if trend else "curiosity-cadence ask (lever: asking the merchant); promises aggregated insight back"
    return {"body": body, "cta": "open_ended",
            "rationale": rationale, "template_name": "vera_curious_ask_v1",
            "template_params": [sal, "weekly_demand_ask"]}


def _h_dormant(cat, mer, trg, cus):
    p = trg.get("payload", {}) or {}
    days = p.get("days_since_last_merchant_message", "")
    topic = _human(p.get("last_topic", ""))
    digest = _digest_item(cat)
    perf = mer.get("performance", {}) or {}
    d7 = (perf.get("delta_7d", {}) or {}).get("views_pct")
    sal = _salutation(mer)
    body = f"{sal}, long time"
    if days:
        body += f" — {days} days since we last spoke"
        if topic:
            body += f" (we were on {topic})"
    body += ". Since then:"
    hook = ""
    if digest.get("title"):
        hook = f" new research landed — \"{digest['title']}\""
    elif d7 is not None:
        hook = f" your profile got {abs(float(d7)) * 100:.0f}% {'more' if d7 >= 0 else 'fewer'} views this week"
    if hook:
        body += hook + "."
    body += (" Worth 2 minutes?" if not _hi_ok(mer) else " 2 minute mil jayenge?")
    rationale = "dormancy re-engagement with fresh, specific hook (new digest/perf data), no guilt-tripping"
    return {"body": body, "cta": "open_ended",
            "rationale": rationale, "template_name": "vera_reengage_v1",
            "template_params": [sal, str(days), topic[:30]]}


def _h_competitor_opened(cat, mer, trg, cus):
    p = trg.get("payload", {}) or {}
    cname = p.get("competitor_name", "A new competitor")
    dist = p.get("distance_km", "")
    their_offer = p.get("their_offer", "")
    opened = _fmt_date(p.get("opened_date", ""))
    offers = _active_offers(mer)
    sal = _salutation(mer)
    body = f"{sal}, heads-up: {cname} opened"
    if dist:
        body += f" just {dist} km away"
    if opened:
        body += f" ({opened})"
    body += "."
    if their_offer:
        body += f" They're pushing \"{their_offer}\"."
    if offers:
        body += f" You already have \"{offers[0]}\" — differentiation beats discounting; I can map a counter-position."
    else:
        body += " An aggressive launch usually means intro pricing — better to have your profile sharp before they ramp up."
    body += (" Reply YES for a 5-min game-plan."
             if not _hi_ok(mer) else
             " Reply YES — 5 minute ka game-plan banati hoon.")
    rationale = f"competitor opening ({cname}, {dist} km) from trigger data; defensive urgency without panic"
    return {"body": body, "cta": "binary",
            "rationale": rationale, "template_name": "vera_competitor_alert_v1",
            "template_params": [sal, cname, str(dist)]}


def _h_gbp_unverified(cat, mer, trg, cus):
    p = trg.get("payload", {}) or {}
    path = _human(p.get("verification_path", "")).replace("postcard or phone call", "postcard or a quick phone call")
    uplift = p.get("estimated_uplift_pct", 0)
    sal = _salutation(mer)
    body = f"{sal}, your Google Business Profile is still unverified — so every update goes through Google's slow review queue."
    if uplift:
        body += f" Verified listings typically see ~{_abs_pct(uplift)} more visibility."
    if path:
        body += f" Verification takes one {path}."
    body += (" Reply YES and I'll walk you through it step by step."
             if not _hi_ok(mer) else
             " Reply YES — main step-by-step guide karwati hoon, 10 minute ka kaam hai.")
    rationale = "unverified GBP blocking updates; quantified uplift + tiny effort ask"
    return {"body": body, "cta": "binary",
            "rationale": rationale, "template_name": "vera_gbp_verify_v1",
            "template_params": [sal, path[:30], _abs_pct(uplift)]}


def _h_supply_alert(cat, mer, trg, cus):
    p = trg.get("payload", {}) or {}
    molecule = _human(p.get("molecule", "a medicine"))
    batches = ", ".join(p.get("affected_batches", []) or [])
    maker = p.get("manufacturer", "")
    alert = _digest_item(cat, p.get("alert_id", ""))
    sal = _salutation(mer)
    body = f"{sal}, urgent: recall alert on {molecule}"
    if batches:
        body += f" — affected batches: {batches}"
    if maker:
        body += f" ({maker})"
    body += "."
    src = alert.get("source", "")
    if alert.get("title"):
        body += f" {alert['title']}"
    if src:
        body += f" ({src})"
    body += (" Please pull these batches from the shelf today. Reply YES once done / if you want the official circular."
             if not _hi_ok(mer) else
             " Aaj hi yeh batches shelf se hatwa lein. Reply YES — official circular bhej deti hoon.")
    rationale = f"urgent supply-chain safety alert (urgency 5); actionable + sourced"
    return {"body": body, "cta": "binary",
            "rationale": rationale, "template_name": "vera_supply_alert_v1",
            "template_params": [sal, molecule, batches[:40]]}


def _h_category_seasonal(cat, mer, trg, cus):
    p = trg.get("payload", {}) or {}
    season = _human(p.get("season", "")).replace("2026", "2026")
    trends = p.get("trends", []) or []
    pretty = []
    for t in trends[:3]:
        tt = _human(t)
        tt = re.sub(r"\+(\d+)", r"+\1%", tt)
        tt = re.sub(r"-(\d+)%?$", r"-\1%", tt)
        pretty.append(tt)
    shelf = p.get("shelf_action_recommended", False)
    sal = _salutation(mer)
    body = f"{sal}, {season} demand shifts are visible in your category:"
    if pretty:
        body += " " + "; ".join(pretty) + "."
    if shelf:
        body += (" Stocking decision window is now — early movers capture the surge."
                 if not _hi_ok(mer) else
                 " Stocking ka sahi window abhi hai — jo pehle ready, usko surge milta hai.")
    body += " Want the full trend sheet?"
    rationale = "seasonal demand trends with concrete % deltas; stocking action framed as timing advantage"
    return {"body": body, "cta": "open_ended",
            "rationale": rationale, "template_name": "vera_seasonal_trends_v1",
            "template_params": [sal, season, "; ".join(pretty)[:50]]}


def _h_planning_intent(cat, mer, trg, cus):
    p = trg.get("payload", {}) or {}
    topic = _human(p.get("intent_topic", "your new program"))
    last_msg = p.get("merchant_last_message", "")
    sal = _salutation(mer)
    catalog = _catalog_offers(cat)
    anchor = next((o.get("title") for o in catalog if o.get("title")), "")
    body = f"{sal}, picking up where we left off"
    if last_msg:
        body += f" — you asked \"{str(last_msg)[:70]}\""
    body += f". Draft outline for {topic}:"
    body += (" (1) entry tier priced like a trial, (2) core package as the flagship, "
             "(3) a bulk/corporate variant for volume.")
    if anchor:
        body += f" Pricing anchor from your category: \"{anchor}\"."
    body += (" Reply CONFIRM and I'll send the full plan + customer-facing WhatsApp copy."
             if not _hi_ok(mer) else
             " Reply CONFIRM — full plan + customer WhatsApp copy bhej deti hoon.")
    rationale = "merchant already signaled planning intent; delivering drafted structure instead of re-qualifying"
    return {"body": body, "cta": "binary",
            "rationale": rationale, "template_name": "vera_plan_draft_v1",
            "template_params": [sal, topic[:40], anchor[:40]]}


def _h_appointment_tomorrow(cat, mer, trg, cus):
    ident = (cus or {}).get("identity", {}) or {}
    name = ident.get("name", "there")
    hi = _cus_hi(cus, mer)
    body = (f"Hi {name}, {_biz_name(mer)} here — reminder for your appointment tomorrow. "
            if not hi else
            f"Hi {name}, {_biz_name(mer)} yaad dila raha hai — aapki appointment kal hai. ")
    body += ("Reply 1 to confirm, or tell us if you need a different time."
             if not hi else
             "Confirm karne ke liye 1 bhejein, ya time badalna ho to batayein.")
    rationale = "appointment confirmation for tomorrow; simple confirm/reschedule"
    return {"body": body, "cta": "multi_choice_slot",
            "rationale": rationale, "send_as": "merchant_on_behalf",
            "template_name": "merchant_appt_confirm_v1",
            "template_params": [name, "tomorrow"]}


def _h_generic(cat, mer, trg, cus):
    sal = _salutation(mer)
    kind = _human(trg.get("kind", "update"))
    perf = mer.get("performance", {}) or {}
    d7 = (perf.get("delta_7d", {}) or {}).get("views_pct")
    signals = [s for s in (mer.get("signals", []) or [])]
    digest = _digest_item(cat)
    parts = [f"{sal},"]
    if digest.get("title") and trg.get("source") == "external":
        parts.append(f"update from your category: \"{digest['title']}\".")
        if digest.get("source"):
            parts[-1] += f" ({digest['source']})"
    elif d7 is not None:
        direction = "up" if float(d7) >= 0 else "down"
        parts.append(f"your profile views are {direction} {abs(float(d7)) * 100:.0f}% this week.")
    elif signals:
        parts.append(f"a quick note on your profile: {_human(signals[0])}.")
    elif _trend_signal(cat).get("query"):
        trend = _trend_signal(cat)
        parts.append(f"searches for \"{trend['query']}\" are up {_abs_pct(trend.get('delta_yoy', 0))} "
                     "YoY in your category — worth knowing.")
    else:
        parts.append("a quick update relevant to your business.")
    body = " ".join(parts)
    body += (" Reply YES and I'll share the details."
             if not _hi_ok(mer) else
             " Reply YES — details bhejti hoon.")
    rationale = f"generic fallback composed for '{kind}' using best available specific hook"
    return {"body": body, "cta": "binary",
            "rationale": rationale, "template_name": "vera_generic_v1",
            "template_params": [sal, kind[:40]]}


KIND_ALIASES = {
    "research_digest_release": "research_digest",
    "weather_heatwave": "_external_event",
    "local_news_event": "_external_event",
    "customer_lapsed_soft": "customer_lapsed_soft",
    "customer_lapsed_hard": "customer_lapsed_hard",
}

HANDLERS = {
    "research_digest": _h_research_digest,
    "regulation_change": _h_regulation_change,
    "cde_opportunity": _h_cde_opportunity,
    "recall_due": _h_recall_due,
    "wedding_package_followup": _h_wedding_followup,
    "trial_followup": _h_trial_followup,
    "chronic_refill_due": _h_chronic_refill,
    "customer_lapsed_hard": lambda c, m, t, x: _h_customer_lapsed(c, m, t, x, hard=True),
    "customer_lapsed_soft": lambda c, m, t, x: _h_customer_lapsed(c, m, t, x, hard=False),
    "perf_spike": _h_perf_spike,
    "perf_dip": _h_perf_dip,
    "seasonal_perf_dip": _h_seasonal_perf_dip,
    "milestone_reached": _h_milestone,
    "review_theme_emerged": _h_review_theme,
    "renewal_due": _h_renewal_due,
    "winback_eligible": _h_winback,
    "festival_upcoming": _h_festival,
    "ipl_match_today": _h_ipl_match,
    "curious_ask_due": _h_curious_ask,
    "dormant_with_vera": _h_dormant,
    "competitor_opened": _h_competitor_opened,
    "gbp_unverified": _h_gbp_unverified,
    "supply_alert": _h_supply_alert,
    "category_seasonal": _h_category_seasonal,
    "active_planning_intent": _h_planning_intent,
    "appointment_tomorrow": _h_appointment_tomorrow,
}


# ── compose (challenge-brief §7.1 contract) ───────────────────────────
def compose(category: dict, merchant: dict, trigger: dict, customer: Optional[dict] = None) -> dict:
    category = category or {}
    merchant = merchant or {}
    trigger = trigger or {}
    kind = trigger.get("kind", "")
    kind = KIND_ALIASES.get(kind, kind)

    if kind == "_external_event":
        handler = _h_festival if (trigger.get("payload", {}) or {}).get("festival") else _h_generic
    else:
        handler = HANDLERS.get(kind, _h_generic)

    out = handler(category, merchant, trigger, customer)
    body = _clean_body(out.get("body", ""))
    body = _scrub_taboos(body, category)
    cta = out.get("cta", "open_ended")
    send_as = out.get("send_as", "vera")
    suppression_key = trigger.get("suppression_key", "") or f"{kind}:{merchant.get('merchant_id', '')}"

    return {
        "body": body,
        "cta": cta,
        "send_as": send_as,
        "suppression_key": suppression_key,
        "rationale": out.get("rationale", f"composed for {kind}"),
        "template_name": out.get("template_name", "vera_generic_v1"),
        "template_params": out.get("template_params", []),
    }


# ── endpoints ─────────────────────────────────────────────────────────
@app.get("/v1/healthz")
async def healthz():
    counts = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
    for scope, _cid in contexts.keys():
        if scope in counts:
            counts[scope] += 1
    return {"status": "ok", "uptime_seconds": int(time.time() - START),
            "contexts_loaded": counts}


@app.get("/v1/metadata")
async def metadata():
    return {
        "team_name": TEAM_NAME, "team_members": TEAM_MEMBERS,
        "model": "rule-composer-v2", "approach": "4-context rule composer: per-trigger-kind handlers, "
                   "category voice-taboo enforcement, trend_signal-anchored curiosity asks, "
                   "bilingual phrasing, conversation-state reply engine",
        "contact_email": CONTACT_EMAIL, "version": "2.1.0",
        "submitted_at": _utcnow().isoformat(),
    }


@app.post("/v1/context")
async def push_context(body: CtxBody):
    if body.scope not in VALID_SCOPES:
        return JSONResponse(status_code=400, content={
            "accepted": False, "reason": "invalid_scope",
            "details": f"scope must be one of {sorted(VALID_SCOPES)}"})
    if body.version < 0:
        return JSONResponse(status_code=400, content={
            "accepted": False, "reason": "invalid_version", "details": "version must be >= 0"})
    key = (body.scope, body.context_id)
    cur = contexts.get(key)
    if cur and cur["version"] >= body.version:
        return JSONResponse(status_code=409, content={
            "accepted": False, "reason": "stale_version", "current_version": cur["version"]})
    contexts[key] = {"version": body.version, "payload": body.payload}
    return {"accepted": True,
            "ack_id": f"ack_{body.context_id}_v{body.version}",
            "stored_at": _utcnow().isoformat()}


def _resolve(trigger_payload: dict):
    mid = trigger_payload.get("merchant_id")
    merchant = contexts.get(("merchant", mid), {}).get("payload") if mid else None
    if not merchant:
        # tolerate triggers nested under payload (some harnesses wrap them)
        inner = trigger_payload.get("payload", {}) or {}
        if not mid and inner.get("merchant_id"):
            mid = inner["merchant_id"]
            merchant = contexts.get(("merchant", mid), {}).get("payload")
    slug = (merchant or {}).get("category_slug", "")
    category = contexts.get(("category", slug), {}).get("payload") if slug else None
    cid = trigger_payload.get("customer_id") or (trigger_payload.get("payload", {}) or {}).get("customer_id")
    customer = contexts.get(("customer", cid), {}).get("payload") if cid else None
    return merchant, category, customer


def _unique_conv_id(base: str) -> str:
    cid, n = base, 2
    while cid in used_conv_ids:
        cid = f"{base}_{n}"
        n += 1
    used_conv_ids.add(cid)
    return cid


@app.post("/v1/tick")
async def tick(body: TickBody):
    now_dt = _parse_dt(body.now) or _utcnow()

    candidates = []
    for tid in body.available_triggers:
        trg = contexts.get(("trigger", tid), {}).get("payload")
        if not isinstance(trg, dict):
            continue
        exp = _parse_dt(trg.get("expires_at", ""))
        if exp and exp < now_dt:
            continue
        skey = trg.get("suppression_key", "") or tid
        if skey in suppressed_keys or tid in suppressed_keys:
            continue
        candidates.append((trg, tid, skey))

    # highest urgency first, cap one action per merchant per tick
    def urgency(pair):
        return pair[0].get("urgency", 1) or 1

    actions, seen_merchants, seen_customers = [], set(), set()
    for trg, tid, skey in sorted(candidates, key=urgency, reverse=True):
        merchant, category, customer = _resolve(trg)
        if not merchant or not category:
            continue
        mid = merchant.get("merchant_id", "")
        if mid in opted_out:
            continue

        comp = compose(category, merchant, trg, customer)

        # one merchant-facing action per merchant per tick; customer-facing
        # messages go to a different chat, so they don't consume the slot
        if comp["send_as"] == "vera":
            if mid in seen_merchants:
                continue
            seen_merchants.add(mid)
        else:
            cid = trg.get("customer_id") or ""
            if cid in seen_customers:
                continue
            seen_customers.add(cid)

        suppressed_keys.add(skey)
        if tid:
            suppressed_keys.add(tid)

        conv_id = _unique_conv_id(f"conv_{mid}_{tid}")
        topic_bits = [trg.get("kind", ""), (trg.get("payload", {}) or {}).get("intent_topic", ""),
                      (_digest_item(category, (trg.get("payload", {}) or {}).get("top_item_id", "")).get("title", ""))]
        conversations[conv_id] = {
            "merchant_id": mid,
            "customer_id": trg.get("customer_id"),
            "trigger_id": tid,
            "topic": next((t for t in topic_bits if t), trg.get("kind", "")),
            "last_bot_body": comp["body"],
            "turns": [{"from": "bot", "msg": comp["body"], "turn": 1}],
        }

        actions.append({
            "conversation_id": conv_id,
            "merchant_id": mid,
            "customer_id": trg.get("customer_id"),
            "send_as": comp["send_as"],
            "trigger_id": tid,
            "template_name": comp["template_name"],
            "template_params": comp["template_params"] or [comp["body"][:50]],
            "body": comp["body"],
            "cta": comp["cta"],
            "suppression_key": skey,
            "rationale": comp["rationale"],
        })
        if len(actions) >= 20:
            break

    return {"actions": actions}


# ── reply engine ──────────────────────────────────────────────────────
AUTO_REPLY_PATTERNS = [
    r"thank you for contacting",
    r"our team will (respond|get back|reach out|contact)",
    r"we (will|'ll) get back to you",
    r"(thanks|thank you) for reaching out",
    r"automated (assistant|message|reply|response)",
    r"auto[- ]?reply",
    r"out of (office|town) until",
    r"will contact you shortly",
    r"received your message",
    r"main (ek )?automated",
]

OPT_OUT_PATTERNS = [
    r"\bstop (messaging|sending|texting)\b",
    r"\bunsubscribe\b", r"\bopt ?out\b",
    r"\bnot interested\b",
    r"\bdon'?t message\b", r"\bdo not message\b",
    r"\bmat bhejo\b", r"\bbakwas\b",
    r"\bharamkhor\b", r"\bidiot\b", r"\bstupid bot\b", r"\buseless\b",
]

INTENT_PATTERNS = [
    r"\blet'?s do it\b", r"\blet'?s go\b", r"\bgo ahead\b",
    r"\bsounds good\b", r"\bi(?:'| a)m in\b",
    r"\bproceed\b", r"\bplease (do|start|send|share|draft)\b",
    r"\byes please\b", r"\bhaan\b.*\bkar(o|wa)\b", r"\bkarwa? do\b", r"\bshuru kar(e|i)?\b",
    r"\bjoin magicpin\b", r"\bmujhe judna\b", r"\bjudrna hai\b", r"\badd karna hai\b",
    r"\bstart my\b", r"\brenew\b", r"\bi want (to|it)\b",
]

AFFIRM_PATTERNS = [r"^\s*(yes|yeah|yep|ok(?:ay)?|sure|haan|ha|ji|theek|thik|chalega|sahi)\b"]
NEGATIVE_PATTERNS = [r"^\s*(no|nope|nahi|nahin|nahi?)\b", r"\babhi nahi\b", r"\bnot now\b"]

OUT_OF_SCOPE_PATTERNS = [r"\bgst\b", r"\bincome tax\b", r"\bca\b.*\bfil", r"\baccounting software\b",
                         r"\bstaff salary\b", r"\bloan\b", r"\bkyc\b"]

PRICING_PATTERNS = [r"\bprice\b", r"\brate\b", r"\bcost\b", r"\bkitn[ea]\b", r"\bcharge\b", r"\bfee\b",
                    r"\bpaisa\b", r"\brupya\b"]


def _matches(text: str, patterns) -> bool:
    return any(re.search(p, text) for p in patterns)


def _action_mode(conv: dict, merchant: Optional[dict]) -> dict:
    topic = (conv or {}).get("topic", "")
    topic_txt = f" on the {topic.replace('_', ' ')} thread" if topic and topic != "recall_due" else ""
    body = (f"On it{topic_txt}. I'm preparing it now - draft will land here in 2 minutes. "
            "Reply CONFIRM to go live, or tell me one tweak.")
    if _matches(body.lower(), QUALIFYING_TRAPS):
        body = "On it. Draft coming in 2 minutes — reply CONFIRM to go live."
    return {"action": "send", "body": body, "cta": "binary",
            "rationale": "explicit commitment detected; switched to execution mode with concrete deliverable + confirm gate"}


QUALIFYING_TRAPS = ["would you", "do you", "can you tell", "what if", "how about"]


def _redirect_body(conv: dict, merchant: Optional[dict]) -> str:
    topic = (conv or {}).get("topic", "").replace("_", " ")
    offers = _active_offers(merchant or {})
    tail = f" Coming back to the {topic} — shall I send it over?" if topic else \
           " Coming back to your growth checklist — want me to pick the top item for you?"
    if offers and not topic:
        tail = f" Meanwhile, your active offer \"{offers[0]}\" could use a push — want me to set it up?"
    return tail


@app.post("/v1/reply")
async def reply(body: ReplyBody):
    msg = body.message or ""
    low = msg.lower().strip()

    conv = conversations.setdefault(body.conversation_id, {
        "merchant_id": body.merchant_id, "customer_id": body.customer_id,
        "trigger_id": None, "topic": "", "last_bot_body": "", "turns": []})
    if body.merchant_id and not conv.get("merchant_id"):
        conv["merchant_id"] = body.merchant_id
    mid = conv.get("merchant_id") or ""

    merchant = contexts.get(("merchant", mid), {}).get("payload") if mid else None
    turn = body.turn_number or (len(conv.get("turns", [])) + 2)
    conv.setdefault("turns", []).append({"from": body.from_role, "msg": msg, "turn": turn})

    def respond(action: str, **kw) -> dict:
        if action == "send":
            out_body = kw.get("body", "")
            if out_body == conv.get("last_bot_body"):
                out_body = out_body.rstrip(".") + " - anything you'd tweak?"
            conv["last_bot_body"] = out_body
            conv["turns"].append({"from": "bot", "msg": out_body, "turn": turn})
            return {"action": "send", "body": _clean_body(out_body),
                    "cta": kw.get("cta", "open_ended"), "rationale": kw.get("rationale", "")}
        if action == "wait":
            return {"action": "wait", "wait_seconds": kw.get("wait_seconds", 1800),
                    "rationale": kw.get("rationale", "")}
        return {"action": "end", "rationale": kw.get("rationale", "")}

    # genuine human message → reset auto-reply streak
    if not _matches(low, AUTO_REPLY_PATTERNS):
        auto_reply_counts.pop(mid, None)

    # 1) hostility / explicit opt-out
    if _matches(low, OPT_OUT_PATTERNS):
        opted_out.add(mid)
        return respond("end", rationale=(
            "Merchant frustration/opt-out is explicit; ending gracefully and suppressing all "
            "future proactive sends for this merchant."))

    # 2) auto-reply detection ladder: 1st → flag it, 2nd → wait 24h, 3rd+ → end
    if _matches(low, AUTO_REPLY_PATTERNS):
        n = auto_reply_counts.get(mid, 0) + 1
        auto_reply_counts[mid] = n
        if n == 1:
            return respond("send",
                           body=("Looks like an auto-reply! When the owner sees this - just reply YES "
                                 "and I'll pick the conversation right back up."),
                           cta="binary",
                           rationale="Detected canned auto-reply; one light prompt so the owner engages when they see it")
        if n == 2:
            return respond("wait", wait_seconds=86400,
                           rationale="Same auto-reply twice in a row — owner not at phone; backing off 24h")
        return respond("end", rationale=(
            f"Auto-reply {n}x in a row with zero engagement signal; closing to avoid burning turns."))

    # 3) explicit intent / commitment → execution mode immediately
    if _matches(low, INTENT_PATTERNS):
        return respond(**_action_mode(conv, merchant))

    # 4) out-of-scope asks → polite decline + redirect to mission
    if _matches(low, OUT_OF_SCOPE_PATTERNS):
        return respond("send",
                       body=("That one's outside my lane — I'd rather not guess at it. "
                             + _redirect_body(conv, merchant)),
                       cta="open_ended",
                       rationale="Out-of-scope request declined honestly; redirected to the original thread")

    # 5) pricing questions → answer from real offers/catalog
    if _matches(low, PRICING_PATTERNS):
        offers = _active_offers(merchant or {})
        if offers:
            return respond("send",
                           body=(f"Current active offer: {offers[0]}."
                                 + (f" Also live: {', '.join(offers[1:3])}." if len(offers) > 1 else "")
                                 + " Want me to feature it more prominently on your listing?"),
                           cta="binary",
                           rationale="Pricing question answered verbatim from merchant's real offers; upsell ask kept binary")
        cat_slug = (merchant or {}).get("category_slug", "")
        cat = contexts.get(("category", cat_slug), {}).get("payload") if cat_slug else None
        catalog = _catalog_offers(cat or {})
        if catalog:
            titles = [o.get("title", "") for o in catalog[:2]]
            return respond("send",
                           body=f"For your category these work well: {'; '.join(titles)}. Want me to set one up for you?",
                           cta="binary",
                           rationale="No merchant offers live; quoted category offer_catalog patterns instead of inventing prices")
        return respond("send",
                       body="Share your menu/rate card and I'll structure an offer that fits your margins.",
                       cta="open_ended",
                       rationale="No pricing data available anywhere; asking instead of fabricating")

    # 6) bare affirmative → advance with a concrete next step (never re-qualify)
    if _matches(low, AFFIRM_PATTERNS):
        topic = conv.get("topic", "").replace("_", " ")
        line = f"Great — moving ahead{(' on ' + topic) if topic else ''}."
        return respond("send",
                       body=line + " I'll prepare it and follow up here. Anything specific you want included?",
                       cta="open_ended",
                       rationale="Affirmative accepted; advancing to preparation instead of asking another qualifying question")

    # 7) bare negative → graceful close
    if _matches(low, NEGATIVE_PATTERNS):
        return respond("end", rationale=(
            "Merchant declined; closing politely and suppressing this thread. Door stays open."))

    # 8) question we might answer from context
    if "?" in msg or low.startswith(("what", "how", "when", "kab", "kaise", "kya")):
        return respond("send",
                       body=("Good question — let me get you the exact numbers rather than guess. "
                             + _redirect_body(conv, merchant)),
                       cta="open_ended",
                       rationale="Question acknowledged; committing to verified data instead of improvising facts")

    # 9) unclear - brief acknowledgment + gentle redirect
    return respond("send",
                   body=("Noted - " + _redirect_body(conv, merchant)),
                   cta="open_ended",
                   rationale="Unclear reply; acknowledging and offering the lowest-friction next step")


@app.post("/v1/teardown")
async def teardown():
    contexts.clear()
    conversations.clear()
    suppressed_keys.clear()
    used_conv_ids.clear()
    opted_out.clear()
    auto_reply_counts.clear()
    return {"wiped": True}


@app.get("/")
async def root():
    return {"status": "magicpin AI Challenge Bot", "version": "2.0.1"}