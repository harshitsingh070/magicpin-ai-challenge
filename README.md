# magicpin AI Challenge - Submission README

## Approach

`bot.py` is a **rule-based, deterministic 4-context composer** - no LLM call in the
composition path. Each trigger `kind` (24 covered, plus aliases for the rest named in
the brief) has its own handler function (`_h_research_digest`, `_h_perf_dip`,
`_h_recall_due`, etc.) that pulls the specific verifiable facts out of `category`,
`merchant`, `trigger`, and optional `customer` and assembles a WhatsApp-ready message:
peer-tone/clinical vocabulary per category, service+price anchors instead of generic
"% off", a single binary or open-ended CTA, and Hindi-English code-mix when the
merchant's `languages` includes `hi`.

`/v1/reply` runs a small conversation-state machine (auto-reply detection ladder,
opt-out/hostility handling, explicit-intent -> action-mode routing, pricing questions
answered from real offer data, anti-repetition guard) so multi-turn behavior doesn't
need a separate `conversation_handlers.py` - it's built into `bot.py`.

## Tradeoffs

- **Determinism over polish**: rules guarantee reproducible, on-spec output (no
  hallucination risk, <30s always) at the cost of the more adaptive phrasing an LLM
  composer could produce for genuinely novel trigger/merchant combinations outside the
  handled kinds (those fall back to `_h_generic`, which is intentionally conservative).
- **URLs are stripped** from every composed body (`_clean_body`), stricter than the
  brief's "URLs allowed when they add value" - chosen to avoid ever emitting a
  malformed or unapproved link, at the cost of not using URLs where they'd genuinely help.
- **No LLM dependency**: `.env` is wired for Groq but unused by `bot.py`, so the bot has
  zero external-call latency/cost and can't leak context to a third-party API - but it
  also can't generalize past its handler map the way a prompted LLM would.

## What additional context would have helped most

- A larger sample of real merchant replies per category (beyond the reference
  excerpts in the brief) to tune the regex-based intent/auto-reply/opt-out detectors,
  which are currently pattern-based and English/Hindi-only.
- Explicit peer-stat deltas over time (not just a snapshot) to make "you're below peer
  median" framings sharper for `perf_dip`/`perf_spike` handlers.

## Fix log

- Corrected an intent-detection regex bug: `\blets? do it\b` didn't match the
  apostrophized `"let's do it"` (only bare `"let"/"lets"`), which meant the exact
  phrase used in the brief's own Phase 4 "Intent transition" replay scenario fell
  through to a generic reply instead of switching to action mode. Fixed to
  `\blet'?s do it\b`, verified against both `"let's do it"` and `"lets do it"`.

## Files

- `bot.py` - the bot (FastAPI, 5 required endpoints + `/v1/teardown`)
- `submission.jsonl` - composed output for all 30 canonical test pairs
- `README.md` - this file
