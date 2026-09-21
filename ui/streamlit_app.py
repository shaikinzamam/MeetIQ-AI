"""Streamlit UI.   Run from the project root:  streamlit run ui/streamlit_app.py"""
import html
import json
import sys
from collections import Counter
from pathlib import Path

import streamlit as st

# Streamlit puts the script's folder (ui/) on sys.path, not the project root,
# so we add the root to be able to `import app...`.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402
from app.errors import MeetIQError  # noqa: E402
from app.schemas import MeetingReport  # noqa: E402
from app.service import MeetingAnalyzer  # noqa: E402

SAMPLE_PATH = ROOT / "data" / "sample_transcript.txt"
PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2}

st.set_page_config(page_title="MeetIQ", page_icon="🧠", layout="wide")

# ----------------------------------------------------------------------------------------
# Styling. Colors use transparency so everything works in both light and dark themes.
# The left "rail" on each row encodes priority/severity: red = high, amber = medium, green = low.
# ----------------------------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Hanken+Grotesk:wght@400;500;600&family=Newsreader:opsz,wght@6..72,400;6..72,600&display=swap');

.stApp, .stMarkdown p, .stMarkdown li, .stTextArea textarea, .stButton button,
[data-testid="stMetric"], [data-testid="stSidebar"] p, [data-testid="stCaptionContainer"] {
    font-family: 'Hanken Grotesk', system-ui, sans-serif;
}
.block-container { padding-top: 4.2rem; max-width: 1150px; }
[data-testid="stAppDeployButton"], .stAppDeployButton { display: none; }

button[kind="primary"], [data-testid="stBaseButton-primary"] { background-color: #0d9488; border-color: #0d9488; color: #fff; }
button[kind="primary"]:hover:not(:disabled), [data-testid="stBaseButton-primary"]:hover:not(:disabled) { background-color: #0f766e; border-color: #0f766e; }
.stTextArea textarea:focus { border-color: #0d9488; box-shadow: 0 0 0 1px #0d9488; }

.app-title { font-family: 'Newsreader', Georgia, serif; font-size: 2.6rem; font-weight: 600;
             line-height: 1.1; margin: 0; }
.app-tagline { font-size: 1.05rem; opacity: .75; margin: .35rem 0 1.4rem 0; }

.brief-title { font-family: 'Newsreader', Georgia, serif; font-size: 2.1rem; font-weight: 600;
               line-height: 1.2; margin: .2rem 0 .2rem 0; }
.brief-meta { font-size: .85rem; opacity: .6; margin-bottom: 1rem; }
.brief-summary { font-family: 'Newsreader', Georgia, serif; font-size: 1.25rem; line-height: 1.65;
                 max-width: 68ch; margin: .4rem 0 1rem 0; }
.section-title { font-family: 'Newsreader', Georgia, serif; font-size: 1.45rem; font-weight: 600;
                 margin: 1.6rem 0 .6rem 0; }

.row { --rail: rgba(128,128,128,.5); border: 1px solid rgba(128,128,128,.25);
       border-left: 5px solid var(--rail); border-radius: 6px; padding: .7rem .95rem;
       margin-bottom: .5rem; }
.row.high   { --rail: #e5484d; }
.row.medium { --rail: #f0a020; }
.row.low    { --rail: #30a46c; }
.row-main { font-weight: 500; line-height: 1.45; }
.row-sub  { font-size: .9rem; opacity: .7; margin-top: .25rem; }
.row-meta { margin-top: .5rem; display: flex; flex-wrap: wrap; gap: .4rem; }

.chip { font-size: .78rem; padding: .08rem .6rem; border-radius: 999px;
        border: 1px solid rgba(128,128,128,.4); white-space: nowrap; }
.chip.warn { border-style: dashed; border-color: #f0a020; color: #f0a020; }

.decision { display: flex; gap: .6rem; padding: .55rem 0;
            border-bottom: 1px solid rgba(128,128,128,.2); }
.decision .tick { color: #30a46c; font-weight: 600; }

.workload { max-width: 540px; margin: .2rem 0 1.1rem 0; }
.bar-row { display: grid; grid-template-columns: 8.5rem 1fr 1.6rem; align-items: center;
           gap: .7rem; margin: .3rem 0; font-size: .92rem; }
.bar-track { height: 10px; border-radius: 5px; background: rgba(128,128,128,.2); overflow: hidden; }
.bar-fill { display: block; height: 100%; border-radius: 5px; background: #0d9488; }
.bar-fill.warn { background: repeating-linear-gradient(45deg, #f0a020 0 4px, transparent 4px 8px); }
.bar-n { text-align: right; opacity: .7; }

.empty { opacity: .6; font-style: italic; padding: .3rem 0; }
.hint  { border: 1px dashed rgba(128,128,128,.4); border-radius: 8px; padding: 1.4rem 1.6rem;
         margin-top: 1.2rem; }
.empty-title { font-family: 'Newsreader', Georgia, serif; font-size: 1.4rem; font-weight: 600; margin-bottom: .3rem; }
.hint b { font-family: 'Newsreader', Georgia, serif; font-size: 1.2rem; font-weight: 600; }
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------- helpers
def esc(text: object) -> str:
    """LLM output is untrusted text, so always escape it before putting it inside HTML."""
    return html.escape(str(text))


def chip(text: str, kind: str = "") -> str:
    return f'<span class="chip {kind}">{esc(text)}</span>'


def action_row(task) -> str:
    owner = chip(task.owner) if task.owner != "Unassigned" else chip("Needs an owner", "warn")
    due = chip(f"Due {task.deadline}") if task.deadline else chip("No deadline", "warn")
    return (
        f'<div class="row {task.priority}"><div class="row-main">{esc(task.task)}</div>'
        f'<div class="row-meta">{owner}{due}{chip(task.priority + " priority")}</div></div>'
    )


def risk_row(risk) -> str:
    mitigation = f'<div class="row-sub">Mitigation: {esc(risk.mitigation)}</div>' if risk.mitigation else ""
    return (
        f'<div class="row {risk.severity}"><div class="row-main">{esc(risk.description)}</div>'
        f'{mitigation}<div class="row-meta">{chip(risk.type)}{chip(risk.severity + " severity")}</div></div>'
    )


def decision_row(d) -> str:
    context = f'<div class="row-sub">{esc(d.context)}</div>' if d.context else ""
    return f'<div class="decision"><span class="tick">✓</span><div>{esc(d.decision)}{context}</div></div>'


def workload_bars(items) -> str:
    """Tasks per owner. Unassigned work is hatched amber so it can't be missed."""
    counts = Counter(t.owner for t in items)
    top = max(counts.values())
    rows = []
    for owner, n in sorted(counts.items(), key=lambda kv: (kv[0] == "Unassigned", -kv[1], kv[0])):
        warn = " warn" if owner == "Unassigned" else ""
        label = "Needs an owner" if owner == "Unassigned" else owner
        rows.append(
            f'<div class="bar-row"><span>{esc(label)}</span>'
            f'<span class="bar-track"><span class="bar-fill{warn}" style="width:{100 * n / top:.0f}%"></span></span>'
            f'<span class="bar-n">{n}</span></div>'
        )
    return '<div class="workload">' + "".join(rows) + "</div>"


def format_time_saved(seconds: int) -> str:
    """0 seconds means the transcript is too short to save time, so say so instead of showing 0.0."""
    if seconds <= 0:
        return "n/a"
    if seconds < 60:
        return f"{seconds} sec"
    return f"{seconds / 60:.1f} min"


def to_markdown(report: MeetingReport) -> str:
    """A shareable plain-text version of the report (for email, Notion, Docs...)."""
    a = report.analysis
    lines = [f"# {a.title}", "", a.summary, "", "## Key points", *[f"- {p}" for p in a.key_points], ""]
    lines += ["## Decisions", *[f"- {d.decision}" + (f" ({d.context})" if d.context else "") for d in a.decisions], ""]
    lines += ["## Action items", "", "| Task | Owner | Deadline | Priority |", "|---|---|---|---|"]
    lines += [f"| {t.task} | {t.owner} | {t.deadline or '-'} | {t.priority} |" for t in a.action_items]
    lines += ["", "## Risks and blockers"]
    lines += [
        f"- **{r.type.title()} ({r.severity})**: {r.description}" + (f" Mitigation: {r.mitigation}" if r.mitigation else "")
        for r in a.risks_blockers
    ]
    return "\n".join(lines) + "\n"


@st.cache_resource
def get_analyzer() -> MeetingAnalyzer:
    return MeetingAnalyzer()


def load_sample() -> None:  # callbacks run BEFORE widgets re-render, which avoids state errors
    st.session_state.transcript = SAMPLE_PATH.read_text(encoding="utf-8")


def load_sample_and_run() -> None:
    load_sample()
    st.session_state.auto_run = True


def load_upload() -> None:
    f = st.session_state.get("uploaded_file")
    if f is not None:
        st.session_state.transcript = f.read().decode("utf-8", errors="ignore")


st.session_state.setdefault("transcript", "")
st.session_state.setdefault("report", None)

settings = get_settings()
try:
    analyzer, config_error = get_analyzer(), None
except Exception as exc:  # MeetIQError or anything unexpected: show a message, never a traceback
    analyzer, config_error = None, str(exc)

# ---------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### How it works")
    st.markdown("1. Paste or upload a transcript\n2. The AI reads it and extracts the key facts\n3. You get a brief you can scan in a minute")
    st.markdown("### Model")
    st.caption(analyzer.client.name if analyzer else "Not connected")
    st.markdown("### Why it matters")
    st.caption(
        "Built for SDG 8 (Decent Work and Economic Growth): less time re-reading meetings, "
        "and every task gets a clear owner and deadline."
    )
    st.caption("Time-saved figures are estimates based on a reading speed of 200 words per minute.")

# ---------------------------------------------------------------------------- header + input
st.markdown('<div class="app-title">MeetIQ</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-tagline">Turn a meeting transcript into decisions, tasks, and risks.</div>',
    unsafe_allow_html=True,
)

if config_error:
    st.error(f"{config_error} Then restart the app.")

st.markdown('<div class="section-title">Your meeting</div>', unsafe_allow_html=True)
with st.container(border=True):
    left, right = st.columns([3, 1])
    with left:
        st.text_area(
            "Meeting transcript",
            key="transcript",
            height=240,
            placeholder="Paste your meeting transcript here…",
            label_visibility="collapsed",
        )
    with right:
        st.button("Load sample transcript", on_click=load_sample, width="stretch")
        st.file_uploader("Or upload a .txt file", type=["txt"], key="uploaded_file", on_change=load_upload)

    n_chars = len(st.session_state.transcript)
    n_words = len(st.session_state.transcript.split())
    too_short = n_chars < settings.min_transcript_chars
    too_long = n_chars > settings.max_transcript_chars

    info, action = st.columns([3, 1], vertical_alignment="center")
    with info:
        if too_long:
            st.warning("This transcript is too long. Split it into smaller parts.")
        elif n_chars and too_short:
            st.caption(f"{n_words:,} words. Add at least {settings.min_transcript_chars} characters to analyze.")
        else:
            st.caption(f"{n_words:,} words · {n_chars:,} of {settings.max_transcript_chars:,} characters")
    with action:
        run = st.button(
            "Analyze meeting", type="primary", width="stretch",
            disabled=bool(config_error) or too_short or too_long,
        )

run = run or st.session_state.pop("auto_run", False)
if run and not (config_error or too_short or too_long):
    try:
        with st.spinner("Reading the transcript…"):
            st.session_state.report = analyzer.analyze(st.session_state.transcript)
    except MeetIQError as exc:  # expected problems: bad input, API failure, invalid AI output
        st.session_state.report = None
        st.error(str(exc))
    except Exception as exc:  # anything else: still show a readable message
        st.session_state.report = None
        st.error("Something unexpected went wrong. Try again, and check the terminal for details.")
        with st.expander("Technical details"):
            st.code(f"{type(exc).__name__}: {exc}")

# ---------------------------------------------------------------------------- results
report: MeetingReport | None = st.session_state.report

if not report:
    with st.container(border=True):
        st.markdown(
            '<div class="empty-title">Ready when you are</div>'
            "<p>Paste a transcript above, or see what MeetIQ produces from a sample meeting. "
            "You will get:</p>"
            "<ul><li>A short summary and the key points</li>"
            "<li>Action items with an owner, a deadline, and a priority</li>"
            "<li>The decisions the team made</li>"
            "<li>Risks and blockers, and how much reading time you saved</li></ul>",
            unsafe_allow_html=True,
        )
        st.button("Try it with a sample meeting", type="primary", on_click=load_sample_and_run,
                  disabled=bool(config_error))
    st.stop()

a, m = report.analysis, report.metrics
st.divider()
st.markdown(f'<div class="brief-title">{esc(a.title)}</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="brief-meta">{m.transcript_words:,} words analyzed with {esc(report.model_used)}</div>',
    unsafe_allow_html=True,
)

stats = [
    ("Decisions", len(a.decisions), None),
    ("Action items", m.action_items_total, None),
    ("Time saved per reader", format_time_saved(m.est_seconds_saved),
     "Estimated reading time of the transcript minus the brief. Shows n/a when the transcript "
     "is too short to save time (a typical 30-minute meeting is about 4,500 words)."),
    ("Accountability", "n/a" if m.accountability_score is None else f"{m.accountability_score:.0f}%",
     "Share of action items that have both an owner and a deadline."),
]
for col, (label, value, help_text) in zip(st.columns(4), stats):
    with col.container(border=True):
        st.metric(label, value, help=help_text)

# Summary
st.markdown(f'<div class="brief-summary">{esc(a.summary)}</div>', unsafe_allow_html=True)
if a.key_points:
    with st.expander("Key points", expanded=True):
        for p in a.key_points:
            st.markdown(f"- {p}")

# Action items (sorted by priority, filterable by owner)
st.markdown('<div class="section-title">Action items</div>', unsafe_allow_html=True)
if not a.action_items:
    st.markdown('<div class="empty">No action items were identified in this meeting.</div>', unsafe_allow_html=True)
else:
    st.caption("Tasks per person. A long bar means one person is carrying a lot.")
    st.markdown(workload_bars(a.action_items), unsafe_allow_html=True)
    owners = sorted({t.owner for t in a.action_items})
    chosen = st.multiselect("Filter by owner", owners, placeholder="Show everyone") if len(owners) > 1 else []
    items = [t for t in a.action_items if not chosen or t.owner in chosen]
    items.sort(key=lambda t: PRIORITY_ORDER[t.priority])
    st.markdown("".join(action_row(t) for t in items), unsafe_allow_html=True)
    missing = m.action_items_total - m.action_items_with_owner
    if missing:
        st.warning(f"{missing} task(s) have no owner. Assign one so they don't fall through the cracks.")

# Decisions and risks side by side
col_dec, col_risk = st.columns(2, gap="large")
with col_dec:
    st.markdown('<div class="section-title">Decisions</div>', unsafe_allow_html=True)
    if a.decisions:
        st.markdown("".join(decision_row(d) for d in a.decisions), unsafe_allow_html=True)
    else:
        st.markdown('<div class="empty">No decisions were recorded.</div>', unsafe_allow_html=True)
with col_risk:
    st.markdown('<div class="section-title">Risks and blockers</div>', unsafe_allow_html=True)
    if a.risks_blockers:
        st.markdown("".join(risk_row(r) for r in a.risks_blockers), unsafe_allow_html=True)
    else:
        st.markdown('<div class="empty">No risks or blockers were raised.</div>', unsafe_allow_html=True)

# Export
st.markdown('<div class="section-title">Export</div>', unsafe_allow_html=True)
payload = json.dumps(report.model_dump(), indent=2, ensure_ascii=False)
d1, d2, _ = st.columns([1, 1, 2])
d1.download_button("Download report (.md)", to_markdown(report), "meetiq_report.md", "text/markdown", width="stretch")
d2.download_button("Download data (.json)", payload, "meetiq_report.json", "application/json", width="stretch")
with st.expander("View raw JSON"):
    st.code(payload, language="json")