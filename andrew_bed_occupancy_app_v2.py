"""
Hospital Bed Occupancy Predictor (v2) — Streamlit App
--------------------------------------------------------
Same pipeline and UI as andrew_bed_occupancy_app.py, but trained on
ANDREW_DATASET_V2.csv — a dataset generated for this app, with a
genuine engineered feature-target relationship (not the real,
recorded ANDREW_DATASET.csv). It is not a measurement of any real
hospital's behavior; do not use its predictions for actual clinical
or staffing decisions.

UI is a chat-thread layout inspired by ChatGPT's interface (dark
sidebar with session history, centered thread, fixed bottom
composer) restyled for a structured prediction form rather than
free text — no OpenAI branding, logos, or assets are used.

Run:
    streamlit run /Users/user/anaconda_projects/andrew_bed_occupancy_app_v2.py
"""

import os
import math
import time as time_module
from datetime import datetime, date, time

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "ANDREW_DATASET_V2.csv")
DATA_FILENAME = "ANDREW_DATASET.csv"

NORMAL_MAX = 71.2
ELEVATED_MAX = 81.0
PREDICTOR_COLUMNS = ["admissions", "discharges", "staff_count", "Month", "Day", "DayOfWeek", "Hour", "net_flow"]

DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MONTH_NAMES = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def html(s):
    """Render a multi-line HTML string via st.markdown.

    Two Streamlit markdown quirks are worked around here:
    1. Leading whitespace on a line is treated as a Markdown code
       block, so nested `with`/`if` blocks (whose Python indentation
       leaks into the string) would otherwise render as literal text.
    2. A blank line inside a large raw-HTML block (e.g. a long
       <style> block) makes the parser drop out of "raw HTML" mode
       partway through, leaking the remainder as visible text — so
       blank lines are stripped entirely, not just dedented.
    """
    cleaned = "\n".join(line.strip() for line in s.strip("\n").splitlines() if line.strip())
    st.markdown(cleaned, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ICONS — inline SVG (Lucide-style outline icons), no emoji anywhere in the UI
# ─────────────────────────────────────────────────────────────────────────────

ICONS = {
    "bed": '<rect x="3" y="11" width="18" height="6" rx="1.5"/><path d="M3 17v3M21 17v3"/><rect x="5" y="8" width="6" height="3.5" rx="1"/>',
    "plus": '<path d="M12 5v14M5 12h14"/>',
    "info": '<circle cx="12" cy="12" r="9"/><path d="M12 16v-5"/><circle cx="12" cy="8" r="0.75" fill="currentColor" stroke="none"/>',
    "arrow-up": '<path d="M12 19V5"/><path d="M5 12l7-7 7 7"/>',
    "activity": '<path d="M3 12h4l3 8 4-16 3 8h4"/>',
    "bar-chart": '<path d="M3 3v18h18"/><rect x="7" y="12" width="3" height="6"/><rect x="12" y="8" width="3" height="10"/><rect x="17" y="5" width="3" height="13"/>',
    "user": '<circle cx="12" cy="8" r="4"/><path d="M4 21c0-4.5 3.5-8 8-8s8 3.5 8 8"/>',
}


def icon(name, size=16, stroke="currentColor", stroke_width=2):
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{stroke}" '
        f'stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round" '
        f'style="display:inline-block;vertical-align:middle;flex-shrink:0;">{ICONS[name]}</svg>'
    )


def icon_data_uri(name, color="black", stroke_width=2.2):
    import urllib.parse

    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="{color}" '
        f'stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round">{ICONS[name]}</svg>'
    )
    return "data:image/svg+xml," + urllib.parse.quote(svg)


def button_icon_css(container_key, icon_name):
    """CSS to prepend an SVG icon to a native st.button via mask-image,
    since st.button's label can't contain raw HTML/SVG."""
    uri = icon_data_uri(icon_name)
    return f"""
    .st-key-{container_key} button::before {{
        content: "";
        display: inline-block; width: 15px; height: 15px;
        background-color: currentColor;
        -webkit-mask: url('{uri}') no-repeat center / contain;
        mask: url('{uri}') no-repeat center / contain;
        margin-right: 8px; vertical-align: -3px;
    }}
    """


# ─────────────────────────────────────────────────────────────────────────────
# FAVICON — generated to match the in-app bed icon (no emoji favicon either)
# ─────────────────────────────────────────────────────────────────────────────

FAVICON_PATH = os.path.join(BASE_DIR, ".bed_occupancy_favicon.png")
if not os.path.exists(FAVICON_PATH):
    from PIL import Image, ImageDraw

    size = 64
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([0, 0, size - 1, size - 1], radius=16, fill=(12, 110, 140, 255))
    draw.rounded_rectangle([10, 32, 54, 46], radius=4, outline="white", width=3)
    draw.line([10, 46, 10, 52], fill="white", width=3)
    draw.line([54, 46, 54, 52], fill="white", width=3)
    draw.rounded_rectangle([14, 22, 28, 32], radius=3, outline="white", width=3)
    img.save(FAVICON_PATH)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG + THEME
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Hospital Bed Occupancy Predictor (v2)", page_icon=FAVICON_PATH, layout="wide")

if "history" not in st.session_state:
    st.session_state["history"] = []
if "pending" not in st.session_state:
    st.session_state["pending"] = None

# Dark theme only (by design — see user request to drop light mode).
T = {
    "bg_main": "#212121",
    "bg_sidebar": "#171717",
    "bg_secondary": "#2F2F2F",
    "border": "#3A3A3A",
    "text_primary": "#ECECEC",
    "text_secondary": "#B4B4B4",
    "text_tertiary": "#8E8EA0",
    "bubble_user": "#2F2F2F",
    "accent_bg": "#FFFFFF",
    "accent_fg": "#0D0D0D",
    "hover": "rgba(255,255,255,0.06)",
    "sidebar_active": "rgba(255,255,255,0.10)",
    "gauge_track": "#3A3A3A",
    "avatar_bg": "linear-gradient(135deg, #0C6E8C 0%, #094F63 100%)",
}

# ─────────────────────────────────────────────────────────────────────────────
# STYLES
# ─────────────────────────────────────────────────────────────────────────────

html(
    f"""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <style>

    html {{ scroll-behavior: smooth; }}
    html, body, [class*="css"], p, span, div, label, h1, h2, h3, h4, button, input, textarea {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    }}
    #MainMenu, footer {{ visibility: hidden; }}
    [data-testid="stHeader"] {{ background: transparent !important; box-shadow: none !important; }}
    .stApp {{ background-color: {T['bg_main']}; transition: background-color 0.25s ease; }}
    .tnum {{ font-variant-numeric: tabular-nums; }}

    /* ── Chat layout: the thread scrolls in its own bounded pane, the
       composer sits below it as a normal (non-floating) flex sibling.
       This is what a real chat UI does — it's the only way for the
       composer to be both always-visible AND never overlap content,
       since raw `position: fixed` overlaps whatever the page has
       scrolled to, regardless of where that is in the document. ──── */
    [data-testid="stMain"] {{
        height: 100vh !important; overflow: hidden !important;
        display: flex !important; flex-direction: column !important;
    }}
    [data-testid="stMainBlockContainer"],
    [data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] {{
        display: flex !important; flex-direction: column !important;
        height: 100% !important; overflow: hidden !important;
        padding-top: 0.6rem !important; padding-bottom: 0.6rem !important;
        max-width: none !important;
    }}
    /* st.container(key=...) wraps its content in an intermediate
       stLayoutWrapper div — THAT is the actual flex item (a direct
       child of the block above), one level up from .st-key-*, so the
       flex-sizing rules have to target the wrapper via :has(), not the
       named class itself. */
    [data-testid="stLayoutWrapper"]:has(.st-key-thread_scroll) {{
        flex: 1 1 auto !important; min-height: 0 !important; overflow: hidden !important;
    }}
    [data-testid="stLayoutWrapper"]:has(.st-key-composer) {{ flex-shrink: 0 !important; }}
    .st-key-thread_scroll {{ height: 100% !important; overflow-y: auto !important; }}
    .st-key-thread_scroll > div {{ max-width: 780px; margin: 0 auto; padding: 0 1.5rem; }}

    .material-icons, .material-icons-round, .material-icons-outlined,
    .material-symbols-rounded, .material-symbols-outlined, [data-testid="stIconMaterial"] {{
        font-size: 0 !important; width: 0 !important; overflow: hidden !important;
    }}

    /* ── Sidebar ───────────────────────────────────────────────────── */
    [data-testid="stSidebar"] {{
        background-color: {T['bg_sidebar']} !important;
        border-right: 1px solid {T['border']} !important;
        transition: background-color 0.25s ease;
    }}
    [data-testid="stSidebar"] * {{ color: {T['text_primary']}; }}
    [data-testid="stSidebarHeader"] {{ padding: 10px 12px !important; }}

    /* ── Sidebar open/close toggle (custom hamburger — Material icon font is
       suppressed above, so both the collapse and expand buttons need their
       own CSS-drawn icon) ────────────────────────────────────────────── */
    [data-testid="stSidebarCollapseButton"] button,
    [data-testid="stExpandSidebarButton"] {{
        width: 32px !important; height: 32px !important;
        display: flex !important; align-items: center !important; justify-content: center !important;
        border-radius: 8px !important; border: none !important; cursor: pointer !important;
        position: relative !important; padding: 0 !important;
        transition: background 0.15s ease !important;
    }}
    [data-testid="stSidebarCollapseButton"] button {{
        background: transparent !important; color: {T['text_secondary']} !important;
    }}
    [data-testid="stSidebarCollapseButton"] button:hover {{ background: {T['hover']} !important; }}
    [data-testid="stExpandSidebarButton"] {{
        background: {T['bg_secondary']} !important; border: 1px solid {T['border']} !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.25) !important;
    }}
    [data-testid="stExpandSidebarButton"]:hover {{ background: {T['hover']} !important; }}
    [data-testid="stSidebarCollapseButton"] svg,
    [data-testid="stExpandSidebarButton"] svg,
    [data-testid="stSidebarCollapseButton"] [data-testid="stIconMaterial"],
    [data-testid="stExpandSidebarButton"] [data-testid="stIconMaterial"] {{ display: none !important; }}
    [data-testid="stSidebarCollapseButton"] button::before,
    [data-testid="stExpandSidebarButton"]::before {{
        content: ""; position: absolute; width: 16px; height: 2px; border-radius: 2px;
        background: currentColor; top: calc(50% - 6px); left: calc(50% - 8px);
        box-shadow: 0 5px 0 currentColor, 0 10px 0 currentColor;
    }}
    [data-testid="stExpandSidebarButton"]::before {{ color: {T['text_primary']}; }}

    .sb-new-btn {{
        display: flex; align-items: center; gap: 9px;
        border: 1px solid {T['border']}; border-radius: 12px;
        padding: 10px 14px; font-size: 13.5px; font-weight: 600;
        color: {T['text_primary']}; cursor: pointer; margin: 4px 12px 14px 12px;
        transition: background 0.15s ease;
    }}
    [data-testid="stSidebar"] .stButton > button {{
        background: transparent !important; border: 1px solid {T['border']} !important;
        border-radius: 12px !important; color: {T['text_primary']} !important;
        font-size: 13.5px !important; font-weight: 600 !important;
        padding: 10px 14px !important; width: 100% !important; text-align: left !important;
        transition: background 0.15s ease !important;
    }}
    [data-testid="stSidebar"] .stButton > button:hover {{ background: {T['hover']} !important; }}

    .sb-section-label {{
        font-size: 11px; font-weight: 700; color: {T['text_tertiary']};
        text-transform: uppercase; letter-spacing: 0.6px; padding: 6px 14px; margin-top: 6px;
    }}
    .sb-row {{
        display: block; text-decoration: none !important;
        padding: 9px 14px; margin: 1px 8px; border-radius: 9px;
        font-size: 13px; font-weight: 500; color: {T['text_primary']} !important;
        white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
        transition: background 0.15s ease;
    }}
    .sb-row:hover {{ background: {T['hover']}; }}
    .sb-row .dot {{ display:inline-block; width:7px; height:7px; border-radius:50%; margin-right:8px; }}
    .sb-row .meta {{ color: {T['text_secondary']}; font-size: 11.5px; margin-left: 15px; }}
    .sb-empty {{ font-size: 12px; color: {T['text_tertiary']}; padding: 8px 14px; line-height: 1.5; }}

    .sb-footer {{
        border-top: 1px solid {T['border']}; margin-top: 10px; padding-top: 10px;
    }}

    /* ── Persona chips (sidebar) ──────────────────────────────────── */
    .persona-chip {{
        display: inline-flex; align-items: center; gap: 5px; font-size: 10.5px; font-weight: 600; color: {T['text_secondary']};
        background: {T['bg_secondary']}; border: 1px solid {T['border']}; border-radius: 20px;
        padding: 4px 10px; margin: 2px 4px 2px 0;
    }}

    /* ── Landing hero (empty thread) ──────────────────────────────── */
    .hero {{
        display: flex; flex-direction: column; align-items: center; justify-content: center;
        text-align: center; padding: 12vh 20px 20px 20px; animation: fadeIn 0.4s ease;
    }}
    .hero-icon {{
        width: 56px; height: 56px; border-radius: 16px; background: {T['avatar_bg']};
        display: flex; align-items: center; justify-content: center; font-size: 27px;
        box-shadow: 0 4px 14px rgba(12,110,140,0.30); margin-bottom: 16px;
    }}
    .hero-title {{ font-size: 27px; font-weight: 800; color: {T['text_primary']}; letter-spacing: -0.5px; }}
    .hero-sub {{ font-size: 14px; color: {T['text_secondary']}; margin-top: 6px; max-width: 420px; line-height: 1.5; }}
    @keyframes fadeIn {{ from {{ opacity: 0; }} to {{ opacity: 1; }} }}

    /* ── Thread ────────────────────────────────────────────────────── */
    .turn {{ margin-bottom: 26px; animation: slideUp 0.32s ease; }}
    @keyframes slideUp {{ from {{ opacity: 0; transform: translateY(10px); }} to {{ opacity: 1; transform: translateY(0); }} }}

    .user-row {{ display: flex; justify-content: flex-end; margin-bottom: 18px; }}
    .user-bubble {{
        background: {T['bubble_user']}; color: {T['text_primary']};
        border-radius: 18px 18px 4px 18px; padding: 11px 16px;
        font-size: 14px; line-height: 1.55; max-width: 78%;
    }}

    .assistant-row {{ display: flex; gap: 12px; align-items: flex-start; }}
    .assistant-avatar {{
        width: 30px; height: 30px; border-radius: 8px; background: {T['avatar_bg']};
        display: flex; align-items: center; justify-content: center; font-size: 15px;
        flex-shrink: 0; margin-top: 2px;
    }}
    .assistant-body {{ flex: 1; min-width: 0; }}
    .assistant-name {{ font-size: 12.5px; font-weight: 700; color: {T['text_primary']}; margin-bottom: 6px; }}

    .thinking-dots {{ display: inline-flex; gap: 4px; padding: 6px 0; }}
    .thinking-dots span {{
        width: 6px; height: 6px; border-radius: 50%; background: {T['text_tertiary']};
        animation: bounce 1.2s infinite ease-in-out;
    }}
    .thinking-dots span:nth-child(2) {{ animation-delay: 0.15s; }}
    .thinking-dots span:nth-child(3) {{ animation-delay: 0.3s; }}
    @keyframes bounce {{ 0%, 60%, 100% {{ transform: translateY(0); opacity: 0.5; }} 30% {{ transform: translateY(-5px); opacity: 1; }} }}

    /* ── Result card content ──────────────────────────────────────── */
    .badge {{
        display: inline-flex; align-items: center; gap: 6px;
        padding: 5px 12px; border-radius: 20px; font-size: 12px; font-weight: 700; border: 1px solid transparent;
    }}
    .badge-dot {{ width: 7px; height: 7px; border-radius: 50%; }}
    .confidence-strip {{
        display: flex; align-items: flex-start; gap: 9px;
        background: {T['bg_secondary']}; border: 1px solid {T['border']}; border-radius: 10px;
        padding: 10px 13px; margin-top: 14px; max-width: 480px;
    }}
    .confidence-strip .ci {{ font-size: 14px; line-height: 1.3; }}
    .confidence-strip .ct {{ font-size: 11.5px; color: {T['text_secondary']}; line-height: 1.5; font-weight: 500; }}
    .confidence-strip .ct b {{ color: {T['text_primary']}; }}
    .gauge-wrap {{ display: flex; flex-direction: column; align-items: flex-start; margin: 4px 0 2px 0; }}
    .gauge-ticks {{ display: flex; justify-content: space-between; width: 220px; margin-top: -6px; }}
    .gauge-ticks span {{ font-size: 10.5px; color: {T['text_tertiary']}; font-weight: 600; }}

    /* ── Sparkline / metrics turn ─────────────────────────────────── */
    .info-card {{
        background: {T['bg_secondary']}; border: 1px solid {T['border']}; border-radius: 14px;
        padding: 16px 18px; max-width: 480px;
    }}
    .metric-row {{
        display: flex; flex-direction: column; gap: 2px;
        padding: 7px 0; border-bottom: 1px solid {T['border']}; font-size: 12.5px;
    }}
    .metric-row:last-child {{ border-bottom: none; }}
    .metric-name {{ color: {T['text_secondary']}; font-weight: 500; font-size: 11px; }}
    .metric-val {{ color: {T['text_primary']}; font-weight: 700; font-size: 14px; }}
    .info-caption {{ font-size: 11.5px; color: {T['text_secondary']}; line-height: 1.6; margin-top: 10px; }}
    .sim-card {{
        border: 1.5px dashed #8B6F1F !important;
        background: rgba(214,154,30,0.08) !important;
    }}
    .sim-tag {{
        display: inline-block; font-size: 10px; font-weight: 800; letter-spacing: 0.5px;
        color: #F0B93D; background: rgba(214,154,30,0.16);
        border: 1px solid rgba(214,154,30,0.35);
        border-radius: 6px; padding: 3px 8px; margin-bottom: 10px;
    }}

    /* ── Composer: a normal (non-floating) flex item docked below the
       scrolling thread pane — see [data-testid="stMain"] above. Because
       it's laid out in-flow rather than `position: fixed`, it can never
       overlap thread content while scrolling, and it's automatically
       full-width-of-stMain (already correctly excluding the sidebar), so
       centering the inner pill needs nothing more than margin: auto. ── */
    .st-key-composer {{
        flex-shrink: 0 !important;
        background: {T['bg_main']};
        padding: 10px 0 4px 0 !important;
    }}
    .st-key-composer > div {{ max-width: 780px; margin: 0 auto; padding: 0 1.5rem; }}
    [data-testid="stForm"] {{
        background: {T['bg_secondary']} !important; border: 1px solid {T['border']} !important;
        border-radius: 26px !important; padding: 10px 18px !important;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06) !important;
        transition: border-color 0.15s ease !important;
    }}
    [data-testid="stForm"]:focus-within {{ border-color: {T['text_secondary']} !important; }}

    .stNumberInput input, .stDateInput input, .stTimeInput input {{
        background: transparent !important; border: none !important; box-shadow: none !important;
        font-size: 13.5px !important; color: {T['text_primary']} !important; padding: 0 !important;
        font-variant-numeric: tabular-nums !important;
    }}
    [data-testid="stForm"] [data-baseweb="input"],
    [data-testid="stForm"] [data-baseweb="base-input"],
    [data-testid="stForm"] [data-baseweb="select"] > div,
    [data-testid="stForm"] [data-testid="stNumberInputContainer"] {{
        background: transparent !important; border: none !important; box-shadow: none !important;
    }}
    [data-testid="stForm"] svg {{ fill: {T['text_secondary']} !important; }}
    .stNumberInput button, [data-testid="stTimeInputField"] {{ background: transparent !important; }}
    .stNumberInput label, .stDateInput label, .stTimeInput label {{
        font-size: 10px !important; font-weight: 700 !important; color: {T['text_tertiary']} !important;
        text-transform: uppercase; letter-spacing: 0.5px;
    }}
    [data-testid="stFormSubmitButton"] > button {{
        background-color: {T['accent_bg']} !important; color: {T['accent_fg']} !important;
        border: none !important; border-radius: 50% !important;
        width: 40px !important; height: 40px !important; min-height: 40px !important;
        padding: 0 !important; font-weight: 700 !important;
        position: relative !important; overflow: hidden !important;
        transition: transform 0.12s ease, opacity 0.15s ease !important;
    }}
    [data-testid="stFormSubmitButton"] > button p {{ font-size: 0 !important; line-height: 0 !important; }}
    [data-testid="stFormSubmitButton"] > button::before {{
        content: ""; position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
        width: 18px; height: 18px; background-color: {T['accent_fg']};
        -webkit-mask: url('{icon_data_uri("arrow-up", color="black", stroke_width=2.5)}') no-repeat center / contain;
        mask: url('{icon_data_uri("arrow-up", color="black", stroke_width=2.5)}') no-repeat center / contain;
    }}
    [data-testid="stFormSubmitButton"] > button:hover {{ transform: scale(1.06) !important; }}
    [data-testid="stFormSubmitButton"] > button:active {{ transform: scale(0.96) !important; }}

    /* ── Top bar controls ─────────────────────────────────────────── */
    .st-key-topbar .stButton > button {{
        background: {T['bg_secondary']} !important; border: 1px solid {T['border']} !important;
        border-radius: 10px !important; color: {T['text_primary']} !important;
        font-size: 13px !important; font-weight: 600 !important; padding: 6px 12px !important;
        transition: background 0.15s ease !important;
    }}
    .st-key-topbar .stButton > button:hover {{ background: {T['hover']} !important; }}

    /* ── Responsive (mobile) ───────────────────────────────────────── */
    html, body {{ overflow-x: hidden !important; }}
    .gauge-wrap svg {{ max-width: 100%; height: auto; }}

    @media (max-width: 640px) {{
        .st-key-thread_scroll > div {{ padding: 0 0.85rem; }}
        .hero {{ padding: 6vh 8px 16px 8px; }}
        .hero-icon {{ width: 44px; height: 44px; border-radius: 13px; margin-bottom: 12px; }}
        .hero-title {{ font-size: 21px; }}
        .hero-sub {{ font-size: 13px; max-width: 320px; }}

        .user-bubble {{ max-width: 88%; font-size: 13.5px; }}
        .assistant-avatar {{ width: 26px; height: 26px; }}
        .confidence-strip, .info-card {{ max-width: 100%; }}
        .gauge-ticks {{ width: 100%; max-width: 220px; }}

        /* The composer already can't overlap content (it's a normal flex
           sibling below the scrolling thread pane, not floated over it) —
           this just keeps it compact on a narrow screen: the 6 inline
           fields wrap into a grid instead of Streamlit's default full
           vertical stack, which would otherwise eat most of the viewport
           height and leave little room for the thread pane above it. */
        .st-key-composer {{ padding: 8px 0 4px 0 !important; }}
        .st-key-composer > div {{ padding: 0 0.6rem; }}
        [data-testid="stForm"] {{ padding: 10px 12px !important; border-radius: 18px !important; }}
        [data-testid="stForm"] [data-testid="stHorizontalBlock"] {{
            flex-wrap: wrap !important; gap: 8px !important;
        }}
        [data-testid="stForm"] [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {{
            flex: 1 1 28% !important; width: auto !important; min-width: 82px !important;
        }}
        [data-testid="stForm"] [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:nth-child(4),
        [data-testid="stForm"] [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:nth-child(5) {{
            flex-basis: 42% !important;
        }}
        [data-testid="stForm"] [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:nth-child(6) {{
            flex: 0 0 40px !important; min-width: 40px !important;
            display: flex !important; align-items: flex-end !important;
        }}
        .stNumberInput label, .stDateInput label, .stTimeInput label {{ font-size: 9px !important; }}
        .stNumberInput input, .stDateInput input, .stTimeInput input {{ font-size: 12.5px !important; }}
    }}

    </style>
    """
)

# ─────────────────────────────────────────────────────────────────────────────
# DATA + MODEL
# ─────────────────────────────────────────────────────────────────────────────


@st.cache_resource
def load_and_train():
    dataset = pd.read_csv(DATA_PATH)
    if "flu_cases" in dataset.columns:
        dataset = dataset.drop(columns=["flu_cases"])
    dataset = dataset.drop_duplicates()
    dataset["timestamp"] = pd.to_datetime(dataset["timestamp"])
    dataset = dataset.sort_values("timestamp").reset_index(drop=True)
    dataset["Month"] = dataset["timestamp"].dt.month
    dataset["Day"] = dataset["timestamp"].dt.day
    dataset["DayOfWeek"] = dataset["timestamp"].dt.dayofweek
    dataset["Hour"] = dataset["timestamp"].dt.hour
    dataset["net_flow"] = dataset["admissions"] - dataset["discharges"]

    X = dataset[PREDICTOR_COLUMNS]
    y = dataset["bed_occupancy"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=200, max_depth=6, min_samples_leaf=10, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    metrics = {
        "mae": mean_absolute_error(y_test, preds),
        "rmse": np.sqrt(mean_squared_error(y_test, preds)),
        "r2": r2_score(y_test, preds),
    }
    return model, metrics, dataset


model, metrics, dataset = load_and_train()


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────


def risk_band(value):
    if value < NORMAL_MAX:
        return {"label": "Normal", "arc": "#12875A", "bg": "rgba(18,135,90,0.15)", "fg": "#3DDC97", "border": "rgba(18,135,90,0.35)"}
    if value < ELEVATED_MAX:
        return {"label": "Elevated", "arc": "#D69A1E", "bg": "rgba(214,154,30,0.16)", "fg": "#F0B93D", "border": "rgba(214,154,30,0.35)"}
    return {"label": "Critical", "arc": "#D14343", "bg": "rgba(209,67,67,0.16)", "fg": "#F17A7A", "border": "rgba(209,67,67,0.35)"}


def gauge_svg(value, gauge_min=50, gauge_max=100):
    r, cx, cy = 90, 100, 96
    fraction = max(0.0, min(1.0, (value - gauge_min) / (gauge_max - gauge_min)))
    circumference = math.pi * r
    dashoffset = circumference * (1 - fraction)
    band = risk_band(value)
    return f"""
    <div class="gauge-wrap">
    <svg width="200" height="112" viewBox="0 0 200 112">
    <path d="M {cx-r},{cy} A {r},{r} 0 0 1 {cx+r},{cy}"
          fill="none" stroke="{T['gauge_track']}" stroke-width="13" stroke-linecap="round" />
    <path d="M {cx-r},{cy} A {r},{r} 0 0 1 {cx+r},{cy}"
          fill="none" stroke="{band['arc']}" stroke-width="13" stroke-linecap="round"
          stroke-dasharray="{circumference:.2f}"
          style="animation: gaugefill-{abs(hash(value)) % 100000} 0.9s ease-out forwards;"
          stroke-dashoffset="{circumference:.2f}" />
    <text x="{cx}" y="{cy-6}" text-anchor="middle" font-size="27" font-weight="800"
          fill="{T['text_primary']}" font-family="Inter" class="tnum">{value:.1f}%</text>
    </svg>
    <div class="gauge-ticks"><span>{gauge_min}%</span><span>{gauge_max}%</span></div>
    </div>
    <style>
    @keyframes gaugefill-{abs(hash(value)) % 100000} {{ to {{ stroke-dashoffset: {dashoffset:.2f}; }} }}
    </style>
    """


def render_assistant_result(value):
    band = risk_band(value)
    free = 100 - value
    badge = (
        f'<span class="badge" style="background:{band["bg"]};color:{band["fg"]};border-color:{band["border"]};">'
        f'<span class="badge-dot" style="background:{band["fg"]};"></span>{band["label"]} capacity</span>'
    )
    return f"""
    <div class="assistant-row">
    <div class="assistant-avatar">{icon('bed', size=15, stroke='white')}</div>
    <div class="assistant-body">
    <div class="assistant-name">Bed Occupancy Forecast</div>
    {gauge_svg(value)}
    <div style="margin-top:8px;">{badge}</div>
    <div class="confidence-strip">
    <div class="ci">{icon('info', size=15)}</div>
    <div class="ct">
    <b>What this means:</b> about {value:.0f} of every 100 beds are expected to be occupied during
    this shift, leaving roughly {free:.0f}% free capacity for new admissions. The
    "{band['label']}" label reflects how this compares to typical occupancy levels seen in the
    historical data (below {NORMAL_MAX:.0f}% is Normal, up to {ELEVATED_MAX:.0f}% is Elevated,
    above that is Critical).
    </div>
    </div>
    </div>
    </div>
    """


def user_summary(inputs):
    d = inputs["date"]
    t = inputs["time"]
    return (
        f"Admissions {inputs['admissions']} · Discharges {inputs['discharges']} · "
        f"Staff {inputs['staff_count']} · {MONTH_NAMES[d.month]} {d.day}, {t.strftime('%H:%M')}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    html(f"<style>{button_icon_css('new_prediction', 'plus')}</style>")

    if st.button("New prediction", key="new_prediction", use_container_width=True):
        st.session_state["history"] = []
        st.rerun()

    html('<div class="sb-section-label">History</div>')
    if st.session_state["history"]:
        rows = []
        for i, turn in enumerate(reversed(st.session_state["history"])):
            idx = len(st.session_state["history"]) - i
            band = risk_band(turn["prediction"])
            rows.append(
                f'<a class="sb-row" href="#turn-{idx}">'
                f'<span class="dot" style="background:{band["arc"]};"></span>'
                f'{turn["prediction"]:.1f}% · {band["label"]}'
                f'<div class="meta">{MONTH_NAMES[turn["inputs"]["date"].month]} {turn["inputs"]["date"].day} · '
                f'{turn["inputs"]["time"].strftime("%H:%M")}</div></a>'
            )
        html("".join(rows))
    else:
        html('<div class="sb-empty">Your predictions this session will appear here.</div>')

# ─────────────────────────────────────────────────────────────────────────────
# MAIN: THREAD (scrolls in its own bounded pane — see the flex-column layout
# in the CSS above, which is what keeps this from ever sliding underneath
# the composer the way a page-level `position: fixed` bar would)
# ─────────────────────────────────────────────────────────────────────────────

with st.container(key="thread_scroll"):
    if not st.session_state["history"]:
        html(
            f"""
            <div class="hero">
            <div class="hero-icon">{icon('bed', size=26, stroke='white')}</div>
            <div class="hero-title">Hospital Bed Occupancy Predictor</div>
            <div class="hero-sub">Enter a shift's operational data below to forecast bed occupancy with a
            Random Forest model trained on {DATA_FILENAME}, sourced from Gaggle.</div>
            </div>
            """
        )
    else:
        for i, turn in enumerate(st.session_state["history"], start=1):
            html(
                f'<div class="turn" id="turn-{i}">'
                f'<div class="user-row"><div class="user-bubble">{user_summary(turn["inputs"])}</div></div>'
                f'{render_assistant_result(turn["prediction"])}'
                f'</div>'
            )

    if st.session_state["pending"] is not None:
        html(
            f"""
            <div class="turn">
            <div class="user-row"><div class="user-bubble">{user_summary(st.session_state["pending"])}</div></div>
            <div class="assistant-row">
            <div class="assistant-avatar">{icon('bed', size=15, stroke='white')}</div>
            <div class="assistant-body">
            <div class="thinking-dots"><span></span><span></span><span></span></div>
            </div>
            </div>
            </div>
            """
        )

        pending = st.session_state["pending"]
        net_flow = pending["admissions"] - pending["discharges"]
        row = pd.DataFrame(
            [
                {
                    "admissions": pending["admissions"],
                    "discharges": pending["discharges"],
                    "staff_count": pending["staff_count"],
                    "Month": pending["date"].month,
                    "Day": pending["date"].day,
                    "DayOfWeek": pending["date"].weekday(),
                    "Hour": pending["time"].hour,
                    "net_flow": net_flow,
                }
            ]
        )[PREDICTOR_COLUMNS]
        time_module.sleep(0.55)
        prediction = float(model.predict(row)[0])
        st.session_state["history"].append({"inputs": pending, "prediction": prediction})
        st.session_state["pending"] = None
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# COMPOSER (fixed bottom)
# ─────────────────────────────────────────────────────────────────────────────

with st.container(key="composer"):
    with st.form("predict_form", border=False):
        cols = st.columns([1, 1, 1, 1, 1, 0.5])
        with cols[0]:
            admissions = st.number_input("Admissions", min_value=0, max_value=80, value=25, step=1, key="in_admissions")
        with cols[1]:
            discharges = st.number_input("Discharges", min_value=0, max_value=80, value=23, step=1, key="in_discharges")
        with cols[2]:
            staff_count = st.number_input("Staff", min_value=10, max_value=100, value=44, step=1, key="in_staff")
        with cols[3]:
            shift_date = st.date_input("Date", value=date.today(), key="in_date")
        with cols[4]:
            shift_time = st.time_input("Time", value=time(datetime.now().hour, 0), key="in_time")
        with cols[5]:
            st.write("")
            submitted = st.form_submit_button("Predict")

if submitted:
    st.session_state["pending"] = {
        "admissions": admissions,
        "discharges": discharges,
        "staff_count": staff_count,
        "date": shift_date,
        "time": shift_time,
    }
    st.rerun()
