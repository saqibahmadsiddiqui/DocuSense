import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import tempfile
import streamlit as st
from docu_sense import load_and_chunk_pdfs, build_vectorstore, build_rag_chain, extract_sources


def get_api_key() -> str:
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return os.getenv("GROQ_API_KEY", "")


st.set_page_config(
    page_title="DocuSense · AI Document Assistant",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded"
)

#  THEME CSS
def get_css(theme: str) -> str:
    d = (theme == "dark")

    if d:
        bg_app          = "#090f1e"
        bg_gradient     = ("radial-gradient(ellipse 80% 50% at 15% 10%, rgba(6,182,212,0.12) 0%, transparent 60%),"
                           "radial-gradient(ellipse 60% 40% at 85% 85%, rgba(16,185,129,0.08) 0%, transparent 60%)")
        bg_sidebar      = "rgba(4,8,20,0.98)"
        sidebar_border  = "rgba(6,182,212,0.14)"
        text_primary    = "#e2e8f0"
        hero_sub        = "rgba(148,163,184,0.60)"
        badge_bg        = "rgba(6,182,212,0.09)"
        badge_border    = "rgba(6,182,212,0.26)"
        badge_text      = "#67e8f9"
        divider_grad    = "linear-gradient(90deg,transparent,rgba(6,182,212,0.38),rgba(52,211,153,0.22),transparent)"
        logo_shadow     = "drop-shadow(0 0 12px rgba(6,182,212,0.55))"
        title_grad      = "linear-gradient(135deg,#38bdf8,#06b6d4,#34d399)"
        hero_gradient   = "linear-gradient(135deg,#38bdf8 0%,#06b6d4 45%,#34d399 100%)"
        sb_sub_color    = "rgba(103,232,249,0.28)"
        sb_label_color  = "rgba(103,232,249,0.36)"
        uploader_bg     = "rgba(6,182,212,0.04)"
        uploader_bdr    = "rgba(6,182,212,0.26)"
        uploader_hbg    = "rgba(6,182,212,0.09)"
        uploader_hbdr   = "rgba(52,211,153,0.48)"
        uploader_text   = "rgba(103,232,249,0.52)"
        uploader_small  = "rgba(103,232,249,0.26)"
        pill_bg         = "linear-gradient(135deg,rgba(6,182,212,0.10),rgba(16,185,129,0.07))"
        pill_bdr        = "rgba(6,182,212,0.20)"
        pill_text       = "#67e8f9"
        pill_dot        = "#34d399"
        pill_shadow     = "0 0 6px rgba(52,211,153,0.60)"
        stat_bg         = "rgba(2,10,22,0.85)"
        stat_bdr        = "rgba(6,182,212,0.17)"
        stat_num        = "#38bdf8"
        stat_lbl        = "rgba(103,232,249,0.36)"
        btn_tog_bg      = "rgba(6,182,212,0.08)"
        btn_tog_text    = "#67e8f9"
        btn_tog_bdr     = "rgba(6,182,212,0.22)"
        btn_tog_hbg     = "rgba(6,182,212,0.15)"
        btn_tog_hbdr    = "rgba(6,182,212,0.42)"
        btn_clr_bg      = "rgba(239,68,68,0.07)"
        btn_clr_text    = "#fca5a5"
        btn_clr_bdr     = "rgba(239,68,68,0.22)"
        btn_clr_hbg     = "rgba(239,68,68,0.13)"
        btn_clr_hbdr    = "rgba(239,68,68,0.42)"
        av_user         = "linear-gradient(135deg,#0ea5e9,#06b6d4)"
        av_ai_bg        = "linear-gradient(135deg,#060c1a,#0a1628)"
        av_ai_bdr       = "rgba(6,182,212,0.38)"
        bub_u_bg        = "linear-gradient(135deg,rgba(14,165,233,0.16),rgba(6,182,212,0.10))"
        bub_u_bdr       = "rgba(14,165,233,0.26)"
        bub_u_text      = "#e2e8f0"
        bub_a_bg        = "rgba(2,12,30,0.75)"
        bub_a_bdr       = "rgba(6,182,212,0.12)"
        bub_a_text      = "#e2e8f0"
        exp_bg          = "rgba(6,182,212,0.04)"
        exp_bdr         = "rgba(6,182,212,0.14)"
        exp_title       = "#38bdf8"
        src_bg          = "rgba(6,182,212,0.06)"
        src_lbdr        = "#0ea5e9"
        src_text        = "#67e8f9"
        src_pg_bg       = "rgba(6,182,212,0.14)"
        src_pg_text     = "#38bdf8"
        inp_bg          = "rgba(2,10,24,0.95)"
        inp_bdr         = "rgba(6,182,212,0.30)"
        inp_fbdr        = "rgba(56,189,248,0.58)"
        inp_fring       = "rgba(6,182,212,0.10)"
        inp_text        = "#e2e8f0"
        inp_ph          = "rgba(103,232,249,0.22)"
        empty_title     = "rgba(148,163,184,0.72)"
        empty_text      = "rgba(103,232,249,0.28)"
        feat_bg         = "rgba(6,182,212,0.05)"
        feat_bdr        = "rgba(6,182,212,0.11)"
        feat_text       = "rgba(103,232,249,0.48)"
        hr_color        = "rgba(6,182,212,0.11)"
        scroll_t        = "rgba(6,182,212,0.28)"
        scroll_h        = "rgba(52,211,153,0.45)"
        code_bg         = "rgba(6,182,212,0.11)"
        code_text       = "#67e8f9"
        pre_bg          = "rgba(2,10,22,0.85)"
        pre_bdr         = "rgba(6,182,212,0.14)"
        spinner_c       = "#38bdf8"
        alert_radius    = "10px"
    else:
        bg_app          = "#f0f4f8"
        bg_gradient     = ("radial-gradient(ellipse 70% 50% at 15% 10%, rgba(79,70,229,0.05) 0%, transparent 60%),"
                           "radial-gradient(ellipse 50% 40% at 85% 80%, rgba(5,150,105,0.04) 0%, transparent 60%)")
        bg_sidebar      = "#ffffff"
        sidebar_border  = "#e2e8f0"
        text_primary    = "#0f172a"
        hero_sub        = "#64748b"
        badge_bg        = "rgba(79,70,229,0.07)"
        badge_border    = "rgba(79,70,229,0.17)"
        badge_text      = "#3730a3"
        divider_grad    = "linear-gradient(90deg,transparent,rgba(79,70,229,0.22),rgba(5,150,105,0.14),transparent)"
        logo_shadow     = "drop-shadow(0 0 10px rgba(79,70,229,0.28))"
        title_grad      = "linear-gradient(135deg,#4f46e5,#2563eb,#059669)"
        hero_gradient   = "linear-gradient(135deg,#4f46e5 0%,#2563eb 50%,#059669 100%)"
        sb_sub_color    = "#94a3b8"
        sb_label_color  = "#94a3b8"
        uploader_bg     = "rgba(79,70,229,0.03)"
        uploader_bdr    = "rgba(79,70,229,0.20)"
        uploader_hbg    = "rgba(79,70,229,0.07)"
        uploader_hbdr   = "rgba(5,150,105,0.38)"
        uploader_text   = "#475569"
        uploader_small  = "#94a3b8"
        pill_bg         = "linear-gradient(135deg,rgba(79,70,229,0.06),rgba(5,150,105,0.04))"
        pill_bdr        = "rgba(79,70,229,0.15)"
        pill_text       = "#3730a3"
        pill_dot        = "#059669"
        pill_shadow     = "0 0 5px rgba(5,150,105,0.38)"
        stat_bg         = "#f8fafc"
        stat_bdr        = "#e2e8f0"
        stat_num        = "#4f46e5"
        stat_lbl        = "#94a3b8"
        btn_tog_bg      = "rgba(79,70,229,0.06)"
        btn_tog_text    = "#3730a3"
        btn_tog_bdr     = "rgba(79,70,229,0.17)"
        btn_tog_hbg     = "rgba(79,70,229,0.12)"
        btn_tog_hbdr    = "rgba(79,70,229,0.32)"
        btn_clr_bg      = "rgba(220,38,38,0.04)"
        btn_clr_text    = "#b91c1c"
        btn_clr_bdr     = "rgba(220,38,38,0.16)"
        btn_clr_hbg     = "rgba(220,38,38,0.09)"
        btn_clr_hbdr    = "rgba(220,38,38,0.30)"
        av_user         = "linear-gradient(135deg,#4f46e5,#2563eb)"
        av_ai_bg        = "linear-gradient(135deg,#f8fafc,#f1f5f9)"
        av_ai_bdr       = "#e2e8f0"
        bub_u_bg        = "rgba(79,70,229,0.07)"
        bub_u_bdr       = "rgba(79,70,229,0.15)"
        bub_u_text      = "#0f172a"
        bub_a_bg        = "#ffffff"
        bub_a_bdr       = "#e2e8f0"
        bub_a_text      = "#1e293b"
        exp_bg          = "rgba(79,70,229,0.03)"
        exp_bdr         = "rgba(79,70,229,0.13)"
        exp_title       = "#4f46e5"
        src_bg          = "rgba(79,70,229,0.05)"
        src_lbdr        = "#4f46e5"
        src_text        = "#334155"
        src_pg_bg       = "rgba(79,70,229,0.08)"
        src_pg_text     = "#4f46e5"
        inp_bg          = "#ffffff"
        inp_bdr         = "#cbd5e1"
        inp_fbdr        = "rgba(79,70,229,0.42)"
        inp_fring       = "rgba(79,70,229,0.07)"
        inp_text        = "#0f172a"
        inp_ph          = "#94a3b8"
        empty_title     = "#334155"
        empty_text      = "#94a3b8"
        feat_bg         = "rgba(79,70,229,0.04)"
        feat_bdr        = "rgba(79,70,229,0.09)"
        feat_text       = "#475569"
        hr_color        = "#e2e8f0"
        scroll_t        = "rgba(79,70,229,0.20)"
        scroll_h        = "rgba(5,150,105,0.35)"
        code_bg         = "rgba(79,70,229,0.07)"
        code_text       = "#3730a3"
        pre_bg          = "#f8fafc"
        pre_bdr         = "#e2e8f0"
        spinner_c       = "#4f46e5"
        alert_radius    = "10px"

    return f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}}

#MainMenu, footer {{ visibility: hidden; }}
header[data-testid="stHeader"] {{ background: transparent !important; }}

/* ── APP BACKGROUND ── */
.stApp {{
    background: {bg_app} !important;
    background-image: {bg_gradient} !important;
}}
.main .block-container {{
    padding: 1.5rem 2.5rem 6rem 2.5rem;
    max-width: 860px;
    margin: 0 auto;
}}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {{
    background: {bg_sidebar} !important;
    border-right: 1px solid {sidebar_border} !important;
}}
[data-testid="stSidebarContent"] {{
    padding: 1.5rem 1.2rem;
}}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div {{
    color: {text_primary};
}}

/* ── BRAND BLOCK ── */
.sb-brand {{
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 1.1rem;
    padding-bottom: 0.95rem;
    border-bottom: 1px solid {hr_color};
}}
.sb-logo {{
    font-size: 1.75rem;
    filter: {logo_shadow};
    line-height: 1;
    flex-shrink: 0;
}}
.sb-title {{
    font-size: 1.0rem;
    font-weight: 700;
    letter-spacing: -0.4px;
    background: {title_grad};
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.2;
}}
.sb-sub {{
    font-size: 0.62rem;
    color: {sb_sub_color};
    letter-spacing: 0.09em;
    text-transform: uppercase;
    margin-top: 2px;
}}

/* ── SECTION LABELS ── */
.sb-label {{
    font-size: 0.62rem;
    font-weight: 600;
    color: {sb_label_color};
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 0.5rem;
}}

/* ── FILE UPLOADER ── */
[data-testid="stFileUploader"] {{
    background: {uploader_bg} !important;
    border: 1.5px dashed {uploader_bdr} !important;
    border-radius: 11px !important;
    transition: all 0.2s ease;
}}
[data-testid="stFileUploader"]:hover {{
    border-color: {uploader_hbdr} !important;
    background: {uploader_hbg} !important;
}}
[data-testid="stFileUploaderDropzoneInstructions"] div span {{
    color: {uploader_text} !important;
    font-size: 0.81rem !important;
}}
[data-testid="stFileUploaderDropzoneInstructions"] div small {{
    color: {uploader_small} !important;
}}

/* ── FILE PILLS ── */
.file-list {{ margin-top: 0.55rem; }}
.file-pill {{
    display: flex;
    align-items: center;
    gap: 8px;
    background: {pill_bg};
    border: 1px solid {pill_bdr};
    border-radius: 7px;
    padding: 5px 10px;
    margin-bottom: 4px;
    color: {pill_text};
    font-size: 0.76rem;
    overflow: hidden;
    white-space: nowrap;
    text-overflow: ellipsis;
    animation: fadeSlide 0.22s ease;
}}
.pill-dot {{
    width: 5px;
    height: 5px;
    border-radius: 50%;
    background: {pill_dot};
    flex-shrink: 0;
    box-shadow: {pill_shadow};
}}
@keyframes fadeSlide {{
    from {{ opacity: 0; transform: translateX(-6px); }}
    to   {{ opacity: 1; transform: translateX(0); }}
}}

/* ── STATS ROW ── */
.stats-row {{
    display: flex;
    gap: 6px;
    margin: 0.65rem 0 0.85rem 0;
}}
.stat-card {{
    flex: 1;
    background: {stat_bg};
    border: 1px solid {stat_bdr};
    border-radius: 8px;
    padding: 8px 6px;
    text-align: center;
}}
.stat-num {{
    font-size: 1.2rem;
    font-weight: 700;
    color: {stat_num};
    line-height: 1;
}}
.stat-lbl {{
    font-size: 0.58rem;
    color: {stat_lbl};
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-top: 3px;
}}

/* ── SIDEBAR BUTTONS — base ── */
[data-testid="stSidebar"] .stButton > button {{
    transition: all 0.2s ease !important;
    border-radius: 8px !important;
    font-size: 0.79rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.01em !important;
    line-height: 1.5 !important;
    padding: 0.38rem 0.80rem !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    white-space: nowrap !important;
}}
[data-testid="stSidebar"] .stButton > button > div,
[data-testid="stSidebar"] .stButton > button > div > p {{
    display: inline !important;
    margin: 0 !important;
    padding: 0 !important;
    white-space: nowrap !important;
    line-height: inherit !important;
    vertical-align: middle !important;
}}

/* Theme toggle — compact (no use_container_width) */
[data-testid="stSidebar"] .stButton > button:not([style*="width: 100%"]) {{
    background: {btn_tog_bg} !important;
    color: {btn_tog_text} !important;
    border: 1px solid {btn_tog_bdr} !important;
}}
[data-testid="stSidebar"] .stButton > button:not([style*="width: 100%"]):hover {{
    background: {btn_tog_hbg} !important;
    border-color: {btn_tog_hbdr} !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 3px 12px rgba(0,0,0,0.12) !important;
}}

/* Clear session button — full width */
[data-testid="stSidebar"] .stButton > button[style*="width: 100%"],
[data-testid="stSidebar"] .stButton > button[style*="width:100%"] {{
    background: {btn_clr_bg} !important;
    color: {btn_clr_text} !important;
    border: 1px solid {btn_clr_bdr} !important;
    width: 100% !important;
}}
[data-testid="stSidebar"] .stButton > button[style*="width: 100%"]:hover,
[data-testid="stSidebar"] .stButton > button[style*="width:100%"]:hover {{
    background: {btn_clr_hbg} !important;
    border-color: {btn_clr_hbdr} !important;
    transform: translateY(-1px) !important;
}}

/* ── HERO ── */
.hero {{
    text-align: center;
    padding: 0.5rem 0 0;
    margin-bottom: 0.15rem;
}}
.hero-logo {{
    font-size: 2.1rem;
    margin-bottom: 0.3rem;
    filter: {logo_shadow};
}}
.hero-title {{
    font-size: 2.45rem;
    font-weight: 700;
    letter-spacing: -1.5px;
    background: {hero_gradient};
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin-bottom: 0.3rem;
}}
.hero-sub {{
    font-size: 0.84rem;
    color: {hero_sub};
    letter-spacing: 0.02em;
    margin-bottom: 0.8rem;
}}
.tech-badges {{
    display: flex;
    justify-content: center;
    gap: 6px;
    flex-wrap: wrap;
    margin-bottom: 1.2rem;
}}
.tech-badge {{
    background: {badge_bg};
    border: 1px solid {badge_border};
    color: {badge_text};
    font-size: 0.69rem;
    font-weight: 500;
    padding: 3px 10px;
    border-radius: 20px;
    letter-spacing: 0.02em;
}}
.divider-line {{
    height: 1px;
    background: {divider_grad};
    margin: 0 0 1.2rem 0;
}}

/* ── CHAT MESSAGES ── */
[data-testid="stChatMessage"] {{
    background: transparent !important;
    border: none !important;
    padding: 0.10rem 0 !important;
}}
[data-testid="stChatMessage"] [data-testid="chatAvatarIcon-user"] {{
    background: {av_user} !important;
    border-radius: 8px !important;
}}
[data-testid="stChatMessage"] [data-testid="chatAvatarIcon-assistant"] {{
    background: {av_ai_bg} !important;
    border: 1px solid {av_ai_bdr} !important;
    border-radius: 8px !important;
}}

/* ── BUBBLES ── */
.user-bubble {{
    background: {bub_u_bg};
    border: 1px solid {bub_u_bdr};
    border-radius: 16px 16px 4px 16px;
    padding: 0.80rem 1.05rem;
    color: {bub_u_text};
    font-size: 0.90rem;
    line-height: 1.65;
}}
.ai-bubble {{
    background: {bub_a_bg};
    border: 1px solid {bub_a_bdr};
    border-radius: 16px 16px 16px 4px;
    padding: 0.80rem 1.05rem;
    color: {bub_a_text};
    font-size: 0.90rem;
    line-height: 1.68;
}}

/* ── SOURCES EXPANDER ── */
[data-testid="stExpander"] {{
    background: {exp_bg} !important;
    border: 1px solid {exp_bdr} !important;
    border-radius: 8px !important;
    margin-top: 4px !important;
}}
[data-testid="stExpander"] details summary {{
    color: {exp_title} !important;
    font-size: 0.75rem !important;
    font-weight: 500 !important;
}}
.source-card {{
    display: flex;
    align-items: center;
    gap: 9px;
    background: {src_bg};
    border-left: 2px solid {src_lbdr};
    border-radius: 0 7px 7px 0;
    padding: 6px 11px;
    margin: 3px 0;
    font-size: 0.76rem;
    color: {src_text};
}}
.source-page {{
    margin-left: auto;
    background: {src_pg_bg};
    border-radius: 4px;
    padding: 2px 7px;
    font-size: 0.66rem;
    color: {src_pg_text};
    font-weight: 600;
    white-space: nowrap;
}}

/* ── CHAT INPUT ── */
[data-testid="stChatInput"] > div {{
    background: {inp_bg} !important;
    border: 1.5px solid {inp_bdr} !important;
    border-radius: 12px !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}}
[data-testid="stChatInput"] > div:focus-within {{
    border-color: {inp_fbdr} !important;
    box-shadow: 0 0 0 3px {inp_fring} !important;
}}
[data-testid="stChatInput"] textarea {{
    color: {inp_text} !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.88rem !important;
    background: transparent !important;
}}
[data-testid="stChatInput"] textarea::placeholder {{
    color: {inp_ph} !important;
}}

/* ── EMPTY STATE ── */
.empty-state {{
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 3.5rem 2rem;
    text-align: center;
    min-height: 50vh;
}}
.empty-icon {{
    font-size: 3rem;
    margin-bottom: 0.95rem;
    animation: float 4s ease-in-out infinite;
}}
@keyframes float {{
    0%,100% {{ transform: translateY(0); }}
    50%      {{ transform: translateY(-10px); }}
}}
.empty-title {{
    font-size: 1.07rem;
    font-weight: 600;
    color: {empty_title};
    margin-bottom: 0.38rem;
}}
.empty-text {{
    font-size: 0.81rem;
    color: {empty_text};
    max-width: 290px;
    line-height: 1.65;
}}
.feature-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    max-width: 360px;
    margin: 1.2rem auto 0;
}}
.feature-item {{
    background: {feat_bg};
    border: 1px solid {feat_bdr};
    border-radius: 8px;
    padding: 10px;
    font-size: 0.75rem;
    color: {feat_text};
    display: flex;
    align-items: flex-start;
    gap: 7px;
    text-align: left;
    line-height: 1.5;
}}
.feat-icon {{
    font-size: 0.94rem;
    flex-shrink: 0;
    margin-top: 1px;
}}

/* ── MISC ── */
[data-testid="stSpinner"] p {{ color: {spinner_c} !important; font-size: 0.82rem !important; }}
[data-testid="stAlert"] {{ border-radius: {alert_radius} !important; font-size: 0.82rem !important; }}

::-webkit-scrollbar {{ width: 3px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: {scroll_t}; border-radius: 3px; }}
::-webkit-scrollbar-thumb:hover {{ background: {scroll_h}; }}

.stMarkdown code {{
    background: {code_bg} !important;
    color: {code_text} !important;
    border-radius: 4px !important;
    padding: 1px 5px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.83em !important;
}}
.stMarkdown pre {{
    background: {pre_bg} !important;
    border: 1px solid {pre_bdr} !important;
    border-radius: 8px !important;
}}
.stMarkdown p, .stMarkdown li, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {{
    color: {bub_a_text} !important;
}}
</style>"""

#  SESSION STATE
def init_session_state():
    defaults = {
        "messages": [],
        "chain": None,
        "processed_files": [],
        "total_queries": 0,
        "theme": "dark",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

#  PDF PROCESSING
def process_pdfs(uploaded_files):
    api_key = get_api_key()
    if not api_key:
        st.error("GROQ_API_KEY not found. Add it to .env (local) or Streamlit secrets (cloud).")
        return

    with st.spinner("Indexing documents — building vector store…"):
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                paths = []
                for f in uploaded_files:
                    p = os.path.join(tmp_dir, f.name)
                    with open(p, "wb") as out:
                        out.write(f.getbuffer())
                    paths.append(p)

                chunks = load_and_chunk_pdfs(paths)
                vector_store = build_vectorstore(chunks)
                st.session_state.chain = build_rag_chain(vector_store, api_key)
                st.session_state.processed_files = [f.name for f in uploaded_files]
                st.session_state.messages = []
                st.session_state.total_queries = 0
        except Exception as e:
            st.error(f"Processing failed: {e}")

#  SIDEBAR
def render_sidebar():
    theme = st.session_state.theme

    with st.sidebar:
        # ── Brand + toggle ──
        col_brand, col_toggle = st.columns([5, 1])
        with col_brand:
            st.markdown(
                '<div class="sb-brand">'
                '<span class="sb-logo">✦</span>'
                '<div>'
                '<div class="sb-title">DocuSense</div>'
                '<div class="sb-sub">AI Document Assistant</div>'
                '</div>'
                '</div>',
                unsafe_allow_html=True,
            )
        with col_toggle:
            tog_icon = "☀" if theme == "dark" else "🌙"
            tog_tip  = "Switch to Light mode" if theme == "dark" else "Switch to Dark mode"
            if st.button(tog_icon, key="theme_toggle", help=tog_tip):
                st.session_state.theme = "light" if theme == "dark" else "dark"
                st.rerun()

        # ── Upload ──
        st.markdown('<div class="sb-label">Upload PDFs</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            label="upload",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        if uploaded:
            file_names = [f.name for f in uploaded]
            if file_names != st.session_state.processed_files:
                process_pdfs(uploaded)

        # ── Stats ──
        if st.session_state.processed_files:
            n_docs  = len(st.session_state.processed_files)
            n_q     = st.session_state.total_queries
            n_turns = len([m for m in st.session_state.messages if m["role"] == "user"])

            st.markdown(
                f'<div class="stats-row">'
                f'<div class="stat-card"><div class="stat-num">{n_docs}</div><div class="stat-lbl">Docs</div></div>'
                f'<div class="stat-card"><div class="stat-num">{n_q}</div><div class="stat-lbl">Queries</div></div>'
                f'<div class="stat-card"><div class="stat-num">{n_turns}</div><div class="stat-lbl">Turns</div></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # ── File pills ──
            st.markdown('<div class="sb-label">Indexed Files</div>', unsafe_allow_html=True)
            pills_html = '<div class="file-list">'
            for name in st.session_state.processed_files:
                short = name if len(name) <= 28 else name[:25] + "…"
                pills_html += (
                    f'<div class="file-pill">'
                    f'<span class="pill-dot"></span>'
                    f'<span title="{name}">{short}</span>'
                    f'</div>'
                )
            pills_html += "</div>"
            st.markdown(pills_html, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Clear ──
            if st.button("✕  Clear session", key="clear_session", use_container_width=True):
                st.session_state.messages        = []
                st.session_state.chain           = None
                st.session_state.processed_files = []
                st.session_state.total_queries   = 0
                st.rerun()

#  CHAT AREA
def render_chat():
    # ── Hero ──
    st.markdown(
        '<div class="hero">'
        '<div class="hero-logo">✦</div>'
        '<div class="hero-title">DocuSense</div>'
        '<div class="hero-sub">Ask anything about your uploaded documents</div>'
        '<div class="tech-badges">'
        '<span class="tech-badge">LangChain</span>'
        '<span class="tech-badge">FAISS</span>'
        '<span class="tech-badge">Groq</span>'
        '<span class="tech-badge">RAG</span>'
        '<span class="tech-badge">Streamlit</span>'
        '</div>'
        '</div>'
        '<div class="divider-line"></div>',
        unsafe_allow_html=True,
    )

    # ── Empty state ──
    if not st.session_state.messages:
        st.markdown(
            '<div class="empty-state">'
            '<div class="empty-icon">📄</div>'
            '<div class="empty-title">No documents indexed yet</div>'
            '<div class="empty-text">Upload one or more PDFs in the sidebar to start chatting with your documents.</div>'
            '<div class="feature-grid">'
            '<div class="feature-item"><span class="feat-icon">🔍</span>Semantic search across all pages</div>'
            '<div class="feature-item"><span class="feat-icon">💬</span>Multi-turn conversation with memory</div>'
            '<div class="feature-item"><span class="feat-icon">📎</span>Page-level source citations</div>'
            '<div class="feature-item"><span class="feat-icon">🚫</span>Grounded — never fabricates answers</div>'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    # ── Message history ──
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(
                    f'<div class="user-bubble">{msg["content"]}</div>',
                    unsafe_allow_html=True,
                )
        else:
            with st.chat_message("assistant"):
                st.markdown(
                    f'<div class="ai-bubble">{msg["content"]}</div>',
                    unsafe_allow_html=True,
                )
                if msg.get("sources"):
                    with st.expander(f"📎  Sources ({len(msg['sources'])})"):
                        src_html = ""
                        for src in msg["sources"]:
                            src_html += (
                                f'<div class="source-card">'
                                f'<span>📄 {src["file"]}</span>'
                                f'<span class="source-page">p. {src["page"]}</span>'
                                f'</div>'
                            )
                        st.markdown(src_html, unsafe_allow_html=True)

    # ── Chat input ──
    if st.session_state.chain is None:
        placeholder = "Upload PDFs first, then ask your question…"
    else:
        placeholder = "Ask anything about your documents…"

    prompt = st.chat_input(placeholder)

    if prompt:
        if st.session_state.chain is None:
            st.warning("Please upload and index PDFs before asking questions.")
            return

        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.total_queries += 1

        with st.chat_message("user"):
            st.markdown(
                f'<div class="user-bubble">{prompt}</div>',
                unsafe_allow_html=True,
            )

        with st.chat_message("assistant"):
            with st.spinner("Searching documents…"):
                try:
                    result  = st.session_state.chain.invoke({"question": prompt})
                    answer  = result.get("answer", "")
                    sources = extract_sources(result.get("source_documents", []))
                except Exception as e:
                    answer  = f"Error: {e}"
                    sources = []

            st.markdown(
                f'<div class="ai-bubble">{answer}</div>',
                unsafe_allow_html=True,
            )
            if sources:
                with st.expander(f"📎  Sources ({len(sources)})"):
                    src_html = ""
                    for src in sources:
                        src_html += (
                            f'<div class="source-card">'
                            f'<span>📄 {src["file"]}</span>'
                            f'<span class="source-page">p. {src["page"]}</span>'
                            f'</div>'
                        )
                    st.markdown(src_html, unsafe_allow_html=True)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
        })

def main():
    init_session_state()
    st.markdown(get_css(st.session_state.theme), unsafe_allow_html=True)
    render_sidebar()
    render_chat()

if __name__ == "__main__":
    main()
