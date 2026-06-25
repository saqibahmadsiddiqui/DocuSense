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

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', -apple-system, sans-serif; }
#MainMenu, footer { visibility: hidden; }

/* ── APP BACKGROUND ─────────────────────────────── */
.stApp {
    background: #020c18;
    background-image:
        radial-gradient(ellipse 80% 50% at 15% 10%, rgba(6,182,212,0.14) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 85% 85%, rgba(16,185,129,0.10) 0%, transparent 60%),
        radial-gradient(ellipse 50% 50% at 50% 50%, rgba(14,165,233,0.05) 0%, transparent 70%);
}
.main .block-container {
    padding: 1.5rem 2.5rem 6rem 2.5rem;
    max-width: 860px;
    margin: 0 auto;
}

/* ── SIDEBAR ────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: rgba(2,10,22,0.98) !important;
    border-right: 1px solid rgba(6,182,212,0.18) !important;
}
[data-testid="stSidebarContent"] { padding: 1.5rem 1.2rem; }

/* ── HERO ───────────────────────────────────────── */
.hero { text-align:center; padding:0.75rem 0 0; margin-bottom:0.25rem; }
.hero-title {
    font-size: 2.7rem;
    font-weight: 700;
    letter-spacing: -1.5px;
    background: linear-gradient(135deg, #38bdf8 0%, #06b6d4 45%, #34d399 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin-bottom: 0.4rem;
}
.hero-sub {
    font-size: 0.87rem;
    color: rgba(103,232,249,0.45);
    letter-spacing: 0.04em;
    margin-bottom: 1rem;
}
.tech-badges { display:flex; justify-content:center; gap:7px; flex-wrap:wrap; margin-bottom:1.5rem; }
.tech-badge {
    background: rgba(6,182,212,0.10);
    border: 1px solid rgba(6,182,212,0.30);
    color: #67e8f9;
    font-size: 0.72rem;
    font-weight: 500;
    padding: 3px 10px;
    border-radius: 20px;
    letter-spacing: 0.03em;
}
.divider-glow {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(6,182,212,0.45), rgba(52,211,153,0.3), transparent);
    margin: 0 0 1.5rem 0;
}

/* ── SIDEBAR LABEL ──────────────────────────────── */
.sb-label {
    font-size: 0.68rem;
    font-weight: 600;
    color: rgba(103,232,249,0.40);
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 0.65rem;
}

/* ── FILE UPLOADER ──────────────────────────────── */
[data-testid="stFileUploader"] {
    background: rgba(6,182,212,0.05) !important;
    border: 1.5px dashed rgba(6,182,212,0.32) !important;
    border-radius: 14px !important;
    transition: all 0.25s ease;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(52,211,153,0.55) !important;
    background: rgba(6,182,212,0.09) !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] div span {
    color: rgba(103,232,249,0.6) !important;
    font-size: 0.85rem !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] div small {
    color: rgba(103,232,249,0.3) !important;
}

/* ── FILE PILLS ─────────────────────────────────── */
.file-list { margin-top:0.75rem; }
.file-pill {
    display: flex;
    align-items: center;
    gap: 8px;
    background: linear-gradient(135deg, rgba(6,182,212,0.13), rgba(16,185,129,0.09));
    border: 1px solid rgba(6,182,212,0.26);
    border-radius: 9px;
    padding: 7px 12px;
    margin-bottom: 5px;
    color: #67e8f9;
    font-size: 0.79rem;
    overflow: hidden;
    white-space: nowrap;
    text-overflow: ellipsis;
    animation: fadeSlide 0.25s ease;
}
.pill-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #34d399;
    flex-shrink: 0;
    box-shadow: 0 0 7px rgba(52,211,153,0.7);
}
@keyframes fadeSlide {
    from { opacity:0; transform:translateX(-8px); }
    to   { opacity:1; transform:translateX(0); }
}

/* ── STATS ──────────────────────────────────────── */
.stats-row { display:flex; gap:8px; margin:0.8rem 0 1rem 0; }
.stat-card {
    flex: 1;
    background: rgba(2,12,24,0.80);
    border: 1px solid rgba(6,182,212,0.20);
    border-radius: 10px;
    padding: 10px 6px;
    text-align: center;
}
.stat-num { font-size:1.3rem; font-weight:700; color:#38bdf8; line-height:1; }
.stat-lbl { font-size:0.63rem; color:rgba(103,232,249,0.40); text-transform:uppercase; letter-spacing:0.06em; margin-top:3px; }

/* ── CLEAR BUTTON — icon + text inline ──────────── */
.stButton > button {
    background: rgba(239,68,68,0.08) !important;
    color: #fca5a5 !important;
    border: 1px solid rgba(239,68,68,0.28) !important;
    border-radius: 10px !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
    /* Force row layout so icon never stacks above text */
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 0 !important;
    white-space: nowrap !important;
    line-height: 1.5 !important;
    padding: 0.45rem 1rem !important;
}
.stButton > button:hover {
    background: rgba(239,68,68,0.17) !important;
    border-color: rgba(239,68,68,0.50) !important;
    transform: translateY(-1px) !important;
}
/* Inline fix: flatten Streamlit's inner p/div so emoji never wraps */
.stButton > button > div,
.stButton > button > div > p {
    display: inline !important;
    margin: 0 !important;
    padding: 0 !important;
    white-space: nowrap !important;
    line-height: inherit !important;
    vertical-align: middle !important;
}
/* Add × icon via CSS so we don't rely on emoji in Python */
.stButton > button::before {
    content: "✕";
    margin-right: 6px;
    font-size: 0.78rem;
    opacity: 0.75;
    vertical-align: middle;
    line-height: 1;
}

/* ── CHAT MESSAGES ──────────────────────────────── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0.2rem 0 !important;
}
[data-testid="stChatMessage"] [data-testid="chatAvatarIcon-user"] {
    background: linear-gradient(135deg,#0ea5e9,#06b6d4) !important;
    border-radius: 10px !important;
}
[data-testid="stChatMessage"] [data-testid="chatAvatarIcon-assistant"] {
    background: linear-gradient(135deg,#020c18,#072035) !important;
    border: 1px solid rgba(6,182,212,0.42) !important;
    border-radius: 10px !important;
}

/* ── MESSAGE BUBBLES ────────────────────────────── */
.user-bubble {
    background: linear-gradient(135deg,rgba(14,165,233,0.20),rgba(6,182,212,0.14));
    border: 1px solid rgba(14,165,233,0.32);
    border-radius: 16px 16px 4px 16px;
    padding: 0.9rem 1.15rem;
    color: #e2e8f0;
    font-size: 0.93rem;
    line-height: 1.65;
    box-shadow: 0 2px 14px rgba(6,182,212,0.12);
}
.ai-bubble {
    background: rgba(2,14,34,0.72);
    border: 1px solid rgba(6,182,212,0.15);
    border-radius: 16px 16px 16px 4px;
    padding: 0.9rem 1.15rem;
    color: #e2e8f0;
    font-size: 0.93rem;
    line-height: 1.7;
    backdrop-filter: blur(10px);
    box-shadow: 0 2px 14px rgba(0,0,0,0.22);
}

/* ── SOURCES EXPANDER ───────────────────────────── */
[data-testid="stExpander"] {
    background: rgba(6,182,212,0.04) !important;
    border: 1px solid rgba(6,182,212,0.17) !important;
    border-radius: 10px !important;
    margin-top: 5px !important;
}
[data-testid="stExpander"] details summary {
    color: #38bdf8 !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
}
.source-card {
    display: flex;
    align-items: center;
    gap: 10px;
    background: rgba(6,182,212,0.08);
    border-left: 3px solid #0ea5e9;
    border-radius: 0 8px 8px 0;
    padding: 7px 12px;
    margin: 4px 0;
    font-size: 0.79rem;
    color: #67e8f9;
}
.source-page {
    margin-left: auto;
    background: rgba(6,182,212,0.18);
    border-radius: 5px;
    padding: 2px 8px;
    font-size: 0.69rem;
    color: #38bdf8;
    font-weight: 600;
    white-space: nowrap;
}

/* ── CHAT INPUT ─────────────────────────────────── */
[data-testid="stChatInput"] > div {
    background: rgba(2,12,24,0.93) !important;
    border: 1px solid rgba(6,182,212,0.36) !important;
    border-radius: 14px !important;
    backdrop-filter: blur(12px) !important;
    box-shadow: 0 4px 24px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.03) !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}
[data-testid="stChatInput"] > div:focus-within {
    border-color: rgba(56,189,248,0.65) !important;
    box-shadow: 0 4px 24px rgba(0,0,0,0.35), 0 0 0 3px rgba(6,182,212,0.13), inset 0 1px 0 rgba(255,255,255,0.03) !important;
}
[data-testid="stChatInput"] textarea {
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.9rem !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: rgba(103,232,249,0.28) !important;
}

/* ── EMPTY STATE ────────────────────────────────── */
.empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 4rem 2rem;
    text-align: center;
    min-height: 52vh;
}
.empty-icon {
    font-size: 3.6rem;
    margin-bottom: 1.2rem;
    animation: float 4s ease-in-out infinite;
    filter: drop-shadow(0 0 22px rgba(6,182,212,0.50));
}
@keyframes float {
    0%,100% { transform:translateY(0px); }
    50%      { transform:translateY(-13px); }
}
.empty-title { font-size:1.15rem; font-weight:600; color:rgba(103,232,249,0.70); margin-bottom:0.5rem; }
.empty-text  { font-size:0.84rem; color:rgba(103,232,249,0.32); max-width:310px; line-height:1.65; }
.feature-grid { display:grid; grid-template-columns:1fr 1fr; gap:10px; max-width:380px; margin:1.5rem auto 0; }
.feature-item {
    background: rgba(6,182,212,0.07);
    border: 1px solid rgba(6,182,212,0.14);
    border-radius: 10px;
    padding: 12px;
    font-size: 0.78rem;
    color: rgba(103,232,249,0.52);
    display: flex;
    align-items: flex-start;
    gap: 8px;
    text-align: left;
}
.feat-icon { font-size:1.05rem; flex-shrink:0; }

/* ── MISC ───────────────────────────────────────── */
[data-testid="stSpinner"] p { color:#38bdf8 !important; font-size:0.85rem !important; }
[data-testid="stAlert"] { border-radius:12px !important; border:none !important; font-size:0.85rem !important; }
::-webkit-scrollbar { width:3px; }
::-webkit-scrollbar-track { background:transparent; }
::-webkit-scrollbar-thumb { background:rgba(6,182,212,0.36); border-radius:3px; }
::-webkit-scrollbar-thumb:hover { background:rgba(52,211,153,0.58); }
.stMarkdown code {
    background: rgba(6,182,212,0.13) !important;
    color: #67e8f9 !important;
    border-radius: 4px !important;
    padding: 1px 5px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85em !important;
}
.stMarkdown pre {
    background: rgba(2,12,24,0.85) !important;
    border: 1px solid rgba(6,182,212,0.18) !important;
    border-radius: 10px !important;
}
</style>
"""


def init_session_state():
    defaults = {
        "messages": [],
        "chain": None,
        "processed_files": [],
        "total_queries": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def process_pdfs(uploaded_files):
    temp_dir = tempfile.mkdtemp()
    temp_paths = []
    for file in uploaded_files:
        path = os.path.join(temp_dir, file.name)
        with open(path, "wb") as f:
            f.write(file.read())
        temp_paths.append(path)
    chunks = load_and_chunk_pdfs(temp_paths)
    vectorstore = build_vectorstore(chunks)
    return build_rag_chain(vectorstore, get_api_key())


def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center;margin-bottom:1.6rem;">
            <div style="font-size:2rem;margin-bottom:0.3rem;
                filter:drop-shadow(0 0 14px rgba(6,182,212,0.6));">✦</div>
            <div style="font-size:1.1rem;font-weight:700;letter-spacing:-0.5px;
                background:linear-gradient(135deg,#38bdf8,#06b6d4,#34d399);
                -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
                DocuSense
            </div>
            <div style="font-size:0.67rem;color:rgba(103,232,249,0.32);
                letter-spacing:0.09em;text-transform:uppercase;margin-top:2px;">
                AI Document Assistant
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sb-label">📂 Upload PDFs</div>', unsafe_allow_html=True)

        uploaded_files = st.file_uploader(
            "Upload PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed"
        )

        if uploaded_files:
            names = [f.name for f in uploaded_files]
            if names != st.session_state.processed_files:
                with st.spinner("Indexing documents…"):
                    st.session_state.chain = process_pdfs(uploaded_files)
                    st.session_state.processed_files = names
                    st.session_state.messages = []
                    st.session_state.total_queries = 0
                st.success(f"✓ {len(uploaded_files)} document(s) ready")

        if st.session_state.processed_files:
            turns = len(st.session_state.messages) // 2
            st.markdown(f"""
            <div class="stats-row">
                <div class="stat-card">
                    <div class="stat-num">{len(st.session_state.processed_files)}</div>
                    <div class="stat-lbl">Docs</div>
                </div>
                <div class="stat-card">
                    <div class="stat-num">{st.session_state.total_queries}</div>
                    <div class="stat-lbl">Queries</div>
                </div>
                <div class="stat-card">
                    <div class="stat-num">{turns}</div>
                    <div class="stat-lbl">Turns</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="sb-label">📄 Indexed Files</div>', unsafe_allow_html=True)
            pills = "".join(
                f'<div class="file-pill"><span class="pill-dot"></span>'
                f'{n if len(n) <= 26 else n[:23] + "…"}</div>'
                for n in st.session_state.processed_files
            )
            st.markdown(f'<div class="file-list">{pills}</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            '<hr style="border-color:rgba(6,182,212,0.14);margin:0.4rem 0 1rem;">',
            unsafe_allow_html=True
        )

        # ← No emoji in Python string; icon comes from CSS ::before
        if st.button("Clear Session", use_container_width=True):
            st.session_state.messages = []
            st.session_state.chain = None
            st.session_state.processed_files = []
            st.session_state.total_queries = 0
            st.rerun()

def render_chat():
    st.markdown("""
    <div class="hero">
        <div style="font-size:2.4rem;margin-bottom:0.35rem;
            filter:drop-shadow(0 0 18px rgba(6,182,212,0.55));">✦</div>
        <div class="hero-title">DocuSense</div>
        <div class="hero-sub">Converse intelligently with your documents</div>
        <div class="tech-badges">
            <span class="tech-badge">⚡ Groq LLM</span>
            <span class="tech-badge">🔗 LangChain</span>
            <span class="tech-badge">🗄 FAISS Vector DB</span>
            <span class="tech-badge">🤖 RAG Pipeline</span>
        </div>
    </div>
    <div class="divider-glow"></div>
    """, unsafe_allow_html=True)

    if not st.session_state.messages:
        if not st.session_state.chain:
            st.markdown("""
            <div class="empty-state">
                <div class="empty-icon">📂</div>
                <div class="empty-title">No documents loaded yet</div>
                <div class="empty-text">
                    Upload your PDFs from the sidebar to begin intelligent conversations.
                </div>
                <div class="feature-grid">
                    <div class="feature-item">
                        <span class="feat-icon">🔍</span>Semantic search across all pages
                    </div>
                    <div class="feature-item">
                        <span class="feat-icon">📑</span>Multi-PDF support
                    </div>
                    <div class="feature-item">
                        <span class="feat-icon">🧠</span>Context-aware answers
                    </div>
                    <div class="feature-item">
                        <span class="feat-icon">🔗</span>Cited sources per reply
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="empty-state">
                <div class="empty-icon">💬</div>
                <div class="empty-title">Ready — ask anything!</div>
                <div class="empty-text">
                    Your documents are indexed. Type your first question below.
                </div>
            </div>
            """, unsafe_allow_html=True)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            cls = "user-bubble" if msg["role"] == "user" else "ai-bubble"
            st.markdown(f'<div class="{cls}">{msg["content"]}</div>', unsafe_allow_html=True)
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander(f"📎 {len(msg['sources'])} source(s) referenced"):
                    for s in msg["sources"]:
                        st.markdown(
                            f'<div class="source-card">📄 {s["file"]}'
                            f'<span class="source-page">p. {s["page"]}</span></div>',
                            unsafe_allow_html=True
                        )

    if prompt := st.chat_input("Ask anything about your documents…"):
        if not st.session_state.chain:
            st.warning("⬅ Please upload at least one PDF from the sidebar first.")
            st.stop()

        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.total_queries += 1

        with st.chat_message("user"):
            st.markdown(f'<div class="user-bubble">{prompt}</div>', unsafe_allow_html=True)

        with st.chat_message("assistant"):
            with st.spinner("Searching through your documents…"):
                result  = st.session_state.chain.invoke({"question": prompt})
                answer  = result["answer"]
                sources = extract_sources(result.get("source_documents", []))

            st.markdown(f'<div class="ai-bubble">{answer}</div>', unsafe_allow_html=True)

            if sources:
                with st.expander(f"📎 {len(sources)} source(s) referenced"):
                    for s in sources:
                        st.markdown(
                            f'<div class="source-card">📄 {s["file"]}'
                            f'<span class="source-page">p. {s["page"]}</span></div>',
                            unsafe_allow_html=True
                        )

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })


def main():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    init_session_state()
    render_sidebar()
    render_chat()


if __name__ == "__main__":
    main()