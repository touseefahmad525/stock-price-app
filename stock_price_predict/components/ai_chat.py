import html
from utils.chatbot import get_ai_response
import streamlit as st

# ── Brand tokens ──────────────────────────────────────────────────────────────
ACCENT       = "#5B47FB"          # electric indigo
ACCENT_LIGHT = "#EEF0FF"
ACCENT_MID   = "#7C6FFC"
DARK         = "#0F0E17"
SURFACE      = "#FFFFFF"
MUTED        = "#6B7280"
BORDER       = "#E5E7EB"
USER_BG      = "#EEF0FF"

# ── SVG assets ────────────────────────────────────────────────────────────────
LOGO_SVG = """
<svg width="26" height="26" viewBox="0 0 26 26" fill="none" xmlns="http://www.w3.org/2000/svg">
  <circle cx="13" cy="13" r="12" fill="#5B47FB"/>
  <path d="M8 13.5C8 10.46 10.46 8 13.5 8C16.54 8 19 10.46 19 13.5" stroke="white" stroke-width="1.8" stroke-linecap="round"/>
  <circle cx="9.5" cy="15.5" r="1.5" fill="white"/>
  <circle cx="17.5" cy="15.5" r="1.5" fill="white"/>
  <path d="M10 18.5C10.83 19.48 12.3 20 13.5 20C14.7 20 16.17 19.48 17 18.5" stroke="white" stroke-width="1.6" stroke-linecap="round"/>
</svg>
"""

ICON_SVG_DATA = (
    "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='22' height='22' "
    "viewBox='0 0 26 26' fill='none'%3E"
    "%3Ccircle cx='13' cy='13' r='12' fill='%235B47FB'/%3E"
    "%3Cpath d='M8 13.5C8 10.46 10.46 8 13.5 8C16.54 8 19 10.46 19 13.5' stroke='white' stroke-width='1.8' stroke-linecap='round'/%3E"
    "%3Ccircle cx='9.5' cy='15.5' r='1.5' fill='white'/%3E"
    "%3Ccircle cx='17.5' cy='15.5' r='1.5' fill='white'/%3E"
    "%3Cpath d='M10 18.5C10.83 19.48 12.3 20 13.5 20C14.7 20 16.17 19.48 17 18.5' stroke='white' stroke-width='1.6' stroke-linecap='round'/%3E"
    "%3C/svg%3E"
)

SEND_ICON = "➤"


# ── State ─────────────────────────────────────────────────────────────────────
def _init_chat_state():
    if "ai_chat_open" not in st.session_state:
        st.session_state.ai_chat_open = False
    if "ai_chat_messages" not in st.session_state:
        st.session_state.ai_chat_messages = []


def _close_chat():
    st.session_state.ai_chat_open = False


# ── CSS ───────────────────────────────────────────────────────────────────────
def _chat_css():
    root = (
        'div[data-testid="stVerticalBlock"]:has(.stock-ai-chat-scope)'
        ':not(:has([data-testid="stHeading"]))'
    )
    return f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');

    /* ── Reset main dashboard layout ── */
    section[data-testid="stMain"] .block-container
    div[data-testid="stVerticalBlock"]:has([data-testid="stHeading"]) {{
        position: static !important;
        bottom: auto !important; right: auto !important;
        left: auto !important; top: auto !important;
        width: 100% !important; max-width: 100% !important;
        z-index: auto !important; transform: none !important;
        pointer-events: auto !important;
    }}

    /* ── Floating container ── */
    {root} {{
        position: fixed !important;
        bottom: 1.75rem !important;
        right: 1.75rem !important;
        left: auto !important; top: auto !important;
        z-index: 999999 !important;
        width: auto !important;
        max-width: calc(100vw - 2rem) !important;
        background: transparent !important;
        border: none !important;
        padding: 0 !important; margin: 0 !important;
        gap: 0.35rem !important;
        transform: none !important;
        pointer-events: none !important;
        font-family: 'DM Sans', system-ui, sans-serif !important;
    }}
    {root} > div {{
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
        gap: 0.35rem !important;
        pointer-events: auto !important;
    }}
    {root} [data-testid="stElementContainer"],
    {root} .element-container,
    {root} form,
    {root} button {{
        pointer-events: auto !important;
    }}

    /* ── Popup card ── */
    {root}:has(.ai-popup-marker) {{
        width: 30rem !important;
        height: 40rem !important;
        background: {SURFACE} !important;
        border-radius: 1.25rem !important;
        box-shadow:
            0 2px 4px rgba(0,0,0,0.04),
            0 8px 24px rgba(91,71,251,0.10),
            0 24px 48px rgba(0,0,0,0.12) !important;
        overflow: hidden !important;
        border: 1px solid rgba(91,71,251,0.10) !important;
        font-family: 'DM Sans', system-ui, sans-serif !important;
    }}
    {root}:has(.ai-popup-marker) > div {{
        background: {SURFACE} !important;
    }}

    /* ── Launcher ── */
    {root}:has(.ai-launcher-marker) div[data-testid="stHorizontalBlock"] {{
        gap: 0.6rem !important;
        justify-content: flex-end !important;
        width: auto !important;
        align-items: center !important;
    }}
    /* Pill button */
    {root}:has(.ai-launcher-marker) div[data-testid="column"]:first-child button {{
        background: {SURFACE} !important;
        color: {ACCENT} !important;
        border: 1.5px solid {ACCENT} !important;
        border-radius: 999px !important;
        padding: 0.65rem 1.4rem !important;
        font-weight: 600 !important;
        font-size: 0.875rem !important;
        box-shadow:
            0 2px 8px rgba(91,71,251,0.12),
            0 4px 20px rgba(91,71,251,0.08) !important;
        min-height: 2.85rem !important;
        letter-spacing: -0.01em !important;
        transition: all 0.2s ease !important;
        font-family: 'DM Sans', system-ui, sans-serif !important;
    }}
    {root}:has(.ai-launcher-marker) div[data-testid="column"]:first-child button:hover {{
        background: {ACCENT} !important;
        color: white !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 24px rgba(91,71,251,0.25) !important;
    }}
    /* Icon button */
    {root}:has(.ai-launcher-marker) div[data-testid="column"]:last-child button {{
        background: {ACCENT} url("{ICON_SVG_DATA}") center / 1.3rem no-repeat !important;
        border: none !important;
        border-radius: 50% !important;
        width: 2.85rem !important;
        min-width: 2.85rem !important;
        height: 2.85rem !important;
        min-height: 2.85rem !important;
        padding: 0 !important;
        box-shadow:
            0 4px 12px rgba(91,71,251,0.3),
            0 8px 32px rgba(91,71,251,0.15) !important;
        color: transparent !important;
        transition: all 0.2s ease !important;
    }}
    {root}:has(.ai-launcher-marker) div[data-testid="column"]:last-child button:hover {{
        transform: scale(1.08) translateY(-1px) !important;
        box-shadow: 0 8px 28px rgba(91,71,251,0.4) !important;
    }}

    /* ── Popup header ── */
    {root}:has(.ai-popup-marker) div[data-testid="stHorizontalBlock"]:first-of-type {{
        padding: 0.9rem 1rem 0.75rem !important;
        border-bottom: 1px solid {BORDER} !important;
        align-items: center !important;
        margin: 0 !important;
        background: linear-gradient(135deg, #FAFAFE 0%, #F5F4FF 100%) !important;
    }}
    .ai-chat-brand-label {{
        display: flex;
        align-items: center;
        gap: 0.55rem;
        font-weight: 700;
        font-size: 0.975rem;
        color: {DARK};
        letter-spacing: -0.02em;
        font-family: 'DM Sans', system-ui, sans-serif;
    }}
    .ai-chat-brand-label .ai-brand-sub {{
        display: block;
        font-size: 0.67rem;
        font-weight: 500;
        color: {MUTED};
        letter-spacing: 0.02em;
        margin-top: -2px;
    }}
    .ai-status-dot {{
        width: 7px; height: 7px;
        background: #22C55E;
        border-radius: 50%;
        display: inline-block;
        margin-left: 2px;
        box-shadow: 0 0 0 2px rgba(34,197,94,0.2);
        animation: ai-pulse 2s infinite;
    }}
    @keyframes ai-pulse {{
        0%, 100% {{ box-shadow: 0 0 0 2px rgba(34,197,94,0.2); }}
        50% {{ box-shadow: 0 0 0 5px rgba(34,197,94,0.0); }}
    }}
    /* End Chat button */
    {root}:has(.ai-popup-marker) div[data-testid="column"]:nth-child(2) button {{
        background: transparent !important;
        color: {MUTED} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 0.5rem !important;
        padding: 0.2rem 0.6rem !important;
        min-height: 1.8rem !important;
        font-size: 0.72rem !important;
        font-weight: 500 !important;
        box-shadow: none !important;
        transition: all 0.15s !important;
        font-family: 'DM Sans', system-ui, sans-serif !important;
    }}
    {root}:has(.ai-popup-marker) div[data-testid="column"]:nth-child(2) button:hover {{
        border-color: #EF4444 !important;
        color: #EF4444 !important;
    }}
    /* Minimise button */
    {root}:has(.ai-popup-marker) div[data-testid="column"]:nth-child(3) button {{
        background: transparent !important;
        border: none !important;
        color: {MUTED} !important;
        font-size: 1.2rem !important;
        padding: 0 !important;
        min-width: 1.6rem !important;
        width: 1.6rem !important;
        box-shadow: none !important;
        line-height: 1 !important;
        transition: color 0.15s !important;
    }}
    {root}:has(.ai-popup-marker) div[data-testid="column"]:nth-child(3) button:hover {{
        color: {DARK} !important;
    }}

    /* ── Chat body ── */
    .ai-chat-body {{
        padding: 1rem 1rem 0.5rem;
        min-height: 13rem;
        max-height: 18rem;
        overflow-y: auto;
        scroll-behavior: smooth;
        background: {SURFACE};
    }}
    .ai-chat-body::-webkit-scrollbar {{ width: 4px; }}
    .ai-chat-body::-webkit-scrollbar-track {{ background: transparent; }}
    .ai-chat-body::-webkit-scrollbar-thumb {{ background: #E5E7EB; border-radius: 4px; }}

    /* Bot message */
    .ai-msg-bot-wrap {{
        display: flex;
        gap: 0.5rem;
        margin: 0 0 0.85rem 0;
        align-items: flex-start;
    }}
    .ai-bot-avatar {{
        width: 26px; height: 26px;
        border-radius: 50%;
        background: {ACCENT};
        flex-shrink: 0;
        display: flex; align-items: center; justify-content: center;
        font-size: 0.7rem;
        margin-top: 2px;
    }}
    .ai-msg-bot {{
        background: #F8F7FF;
        border: 1px solid rgba(91,71,251,0.08);
        border-radius: 0 0.85rem 0.85rem 0.85rem;
        padding: 0.6rem 0.85rem;
        color: #1f2937;
        font-size: 0.875rem;
        line-height: 1.55;
        max-width: 82%;
        font-family: 'DM Sans', system-ui, sans-serif;
    }}
    /* User message */
    .ai-msg-user-wrap {{
        display: flex;
        justify-content: flex-end;
        margin: 0 0 0.85rem 0;
    }}
    .ai-msg-user {{
        background: linear-gradient(135deg, {ACCENT} 0%, {ACCENT_MID} 100%);
        color: white;
        padding: 0.6rem 0.9rem;
        border-radius: 0.85rem 0.85rem 0 0.85rem;
        font-size: 0.875rem;
        max-width: 75%;
        line-height: 1.45;
        box-shadow: 0 2px 8px rgba(91,71,251,0.2);
        font-family: 'DM Sans', system-ui, sans-serif;
    }}
    /* Feedback row */
    .ai-feedback-row {{
        display: flex;
        gap: 0.35rem;
        margin: -0.45rem 0 0.85rem 2.5rem;
    }}
    .ai-feedback-btn {{
        width: 1.75rem; height: 1.75rem;
        border: 1px solid {BORDER};
        border-radius: 50%;
        display: inline-flex;
        align-items: center; justify-content: center;
        font-size: 0.75rem;
        color: {MUTED};
        background: {SURFACE};
        cursor: pointer;
        transition: all 0.15s;
    }}
    .ai-feedback-btn:hover {{ border-color: {ACCENT}; color: {ACCENT}; }}

    /* Welcome state */
    .ai-welcome {{
        text-align: center;
        padding: 1.25rem 0.5rem 0.75rem;
    }}
    .ai-welcome-icon {{
        width: 48px; height: 48px;
        border-radius: 50%;
        background: linear-gradient(135deg, {ACCENT} 0%, {ACCENT_MID} 100%);
        margin: 0 auto 0.75rem;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.4rem;
        box-shadow: 0 4px 16px rgba(91,71,251,0.3);
    }}
    .ai-welcome-title {{
        font-size: 0.9rem;
        font-weight: 700;
        color: {DARK};
        margin-bottom: 0.3rem;
        letter-spacing: -0.02em;
        font-family: 'DM Sans', system-ui, sans-serif;
    }}
    .ai-welcome-sub {{
        font-size: 0.78rem;
        color: {MUTED};
        line-height: 1.5;
        font-family: 'DM Sans', system-ui, sans-serif;
    }}

    /* Suggestions */
    .ai-suggestions {{
        display: flex;
        flex-wrap: wrap;
        gap: 0.4rem;
        padding: 0 1rem 0.75rem;
        justify-content: center;
    }}
    .ai-suggestion-chip {{
        background: {ACCENT_LIGHT};
        border: 1px solid rgba(91,71,251,0.15);
        color: {ACCENT};
        font-size: 0.72rem;
        font-weight: 500;
        padding: 0.3rem 0.7rem;
        border-radius: 999px;
        cursor: pointer;
        transition: all 0.15s;
        font-family: 'DM Sans', system-ui, sans-serif;
    }}
    .ai-suggestion-chip:hover {{
        background: {ACCENT};
        color: white;
    }}

    /* ── Input area ── */
    {root}:has(.ai-popup-marker) form {{
        padding: 0.5rem 0.85rem 0.6rem !important;
        border: none !important;
        background: {SURFACE} !important;
        border-top: 1px solid {BORDER} !important;
        position: relative !important;
    }}
    {root}:has(.ai-popup-marker) form input[type="text"] {{
        background: #F9FAFB !important;
        border: 1.5px solid {BORDER} !important;
        border-radius: 0.75rem !important;
        padding: 0.65rem 2.6rem 0.65rem 0.9rem !important;
        font-size: 0.85rem !important;
        font-family: 'DM Sans', system-ui, sans-serif !important;
        color: {DARK} !important;
        transition: border-color 0.2s !important;
        width: 100% !important;
        box-sizing: border-box !important;
    }}
    {root}:has(.ai-popup-marker) form input[type="text"]:focus {{
        border-color: {ACCENT} !important;
        background: {SURFACE} !important;
        outline: none !important;
        box-shadow: 0 0 0 3px rgba(91,71,251,0.08) !important;
    }}
    {root}:has(.ai-popup-marker) form button {{
        position: absolute !important;
        right: 1.15rem !important;
        top: 50% !important;
        transform: translateY(-50%) !important;
        background: {ACCENT} !important;
        border: none !important;
        color: white !important;
        box-shadow: none !important;
        min-height: auto !important;
        padding: 0 !important;
        font-size: 0.75rem !important;
        width: 1.85rem !important;
        height: 1.85rem !important;
        border-radius: 0.45rem !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        transition: all 0.15s !important;
    }}
    {root}:has(.ai-popup-marker) form button:hover {{
        background: {ACCENT_MID} !important;
        transform: translateY(-50%) scale(1.05) !important;
    }}

    /* ── Footer ── */
    .ai-chat-footer-note {{
        padding: 0.55rem 1rem 0.75rem;
        font-size: 0.65rem;
        color: #9CA3AF;
        line-height: 1.4;
        text-align: center;
        font-family: 'DM Sans', system-ui, sans-serif;
    }}
    .ai-chat-footer-note a {{ color: {ACCENT}; text-decoration: none; }}
    .ai-chat-footer-note a:hover {{ text-decoration: underline; }}

    /* Typing dots */
    .ai-typing {{
        display: inline-flex;
        gap: 3px;
        align-items: center;
        padding: 4px 2px;
    }}
    .ai-typing span {{
        width: 5px; height: 5px;
        border-radius: 50%;
        background: {ACCENT};
        display: inline-block;
        animation: ai-dot-bounce 1.1s infinite ease-in-out;
        opacity: 0.7;
    }}
    .ai-typing span:nth-child(2) {{ animation-delay: 0.18s; }}
    .ai-typing span:nth-child(3) {{ animation-delay: 0.36s; }}
    @keyframes ai-dot-bounce {{
        0%, 80%, 100% {{ transform: translateY(0); opacity: 0.5; }}
        40% {{ transform: translateY(-5px); opacity: 1; }}
    }}
    </style>
    """


# ── Message renderer ──────────────────────────────────────────────────────────
def _render_messages_html():
    messages = st.session_state.ai_chat_messages

    if not messages:
        return """
        <div class="ai-welcome">
            <div class="ai-welcome-icon">🤖</div>
            <div class="ai-welcome-title">Hi there! I'm your Stock AI</div>
            <div class="ai-welcome-sub">Ask me about markets, portfolios,<br>or any financial question.</div>
        </div>
        """

    last_bot_index = max(
        (i for i, m in enumerate(messages) if m["role"] == "bot"),
        default=None,
    )

    parts = []
    bot_avatar = f'<div class="ai-bot-avatar">{LOGO_SVG}</div>'

    for i, message in enumerate(messages):
        text = html.escape(message["text"])
        if message["role"] == "user":
            parts.append(
                f'<div class="ai-msg-user-wrap">'
                f'<div class="ai-msg-user">{text}</div></div>'
            )
        else:
            parts.append(
                f'<div class="ai-msg-bot-wrap">'
                f'{bot_avatar}'
                f'<div class="ai-msg-bot">{text}</div>'
                f'</div>'
            )
            if i == last_bot_index:
                parts.append(
                    '<div class="ai-feedback-row">'
                    '<span class="ai-feedback-btn" title="Helpful">👍</span>'
                    '<span class="ai-feedback-btn" title="Not helpful">👎</span>'
                    "</div>"
                )

    return "\n".join(parts)


# ── Launcher ──────────────────────────────────────────────────────────────────
def _render_launcher():
    st.markdown('<div class="ai-launcher-marker"></div>', unsafe_allow_html=True)
    pill_col, icon_col = st.columns([2.4, 0.55])
    with pill_col:
        if st.button("AI Assistant", key="stock_ai_launcher_pill"):
            st.session_state.ai_chat_open = True
            st.rerun()
    with icon_col:
        if st.button(" ", key="stock_ai_launcher_icon"):
            st.session_state.ai_chat_open = True
            st.rerun()


# ── Popup ─────────────────────────────────────────────────────────────────────
def _render_popup():
    st.markdown('<div class="ai-popup-marker"></div>', unsafe_allow_html=True)

    # Header
    brand_col, end_col, min_col = st.columns([3.4, 1.3, 0.42])
    with brand_col:
        st.markdown(
            f'<div class="ai-chat-brand-label">'
            f'{LOGO_SVG}'
            f'<div>'
            f'<span>Stock AI Assistant</span>'
            f'<span class="ai-brand-sub">Market Intelligence <span class="ai-status-dot"></span></span>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with end_col:
        if st.button("End Chat", key="stock_ai_end_chat"):
            _close_chat()
            st.rerun()
    with min_col:
        if st.button("⌄", key="stock_ai_minimize"):
            _close_chat()
            st.rerun()

    # Messages
    no_messages = not st.session_state.ai_chat_messages
    st.markdown(
        f'<div class="ai-chat-body">{_render_messages_html()}</div>',
        unsafe_allow_html=True,
    )

    # Quick-start suggestions when no messages yet
    if no_messages:
        st.markdown(
            '<div class="ai-suggestions">'
            '<span class="ai-suggestion-chip">📈 Top gainers today</span>'
            '<span class="ai-suggestion-chip">📊 Portfolio review</span>'
            '<span class="ai-suggestion-chip">🔍 Analyse a stock</span>'
            '<span class="ai-suggestion-chip">📰 Latest news</span>'
            '</div>',
            unsafe_allow_html=True,
        )

    # Input form
    with st.form("stock_ai_chat_form", clear_on_submit=True):

        input_col, send_col = st.columns([5, 1])

        with input_col:
            user_message = st.text_input(
                "Message",
                placeholder="Ask about stocks, markets, news…",
                label_visibility="collapsed",
            )

        with send_col:
            sent = st.form_submit_button(SEND_ICON)

        if sent and user_message.strip():
            st.session_state.ai_chat_messages.append(
                {"role": "user", "text": user_message.strip()}
            )

            try:
                bot_response = get_ai_response(user_message.strip())
            except Exception as exc:
                bot_response = f"Chat setup error: {exc}"

            st.session_state.ai_chat_messages.append(
                {
                    "role": "bot",
                    "text": bot_response,
                }
            )

            st.rerun()

    # Footer
    st.markdown(
        '<div class="ai-chat-footer-note">'
        'By using this assistant you agree to our '
        '<a href="#">Terms of Use</a>. '
        'Not financial advice.'
        '</div>',
        unsafe_allow_html=True,
    )


# ── Public entry-point ────────────────────────────────────────────────────────
@st.fragment
def render_ai_chat():
    """Floating launcher + popup chat UI — redesigned."""
    _init_chat_state()

    with st.container():
        st.markdown(_chat_css(), unsafe_allow_html=True)
        st.markdown(
            '<div id="stock-ai-chat-root" class="stock-ai-chat-scope"></div>',
            unsafe_allow_html=True,
        )

        if st.session_state.ai_chat_open:
            _render_popup()
        else:
            _render_launcher()
