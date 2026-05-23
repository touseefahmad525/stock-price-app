import html

import streamlit as st

PURPLE = "#702D91"
USER_BUBBLE = "#EDE4F7"

LOGO_SVG = """
<svg width="28" height="28" viewBox="0 0 28 28" fill="none" xmlns="http://www.w3.org/2000/svg">
  <path d="M14 2L24 8V20L14 26L4 20V8L14 2Z" stroke="#702D91" stroke-width="1.6" fill="none"/>
  <path d="M14 8L19 11V17L14 20L9 17V11L14 8Z" fill="#702D91"/>
</svg>
"""

ICON_SVG_DATA = (
    "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='22' height='22' "
    "viewBox='0 0 28 28' fill='none'%3E%3Cpath d='M14 2L24 8V20L14 26L4 20V8L14 2Z' "
    "stroke='%23702D91' stroke-width='1.6' fill='none'/%3E%3Cpath d='M14 8L19 11V17L14 20"
    "L9 17V11L14 8Z' fill='%23702D91'/%3E%3C/svg%3E"
)


def _init_chat_state():
    if "ai_chat_open" not in st.session_state:
        st.session_state.ai_chat_open = False
    if "ai_chat_messages" not in st.session_state:
        st.session_state.ai_chat_messages = []


def _close_chat():
    st.session_state.ai_chat_open = False


def _chat_css():
    # Chat-only nested block (has marker, no page title heading).
    root = (
        'div[data-testid="stVerticalBlock"]:has(.stock-ai-chat-scope)'
        ':not(:has([data-testid="stHeading"]))'
    )
    return f"""
    <style>
    /* Keep main dashboard layout normal even if a broad rule matched earlier */
    section[data-testid="stMain"] .block-container
    div[data-testid="stVerticalBlock"]:has([data-testid="stHeading"]) {{
        position: static !important;
        bottom: auto !important;
        right: auto !important;
        left: auto !important;
        top: auto !important;
        width: 100% !important;
        max-width: 100% !important;
        z-index: auto !important;
        transform: none !important;
        pointer-events: auto !important;
    }}

    {root} {{
        position: fixed !important;
        bottom: 1.5rem !important;
        right: 1.5rem !important;
        left: auto !important;
        top: auto !important;
        z-index: 999999 !important;
        width: auto !important;
        max-width: calc(100vw - 2rem) !important;
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
        margin: 0 !important;
        gap: 0.35rem !important;
        transform: none !important;
        pointer-events: none !important;
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

    /* Popup open: white card */
    {root}:has(.ai-popup-marker) {{
        width: 22rem !important;
        background: #ffffff !important;
        border-radius: 1rem !important;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.18) !important;
        overflow: hidden !important;
        font-family: "Segoe UI", system-ui, sans-serif;
    }}
    {root}:has(.ai-popup-marker) > div {{
        background: #ffffff !important;
    }}

    /* Launcher: pill + icon */
    {root}:has(.ai-launcher-marker) div[data-testid="stHorizontalBlock"] {{
        gap: 0.5rem !important;
        justify-content: flex-end !important;
        width: auto !important;
    }}
    {root}:has(.ai-launcher-marker) div[data-testid="column"]:first-child button {{
        background: #ffffff !important;
        color: {PURPLE} !important;
        border: none !important;
        border-radius: 999px !important;
        padding: 0.65rem 1.35rem !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.14) !important;
        min-height: 2.75rem !important;
    }}
    {root}:has(.ai-launcher-marker) div[data-testid="column"]:last-child button {{
        background: #ffffff url("{ICON_SVG_DATA}") center / 1.25rem no-repeat !important;
        border: none !important;
        border-radius: 50% !important;
        width: 2.75rem !important;
        min-width: 2.75rem !important;
        height: 2.75rem !important;
        min-height: 2.75rem !important;
        padding: 0 !important;
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.14) !important;
        color: transparent !important;
    }}

    /* Popup header row */
    {root}:has(.ai-popup-marker) div[data-testid="stHorizontalBlock"]:first-of-type {{
        padding: 0.75rem 0.85rem 0.55rem !important;
        border-bottom: 1px solid #ececec !important;
        align-items: center !important;
        margin: 0 !important;
    }}
    .ai-chat-brand-label {{
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-weight: 700;
        font-size: 1rem;
        color: #111827;
    }}
    {root}:has(.ai-popup-marker) div[data-testid="column"]:nth-child(2) button {{
        background: #ffffff !important;
        color: {PURPLE} !important;
        border: 1px solid {PURPLE} !important;
        border-radius: 0.45rem !important;
        padding: 0.2rem 0.55rem !important;
        min-height: 1.75rem !important;
        font-size: 0.72rem !important;
        font-weight: 600 !important;
        box-shadow: none !important;
        margin-left: auto;
    }}
    {root}:has(.ai-popup-marker) div[data-testid="column"]:nth-child(3) button {{
        background: transparent !important;
        border: none !important;
        color: #374151 !important;
        font-size: 1.15rem !important;
        padding: 0 !important;
        min-width: 1.5rem !important;
        width: 1.5rem !important;
        box-shadow: none !important;
    }}

    /* Chat body */
    .ai-chat-body {{
        padding: 0.85rem 1rem 0.5rem;
        min-height: 11rem;
        max-height: 17rem;
        overflow-y: auto;
    }}
    .ai-msg-bot {{
        text-align: left;
        color: #1f2937;
        font-size: 0.9rem;
        line-height: 1.45;
        margin: 0 0 0.75rem 0;
        max-width: 92%;
    }}
    .ai-msg-user-wrap {{
        display: flex;
        justify-content: flex-end;
        margin: 0 0 0.75rem 0;
    }}
    .ai-msg-user {{
        background: {USER_BUBBLE};
        color: #111827;
        padding: 0.55rem 0.85rem;
        border-radius: 0.85rem;
        font-size: 0.9rem;
        max-width: 75%;
        line-height: 1.4;
    }}
    .ai-feedback-row {{
        display: flex;
        gap: 0.5rem;
        margin: -0.25rem 0 0.85rem 0;
    }}
    .ai-feedback-btn {{
        width: 2rem;
        height: 2rem;
        border: 1px solid #d1d5db;
        border-radius: 50%;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 0.85rem;
        color: #6b7280;
        background: #fff;
    }}

    /* Input */
    {root}:has(.ai-popup-marker) form {{
        padding: 0.35rem 1rem 0.5rem !important;
        border: none !important;
        background: #ffffff !important;
        position: relative;
    }}
    {root}:has(.ai-popup-marker) form input {{
        background: #f3f4f6 !important;
        border: none !important;
        border-radius: 999px !important;
        padding: 0.7rem 2.4rem 0.7rem 1rem !important;
        font-size: 0.85rem !important;
    }}
    {root}:has(.ai-popup-marker) form button {{
        position: absolute !important;
        right: 1.2rem !important;
        top: 0.65rem !important;
        background: transparent !important;
        border: none !important;
        color: #6b7280 !important;
        box-shadow: none !important;
        min-height: auto !important;
        padding: 0 !important;
        font-size: 1.05rem !important;
        width: 2rem !important;
    }}

    /* Footer */
    .ai-chat-footer-note {{
        padding: 0.55rem 1rem 0.85rem;
        border-top: 1px solid #ececec;
        font-size: 0.68rem;
        color: #9ca3af;
        line-height: 1.35;
    }}
    .ai-chat-footer-note a {{
        color: #2563eb;
        text-decoration: none;
    }}
    </style>
    """


def _render_messages_html():
    messages = st.session_state.ai_chat_messages
    if not messages:
        return (
            '<p class="ai-msg-bot">Welcome to Stock AI Assistant. '
            "How can I help you today?</p>"
        )

    parts = []
    last_bot_index = max(
        (i for i, m in enumerate(messages) if m["role"] == "bot"),
        default=None,
    )

    for i, message in enumerate(messages):
        text = html.escape(message["text"])
        if message["role"] == "user":
            parts.append(
                f'<div class="ai-msg-user-wrap">'
                f'<div class="ai-msg-user">{text}</div></div>'
            )
        else:
            parts.append(f'<p class="ai-msg-bot">{text}</p>')
            if i == last_bot_index:
                parts.append(
                    '<div class="ai-feedback-row">'
                    '<span class="ai-feedback-btn">👍</span>'
                    '<span class="ai-feedback-btn">👎</span>'
                    "</div>"
                )
    return "\n".join(parts)


def _render_launcher():
    st.markdown('<div class="ai-launcher-marker"></div>', unsafe_allow_html=True)
    pill_col, icon_col = st.columns([2.2, 0.55])
    with pill_col:
        if st.button("AI Assistant", key="stock_ai_launcher_pill"):
            st.session_state.ai_chat_open = True
            st.rerun()
    with icon_col:
        if st.button(" ", key="stock_ai_launcher_icon"):
            st.session_state.ai_chat_open = True
            st.rerun()


def _render_popup():
    st.markdown('<div class="ai-popup-marker"></div>', unsafe_allow_html=True)

    brand_col, end_col, min_col = st.columns([3.2, 1.3, 0.4])
    with brand_col:
        st.markdown(
            f'<div class="ai-chat-brand-label">{LOGO_SVG}<span>Stock AI Assistant</span></div>',
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

    st.markdown(
        f'<div class="ai-chat-body">{_render_messages_html()}</div>',
        unsafe_allow_html=True,
    )

    with st.form("stock_ai_chat_form", clear_on_submit=True):
        user_message = st.text_input(
            "Message",
            placeholder="Type your message here...",
            label_visibility="collapsed",
        )
        sent = st.form_submit_button("➤")
        if sent and user_message.strip():
            st.session_state.ai_chat_messages.append(
                {"role": "user", "text": user_message.strip()}
            )
            st.session_state.ai_chat_messages.append(
                {
                    "role": "bot",
                    "text": "I received your message. AI feature coming soon.",
                }
            )

    st.markdown(
        """
        <div class="ai-chat-footer-note">
            By using this AI assistant, you acknowledge and agree that...
            <a href="#">read more</a>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.fragment
def render_ai_chat():
    """Floating launcher + popup chat UI (reference-style)."""
    _init_chat_state()

    # Nested container = separate DOM block from dashboard content.
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
