"""
Professional AI Chatbot Application
Built with Python, Streamlit, and the official Groq API.
Soft Minimal Light Theme with Muted Teal Identity & Monochromatic Outlined Icons.

Features:
- Soft Minimal Light Theme: Pure White (#FFFFFF) canvas, Light Grey (#F3F4F6) sidebar, Muted Teal (#477D78) accent, Charcoal (#262626) text
- Inter typography with consistent weights, hierarchy, and 15-16px chat readability
- Monochromatic, outlined Lucide-style SVG vector icons throughout (strictly zero emojis)
- ChatGPT-inspired clean message layout, floating pill chat input, and spacious conversation flow
- Dynamic live model fetching directly from GroqCloud API (prevents decommission errors)
- Fully updated catalog of current Production & Preview models
- Secure API key management (never hardcoded, exposed, or logged)
- Connection validation with live Groq models endpoint
- Custom model ID support
- 5 Predefined system personas + Custom role editor
- Full hyperparameter controls (Temperature, Max Tokens, Top-P, Penalties)
- Streaming and non-streaming response modes
- Context-aware conversation history toggle
- Session management: New Chat, Clear History, Reset Settings
- Export transcripts as TXT or structured JSON
- Robust exception handling for decommissioned models, authentication, rate limits, and network errors
"""

import os
import json
from datetime import datetime
from typing import List, Dict, Any, Generator, Tuple

import streamlit as st

# Safe import of official Groq SDK with graceful fallback
try:
    from groq import (
        Groq,
        GroqError,
        AuthenticationError,
        RateLimitError,
        APIConnectionError,
        BadRequestError,
    )
except ImportError:
    Groq = None
    GroqError = Exception
    AuthenticationError = Exception
    RateLimitError = Exception
    APIConnectionError = Exception
    BadRequestError = Exception

# ==============================================================================
# Monochromatic Lucide-Style SVG Outlined Icons
# ==============================================================================

ICON_SPARK = """<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round" class="lucide-icon"><path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z"/></svg>"""

ICON_CPU = """<svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide-icon"><rect width="16" height="16" x="4" y="4" rx="2"/><rect width="6" height="6" x="9" y="9" rx="1"/><path d="M15 2v2"/><path d="M15 20v2"/><path d="M2 15h2"/><path d="M2 9h2"/><path d="M20 15h2"/><path d="M20 9h2"/><path d="M9 2v2"/><path d="M9 20v2"/></svg>"""

ICON_USER = """<svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide-icon"><path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>"""

ICON_MESSAGE = """<svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide-icon"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>"""

ICON_BOT_HERO = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 8V4H8"/><rect width="16" height="12" x="4" y="8" rx="2"/><path d="M2 14h2"/><path d="M20 14h2"/><path d="M15 13v2"/><path d="M9 13v2"/></svg>"""

ICON_KEY = """<svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide-icon"><circle cx="7.5" cy="15.5" r="5.5"/><path d="m21 2-9.6 9.6"/><path d="m15.5 7.5 3 3L22 7l-3-3"/></svg>"""

ICON_EXTERNAL = """<svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide-icon"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/><polyline points="15 3 21 3 21 9"/><line x1="10" y1="14" x2="21" y2="3"/></svg>"""

# ==============================================================================
# Constants & Model Catalogs
# ==============================================================================

# Curated active production and preview chat models from official GroqCloud Docs
DEFAULT_ACTIVE_MODELS = [
    "llama-3.3-70b-versatile",    # Production: Meta Llama 3.3 70B (Recommended)
    "llama-3.1-8b-instant",        # Production: Meta Llama 3.1 8B (560 t/s ultra-fast)
    "openai/gpt-oss-120b",         # Production: OpenAI GPT-OSS 120B (500 t/s)
    "openai/gpt-oss-20b",          # Production: OpenAI GPT-OSS 20B (1000 t/s)
    "qwen/qwen3.8-27b",            # Preview: Alibaba Qwen 3.8 27B (450 t/s)
    "meta-models/muse-spark-open", # Preview: Meta Muse Spark Open (350 t/s)
    "minimaxai/minimax-m2.7",      # Preview: MiniMax M2.7 (260 t/s)
]

# Patterns for models that do NOT support text chat completions
EXCLUDED_MODEL_PATTERNS = [
    "whisper",       # Audio speech-to-text
    "orpheus",       # Audio text-to-speech
    "prompt-guard",  # Guardrail classification
    "safeguard",     # Safety moderation
]

ROLE_TEMPLATES: Dict[str, str] = {
    "General Assistant": (
        "You are an intelligent, courteous, and highly competent AI assistant. "
        "Provide clear, well-structured, insightful, and accurate answers to all queries."
    ),
    "Python Tutor": (
        "You are an expert Python programming mentor and senior developer. You explain core "
        "and advanced concepts with crystal clarity, provide idiomatic PEP 8 compliant code examples, "
        "highlight common pitfalls, and guide the user through debugging step-by-step with patience."
    ),
    "Machine Learning Expert": (
        "You are a Senior Machine Learning Engineer and AI Researcher. You possess deep expertise "
        "in deep learning architectures (Transformers, CNNs, GNNs), statistical modeling, loss functions, "
        "optimization techniques, MLOps, model evaluation, and deployment best practices with mathematical rigor."
    ),
    "AI & Robotics Trainer": (
        "You are an advanced AI & Robotics systems specialist. You explain robotic kinematics, ROS/ROS 2, "
        "sensor fusion, SLAM, computer vision, reinforcement learning, trajectory planning, and embedded "
        "hardware integration with practical engineering precision and code examples."
    ),
    "Data Analyst": (
        "You are a Senior Data Analyst and Business Intelligence strategist. You specialize in data "
        "cleaning, exploratory data analysis (EDA), complex SQL queries, statistical hypothesis testing, "
        "data visualization principles, and translating raw data metrics into strategic business insights."
    ),
    "Custom Role": (
        "You are a specialized AI assistant tailored to user specifications. Follow all user "
        "directives precisely while maintaining logical reasoning, safety, and factual accuracy."
    ),
}

DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TOP_P = 1.0
DEFAULT_FREQ_PENALTY = 0.0
DEFAULT_PRES_PENALTY = 0.0


# ==============================================================================
# CSS Theme Engine: Soft Minimal Light Theme (Inter & Muted Teal)
# ==============================================================================

def inject_custom_styles() -> None:
    """Inject polished Soft Minimal Light Theme CSS styling."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        /* Soft Minimal Light Palette */
        :root {
            --font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            --bg-main: #FFFFFF;
            --bg-sidebar: #F3F4F6;
            --primary-accent: #477D78;
            --primary-accent-light: rgba(71, 125, 120, 0.08);
            --primary-accent-border: rgba(71, 125, 120, 0.25);
            --primary-accent-hover: #3b6864;
            --text-primary: #262626;
            --text-secondary: #525252;
            --text-muted: #737373;
            --border-subtle: #E5E7EB;
            --border-hover: #D1D5DB;
            --bg-bubble-user: #F9FAFB;
            --bg-bubble-assistant: #FFFFFF;
            --shadow-subtle: 0 1px 3px rgba(0, 0, 0, 0.03);
            --shadow-card: 0 2px 8px rgba(0, 0, 0, 0.04);
            --radius-sm: 8px;
            --radius-md: 12px;
            --radius-lg: 14px;
            --radius-pill: 24px;
        }

        /* Pure White Main Background & Inter Typography */
        html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"], [data-testid="stMainBlockContainer"] {
            background-color: var(--bg-main) !important;
            color: var(--text-primary) !important;
            font-family: var(--font-family) !important;
            -webkit-font-smoothing: antialiased;
        }

        /* Subtle Light Grey Sidebar (#F3F4F6) with Clean Minimal Border */
        section[data-testid="stSidebar"], 
        section[data-testid="stSidebar"] > div {
            background-color: var(--bg-sidebar) !important;
            border-right: 1px solid var(--border-subtle) !important;
        }

        /* Top Header Navigation Card */
        .chat-header-container {
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 12px;
            padding: 0.95rem 1.4rem;
            background: #FFFFFF;
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-lg);
            margin-bottom: 1.5rem;
            box-shadow: var(--shadow-subtle);
        }

        .header-title-box {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .header-icon-badge {
            width: 36px;
            height: 36px;
            border-radius: 50%;
            background: var(--primary-accent-light);
            border: 1px solid var(--primary-accent-border);
            display: flex;
            align-items: center;
            justify-content: center;
            color: var(--primary-accent);
        }

        .header-text h1 {
            color: var(--text-primary) !important;
            font-size: 1.15rem !important;
            font-weight: 600 !important;
            margin: 0 !important;
            letter-spacing: -0.01em;
            font-family: var(--font-family) !important;
        }

        .header-text p {
            color: var(--text-muted) !important;
            font-size: 0.8rem !important;
            margin: 1px 0 0 0 !important;
            font-family: var(--font-family) !important;
        }

        .header-badges {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            align-items: center;
        }

        .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: #FFFFFF;
            border: 1px solid var(--border-subtle);
            border-radius: 20px;
            padding: 4px 11px;
            font-size: 0.76rem;
            color: var(--text-secondary);
            font-weight: 500;
            font-family: var(--font-family) !important;
        }

        .status-pill.online {
            border-color: #bbf7d0;
            background: #f0fdf4;
            color: #15803d;
        }

        .status-pill.offline {
            border-color: #fecaca;
            background: #fef2f2;
            color: #b91c1c;
        }

        .pulse-dot {
            width: 6.5px;
            height: 6.5px;
            border-radius: 50%;
            background-color: currentColor;
            display: inline-block;
        }

        /* Sidebar Section Titles */
        section[data-testid="stSidebar"] .stMarkdown h3 {
            color: var(--text-muted) !important;
            font-size: 0.72rem !important;
            font-weight: 600 !important;
            margin-top: 1.1rem !important;
            margin-bottom: 0.35rem !important;
            text-transform: uppercase;
            letter-spacing: 0.06em !important;
            font-family: var(--font-family) !important;
        }

        /* Lucide icon alignment */
        .lucide-icon {
            display: inline-block;
            vertical-align: -2px;
            stroke: currentColor;
            flex-shrink: 0;
        }

        /* Chat Message Container Layout (15-16px Clean Readability) */
        div[data-testid="stChatMessage"] {
            background-color: #FFFFFF !important;
            border: 1px solid var(--border-subtle) !important;
            border-radius: var(--radius-lg) !important;
            padding: 1.15rem 1.4rem !important;
            margin-bottom: 0.95rem !important;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.02) !important;
            color: var(--text-primary) !important;
            line-height: 1.65 !important;
            font-size: 15.5px !important;
            font-family: var(--font-family) !important;
        }

        /* Subtle distinguish for user message container */
        div[data-testid="stChatMessage"]:has([aria-label="user message"]),
        div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
            background-color: var(--bg-bubble-user) !important;
            border: 1px solid var(--border-subtle) !important;
        }

        /* Code blocks within chat */
        div[data-testid="stChatMessage"] pre {
            background-color: #F9FAFB !important;
            border: 1px solid var(--border-subtle) !important;
            border-radius: var(--radius-sm) !important;
            padding: 0.9rem !important;
        }

        div[data-testid="stChatMessage"] code {
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace !important;
            font-size: 0.88rem !important;
        }

        /* Empty State / Welcome Screen */
        .empty-hero-container {
            text-align: center;
            padding: 3rem 1rem 1.6rem 1rem;
            max-width: 680px;
            margin: 0 auto;
        }

        .empty-hero-icon {
            width: 48px;
            height: 48px;
            background: var(--primary-accent-light);
            border: 1px solid var(--primary-accent-border);
            border-radius: 50%;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            color: var(--primary-accent);
            margin-bottom: 1rem;
        }

        .empty-hero-title {
            color: var(--text-primary);
            font-size: 1.6rem;
            font-weight: 600;
            letter-spacing: -0.02em;
            margin-bottom: 0.35rem;
            font-family: var(--font-family) !important;
        }

        .empty-hero-subtitle {
            color: var(--text-muted);
            font-size: 0.92rem;
            line-height: 1.5;
            margin: 0 auto 1.8rem auto;
            font-family: var(--font-family) !important;
        }

        .section-subhead {
            color: #9CA3AF;
            font-size: 0.72rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            text-align: center;
            margin-bottom: 0.85rem;
            font-family: var(--font-family) !important;
        }

        /* Streamlit Button Styling (Soft Minimal & Muted Teal Accent) */
        .stButton button {
            border-radius: var(--radius-sm) !important;
            font-weight: 500 !important;
            font-family: var(--font-family) !important;
            border: 1px solid var(--border-subtle) !important;
            background-color: #FFFFFF !important;
            color: var(--text-primary) !important;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03) !important;
            transition: all 0.15s ease-in-out !important;
        }

        .stButton button:hover {
            background-color: var(--primary-accent-light) !important;
            border-color: var(--primary-accent) !important;
            color: var(--primary-accent) !important;
        }

        .stButton button:active {
            background-color: rgba(71, 125, 120, 0.12) !important;
        }

        /* Prominent New Chat Button in Sidebar */
        .sidebar-new-chat-btn button {
            background-color: #FFFFFF !important;
            border: 1px solid var(--border-subtle) !important;
            color: var(--text-primary) !important;
            font-weight: 500 !important;
            padding: 0.55rem 1rem !important;
            font-family: var(--font-family) !important;
        }

        .sidebar-new-chat-btn button:hover {
            background-color: var(--primary-accent-light) !important;
            border-color: var(--primary-accent) !important;
            color: var(--primary-accent) !important;
            box-shadow: 0 2px 6px rgba(71, 125, 120, 0.1) !important;
        }

        /* Modern Rounded Chat Input (ChatGPT-Style with Muted Teal Focus) */
        div[data-testid="stChatInput"] {
            border-radius: var(--radius-pill) !important;
            border: 1px solid var(--border-subtle) !important;
            box-shadow: var(--shadow-card) !important;
            background-color: #FFFFFF !important;
            padding: 3px 8px !important;
            transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
        }

        div[data-testid="stChatInput"]:focus-within {
            border-color: var(--primary-accent) !important;
            box-shadow: 0 0 0 2px rgba(71, 125, 120, 0.15), var(--shadow-card) !important;
        }

        div[data-testid="stChatInput"] textarea {
            color: var(--text-primary) !important;
            font-size: 0.95rem !important;
            font-family: var(--font-family) !important;
            background-color: transparent !important;
        }

        div[data-testid="stChatInput"] button {
            color: var(--primary-accent) !important;
        }

        /* Sliders, Inputs & Selectbox Overrides */
        div[data-baseweb="input"] input, 
        div[data-baseweb="textarea"] textarea {
            background-color: #FFFFFF !important;
            border-color: var(--border-subtle) !important;
            color: var(--text-primary) !important;
            border-radius: var(--radius-sm) !important;
            font-family: var(--font-family) !important;
        }

        div[data-baseweb="input"]:focus-within,
        div[data-baseweb="textarea"]:focus-within {
            border-color: var(--primary-accent) !important;
            box-shadow: 0 0 0 2px rgba(71, 125, 120, 0.15) !important;
        }

        div[data-baseweb="select"] > div {
            background-color: #FFFFFF !important;
            border-color: var(--border-subtle) !important;
            border-radius: var(--radius-sm) !important;
            color: var(--text-primary) !important;
            font-family: var(--font-family) !important;
        }

        div[data-baseweb="select"] > div:focus-within {
            border-color: var(--primary-accent) !important;
        }

        /* Slider Thumb & Active Accents */
        div[data-testid="stSlider"] div[role="slider"] {
            background-color: var(--primary-accent) !important;
            border-color: var(--primary-accent) !important;
        }

        /* Toggle Active Accent */
        div[data-testid="stCheckbox"] input:checked + div,
        div[data-testid="stToggle"] input:checked + div {
            background-color: var(--primary-accent) !important;
        }

        /* Disclaimer footer at bottom of chat */
        .input-disclaimer {
            text-align: center;
            color: #A3A3A3;
            font-size: 0.73rem;
            margin-top: 0.5rem;
            margin-bottom: 0.2rem;
            font-family: var(--font-family) !important;
        }

        /* Sidebar Footer */
        .sidebar-footer {
            text-align: center;
            color: #A3A3A3;
            font-size: 0.72rem;
            margin-top: 1.8rem;
            padding-top: 0.8rem;
            border-top: 1px solid var(--border-subtle);
            font-family: var(--font-family) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# State Initialization & Callback Handlers
# ==============================================================================

def initialize_session_state() -> None:
    """Initialize all persistent session state variables safely."""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "api_key" not in st.session_state:
        st.session_state.api_key = os.environ.get("GROQ_API_KEY", "")

    if "api_key_valid" not in st.session_state:
        st.session_state.api_key_valid = None

    if "available_models" not in st.session_state:
        st.session_state.available_models = list(DEFAULT_ACTIVE_MODELS)

    if "selected_model" not in st.session_state:
        st.session_state.selected_model = DEFAULT_ACTIVE_MODELS[0]

    if "custom_model_id" not in st.session_state:
        st.session_state.custom_model_id = ""

    if "selected_role" not in st.session_state:
        st.session_state.selected_role = "General Assistant"

    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = ROLE_TEMPLATES["General Assistant"]

    if "temperature" not in st.session_state:
        st.session_state.temperature = DEFAULT_TEMPERATURE

    if "max_tokens" not in st.session_state:
        st.session_state.max_tokens = DEFAULT_MAX_TOKENS

    if "top_p" not in st.session_state:
        st.session_state.top_p = DEFAULT_TOP_P

    if "frequency_penalty" not in st.session_state:
        st.session_state.frequency_penalty = DEFAULT_FREQ_PENALTY

    if "presence_penalty" not in st.session_state:
        st.session_state.presence_penalty = DEFAULT_PRES_PENALTY

    if "enable_streaming" not in st.session_state:
        st.session_state.enable_streaming = True

    if "include_history" not in st.session_state:
        st.session_state.include_history = True

    if "chat_created_at" not in st.session_state:
        st.session_state.chat_created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = None


def reset_conversation() -> None:
    """Clear active chat messages and restart session timer."""
    st.session_state.messages = []
    st.session_state.chat_created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.pending_prompt = None


def clear_history() -> None:
    """Clear message history while preserving other settings."""
    st.session_state.messages = []
    st.session_state.pending_prompt = None


def reset_all_settings() -> None:
    """Reset all parameters and persona selections to factory defaults."""
    st.session_state.selected_model = st.session_state.available_models[0] if st.session_state.available_models else DEFAULT_ACTIVE_MODELS[0]
    st.session_state.custom_model_id = ""
    st.session_state.selected_role = "General Assistant"
    st.session_state.system_prompt = ROLE_TEMPLATES["General Assistant"]
    st.session_state.temperature = DEFAULT_TEMPERATURE
    st.session_state.max_tokens = DEFAULT_MAX_TOKENS
    st.session_state.top_p = DEFAULT_TOP_P
    st.session_state.frequency_penalty = DEFAULT_FREQ_PENALTY
    st.session_state.presence_penalty = DEFAULT_PRES_PENALTY
    st.session_state.enable_streaming = True
    st.session_state.include_history = True


def set_pending_prompt(prompt_text: str) -> None:
    """Set a quick-start prompt to be triggered on the next rerun."""
    st.session_state.pending_prompt = prompt_text


# ==============================================================================
# Groq Client & Dynamic Model Discovery Handlers
# ==============================================================================

def get_groq_client(api_key: str) -> Groq:
    """
    Instantiate and return the Groq SDK client.
    Guarantees that credentials are not exposed to logs or disk.
    """
    if not api_key or not api_key.strip():
        raise ValueError("Groq API Key is missing. Please provide a key in the sidebar.")
    if Groq is None:
        raise ImportError(
            "The official 'groq' package is not installed. Run: pip install -r requirements.txt"
        )
    return Groq(api_key=api_key.strip())


def fetch_live_models_from_groq(api_key: str) -> Tuple[bool, List[str], str]:
    """
    Query Groq's official GET /models endpoint to fetch live active models.
    Filters out non-chat models (audio, whisper, moderation) and sorts production first.
    Returns (success, list_of_model_ids, status_message).
    """
    if not api_key or not api_key.strip():
        return False, list(DEFAULT_ACTIVE_MODELS), "API key cannot be empty."

    try:
        client = get_groq_client(api_key)
        response = client.models.list()
        raw_models = getattr(response, "data", [])

        discovered_models: List[str] = []
        for m in raw_models:
            m_id = getattr(m, "id", "")
            if not m_id:
                continue
            lower_id = m_id.lower()
            # Filter out non-chat models
            if any(pat in lower_id for pat in EXCLUDED_MODEL_PATTERNS):
                continue
            # Filter out known decommissioned models
            if any(dec in lower_id for dec in ["gemma2", "llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b"]):
                continue
            # Check active status if present
            is_active = getattr(m, "active", True)
            if is_active and m_id not in discovered_models:
                discovered_models.append(m_id)

        # Priority sorter: production Llama & GPT-OSS at the top
        def model_rank(mid: str) -> Tuple[int, str]:
            if "llama-3.3-70b" in mid:
                return (0, mid)
            if "llama-3.1-8b" in mid:
                return (1, mid)
            if "gpt-oss-120b" in mid:
                return (2, mid)
            if "gpt-oss-20b" in mid:
                return (3, mid)
            if "qwen" in mid:
                return (4, mid)
            return (5, mid)

        if discovered_models:
            discovered_models.sort(key=model_rank)
            return True, discovered_models, f"Successfully loaded {len(discovered_models)} active models from GroqCloud."

        return True, list(DEFAULT_ACTIVE_MODELS), "Connected. Loaded verified production models."

    except AuthenticationError:
        return False, list(DEFAULT_ACTIVE_MODELS), "Authentication failed. Invalid or expired Groq API key."
    except APIConnectionError:
        return False, list(DEFAULT_ACTIVE_MODELS), "Connection error: Unable to reach Groq servers. Check your network."
    except RateLimitError:
        return False, list(DEFAULT_ACTIVE_MODELS), "Rate limit reached. Please wait a moment before retrying."
    except Exception as e:
        return False, list(DEFAULT_ACTIVE_MODELS), f"Error connecting to Groq: {str(e)}"


def get_effective_model_id() -> str:
    """Return the final model identifier to send to Groq API."""
    if st.session_state.selected_model == "Custom Model...":
        custom = st.session_state.custom_model_id.strip()
        return custom if custom else "llama-3.3-70b-versatile"
    return st.session_state.selected_model


def prepare_messages_payload(
    user_query: str,
    system_prompt: str,
    history: List[Dict[str, Any]],
    include_history: bool,
) -> List[Dict[str, str]]:
    """
    Construct the messages array for the Groq Chat Completion API.
    Injects system prompt, conversation history (if enabled), and current prompt.
    """
    payload: List[Dict[str, str]] = []

    if system_prompt and system_prompt.strip():
        payload.append({"role": "system", "content": system_prompt.strip()})

    if include_history:
        for turn in history:
            role = turn.get("role")
            content = turn.get("content", "")
            if role in ("user", "assistant") and content:
                payload.append({"role": role, "content": content})

    payload.append({"role": "user", "content": user_query})
    return payload


def stream_groq_completion(
    client: Groq,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    top_p: float,
    frequency_penalty: float,
    presence_penalty: float,
) -> Generator[str, None, None]:
    """Generator yielding streamed tokens from Groq API in real-time."""
    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "top_p": float(top_p),
        "stream": True,
    }

    if frequency_penalty != 0.0:
        kwargs["frequency_penalty"] = float(frequency_penalty)
    if presence_penalty != 0.0:
        kwargs["presence_penalty"] = float(presence_penalty)

    completion_stream = client.chat.completions.create(**kwargs)
    for chunk in completion_stream:
        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


# ==============================================================================
# Export Capabilities
# ==============================================================================

def export_conversation_txt() -> str:
    """Generate a clean formatted plain-text transcript."""
    lines = [
        "=" * 60,
        "GROQ AI CHATBOT TRANSCRIPT",
        f"Export Timestamp : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Session Started  : {st.session_state.get('chat_created_at', 'N/A')}",
        f"Model Deployed   : {get_effective_model_id()}",
        f"Role Archetype   : {st.session_state.get('selected_role', 'Custom')}",
        "=" * 60,
        "",
        f"[SYSTEM PROMPT]\n{st.session_state.get('system_prompt', '').strip()}\n",
        "-" * 60,
    ]

    for i, msg in enumerate(st.session_state.messages, 1):
        role_label = "USER" if msg["role"] == "user" else "ASSISTANT"
        ts = msg.get("timestamp", "")
        ts_part = f" ({ts})" if ts else ""
        lines.append(f"\n[{i}] {role_label}{ts_part}\n{msg['content']}\n")

    lines.append("-" * 60)
    lines.append("End of Conversation")
    return "\n".join(lines)


def export_conversation_json() -> str:
    """Generate structured JSON containing full conversation and session parameters."""
    export_obj = {
        "metadata": {
            "application": "Groq Streamlit AI Chatbot",
            "exported_at": datetime.now().isoformat(),
            "session_created_at": st.session_state.get("chat_created_at"),
            "model": get_effective_model_id(),
            "role": st.session_state.get("selected_role"),
            "system_prompt": st.session_state.get("system_prompt"),
            "parameters": {
                "temperature": st.session_state.get("temperature"),
                "max_tokens": st.session_state.get("max_tokens"),
                "top_p": st.session_state.get("top_p"),
                "frequency_penalty": st.session_state.get("frequency_penalty"),
                "presence_penalty": st.session_state.get("presence_penalty"),
                "streaming": st.session_state.get("enable_streaming"),
                "history_enabled": st.session_state.get("include_history"),
            },
            "message_count": len(st.session_state.messages),
        },
        "messages": st.session_state.messages,
    }
    return json.dumps(export_obj, indent=2, ensure_ascii=False)


# ==============================================================================
# UI Component: Left Sidebar Configuration Panel (Soft Minimal Light)
# ==============================================================================

def render_sidebar() -> None:
    """Render comprehensive configuration controls in the left sidebar."""
    with st.sidebar:
        # 1. Prominent Soft Minimal New Chat button at top of sidebar
        st.markdown('<div class="sidebar-new-chat-btn">', unsafe_allow_html=True)
        st.button("+ New Chat", on_click=reset_conversation, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("### Authentication & Key")

        # API Key Field (Masked & Protected)
        st.text_input(
            "Groq API Key",
            type="password",
            key="api_key",
            help="Stored securely in memory. Never exposed, written to disk, or logged.",
            placeholder="gsk_...",
        )

        col_val, col_link = st.columns([1.3, 1])
        with col_val:
            if st.button("Test Connection", use_container_width=True):
                if not st.session_state.api_key:
                    st.warning("Please enter your Groq API key first.")
                else:
                    with st.spinner("Connecting to Groq..."):
                        valid, live_models, feedback = fetch_live_models_from_groq(st.session_state.api_key)
                        st.session_state.api_key_valid = valid
                        if valid:
                            st.session_state.available_models = live_models
                            if st.session_state.selected_model not in live_models:
                                st.session_state.selected_model = live_models[0]
                            st.success(feedback)
                        else:
                            st.error(feedback)

        with col_link:
            st.markdown(
                '<a href="https://console.groq.com/keys" target="_blank" style="text-decoration:none;">'
                '<button style="width:100%; border:1px solid #E5E7EB; background:#FFFFFF; color:#262626; '
                'border-radius:8px; padding:0.42rem 0.2rem; font-size:0.8rem; cursor:pointer; font-weight:500;'
                'font-family:Inter,sans-serif; transition:all 0.15s ease;">Get API Key</button></a>',
                unsafe_allow_html=True,
            )

        # Connection status notification
        if st.session_state.api_key_valid is True:
            st.markdown(
                '<div style="color:#16a34a; font-size:0.78rem; font-weight:500; margin-top:2px;">Connected to GroqCloud</div>',
                unsafe_allow_html=True,
            )
        elif st.session_state.api_key_valid is False:
            st.markdown(
                '<div style="color:#dc2626; font-size:0.78rem; font-weight:500; margin-top:2px;">Disconnected / Invalid Key</div>',
                unsafe_allow_html=True,
            )

        st.divider()

        # 2. Dynamic Model Selection
        st.markdown("### Model Selection")
        
        model_options = list(st.session_state.available_models) + ["Custom Model..."]
        
        if st.session_state.selected_model not in model_options:
            st.session_state.selected_model = model_options[0]

        col_m_sel, col_m_sync = st.columns([3.8, 1.2])
        with col_m_sel:
            st.selectbox(
                "Active Models",
                options=model_options,
                key="selected_model",
                help="Select an active production or preview model hosted on GroqCloud LPUs.",
                label_visibility="collapsed",
            )
        with col_m_sync:
            if st.button("Sync", help="Sync active models from your Groq account", use_container_width=True):
                if st.session_state.api_key:
                    with st.spinner("Syncing..."):
                        valid, live_models, feedback = fetch_live_models_from_groq(st.session_state.api_key)
                        if valid:
                            st.session_state.available_models = live_models
                            st.session_state.api_key_valid = True
                            st.toast(feedback)
                            st.rerun()
                        else:
                            st.toast(feedback)
                else:
                    st.toast("Enter API Key to sync live models")

        if st.session_state.selected_model == "Custom Model...":
            st.text_input(
                "Enter Custom Model ID",
                key="custom_model_id",
                placeholder="e.g. llama-3.3-70b-versatile",
                help="Enter any valid model endpoint supported on your Groq tier.",
            )

        st.divider()

        # 3. System Persona Templates
        st.markdown("### Persona & System Role")

        def handle_role_selection() -> None:
            """Synchronize system prompt when predefined template changes."""
            selected = st.session_state.selected_role
            if selected in ROLE_TEMPLATES and selected != "Custom Role":
                st.session_state.system_prompt = ROLE_TEMPLATES[selected]

        st.selectbox(
            "Predefined Template",
            options=list(ROLE_TEMPLATES.keys()),
            key="selected_role",
            on_change=handle_role_selection,
            help="Predefined domain personas automatically adapt the system instructions.",
            label_visibility="collapsed",
        )

        st.text_area(
            "System Prompt",
            key="system_prompt",
            height=110,
            help="Customize these instructions to adjust tone, domain specialization, or response rules.",
        )

        st.divider()

        # 4. Hyperparameter Tuning
        st.markdown("### Model Parameters")

        c1, c2 = st.columns(2)
        with c1:
            st.slider(
                "Temperature",
                min_value=0.0,
                max_value=2.0,
                step=0.05,
                key="temperature",
                help="0.0 is deterministic and focused; higher values increase variety and creativity.",
            )
        with c2:
            st.slider(
                "Top-P",
                min_value=0.0,
                max_value=1.0,
                step=0.05,
                key="top_p",
                help="Nucleus sampling probability threshold.",
            )

        st.slider(
            "Max Response Tokens",
            min_value=128,
            max_value=8192,
            step=128,
            key="max_tokens",
            help="Maximum upper bound of tokens generated per turn.",
        )

        with st.expander("Advanced Penalties", expanded=False):
            st.slider(
                "Frequency Penalty",
                min_value=-2.0,
                max_value=2.0,
                step=0.1,
                key="frequency_penalty",
                help="Positive values penalize tokens based on their frequency so far.",
            )
            st.slider(
                "Presence Penalty",
                min_value=-2.0,
                max_value=2.0,
                step=0.1,
                key="presence_penalty",
                help="Positive values penalize tokens based on whether they appeared so far.",
            )

        st.divider()

        # 5. Session Controls & Toggles
        st.markdown("### Chat Preferences")

        st.toggle(
            "Enable Streaming",
            key="enable_streaming",
            help="Stream tokens to screen dynamically as they are produced.",
        )

        st.toggle(
            "Include Conversation History",
            key="include_history",
            help="Pass previous conversation turns for multi-turn context awareness.",
        )

        col_clear, col_reset = st.columns(2)
        with col_clear:
            st.button("Clear Chat", on_click=clear_history, use_container_width=True)
        with col_reset:
            st.button("Reset Settings", on_click=reset_all_settings, use_container_width=True)

        st.divider()

        # 6. Conversation Export
        st.markdown("### Export Conversation")
        if st.session_state.messages:
            now_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            col_exp_txt, col_exp_json = st.columns(2)
            with col_exp_txt:
                st.download_button(
                    label="Download TXT",
                    data=export_conversation_txt(),
                    file_name=f"groq_chat_{now_stamp}.txt",
                    mime="text/plain",
                    use_container_width=True,
                )
            with col_exp_json:
                st.download_button(
                    label="Download JSON",
                    data=export_conversation_json(),
                    file_name=f"groq_chat_{now_stamp}.json",
                    mime="application/json",
                    use_container_width=True,
                )
        else:
            st.caption("No messages to export yet.")

        st.markdown(
            '<div class="sidebar-footer">Powered by GroqCloud LPUs & Streamlit</div>',
            unsafe_allow_html=True,
        )


# ==============================================================================
# UI Component: Main Screen Chat & Header (Soft Minimal Light)
# ==============================================================================

def render_header() -> None:
    """Render top clean status bar with active model, persona, and message metrics."""
    effective_model = get_effective_model_id()
    msg_count = len(st.session_state.messages)
    is_valid = st.session_state.api_key_valid is True

    status_class = "online" if is_valid else "offline"
    status_text = "API Ready" if is_valid else ("Key Entered" if st.session_state.api_key else "No Key")

    st.markdown(
        f"""
        <div class="chat-header-container">
            <div class="header-title-box">
                <div class="header-icon-badge">
                    {ICON_SPARK}
                </div>
                <div class="header-text">
                    <h1>Groq Assistant</h1>
                    <p>Ultra-Fast Open Model Interface</p>
                </div>
            </div>
            <div class="header-badges">
                <div class="status-pill {status_class}">
                    <span class="pulse-dot"></span> {status_text}
                </div>
                <div class="status-pill">
                    {ICON_CPU} {effective_model}
                </div>
                <div class="status-pill">
                    {ICON_USER} {st.session_state.selected_role}
                </div>
                <div class="status-pill">
                    {ICON_MESSAGE} {msg_count} turn{'s' if msg_count != 1 else ''}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_state() -> None:
    """Display minimalist Soft Minimal welcome hero with prompt suggestion cards."""
    st.markdown(
        f"""
        <div class="empty-hero-container">
            <div class="empty-hero-icon">
                {ICON_BOT_HERO}
            </div>
            <div class="empty-hero-title">What can I help with today?</div>
            <div class="empty-hero-subtitle">
                Experience ultra-low latency inference powered by Groq LPUs.
                Select a starter prompt below or enter your query.
            </div>
            <div class="section-subhead">Suggested Prompts</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    suggestions = [
        ("Python Mentor", "Explain Python's asyncio event loop with a concise, real-world example."),
        ("Machine Learning", "Compare Self-Attention vs Mamba State-Space Models in complexity."),
        ("Robotics Specialist", "How does Extended Kalman Filter (EKF) sensor fusion work in robotics?"),
        ("Data Analyst", "Write an optimized SQL window function to calculate monthly cohort retention."),
    ]

    col1, col2 = st.columns(2)
    for i, (title, prompt_text) in enumerate(suggestions):
        target_col = col1 if i % 2 == 0 else col2
        with target_col:
            st.button(
                f"**{title}**\n\n_{prompt_text}_",
                key=f"quick_prompt_{i}",
                on_click=set_pending_prompt,
                args=(prompt_text,),
                use_container_width=True,
            )


def render_message_history() -> None:
    """Display past turns with clean monochromatic styling."""
    for message in st.session_state.messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        with st.chat_message(role, avatar=None):
            st.markdown(content)


def handle_user_submission(user_prompt: str) -> None:
    """
    Process incoming query: validates credentials, queries Groq API,
    handles streaming/non-streaming, catches decommission errors, and updates session history.
    """
    api_key = st.session_state.api_key.strip()

    if not api_key:
        st.error(
            "Groq API Key Required: Please enter your Groq API key in the left sidebar to start chatting. "
            "You can acquire an API key at console.groq.com/keys."
        )
        return

    # Record user message in history
    now_str = datetime.now().strftime("%H:%M:%S")
    st.session_state.messages.append({
        "role": "user",
        "content": user_prompt,
        "timestamp": now_str,
    })

    # Render user turn
    with st.chat_message("user", avatar=None):
        st.markdown(user_prompt)

    # Initialize client
    try:
        client = get_groq_client(api_key)
    except Exception as e:
        st.error(f"Failed to initialize Groq client: {str(e)}")
        return

    # Build context payload
    effective_model = get_effective_model_id()
    messages_payload = prepare_messages_payload(
        user_query=user_prompt,
        system_prompt=st.session_state.system_prompt,
        history=st.session_state.messages[:-1],
        include_history=st.session_state.include_history,
    )

    # Synthesize & display assistant response
    with st.chat_message("assistant", avatar=None):
        response_text = ""
        try:
            if st.session_state.enable_streaming:
                stream_generator = stream_groq_completion(
                    client=client,
                    model=effective_model,
                    messages=messages_payload,
                    temperature=st.session_state.temperature,
                    max_tokens=st.session_state.max_tokens,
                    top_p=st.session_state.top_p,
                    frequency_penalty=st.session_state.frequency_penalty,
                    presence_penalty=st.session_state.presence_penalty,
                )
                response_text = st.write_stream(stream_generator)
            else:
                with st.spinner("Generating response..."):
                    kwargs: Dict[str, Any] = {
                        "model": effective_model,
                        "messages": messages_payload,
                        "temperature": float(st.session_state.temperature),
                        "max_tokens": int(st.session_state.max_tokens),
                        "top_p": float(st.session_state.top_p),
                        "stream": False,
                    }
                    if st.session_state.frequency_penalty != 0.0:
                        kwargs["frequency_penalty"] = float(st.session_state.frequency_penalty)
                    if st.session_state.presence_penalty != 0.0:
                        kwargs["presence_penalty"] = float(st.session_state.presence_penalty)

                    completion = client.chat.completions.create(**kwargs)
                    response_text = completion.choices[0].message.content or ""
                    st.markdown(response_text)

            # Persist assistant response
            if response_text:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_text,
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "model": effective_model,
                })

        except BadRequestError as e:
            err_msg = str(e)
            if "model_decommissioned" in err_msg or "decommissioned" in err_msg.lower():
                fallback_model = "llama-3.3-70b-versatile"
                st.error(
                    f"Model Decommissioned by Groq: The model `{effective_model}` is no longer supported on GroqCloud. "
                    f"Automatically switched default to `{fallback_model}` (Production Llama 3.3 70B). "
                    f"Click 'Sync' in the sidebar to refresh active models."
                )
                if effective_model in st.session_state.available_models:
                    st.session_state.available_models.remove(effective_model)
                st.session_state.selected_model = fallback_model
            else:
                st.error(f"Bad Request Error: {err_msg}")

        except AuthenticationError:
            st.error(
                "Authentication Failed: Invalid or expired Groq API key. "
                "Please verify your credentials in the sidebar."
            )
            st.session_state.api_key_valid = False
        except RateLimitError:
            st.error(
                "Rate Limit Exceeded: You have hit the Groq API rate limit. "
                "Please wait briefly before sending another message."
            )
        except APIConnectionError:
            st.error(
                "Connection Error: Unable to reach Groq API. "
                "Please check your internet connection or proxy settings."
            )
        except GroqError as e:
            st.error(f"Groq Error: {str(e)}")
        except Exception as e:
            st.error(f"Unexpected Error: {str(e)}")


# ==============================================================================
# Application Mainline
# ==============================================================================

def main() -> None:
    """Application main execution path."""
    st.set_page_config(
        page_title="Groq AI Chatbot",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Initialize appearance and state
    inject_custom_styles()
    initialize_session_state()

    # Render UI layout
    render_sidebar()
    render_header()

    # Render main chat area or empty state
    if not st.session_state.messages and not st.session_state.pending_prompt:
        render_empty_state()
    else:
        render_message_history()

    # Determine incoming query (either from suggestion chip or chat_input)
    incoming_query = None
    if st.session_state.pending_prompt:
        incoming_query = st.session_state.pending_prompt
        st.session_state.pending_prompt = None

    user_input = st.chat_input("Message Groq Assistant...")
    if user_input:
        incoming_query = user_input

    # Small disclaimer under input
    st.markdown(
        '<div class="input-disclaimer">Groq Assistant can make mistakes. Check important info.</div>',
        unsafe_allow_html=True,
    )

    # Process query if present
    if incoming_query:
        handle_user_submission(incoming_query)


if __name__ == "__main__":
    main()
