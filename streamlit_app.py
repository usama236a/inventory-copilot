import json
from pathlib import Path
import pandas as pd
import streamlit as st

from app.agent import ask
from app import llm, tools

# Minimalist Monochrome Configuration
st.set_page_config(
    page_title="Inventory Copilot",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Minimalist Black & White CSS
st.markdown(
    """
    <style>
    /* Global Background & Font */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #000000 !important;
        color: #FFFFFF !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
    }

    /* Top Header & Toolbar */
    [data-testid="stHeader"] {
        background-color: #000000 !important;
        color: #FFFFFF !important;
    }

    /* Left Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #080808 !important;
        border-right: 1px solid #222222 !important;
    }
    [data-testid="stSidebar"] * {
        color: #FFFFFF !important;
    }

    /* Huge Title in Left Section */
    .huge-sidebar-title {
        font-size: 2.8rem !important;
        font-weight: 900 !important;
        line-height: 0.95 !important;
        letter-spacing: -0.05em !important;
        color: #FFFFFF !important;
        text-transform: uppercase !important;
        margin-top: 0.5rem !important;
        margin-bottom: 1.5rem !important;
        border-bottom: 2px solid #FFFFFF !important;
        padding-bottom: 1rem !important;
    }

    .main-huge-title {
        font-size: 3.2rem !important;
        font-weight: 900 !important;
        line-height: 1.0 !important;
        letter-spacing: -0.05em !important;
        color: #FFFFFF !important;
        text-transform: uppercase !important;
        margin-bottom: 0.25rem !important;
    }

    .sub-caption {
        font-size: 0.95rem !important;
        letter-spacing: 0.05em !important;
        color: #888888 !important;
        text-transform: uppercase !important;
        margin-bottom: 1.5rem !important;
    }

    /* Chat Messages */
    [data-testid="stChatMessage"] {
        background-color: #0A0A0A !important;
        border: 1px solid #222222 !important;
        border-radius: 4px !important;
        color: #FFFFFF !important;
        margin-bottom: 0.8rem !important;
    }

    /* Input Field */
    [data-testid="stChatInput"] textarea {
        background-color: #0A0A0A !important;
        color: #FFFFFF !important;
        border: 1px solid #333333 !important;
        border-radius: 4px !important;
    }
    [data-testid="stChatInput"] textarea:focus {
        border-color: #FFFFFF !important;
    }

    /* Minimalist Buttons */
    .stButton > button {
        background-color: #000000 !important;
        color: #FFFFFF !important;
        border: 1px solid #333333 !important;
        border-radius: 3px !important;
        font-weight: 500 !important;
        transition: all 0.15s ease-in-out !important;
        text-align: left !important;
    }
    .stButton > button:hover {
        background-color: #FFFFFF !important;
        color: #000000 !important;
        border-color: #FFFFFF !important;
    }

    /* Expanders & Cards */
    [data-testid="stExpander"] {
        background-color: #080808 !important;
        border: 1px solid #222222 !important;
        border-radius: 4px !important;
    }

    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }
    [data-testid="stMetricLabel"] {
        color: #888888 !important;
        text-transform: uppercase !important;
        font-size: 0.75rem !important;
        letter-spacing: 0.05em !important;
    }

    /* Dividers */
    hr {
        border-color: #222222 !important;
    }

    /* Code blocks */
    code, pre {
        background-color: #0A0A0A !important;
        color: #E5E5E5 !important;
        border: 1px solid #222222 !important;
    }

    /* Hide unnecessary decorations */
    #MainMenu, footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True
)

# --- Left Section (Sidebar) ---
with st.sidebar:
    # Huge Text in the top left section
    st.markdown('<div class="huge-sidebar-title">INVENTORY<br>COPILOT</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-caption">LOCAL AGENTIC SYSTEM</div>', unsafe_allow_html=True)

    # Status Indicator
    is_ready, status_msg = llm.check_status()
    if is_ready:
        st.markdown(f"**STATUS:** `ONLINE` — {status_msg}")
    else:
        st.markdown(f"**STATUS:** `OFFLINE`")
        st.caption(status_msg)
        with st.expander("START OLLAMA"):
            st.code("ollama serve\nollama pull llama3.2:3b", language="bash")

    st.markdown("---")

    # Inventory Metrics
    st.markdown("**DATABASE OVERVIEW**")
    m1, m2 = st.columns(2)
    m1.metric("Products", "32")
    m1.metric("Suppliers", "4")
    m2.metric("Categories", "4")
    m2.metric("Sales Records", "11,500+")

    st.markdown("---")

    # Quick Prompts
    st.markdown("**EXAMPLE QUERIES**")
    sample_queries = [
        ("SQL", "Which products are below their reorder level?"),
        ("SQL", "Which 3 products earned the most revenue overall?"),
        ("FORECAST", "Will Earbuds run out in the next 2 weeks?"),
        ("FORECAST", "Forecast demand for Green Tea for the next 30 days."),
        ("POLICY", "What is the return window for damaged goods?"),
        ("POLICY", "Who has to approve emergency purchase orders?"),
    ]

    for tool_tag, query_text in sample_queries:
        if st.button(f"[{tool_tag}] {query_text}", key=f"btn_{query_text}"):
            st.session_state.selected_prompt = query_text

    st.markdown("---")
    if st.button("CLEAR CONVERSATION", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --- Main Workspace ---
st.markdown('<div class="main-huge-title">INVENTORY COPILOT</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-caption">AUTONOMOUS MULTI-TOOL DECISION ENGINE & FORECASTING</div>', unsafe_allow_html=True)

# Message History
if "messages" not in st.session_state:
    st.session_state.messages = []


def render_evidence(msg_data: dict):
    tool = msg_data.get("tool")
    arg = msg_data.get("arg")
    result = msg_data.get("result")

    if not tool:
        return

    with st.expander(f"EVIDENCE LOG // [{tool.upper()}]"):
        st.markdown(f"**TOOL:** `{tool.upper()}`")
        if tool == "sql":
            st.markdown("**QUERY:**")
            st.code(arg if isinstance(arg, str) else json.dumps(arg), language="sql")
            if isinstance(result, list) and len(result) > 0:
                st.markdown(f"**RESULTS ({len(result)} ROWS):**")
                st.dataframe(pd.DataFrame(result), use_container_width=True)
            elif isinstance(result, dict) and "error" in result:
                st.error(result["error"])
            else:
                st.info("0 rows returned.")

        elif tool == "forecast":
            st.markdown("**PARAMETERS:**")
            st.code(json.dumps(arg, indent=2) if isinstance(arg, dict) else str(arg), language="json")
            if isinstance(result, dict) and "holdout_mae" in result:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("STOCK ON HAND", result.get("stock"))
                c2.metric("LEAD TIME", f"{result.get('lead_days')} d")
                fc_key = [k for k in result.keys() if k.startswith("forecast_units")][0]
                c3.metric("DEMAND FORECAST", f"{result.get(fc_key)} u")
                c4.metric(
                    "REORDER TRIGGER",
                    "ACTION REQUIRED" if result.get("reorder_now") else "HEALTHY"
                )
                st.caption(
                    f"LightGBM Holdout MAE: {result.get('holdout_mae')} | Naive 28d MAE: {result.get('naive_mae')}"
                )
            else:
                st.json(result)

        elif tool == "docs":
            st.markdown(f"**SEARCH STRING:** `{arg}`")
            if isinstance(result, list):
                st.markdown("**MATCHED POLICY FRAGMENTS:**")
                for item in result:
                    st.markdown(f"**SOURCE:** `{item.get('source')}` | **COSINE SCORE:** `{item.get('score')}`")
                    st.code(item.get("text"), language="markdown")
            else:
                st.json(result)


# Render Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and msg.get("tool"):
            render_evidence(msg)

# Input Prompt Handling
user_prompt = st.chat_input("Enter inquiry...")
if getattr(st.session_state, "selected_prompt", None):
    user_prompt = st.session_state.selected_prompt
    st.session_state.selected_prompt = None

if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.write(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("PROCESSING..."):
            response = ask(user_prompt)

        answer_text = response["answer"]
        st.write(answer_text)
        render_evidence(response)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer_text,
            "tool": response.get("tool"),
            "arg": response.get("arg"),
            "result": response.get("result")
        })
