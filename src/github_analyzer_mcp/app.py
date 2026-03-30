"""
app.py — Streamlit UI for LangGraph Multi-Agent Research Engine
"""

import streamlit as st
import os
from engine import run_research, ResearchState

st.set_page_config(page_title="Multi-Agent Research Engine", page_icon="🔬", layout="wide")

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  .main { background: #07080d; }
  .hero { background: linear-gradient(135deg,#07080d 0%,#0d0a1a 50%,#07080d 100%);
          border:1px solid #1a1a30; border-radius:16px; padding:32px 40px;
          text-align:center; margin-bottom:24px; }
  .hero h1 { font-size:36px; font-weight:700; color:#fff; margin:0 0 6px; }
  .hero p  { color:#64748b; font-size:14px; margin:0; }
  .agent-box { background:#080a14; border:1px solid #1a1a30; border-radius:10px;
               padding:14px 18px; margin:8px 0; font-size:13px; }
  .report-box { background:#080a14; border:1px solid #1a1a30; border-left:4px solid #7c3aed;
                border-radius:0 12px 12px 0; padding:20px 24px; line-height:1.85; color:#cbd5e1; }
  div.stButton > button { background:linear-gradient(135deg,#4c1d95,#7c3aed); color:white;
                          font-weight:700; border:none; border-radius:10px; padding:12px 28px;
                          font-size:15px; width:100%; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>🔬 Multi-Agent Research Engine</h1>
  <p>Supervisor → Researcher → Synthesizer → Fact-Checker → Writer · Powered by LangGraph</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 🔑 API Keys")
    openai_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
    tavily_key = st.text_input("Tavily API Key (optional)", type="password", placeholder="tvly-...")
    st.info("Tavily enables real web search. Without it, the engine uses GPT knowledge.")
    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    max_iter = st.slider("Research depth (iterations)", 2, 6, 3)
    st.markdown("---")
    st.markdown("### 🤖 Agent Pipeline")
    st.markdown("""
1. **Supervisor** — routes tasks
2. **Researcher** — gathers info
3. **Synthesizer** — organizes findings
4. **Fact-Checker** — validates claims
5. **Writer** — final report
    """)

topic = st.text_input("Research topic", placeholder="e.g. What are the latest advances in LLM reasoning and chain-of-thought prompting?")
run_clicked = st.button("🚀 Start Research")

if run_clicked:
    if not topic.strip():
        st.warning("Please enter a research topic.")
    elif not openai_key:
        st.error("Please add your OpenAI API key in the sidebar.")
    else:
        os.environ["OPENAI_API_KEY"] = openai_key
        if tavily_key:
            os.environ["TAVILY_API_KEY"] = tavily_key

        progress_container = st.empty()
        log_container      = st.empty()
        agent_logs = []

        with st.spinner("🧠 Research agents working..."):
            try:
                result = run_research(topic, max_iterations=max_iter, verbose=False)

                # Show agent messages
                st.markdown("### 📋 Agent Activity Log")
                for msg in result.get("messages", []):
                    if hasattr(msg, "content") and msg.content.startswith("["):
                        st.markdown(f'<div class="agent-box">{msg.content}</div>', unsafe_allow_html=True)

                st.markdown("---")
                st.markdown("### 📄 Research Report")
                report = result.get("final_report", "No report generated.")
                st.markdown(f'<div class="report-box">{report}</div>', unsafe_allow_html=True)

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Facts Gathered", len(result.get("verified_facts", [])))
                with col2:
                    st.metric("Contradictions Found", len(result.get("contradictions", [])))

                st.download_button(
                    "⬇️ Download Report (.txt)",
                    data=report,
                    file_name=f"research_{topic[:30].replace(' ','_')}.txt",
                    mime="text/plain",
                )

            except Exception as e:
                st.error(f"Research failed: {str(e)}")
