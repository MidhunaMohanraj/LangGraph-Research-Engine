# 🔬 LangGraph Multi-Agent Research Engine

<div align="center">

![Banner](https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=10,14,20&height=180&section=header&text=Multi-Agent%20Research%20Engine&fontSize=38&fontColor=fff&animation=twinkling&fontAlignY=38&desc=LangGraph%20%E2%80%A2%20Supervisor-Worker%20Pattern%20%E2%80%A2%205%20Specialized%20Agents%20%E2%80%A2%20PyPI%20Package&descAlignY=58&descSize=13)

<p>
  <img src="https://img.shields.io/badge/LangGraph-0.2%2B-1C3C3C?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/LangChain-0.3%2B-1C3C3C?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/PyPI-langgraph--research--engine-3776AB?style=for-the-badge&logo=pypi&logoColor=white"/>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/OpenAI-GPT--4o-412991?style=for-the-badge&logo=openai&logoColor=white"/>
  <img src="https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge"/>
</p>

<p>
  <b>An autonomous multi-agent research system that orchestrates 5 specialized AI agents to perform deep, multi-step research on any topic — producing verified, fact-checked reports.</b>
</p>

</div>

---

## 🌟 What This Does

Ask it any complex research question. The agents collaborate to produce a professional report:

```
Input:  "What are the latest advances in LLM reasoning and chain-of-thought?"

Output: Full research report with:
  ✅ Executive summary
  ✅ Key findings with confidence scores
  ✅ Detailed analysis
  ✅ Fact-checked claims
  ✅ Contradictions identified
  ✅ Conclusions and recommendations
```

---

## 🤖 Agent Architecture

```
┌─────────────────────────────────────────────────────────┐
│  SUPERVISOR (routes tasks, decides when done)           │
└──────┬──────────────────────────────────────────────────┘
       │
       ├──► RESEARCHER   — web search + info gathering
       │
       ├──► SYNTHESIZER  — organizes findings into themes
       │
       ├──► FACT-CHECKER — validates claims, finds contradictions
       │
       └──► WRITER       — produces final structured report
```

**Pattern:** Supervisor-Worker with a stateful `StateGraph` (LangGraph)

Each agent reads shared state, performs its task, and returns to the Supervisor which decides the next step. The loop continues until the final report is complete.

---

## 📦 Installation

```bash
pip install langgraph-research-engine
```

Or from source:
```bash
git clone https://github.com/MidhunaMohanraj/langgraph-research-engine
cd langgraph-research-engine
pip install -e .
```

---

## 🚀 Quick Start

### Python API

```python
from langgraph_research_engine import run_research

result = run_research(
    topic="What are the key challenges in deploying LLMs in production?",
    max_iterations=4,
    verbose=True,
)

print(result["final_report"])
print(f"Facts gathered: {len(result['verified_facts'])}")
print(f"Contradictions: {result['contradictions']}")
```

### CLI

```bash
# Set your API keys
export OPENAI_API_KEY=sk-...
export TAVILY_API_KEY=tvly-...  # optional, enables real web search

# Run research
langgraph-research "Latest advances in multimodal AI systems"
```

### Streamlit UI

```bash
cd src/langgraph_research_engine
streamlit run app.py
```

---

## ⚙️ Configuration

### Required
```bash
export OPENAI_API_KEY=sk-...        # Get at platform.openai.com
```

### Optional (enables real web search)
```bash
export TAVILY_API_KEY=tvly-...      # Free tier at tavily.com
```

Without Tavily, the engine uses GPT knowledge which may be outdated for recent events.

---

## 🧠 State Schema

The agents share a typed state object:

```python
class ResearchState(TypedDict):
    messages:        list[BaseMessage]   # full conversation history
    research_topic:  str                 # the research question
    raw_findings:    list[str]           # gathered information
    verified_facts:  list[str]           # extracted key facts
    contradictions:  list[str]           # conflicting claims found
    final_report:    str                 # the finished report
    next_agent:      str                 # supervisor's routing decision
    iteration:       int                 # current cycle count
    max_iterations:  int                 # stop condition
```

---

## 🔧 Extend It

Add your own agent in 3 steps:

```python
# 1. Define the agent function
def my_custom_agent(state: ResearchState) -> dict:
    llm = get_llm()
    response = llm.invoke([HumanMessage(content=f"Do X with: {state['research_topic']}")])
    return {"messages": [AIMessage(content=response.content)]}

# 2. Add to the graph
graph.add_node("my_agent", my_custom_agent)
graph.add_edge("my_agent", "supervisor")

# 3. Update supervisor routing logic to include "my_agent"
```

---

## 📁 Project Structure

```
langgraph-research-engine/
├── src/langgraph_research_engine/
│   ├── __init__.py
│   ├── engine.py      # 5 agents + StateGraph + run_research()
│   └── app.py         # Streamlit UI
├── pyproject.toml      # PyPI config
├── README.md
└── LICENSE
```

---

## 🗺️ Roadmap

- [ ] Add Tavily + Arxiv + Wikipedia as specialized data sources
- [ ] Citation tracking — link every fact to its source
- [ ] Async parallel research across multiple sub-topics
- [ ] Memory persistence between research sessions
- [ ] Export as PDF with citations
- [ ] Streaming output for real-time progress updates

---

## 📄 License

MIT — see [LICENSE](LICENSE)

---

<div align="center">

**⭐ Star this repo if you find it useful!**

Built with [LangGraph](https://github.com/langchain-ai/langgraph) · [LangChain](https://github.com/langchain-ai/langchain)

![Footer](https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=10,14,20&height=80&section=footer)

</div>
