"""
LangGraph Multi-Agent Research Engine
Orchestrates specialized agents to perform deep, multi-step research tasks.

Agents:
  - Supervisor    : routes tasks and decides when research is complete
  - Researcher    : gathers information via web search
  - Synthesizer   : combines findings into coherent knowledge
  - Fact-Checker  : validates claims and identifies contradictions
  - Writer        : produces the final structured report

Pattern: Supervisor-Worker with stateful graph (LangGraph StateGraph)
"""

from __future__ import annotations

import os
from typing import Annotated, TypedDict, Literal
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
import functools
import operator


# ── State definition ───────────────────────────────────────────────────────────
class ResearchState(TypedDict):
    """Shared state passed between all agents."""
    messages: Annotated[list[BaseMessage], add_messages]
    research_topic: str
    raw_findings: Annotated[list[str], operator.add]
    verified_facts: Annotated[list[str], operator.add]
    contradictions: Annotated[list[str], operator.add]
    final_report: str
    next_agent: str
    iteration: int
    max_iterations: int


# ── LLM + Tools setup ──────────────────────────────────────────────────────────
def get_llm(model: str = "gpt-4o-mini", temperature: float = 0.1) -> ChatOpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable not set. "
            "Get one at https://platform.openai.com/api-keys"
        )
    return ChatOpenAI(model=model, temperature=temperature, api_key=api_key)


def get_search_tool(max_results: int = 5) -> TavilySearchResults:
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        raise ValueError(
            "TAVILY_API_KEY environment variable not set. "
            "Get a free key at https://tavily.com"
        )
    return TavilySearchResults(max_results=max_results)


# ── Agent nodes ────────────────────────────────────────────────────────────────
def supervisor_node(state: ResearchState) -> dict:
    """
    Supervisor: Reads current state and decides which agent to call next.
    Routes: researcher → synthesizer → fact_checker → writer → END
    """
    llm = get_llm(temperature=0)
    iteration = state.get("iteration", 0)
    max_iter   = state.get("max_iterations", 3)

    # Force completion if max iterations reached
    if iteration >= max_iter and state.get("final_report"):
        return {"next_agent": "END", "iteration": iteration}

    # Decide routing based on what exists
    has_findings  = len(state.get("raw_findings", [])) > 0
    has_facts     = len(state.get("verified_facts", [])) > 0
    has_synthesis = any("synthesis" in str(m.content).lower()
                        for m in state.get("messages", [])
                        if hasattr(m, "content"))
    has_report    = bool(state.get("final_report", ""))

    if has_report:
        return {"next_agent": "END"}

    prompt = f"""You are a research supervisor. Current research state:
Topic: {state['research_topic']}
Iteration: {iteration}/{max_iter}
Raw findings collected: {len(state.get('raw_findings', []))}
Verified facts: {len(state.get('verified_facts', []))}
Has synthesis: {has_synthesis}
Has final report: {has_report}

Choose the next agent:
- "researcher" — if we need more information (do this first, up to {max_iter} times)
- "synthesizer" — if we have enough findings to synthesize (after researcher runs)
- "fact_checker" — if we have a synthesis to verify
- "writer" — if facts are verified, write the final report
- "END" — only if final_report is complete

Respond with ONLY one word: researcher, synthesizer, fact_checker, writer, or END"""

    response = llm.invoke([HumanMessage(content=prompt)])
    next_agent = response.content.strip().lower().strip('"').strip("'")

    # Validate response
    valid = {"researcher", "synthesizer", "fact_checker", "writer", "END"}
    if next_agent not in valid:
        # Fallback logic
        if not has_findings:
            next_agent = "researcher"
        elif not has_synthesis:
            next_agent = "synthesizer"
        elif not has_facts:
            next_agent = "fact_checker"
        else:
            next_agent = "writer"

    return {
        "next_agent": next_agent,
        "iteration": iteration + 1,
        "messages": [AIMessage(content=f"[Supervisor] Routing to: {next_agent} (iteration {iteration+1})")],
    }


def researcher_node(state: ResearchState) -> dict:
    """
    Researcher: Uses web search to gather information on the topic.
    Collects multiple search queries for breadth.
    """
    llm   = get_llm()
    topic = state["research_topic"]
    iteration = state.get("iteration", 1)

    try:
        search = get_search_tool(max_results=4)
        search_available = True
    except ValueError:
        search_available = False

    # Generate targeted search queries
    query_prompt = f"""Generate {min(iteration + 1, 3)} distinct search queries for researching:
"{topic}"

Make them specific and complementary. Return ONLY the queries, one per line."""

    queries_response = llm.invoke([HumanMessage(content=query_prompt)])
    queries = [q.strip() for q in queries_response.content.strip().split("\n") if q.strip()][:3]

    findings = []
    if search_available:
        for query in queries:
            try:
                results = search.invoke(query)
                if isinstance(results, list):
                    for r in results:
                        if isinstance(r, dict):
                            content = r.get("content", r.get("snippet", ""))
                            url     = r.get("url", "")
                            if content:
                                findings.append(f"[Source: {url}]\n{content[:500]}")
            except Exception as e:
                findings.append(f"[Search error for '{query}': {str(e)}]")
    else:
        # Fallback: use LLM knowledge
        knowledge_prompt = f"""Provide detailed research findings about: "{topic}"
Include key facts, recent developments, statistics, and expert perspectives.
Format as bullet points."""
        knowledge = llm.invoke([HumanMessage(content=knowledge_prompt)])
        findings.append(f"[LLM Knowledge]\n{knowledge.content}")

    msg = f"[Researcher] Gathered {len(findings)} findings from {len(queries)} queries"
    return {
        "raw_findings": findings,
        "messages": [AIMessage(content=msg)],
    }


def synthesizer_node(state: ResearchState) -> dict:
    """
    Synthesizer: Combines raw findings into structured, coherent knowledge.
    Removes duplicates, organizes themes, identifies key insights.
    """
    llm = get_llm(temperature=0.2)
    topic = state["research_topic"]
    findings = state.get("raw_findings", [])

    findings_text = "\n\n---\n\n".join(findings[:15])  # cap to avoid token limits

    prompt = f"""You are a research synthesizer. Combine these findings about "{topic}" into structured knowledge.

RAW FINDINGS:
{findings_text}

Create a synthesis with:
1. **Key Themes** (3-5 main themes identified)
2. **Core Facts** (bullet list of the most important verified-seeming facts)
3. **Recent Developments** (what's new or changing)
4. **Conflicting Information** (any contradictions you noticed)
5. **Gaps** (what's missing or unclear)

Be concise and structured."""

    response = llm.invoke([HumanMessage(content=prompt)])

    # Extract facts as separate items for fact-checking
    facts_prompt = f"""From this synthesis, extract 5-8 specific, verifiable factual claims as a numbered list.
Each should be a single, concrete statement.

SYNTHESIS:
{response.content}

Return ONLY the numbered list."""

    facts_response = llm.invoke([HumanMessage(content=facts_prompt)])
    facts = [
        line.strip().lstrip("0123456789.-) ").strip()
        for line in facts_response.content.strip().split("\n")
        if line.strip() and any(c.isalpha() for c in line)
    ]

    return {
        "verified_facts": facts,
        "messages": [
            AIMessage(content=f"[Synthesizer] Synthesized {len(findings)} findings into {len(facts)} key facts"),
            AIMessage(content=f"[Synthesis]\n{response.content}"),
        ],
    }


def fact_checker_node(state: ResearchState) -> dict:
    """
    Fact-Checker: Validates claims, identifies contradictions,
    flags uncertainty, and assigns confidence scores.
    """
    llm   = get_llm(temperature=0)
    topic = state["research_topic"]
    facts = state.get("verified_facts", [])

    if not facts:
        return {
            "messages": [AIMessage(content="[Fact-Checker] No facts to check.")],
        }

    facts_text = "\n".join(f"{i+1}. {f}" for i, f in enumerate(facts))

    prompt = f"""You are a fact-checker reviewing claims about "{topic}".

CLAIMS TO VERIFY:
{facts_text}

For each claim:
- Mark as ✅ LIKELY TRUE, ⚠️ UNCERTAIN, or ❌ LIKELY FALSE
- Give a brief reason (1 sentence)
- Note any contradictions between claims

Then provide:
- CONTRADICTIONS: list any claims that conflict with each other
- CONFIDENCE: overall confidence in the research (High/Medium/Low)
- MISSING: what key facts are missing"""

    response = llm.invoke([HumanMessage(content=prompt)])

    # Extract contradictions
    contradictions = []
    in_contradictions = False
    for line in response.content.split("\n"):
        if "CONTRADICTIONS:" in line.upper():
            in_contradictions = True
        elif in_contradictions and line.strip().startswith("-"):
            contradictions.append(line.strip().lstrip("- "))
        elif in_contradictions and line.strip() and not line.strip().startswith("-"):
            if any(keyword in line.upper() for keyword in ["CONFIDENCE:", "MISSING:"]):
                in_contradictions = False

    return {
        "contradictions": contradictions,
        "messages": [
            AIMessage(content=f"[Fact-Checker] Checked {len(facts)} claims, found {len(contradictions)} contradictions"),
            AIMessage(content=f"[Fact-Check Report]\n{response.content}"),
        ],
    }


def writer_node(state: ResearchState) -> dict:
    """
    Writer: Produces the final polished research report.
    Incorporates all findings, facts, and corrections into a structured document.
    """
    llm   = get_llm(temperature=0.3)
    topic = state["research_topic"]

    # Collect everything
    all_msgs = state.get("messages", [])
    synthesis = next(
        (m.content for m in reversed(all_msgs)
         if hasattr(m, "content") and "[Synthesis]" in str(m.content)),
        ""
    )
    fact_check = next(
        (m.content for m in reversed(all_msgs)
         if hasattr(m, "content") and "[Fact-Check Report]" in str(m.content)),
        ""
    )
    contradictions = state.get("contradictions", [])

    prompt = f"""You are a research writer. Write a comprehensive, professional research report.

TOPIC: {topic}

SYNTHESIS:
{synthesis[:2000] if synthesis else "See raw findings"}

FACT-CHECK:
{fact_check[:1500] if fact_check else "No fact-check available"}

CONTRADICTIONS FOUND: {', '.join(contradictions) if contradictions else 'None'}

Write a complete research report with:
# {topic} — Research Report

## Executive Summary
(2-3 paragraphs — the most important takeaways)

## Key Findings
(Bullet points of the most important facts, with confidence levels)

## Detailed Analysis
(3-4 paragraphs of in-depth analysis)

## Areas of Uncertainty
(What is unclear, debated, or not well-established)

## Conclusions & Recommendations
(Based on the research, what should the reader take away?)

---
*Generated by LangGraph Multi-Agent Research Engine*"""

    response = llm.invoke([HumanMessage(content=prompt)])

    return {
        "final_report": response.content,
        "messages": [AIMessage(content="[Writer] Final report complete")],
    }


# ── Routing function ───────────────────────────────────────────────────────────
def route_next(state: ResearchState) -> Literal["researcher", "synthesizer", "fact_checker", "writer", "__end__"]:
    next_agent = state.get("next_agent", "researcher")
    if next_agent == "END":
        return "__end__"
    return next_agent  # type: ignore


# ── Build the graph ────────────────────────────────────────────────────────────
def build_research_graph() -> StateGraph:
    graph = StateGraph(ResearchState)

    # Add all agent nodes
    graph.add_node("supervisor",   supervisor_node)
    graph.add_node("researcher",   researcher_node)
    graph.add_node("synthesizer",  synthesizer_node)
    graph.add_node("fact_checker", fact_checker_node)
    graph.add_node("writer",       writer_node)

    # Entry point: always start with supervisor
    graph.set_entry_point("supervisor")

    # Supervisor decides who goes next
    graph.add_conditional_edges(
        "supervisor",
        route_next,
        {
            "researcher":   "researcher",
            "synthesizer":  "synthesizer",
            "fact_checker": "fact_checker",
            "writer":       "writer",
            "__end__":      END,
        },
    )

    # After each agent, return to supervisor
    for agent in ["researcher", "synthesizer", "fact_checker", "writer"]:
        graph.add_edge(agent, "supervisor")

    return graph.compile()


# ── Main run function ──────────────────────────────────────────────────────────
def run_research(
    topic: str,
    max_iterations: int = 4,
    verbose: bool = True,
) -> dict:
    """
    Run the full multi-agent research pipeline on a topic.

    Args:
        topic: The research question or topic
        max_iterations: Max supervisor cycles (default 4)
        verbose: Print agent progress

    Returns:
        dict with final_report, verified_facts, contradictions, messages
    """
    graph = build_research_graph()

    initial_state: ResearchState = {
        "messages":       [HumanMessage(content=f"Research this topic thoroughly: {topic}")],
        "research_topic": topic,
        "raw_findings":   [],
        "verified_facts": [],
        "contradictions": [],
        "final_report":   "",
        "next_agent":     "researcher",
        "iteration":      0,
        "max_iterations": max_iterations,
    }

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Research Engine Starting")
        print(f"  Topic: {topic}")
        print(f"  Max iterations: {max_iterations}")
        print(f"{'='*60}\n")

    final_state = graph.invoke(initial_state)

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Research Complete!")
        print(f"  Facts gathered: {len(final_state.get('verified_facts', []))}")
        print(f"  Contradictions: {len(final_state.get('contradictions', []))}")
        print(f"{'='*60}\n")

    return final_state


# ── CLI entry point ────────────────────────────────────────────────────────────
def main():
    import sys
    if len(sys.argv) < 2:
        print("Usage: langgraph-research 'Your research topic here'")
        print("Example: langgraph-research 'What are the latest advances in LLM reasoning?'")
        sys.exit(1)

    topic = " ".join(sys.argv[1:])
    result = run_research(topic, verbose=True)

    print("\n" + "="*60)
    print("FINAL RESEARCH REPORT")
    print("="*60)
    print(result.get("final_report", "No report generated"))


if __name__ == "__main__":
    main()
