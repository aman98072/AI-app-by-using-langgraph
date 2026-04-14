"""
ReAct Agent using LangGraph
============================
ReAct = Reasoning + Acting

Concept:
  Agent THINKS what to do → ACTS (calls a tool) → OBSERVES result
  → THINKS again → ACTS again → ... → gives FINAL ANSWER

Flow:
  START → agent (think) → should_continue?
                              ├── "tools" → run_tools → back to agent
                              └── "end"  → END

Tools available:
  - calculator : does math
  - weather    : fake weather data
  - search     : fake search results
"""

import json
from typing import TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from dotenv import load_dotenv
import os

load_dotenv()


# =============================================================
# TOOLS DEFINITION
# =============================================================

@tool
def calculator(expression: str) -> str:
    """
    Evaluates a math expression.
    Example: calculator("2 + 2") → "4"
    """
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def weather(city: str) -> str:
    """
    Returns fake weather data for a city.
    Example: weather("Delhi") → "Delhi: 35°C, Sunny"
    """
    fake_data = {
        "delhi":   "35°C, Sunny ☀️",
        "mumbai":  "30°C, Humid 🌫️",
        "london":  "12°C, Rainy 🌧️",
        "new york":"22°C, Cloudy ⛅",
    }
    city_lower = city.lower()
    result = fake_data.get(city_lower, f"20°C, Clear 🌤️")
    return f"{city}: {result}"


@tool
def search(query: str) -> str:
    """
    Searches for information about a query.
    Example: search("Python creator") → "Python was created by Guido van Rossum"
    """
    fake_results = {
        "python creator":      "Python was created by Guido van Rossum in 1991.",
        "langgraph":           "LangGraph is a library for building stateful multi-actor apps with LLMs.",
        "react agent":         "ReAct stands for Reasoning + Acting. It is a pattern for AI agents.",
        "capital of france":   "The capital of France is Paris.",
        "capital of india":    "The capital of India is New Delhi.",
    }
    query_lower = query.lower()
    for key, value in fake_results.items():
        if key in query_lower:
            return value
    return f"No results found for '{query}'. Try a different search term."


# All tools in a list
tools = [calculator, weather, search]

# Tool name → function mapping (for execution)
tools_map = {
    "calculator": calculator,
    "weather":    weather,
    "search":     search,
}


# =============================================================
# LLM SETUP — bind tools so LLM knows what it can use
# =============================================================

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY"),
)

llm_with_tools = llm.bind_tools(tools)


# =============================================================
# STATE
# =============================================================

class AgentState(TypedDict):
    # Annotated with operator.add means messages accumulate (don't overwrite)
    messages: Annotated[list[BaseMessage], operator.add]


# =============================================================
# NODES
# =============================================================

def agent_node(state: AgentState) -> AgentState:
    """
    THINK step — LLM decides:
      - Call a tool? → returns ToolCall
      - Give final answer? → returns plain text
    """
    print("\n🤔 Agent THINKING...")

    messages = state["messages"]
    response = llm_with_tools.invoke(messages)

    # Show what agent decided
    if response.tool_calls:
        for tc in response.tool_calls:
            print(f"   → Wants to call tool: '{tc['name']}' with args: {tc['args']}")
    else:
        print(f"   → Final Answer ready!")

    return {"messages": [response]}


def tools_node(state: AgentState) -> AgentState:
    """
    ACT step — actually runs the tool the agent requested
    """
    messages = state["messages"]
    last_message = messages[-1]  # latest AI message with tool_calls

    tool_results = []

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        print(f"\n🔧 Running tool: '{tool_name}' with args: {tool_args}")

        # Find and run the tool
        tool_fn = tools_map.get(tool_name)
        if tool_fn:
            result = tool_fn.invoke(tool_args)
        else:
            result = f"Tool '{tool_name}' not found."

        print(f"   → Tool result: {result}")

        # Wrap result in ToolMessage so LLM can read it
        tool_results.append(ToolMessage(
            content=str(result),
            tool_call_id=tool_call["id"]
        ))

    return {"messages": tool_results}


# =============================================================
# CONDITIONAL EDGE — should agent stop or call more tools?
# =============================================================

def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]

    # If last message has tool calls → go to tools node
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        print("\n   ↪ Decision: Call tools")
        return "tools"

    # Otherwise → agent has final answer → END
    print("\n   ↪ Decision: Done!")
    return "end"


# =============================================================
# BUILD GRAPH
# =============================================================

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)

    graph.add_edge(START, "agent")

    # After agent: decide to call tools or end
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end":   END
        }
    )

    # After tools: always go back to agent (think again)
    graph.add_edge("tools", "agent")

    return graph.compile()


# =============================================================
# RUN
# =============================================================

def ask(question: str):
    print(f"\n{'=' * 55}")
    print(f"❓ Question: {question}")
    print(f"{'=' * 55}")

    graph = build_graph()

    result = graph.invoke({
        "messages": [HumanMessage(content=question)]
    })

    # Last message = final answer
    final = result["messages"][-1].content
    print(f"\n✅ Final Answer: {final}")
    print(f"{'=' * 55}\n")
    return final


def main():
    print("\n🤖 ReAct Agent Demo — LangGraph")
    print("Tools available: calculator, weather, search\n")

    # Test questions — agent will THINK + USE TOOLS to answer
    questions = [
        "What is 25 multiplied by 4?",
        "What is the weather in Delhi?",
        "Who created Python programming language?",
        "What is the weather in London and also calculate 100 divided by 4?",  # uses 2 tools!
    ]

    for q in questions:
        ask(q)


if __name__ == "__main__":
    main()