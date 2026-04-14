"""
Simple Sub-Graphs Example using LangGraph
==========================================
Concept: A graph inside another graph

Main Graph Flow:
  1. Receive a number
  2. Send to "Math Subgraph" → doubles + adds 10
  3. Send to "Text Subgraph" → converts number to a fun message
  4. Print final result

Sub-Graph 1 (Math):
  double_it → add_ten

Sub-Graph 2 (Text):
  build_message → add_emoji
"""

from typing import TypedDict
from langgraph.graph import StateGraph, END


# =============================================================
# STATES
# =============================================================

class MainState(TypedDict):
    number: int        # input number
    result: int        # after math subgraph
    message: str       # after text subgraph


class MathState(TypedDict):
    number: int
    result: int


class TextState(TypedDict):
    result: int
    message: str


# =============================================================
# SUB-GRAPH 1: Math Subgraph
# (doubles the number, then adds 10)
# =============================================================

def double_it(state: MathState) -> MathState:
    doubled = state["number"] * 2
    print(f"  [Math Subgraph] double_it: {state['number']} × 2 = {doubled}")
    return {"result": doubled}


def add_ten(state: MathState) -> MathState:
    added = state["result"] + 10
    print(f"  [Math Subgraph] add_ten: {state['result']} + 10 = {added}")
    return {"result": added}


def build_math_subgraph():
    graph = StateGraph(MathState)

    graph.add_node("double_it", double_it)
    graph.add_node("add_ten", add_ten)

    graph.set_entry_point("double_it")
    graph.add_edge("double_it", "add_ten")
    graph.add_edge("add_ten", END)

    return graph.compile()


# =============================================================
# SUB-GRAPH 2: Text Subgraph
# (converts number to a fun message with emoji)
# =============================================================

def build_message(state: TextState) -> TextState:
    msg = f"Your final number is {state['result']}"
    print(f"  [Text Subgraph] build_message: '{msg}'")
    return {"message": msg}


def add_emoji(state: TextState) -> TextState:
    emoji = "🔥" if state["result"] > 50 else "😊"
    final = f"{state['message']} {emoji}"
    print(f"  [Text Subgraph] add_emoji: '{final}'")
    return {"message": final}


def build_text_subgraph():
    graph = StateGraph(TextState)

    graph.add_node("build_message", build_message)
    graph.add_node("add_emoji", add_emoji)

    graph.set_entry_point("build_message")
    graph.add_edge("build_message", "add_emoji")
    graph.add_edge("add_emoji", END)

    return graph.compile()


# =============================================================
# MAIN GRAPH
# (calls both subgraphs one by one)
# =============================================================

# Compile subgraphs once
math_subgraph = build_math_subgraph()
text_subgraph = build_text_subgraph()


def run_math_subgraph(state: MainState) -> MainState:
    """Step 1: Pass number to Math Subgraph"""
    print("\n📐 Running Math Subgraph...")

    # Call subgraph with its own state format
    output = math_subgraph.invoke({
        "number": state["number"],
        "result": 0
    })

    print(f"  Math Subgraph Output: {output['result']}")
    return {"result": output["result"]}


def run_text_subgraph(state: MainState) -> MainState:
    """Step 2: Pass result to Text Subgraph"""
    print("\n✍️  Running Text Subgraph...")

    # Call subgraph with its own state format
    output = text_subgraph.invoke({
        "result": state["result"],
        "message": ""
    })

    print(f"  Text Subgraph Output: {output['message']}")
    return {"message": output["message"]}


def build_main_graph():
    graph = StateGraph(MainState)

    graph.add_node("math_subgraph", run_math_subgraph)
    graph.add_node("text_subgraph", run_text_subgraph)

    graph.set_entry_point("math_subgraph")
    graph.add_edge("math_subgraph", "text_subgraph")
    graph.add_edge("text_subgraph", END)

    return graph.compile()


# =============================================================
# RUN
# =============================================================

def main():
    print("=" * 50)
    print("   Sub-Graphs Demo")
    print("=" * 50)

    main_graph = build_main_graph()

    # Test with different numbers
    for number in [5, 30]:
        print(f"\n{'─' * 40}")
        print(f"📥 Input Number: {number}")

        result = main_graph.invoke({
            "number": number,
            "result": 0,
            "message": ""
        })

        print(f"\n✅ Final Result:")
        print(f"   Number after math : {result['result']}")
        print(f"   Final message      : {result['message']}")

    print(f"\n{'=' * 50}")
    print("Done!")


if __name__ == "__main__":
    main()