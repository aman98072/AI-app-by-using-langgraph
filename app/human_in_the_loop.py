"""
Simple Human-in-the-Loop Example using LangGraph
=================================================
Flow:
  1. AI generates a joke
  2. Asks human — did you like it?
  3. If "yes" → END
  4. If "no" → generates another joke (max 3 attempts)
"""

from typing import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt


# ─────────────────────────────────────────────────
# State
# ─────────────────────────────────────────────────
class State(TypedDict):
    joke: str
    attempt: int
    human_approved: bool


# ─────────────────────────────────────────────────
# Nodes
# ─────────────────────────────────────────────────
def generate_joke(state: State) -> State:
    """AI generates a joke (using a simple list instead of LLM)"""
    jokes = [
        "Why do programmers prefer dark mode? Because light attracts bugs! 🐛",
        "Why did the developer go broke? Because he used up all his cache! 💸",
        "Why do Java developers wear glasses? Because they don't C#! 👓",
    ]
    attempt = state.get("attempt", 0)
    joke = jokes[attempt % len(jokes)]

    print(f"\n🤖 AI: Here is your joke (attempt {attempt + 1}):\n   {joke}")

    return {
        "joke": joke,
        "attempt": attempt + 1,
        "human_approved": False
    }


def human_review(state: State) -> State:
    """
    Graph PAUSES here and waits for human input.
    Execution resumes only after human responds.
    """
    # ⏸️ This line pauses the graph
    decision = interrupt({
        "joke": state["joke"],
        "question": "Did you like the joke? (yes/no)"
    })

    approved = decision.strip().lower() in ["yes", "y"]
    print(f"✅ Human response: {decision} → Approved: {approved}")

    return {"human_approved": approved}


# ─────────────────────────────────────────────────
# Conditional Edge
# ─────────────────────────────────────────────────
def should_continue(state: State) -> str:
    if state.get("human_approved"):
        return "end"
    elif state.get("attempt", 0) >= 3:
        return "give_up"
    else:
        return "retry"


# ─────────────────────────────────────────────────
# Graph Build
# ─────────────────────────────────────────────────
def build_graph():
    graph = StateGraph(State)

    graph.add_node("generate_joke", generate_joke)
    graph.add_node("human_review", human_review)

    graph.set_entry_point("generate_joke")
    graph.add_edge("generate_joke", "human_review")

    graph.add_conditional_edges(
        "human_review",
        should_continue,
        {
            "end":     END,
            "give_up": END,
            "retry":   "generate_joke"  # generate a new joke
        }
    )

    memory = MemorySaver()  # required for interrupt to work
    return graph.compile(checkpointer=memory)


# ─────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────
def main():
    graph = build_graph()

    # A unique thread_id is required for each session
    config = {"configurable": {"thread_id": "demo-thread-1"}}

    initial_state = {
        "joke": "",
        "attempt": 0,
        "human_approved": False
    }

    print("=" * 50)
    print("   Human-in-the-Loop Demo")
    print("=" * 50)

    # ── Step 1: Run the graph — it will pause at interrupt ──
    result = graph.invoke(initial_state, config)
    print(f"\n⏸️  Graph paused! Interrupted state: {result}")

    # ── Step 2: Loop until human approves ──
    while True:
        # Get human input from terminal
        human_input = input("\n👤 Your answer (yes/no): ").strip()

        # ── Step 3: Resume the graph with human input ──
        result = graph.invoke(
            {"human_decision": human_input},  # human's response
            config                            # same thread_id — resumes from where it paused
        )

        # Check if graph has finished
        if result.get("human_approved"):
            print("\n🎉 Joke approved! Graph finished.")
            break
        elif result.get("attempt", 0) >= 3:
            print("\n😅 3 attempts done. No more jokes!")
            break

        print(f"\n⏸️  Graph paused again. New joke incoming!")


if __name__ == "__main__":
    main()