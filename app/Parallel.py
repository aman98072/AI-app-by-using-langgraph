"""
Parallel Nodes Example using LangGraph
=======================================
Concept: Multiple nodes running at the SAME TIME

Main Graph Flow:
  1. Receive a topic (e.g. "Python")
  2. Run 3 nodes IN PARALLEL:
       - node_facts     → generates a fun fact
       - node_quiz      → generates a quiz question
       - node_tip       → generates a learning tip
  3. Combine all results and print

Visual Flow:
                    ┌── node_facts ──┐
  start ── split ──├── node_quiz  ──┤── combine ── END
                    └── node_tip  ──┘
"""

import time
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END


# =============================================================
# STATE
# =============================================================

class State(TypedDict):
    topic: str
    fact: str        # filled by node_facts
    quiz: str        # filled by node_quiz
    tip: str         # filled by node_tip
    final: str       # filled by combine node


# =============================================================
# PARALLEL NODES
# (these 3 run at the same time)
# =============================================================

def node_facts(state: State) -> State:
    """Generates a fun fact about the topic"""
    print(f"  [node_facts] Started...")
    time.sleep(1)  # simulating some work

    facts = {
        "python": "Python was created by Guido van Rossum in 1991. 🐍",
        "ai":     "The term 'Artificial Intelligence' was coined in 1956. 🤖",
        "space":  "There are more stars in the universe than grains of sand on Earth. 🌌",
    }
    topic = state["topic"].lower()
    fact = facts.get(topic, f"'{state['topic']}' is a very interesting topic!")

    print(f"  [node_facts] Done! → {fact}")
    return {"fact": fact}


def node_quiz(state: State) -> State:
    """Generates a quiz question about the topic"""
    print(f"  [node_quiz] Started...")
    time.sleep(1)  # simulating some work

    quizzes = {
        "python": "Q: What does 'PEP' stand for in Python? A: Python Enhancement Proposal 📝",
        "ai":     "Q: What is the full form of 'GPU'? A: Graphics Processing Unit 💻",
        "space":  "Q: How far is the Moon from Earth? A: ~384,400 km 🌙",
    }
    topic = state["topic"].lower()
    quiz = quizzes.get(topic, f"Q: What is the most important thing about '{state['topic']}'?")

    print(f"  [node_quiz] Done! → {quiz}")
    return {"quiz": quiz}


def node_tip(state: State) -> State:
    """Generates a learning tip about the topic"""
    print(f"  [node_tip] Started...")
    time.sleep(1)  # simulating some work

    tips = {
        "python": "Tip: Practice daily with small scripts. Consistency beats intensity! 💡",
        "ai":     "Tip: Start with linear regression before jumping to neural networks! 💡",
        "space":  "Tip: Use NASA's free resources to learn astronomy online! 💡",
    }
    topic = state["topic"].lower()
    tip = tips.get(topic, f"Tip: Read books and build projects about '{state['topic']}'!")

    print(f"  [node_tip] Done! → {tip}")
    return {"tip": tip}


# =============================================================
# COMBINE NODE
# (runs AFTER all parallel nodes finish)
# =============================================================

def combine(state: State) -> State:
    """Combines all parallel results into one final output"""
    print(f"\n  [combine] Merging all results...")

    final = f"""
╔══════════════════════════════════════════╗
   Topic: {state['topic'].upper()}
╠══════════════════════════════════════════╣
 📚 Fun Fact:
   {state['fact']}

 ❓ Quiz:
   {state['quiz']}

 💡 Learning Tip:
   {state['tip']}
╚══════════════════════════════════════════╝
    """.strip()

    return {"final": final}


# =============================================================
# MAIN GRAPH
# =============================================================

def build_graph():
    graph = StateGraph(State)

    # Add all nodes
    graph.add_node("node_facts", node_facts)
    graph.add_node("node_quiz",  node_quiz)
    graph.add_node("node_tip",   node_tip)
    graph.add_node("combine",    combine)

    # Entry point — all 3 start at the same time
    graph.set_entry_point("node_facts")  # LangGraph handles parallelism via fan-out

    # Fan-out: start → 3 parallel nodes
    # In LangGraph, add all 3 as entry points using add_edge from START
    from langgraph.graph import START
    graph.add_edge(START, "node_facts")
    graph.add_edge(START, "node_quiz")
    graph.add_edge(START, "node_tip")

    # Fan-in: all 3 nodes → combine
    graph.add_edge("node_facts", "combine")
    graph.add_edge("node_quiz",  "combine")
    graph.add_edge("node_tip",   "combine")

    # combine → END
    graph.add_edge("combine", END)

    return graph.compile()


# =============================================================
# RUN
# =============================================================

def main():
    print("=" * 50)
    print("   Parallel Nodes Demo")
    print("=" * 50)

    graph = build_graph()

    topics = ["Python", "AI", "Space"]

    for topic in topics:
        print(f"\n{'─' * 50}")
        print(f"📥 Topic: {topic}")
        print(f"🚀 Running 3 nodes in parallel...\n")

        start_time = time.time()

        result = graph.invoke({
            "topic": topic,
            "fact":  "",
            "quiz":  "",
            "tip":   "",
            "final": ""
        })

        elapsed = time.time() - start_time

        print(f"\n✅ Final Output:")
        print(result["final"])
        print(f"\n⏱️  Time taken: {elapsed:.2f}s")

    print(f"\n{'=' * 50}")
    print("Done!")


if __name__ == "__main__":
    main()