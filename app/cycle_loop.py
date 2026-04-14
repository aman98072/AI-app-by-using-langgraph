"""
Cycles & Loops Example using LangGraph
========================================
Concept: Agent keeps trying until it gets a good answer

Story:
  - User asks a math question
  - Agent gives an answer
  - Validator checks if answer is correct
  - If WRONG → agent tries again (loop!)
  - If CORRECT → END
  - If max attempts reached → give up → END

Flow:
  START → agent → validator → correct?  
                                ├── YES  → END ✅
                                ├── NO   → agent (loop back!) 🔄
                                └── GIVE UP (3 attempts done) → END 😅
"""

import random
from typing import TypedDict
from langgraph.graph import StateGraph, END, START


# =============================================================
# STATE
# =============================================================

class State(TypedDict):
    question: str       # math question asked
    correct_answer: int # actual correct answer
    agent_answer: int   # agent's attempt
    attempt: int        # how many times tried
    feedback: str       # feedback from validator
    status: str         # "success" / "failed" / "in_progress"


# =============================================================
# NODES
# =============================================================

def agent_node(state: State) -> State:
    """
    Agent tries to answer the question.
    Sometimes gets it wrong (simulated with random error).
    """
    attempt = state.get("attempt", 0) + 1
    correct = state["correct_answer"]

    print(f"\n🤖 Agent [Attempt {attempt}]: Thinking...")

    # Simulate agent making mistakes sometimes
    # Attempt 1 → 60% chance of wrong answer
    # Attempt 2 → 40% chance of wrong answer
    # Attempt 3 → correct answer always
    if attempt == 1:
        wrong = random.random() < 0.6  # 60% chance wrong
    elif attempt == 2:
        wrong = random.random() < 0.4  # 40% chance wrong
    else:
        wrong = False  # attempt 3: always correct

    if wrong:
        # Give a wrong answer (off by some random amount)
        error = random.choice([-5, -3, -2, 2, 3, 5])
        agent_answer = correct + error
        print(f"   → Agent's answer: {agent_answer}  (wrong! correct is {correct})")
    else:
        agent_answer = correct
        print(f"   → Agent's answer: {agent_answer}  (correct! ✅)")

    return {
        "agent_answer": agent_answer,
        "attempt": attempt,
        "status": "in_progress"
    }


def validator_node(state: State) -> State:
    """
    Checks if agent's answer is correct.
    Gives feedback for the next attempt.
    """
    correct = state["correct_answer"]
    given   = state["agent_answer"]
    attempt = state["attempt"]

    print(f"\n🔍 Validator checking: agent said {given}, correct is {correct}")

    if given == correct:
        feedback = "Perfect! Answer is correct."
        status   = "success"
        print(f"   ✅ CORRECT after {attempt} attempt(s)!")
    elif attempt >= 3:
        feedback = "Max attempts reached. Giving up."
        status   = "failed"
        print(f"   ❌ GAVE UP after {attempt} attempts.")
    else:
        diff     = correct - given
        feedback = f"Wrong! Your answer {given} is off by {diff}. Try again."
        status   = "in_progress"
        print(f"   ❌ WRONG. Feedback: {feedback}")

    return {
        "feedback": feedback,
        "status":   status
    }


# =============================================================
# CONDITIONAL EDGE — loop or stop?
# =============================================================

def should_continue(state: State) -> str:
    status  = state.get("status", "in_progress")
    attempt = state.get("attempt", 0)

    if status == "success":
        return "correct"       # answer is right → END
    elif attempt >= 3:
        return "give_up"       # too many attempts → END
    else:
        return "retry"         # wrong but can try again → loop back


# =============================================================
# BUILD GRAPH
# =============================================================

def build_graph():
    graph = StateGraph(State)

    graph.add_node("agent",     agent_node)
    graph.add_node("validator", validator_node)

    # START → agent
    graph.add_edge(START, "agent")

    # agent → validator (always)
    graph.add_edge("agent", "validator")

    # validator → decide what to do next
    graph.add_conditional_edges(
        "validator",
        should_continue,
        {
            "correct":  END,      # success → stop
            "give_up":  END,      # failed  → stop
            "retry":    "agent"   # wrong   → loop back to agent ♻️
        }
    )

    return graph.compile()


# =============================================================
# RUN
# =============================================================

def ask(question: str, correct_answer: int):
    print(f"\n{'=' * 55}")
    print(f"❓ Question: {question}")
    print(f"   Correct Answer: {correct_answer}")
    print(f"{'=' * 55}")

    graph = build_graph()

    result = graph.invoke({
        "question":       question,
        "correct_answer": correct_answer,
        "agent_answer":   0,
        "attempt":        0,
        "feedback":       "",
        "status":         "in_progress"
    })

    print(f"\n📊 Final Summary:")
    print(f"   Attempts made : {result['attempt']}")
    print(f"   Final answer  : {result['agent_answer']}")
    print(f"   Status        : {result['status'].upper()}")
    print(f"   Feedback      : {result['feedback']}")


def main():
    print("\n♻️  Cycles & Loops Demo — LangGraph")
    print("Agent will keep trying until correct or 3 attempts done\n")

    questions = [
        ("What is 12 + 15?",  27),
        ("What is 8 × 9?",    72),
        ("What is 100 - 37?", 63),
    ]

    for question, answer in questions:
        ask(question, answer)

    print(f"\n{'=' * 55}")
    print("Done!")


if __name__ == "__main__":
    main()