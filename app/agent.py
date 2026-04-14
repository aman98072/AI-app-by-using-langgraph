import json
import os
import re
from typing import Dict, TypedDict
import traceback
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# LLM Setup
# ─────────────────────────────────────────────────────────────────────────────
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.7,
    max_tokens=4096,
    top_p=0.9,
    api_key=os.getenv("OPENAI_API_KEY"),
)

# ─────────────────────────────────────────────────────────────────────────────
# State Definition
# ─────────────────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    message: str
    article: str
    blog_rewrite_cnt: int
    rating_score: int


# ─────────────────────────────────────────────────────────────────────────────
# Nodes
# ─────────────────────────────────────────────────────────────────────────────
def generate_article(state: AgentState) -> AgentState:
    """
    Generate an article based on the input message.
    If rewrite_count > 0, use previous rating feedback to improve.
    """
    rewrite_count = state.get("blog_rewrite_cnt", 0)
    message = state["message"]

    # If this is a regeneration, add context from previous attempt
    print('rewrite_count : ', rewrite_count, message)
    if rewrite_count > 0:
        previous_article = state.get("article", "")
        previous_rating = state.get("rating_score", "")

        improved_prompt = f"""Topic: {message}

        Previous article (score was below 9): {previous_article}

        Rating feedback: {previous_rating}

        Please generate a much better, more comprehensive article on the same topic, addressing the feedback above."""

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert content writer. Create detailed, high-quality articles that are well-structured and informative."),
            ("user", improved_prompt),
        ])
    else:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that generates articles."),
            ("user", message),
        ])

    response = llm.invoke(prompt.format_messages())
    print('response : ', response)

    # Increment rewrite count
    new_count = rewrite_count + 1

    return {
        "article": response.content,
        "blog_rewrite_cnt": new_count
    }

def generate_rating(state: AgentState) -> AgentState:
    try:
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert article evaluator. Rate the article from 1-10.

            IMPORTANT: Respond ONLY with valid JSON in this exact format, no extra text:
            {{
                "score": <number from 1 to 10>,
                "content": "<brief comment>",
                "structure": "<brief comment>",
                "recommendation": "<yes or no>"
            }}"""),
            ("user", f"Rate this article:\n\n{state['article']}"),
        ])

        response = llm.invoke(prompt.format_messages())
        raw = response.content
        print('raw response:', raw)

        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if not match:
            raise ValueError(f"No JSON found in response: {raw}")

        data = json.loads(match.group())
        print('parsed rating:', data)

        return {
            "rating_score": data.get("score", 0),
            "rating_data": data
        }
    
    except Exception as e:
        print("Error in generate_rating:", str(e))
        traceback.print_exc()
        return {}        

# ─────────────────────────────────────────────────────────────────────────────
# Workflow Graph
# ─────────────────────────────────────────────────────────────────────────────
def should_continue(state: AgentState) -> str:
    """
    Decision logic:
    - If score >= 9 -> END
    - If rewrite_count >= 3 -> END with give_up message
    - If score < 9 -> regenerate article
    """
    score = state.get("rating_score", 0)
    count = state.get("blog_rewrite_cnt", 0)

    print(f"[DEBUG] Score: {score}, Rewrite Count: {count}")

    if score >= 9:
        return "end"
    elif count >= 3:
        return "give_up"
    else:
        return "regenerate"


def workflow():
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("generate_article", generate_article)
    graph.add_node("generate_rating", generate_rating)

    # Set entry point
    graph.set_entry_point("generate_article")

    # article -> rating
    graph.add_edge("generate_article", "generate_rating")

    # After rating: conditional check
    graph.add_conditional_edges(
        "generate_rating",
        should_continue,
        {
            "end": END,
            "give_up": END,
            "regenerate": "generate_article"
        }
    )

    return graph.compile()


def response_generator(message: str) -> Dict[str, str]:
    try:
        graph = workflow()

        # Initial state
        initial_state = {
            "message": message,
            "article": "",
            "rating_score": 0,
            "blog_rewrite_cnt": 0
        }

        result = graph.invoke(initial_state)

        final_score = result.get("rating_score", 0)
        final_count = result.get("blog_rewrite_cnt", 0)

        # Check if we gave up
        if final_score < 9 and final_count >= 3:
            return {
                "article": result.get("article", ""),
                "rating": result.get("rating_score", ""),
                "status": "failed",
                "message": "We could not generate a satisfactory article after 3 attempts. Please try a different topic."
            }

        return {
            "article": result.get("article", ""),
            "rating": result.get("rating_score", ""),
            "status": "success"
        }

    except Exception as e:
        print("Error in response_generator:", str(e))
        return {"error": str(e)}