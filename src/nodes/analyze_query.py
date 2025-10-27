from .graph_state import GraphState


def analyze_query(state: "GraphState") -> "GraphState":
    """
    Analyze and potentially expand the user's query.

    Args:
        state: Current graph state

    Returns:
        Updated graph state with analysis steps
    """
    steps = [f"Analyzing query: {state['question']}"]

    # Simple query analysis - you can make this more sophisticated
    return {
        **state,
        "steps": steps,
        "rewrite_count": 0
    }