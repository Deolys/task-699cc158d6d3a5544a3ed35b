import os
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from typing import Dict, Any, List

# Define a simple tool that simulates getting price of a product in a city

def get_price(product: str, city: str) -> str:
    """Mock function to return a fake price table."""
    # In real scenario this would call an API or database
    return f"| Продукт | Цена (руб.) | Магазин  |
| {product} | 89          | Магнит   |"

# Wrap the tool for LangChain
from langchain.tools import StructuredTool
price_tool = StructuredTool.from_function(
    func=get_price,
    name="get_price",
    description="Get price of a product in a city. Returns markdown table.",
)

# Define the agent state and graph
class AgentState(dict):
    pass

def main():
    # Initialize LLM (use environment variable for API key)
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.2, api_key=os.getenv("OPENAI_API_KEY"))

    # Build the graph with a single tool node
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", lambda state: llm.invoke(state["messages"]))
    workflow.add_node("tool", ToolNode([price_tool]))

    # Connect agent to tool and back, finish when no more tool calls
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        lambda x: END if not x["messages"][-1].tool_calls else "tool",
    )
    workflow.add_edge("tool", "agent")

    graph = workflow.compile()

    # Prepare initial user message
    init_message = {"role": "human", "content": "Покажи цену молока в Казани."}

    # Stream the response
    stream = graph.stream(
        {"messages": [init_message]},
        stream_mode=["messages", "updates"],
    )

    step = 1

    def format_chunk_message(chunk):
        message, meta = chunk
        nonlocal step
        if meta.get("langgraph_step") != step:
            step = meta["langgraph_step"]
            print("\n --- --- --- \n")
        if message.content:
            print(message.content, end="", flush=True)

    def format_message(message):
        if message.content:
            return message.content
        # Assume single tool call for simplicity
        tc = message.tool_calls[0]
        return f"{tc['name']}({tc['args']})"

    for chunk_type, chunk_data in stream:
        if chunk_type == "messages":
            format_chunk_message(chunk_data)
        elif chunk_type == "updates":
            if chunk_data.get("model"):
                last_msg = chunk_data["model"]["messages"][-1]
                print(format_message(last_msg))

if __name__ == "__main__":
    main()
