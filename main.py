import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from typing import Dict, Any

# Load environment variables (e.g., OPENAI_API_KEY)
load_dotenv()

# Define a simple tool that simulates price lookup
async def get_price(product: str, city: str) -> str:
    """Mock function to return a price table for a product in a city."""
    # In a real scenario this would query an API or database.
    prices = {
        "молоко": {"Казань": 89, "Москва": 95},
        "хлеб": {"Казань": 30, "Москва": 35},
    }
    price = prices.get(product, {}).get(city, "неизвестно")
    return f"| Продукт | Цена (руб.) | Магазин |
| {product} | {price} | Магнит |"

# Create the LLM instance
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

# Define the agent graph
def main_agent() -> StateGraph:
    # The state will just hold messages
    def add_messages(state: Dict[str, Any]):
        return state

    graph = StateGraph(add_messages)
    graph.add_node("agent", llm)
    graph.set_entry_point("agent")
    graph.add_edge("agent", END)
    return graph.compile()

# Instantiate the agent
agent = main_agent()

# Stream the response
stream = agent.stream(
    {"messages": [{"role": "human", "content": "Покажи цену молока в Казани и хлеба."}]},
    stream_mode=["messages", "updates"]
)

step = 1

def format_chunk_message(chunk):
    message, meta = chunk
    global step
    if meta.get("langgraph_step") != step:
        step = meta["langgraph_step"]
        print("\n --- --- --- \n")
    if message.content:
        print(message.content, end="", flush=True)

# Helper to format final messages (tool calls)
def format_message(message):
    if message.content:
        return message.content
    # Assume single tool call for simplicity
    tc = message.tool_calls[0]
    name = tc["name"]
    args = tc["args"]
    return f"{name}({args})"

for chunk in stream:
    chunk_type, chunk_data = chunk
    if chunk_type == "messages":
        format_chunk_message(chunk_data)
    elif chunk_type == "updates":
        if chunk_data.get("model"):
            last_msg = chunk_data["model"]["messages"][-1]
            print(format_message(last_msg))

print("\n--- Завершено ---")
