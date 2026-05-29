import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from typing import Dict, List

# Load environment variables (e.g., OPENAI_API_KEY)
load_dotenv()

# Define a simple tool that returns a price for a product in a city
async def get_price(product: str, city: str) -> str:
    """Mock function to simulate fetching a product price."""
    # In a real scenario this would query an API or database.
    prices = {
        ("молоко", "Казань"): 89,
        ("хлеб", "Казань"): 35,
        ("чай", "Москва"): 120,
    }
    key = (product, city)
    price = prices.get(key, None)
    if price is None:
        return f"Цена для {product} в городе {city} не найдена."
    return f"{price} руб."

# Create the LLM instance
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Define the agent graph
def main_agent() -> StateGraph:
    # The state will contain a list of messages
    def add_message(state: Dict[str, List], message) -> Dict[str, List]:
        state["messages"].append(message)
        return state

    graph = StateGraph(add_message)
    graph.add_node("agent", lambda state: llm.invoke({"messages": state["messages"]}))
    graph.set_entry_point("agent")
    graph.add_edge("agent", END)
    return graph.compile()

# Instantiate the agent
agent = main_agent()

# Stream the conversation
stream = agent.stream(
    {"messages": [{"role": "human", "content": "Покажи цены на молоко и хлеб в Казани."}]},
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


def format_message(message) -> str:
    if message.content:
        return message.content
    # If the message is a tool call, display it nicely
    tool_call = message.tool_calls[0]
    name = tool_call.get("name")
    args = tool_call.get("args")
    return f"{name}({args})"

for chunk in stream:
    chunk_type, chunk_data = chunk
    if chunk_type == "messages":
        format_chunk_message(chunk_data)
    elif chunk_type == "updates":
        if chunk_data.get("model"):
            last_msg = chunk_data["model"]["messages"][-1]
            print(format_message(last_msg))

print("\n--- Conversation finished ---")
