import os
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
import json

# Simple tool that returns a mock price table

def get_price(product: str, city: str) -> str:
    """Mock function to return a price table for a product in a city."""
    # In a real scenario this would query an API or database.
    prices = {
        "молоко": {"Казань": 89, "Москва": 95},
        "хлеб": {"Казань": 30, "Москва": 35},
    }
    price = prices.get(product, {}).get(city, None)
    if price is None:
        return f"{product} not found in {city}."
    table = f"| Продукт | Цена (руб.) | Магазин |
|---------|-------------|----------|
| {product} | {price} | Магнит |
"
    return table

# Define the tool specification for LangGraph
from langgraph.tools import ToolSpec

tool_spec = ToolSpec.from_function(get_price, name="get_price", description="Get price of a product in a city")

# Create the LLM and graph
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.2)

# Define the state: just messages for simplicity
class State(dict):
    pass

# Build the graph
graph_builder = StateGraph(State)

# Add tool node
tool_node = ToolNode([tool_spec])

# Define a simple chain: user -> llm -> tool -> llm -> end
@graph_builder.add_start("user")
def start(state):
    return state

@graph_builder.add_edge("user", "llm")
async def llm_node(state, messages=None):
    # Append user message if provided
    if messages:
        state["messages"] = messages
    else:
        state.setdefault("messages", [])
    # Run LLM with current messages
    response = await llm.ainvoke({"messages": state["messages"]})
    state["messages"].append(response)
    return "llm"

@graph_builder.add_edge("llm", "tool")
def tool_node_wrapper(state):
    # Extract last message for tool calls
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tool"
    return "end"

@graph_builder.add_edge("tool", "llm")
def after_tool(state):
    # Append tool result to messages
    state["messages"].append(state["tool_result"])
    return "llm"

# End node simply returns final message
@graph_builder.add_end("end")
def end_node(state):
    return state

# Build the graph and create an agent with stream support
agent = graph_builder.compile(checkpointer=MemorySaver(), interrupt_before=["tool"])

# Function to format messages for streaming output
step_counter = 1

def format_chunk_message(chunk):
    global step_counter
    message, meta = chunk
    if meta.get("langgraph_step") != step_counter:
        step_counter = meta["langgraph_step"]
        print("\n --- --- --- \n")
    if message.content:
        print(message.content, end="", flush=True)

# Function to format final messages or tool calls
def format_message(message):
    if getattr(message, "content", None):
        return message.content
    # Assume single tool call for simplicity
    tc = message.tool_calls[0]
    return f"{tc['name']}({json.dumps(tc['args'])})"

# Run the agent in stream mode
stream = agent.stream(
    {"messages": [{"role": "human", "content": "Покажи цены на молоко и хлеб в Казани."}]},
    stream_mode=["messages", "updates"]
)

for chunk_type, chunk_data in stream:
    if chunk_type == "messages":
        format_chunk_message(chunk_data)
    elif chunk_type == "updates":
        if chunk_data.get("model"):
            last_msg = chunk_data["model"]["messages"][-1]
            print(format_message(last_msg))

print("\n--- Finished ---")
