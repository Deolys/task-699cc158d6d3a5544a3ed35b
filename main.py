import os
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START
from typing import Dict, Any

# Define a simple tool that returns a price for a product in a city
async def get_price(product: str, city: str) -> str:
    """Mock function to simulate fetching a price."""
    # In a real scenario this would query an API or database.
    prices = {
        ("молоко", "Казань"): 89,
        ("хлеб", "Казань"): 45,
    }
    key = (product, city)
    price = prices.get(key, 0)
    return f"{price}"

# Create the LLM and tool registry
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

# Define a simple state graph that calls get_price when needed
def main_node(state: Dict[str, Any]) -> Dict[str, Any]:
    # The node simply returns the current state unchanged.
    return state

graph = StateGraph()
graph.add_node("main", main_node)
# For simplicity we use a single edge that loops back to start
graph.set_entry_point("main")
flow = graph.compile()

# Helper functions for streaming output
step_counter = 1

def format_chunk_message(chunk):
    global step_counter
    message, meta = chunk
    if meta.get('langgraph_step') != step_counter:
        step_counter = meta.get('langgraph_step', step_counter)
        print('\n --- --- --- \n')
    if message.content:
        print(message.content, end='', flush=True)

def format_message(message):
    if message.content:
        return message.content
    # If the message is a tool call, display it nicely
    if hasattr(message, 'tool_calls') and message.tool_calls:
        tc = message.tool_calls[0]
        return f"{tc['name']}({tc['args']})"
    return ''

# Run the agent in stream mode
stream = flow.stream(
    {"messages": [{"role": "human", "content": "Покажи цену молока и хлеба в Казани."}]},
    stream_mode=["messages", "updates"]
)

for chunk_type, chunk_data in stream:
    if chunk_type == 'messages':
        format_chunk_message(chunk_data)
    elif chunk_type == 'updates':
        # When the model finishes a step and calls a tool
        if chunk_data.get('model'):
            last_msg = chunk_data['model']['messages'][-1]
            print(format_message(last_msg))

print('\n--- Завершено ---\n')
