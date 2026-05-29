import os
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

# 1. Define a simple tool that simulates an API call

def get_price(product: str, city: str) -> str:
    """Mock function to return price for a product in a city."""
    prices = {
        ("молоко", "Казань"): 89,
        ("хлеб", "Казань"): 45,
    }
    key = (product, city)
    price = prices.get(key, 0)
    return f"{price}"

# 2. Create a tool node for the agent
get_price_node = ToolNode(get_price)

# 3. Define the graph state and nodes
def main() -> None:
    # LLM configuration – use environment variable for API key
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, api_key=os.getenv("OPENAI_API_KEY"))

    # Graph definition
    graph = StateGraph()
    graph.add_node("agent", lambda state: llm.invoke(state["messages"]))
    graph.add_node("get_price", get_price_node)

    # Define transitions
    def agent_to_next(state):
        last_message = state["messages"][-1]
        if last_message.tool_calls:
            return "get_price"
        return END

    graph.set_entry_point("agent")
    graph.add_edge("agent", "get_price", condition=agent_to_next)
    graph.add_edge("get_price", "agent")

    # Compile the graph with a memory saver for streaming
    app = graph.compile(checkpointer=MemorySaver())

    # 4. Stream the response
    stream = app.stream(
        {"messages": [{"role": "human", "content": "Покажи цену молока и хлеба в Казани."}]},
        stream_mode=["messages", "updates"],
    )

    step = 1

    def format_chunk_message(chunk):
        nonlocal step
        message, meta = chunk
        if meta["langgraph_step"] != step:
            step = meta["langgraph_step"]
            print("\n --- --- --- \n")
        if message.content:
            print(message.content, end="", flush=True)

    def format_message(message):
        if message.content:
            return message.content
        # Assume single tool call for simplicity
        name = message.tool_calls[0]["name"]
        args = message.tool_calls[0]["args"]
        return f"{name}({args})"

    for chunk in stream:
        chunk_type, chunk_data = chunk
        if chunk_type == "messages":
            format_chunk_message(chunk_data)
        elif chunk_type == "updates":
            if chunk_data.get("model"):
                last_msg = chunk_data["model"]["messages"][-1]
                print(format_message(last_msg))

if __name__ == "__main__":
    main()
