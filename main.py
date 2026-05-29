import os
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from typing import Dict, Any

# LLM configuration
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Dummy tool implementation

def get_price(product: str, city: str) -> str:
    return f"{product} in {city}: 100"

tools = {"get_price": get_price}

class AgentState(dict):
    pass

# Build the graph
graph = StateGraph(AgentState)
graph.add_node("agent", lambda state: llm.invoke(state["messages"]))
graph.add_node("tool", ToolNode(tools))
graph.set_entry_point("agent")
graph.add_edge("agent", "tool")
graph.add_edge("tool", END)

app = graph.compile()

# Helper to format messages for updates

def format_message(message):
    if message.content:
        return message.content
    return f"{message.tool_calls[0]['name']}({message.tool_calls[0]['args']})"

step = 1

def main():
    stream = app.stream(
        {"messages": [{"role":"human","content":"Tell me the price of milk in Kazan"}]},
        stream_mode=["messages", "updates"]
    )
    for chunk_type, chunk_data in stream:
        if chunk_type == "messages":
            message, meta = chunk_data
            if meta["langgraph_step"] != step:
                global step
                step = meta["langgraph_step"]
                print("\n --- --- --- \n")
            if message.content:
                print(message.content, end="", flush=True)
        elif chunk_type == "updates":
            if chunk_data.get("model"):
                last_message = chunk_data["model"]["messages"][-1]
                print(format_message(last_message))

if __name__ == "__main__":
    main()
