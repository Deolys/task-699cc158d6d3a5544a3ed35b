import os
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from typing import Dict, Any

# 1. LLM setup
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

# 2. Define a simple tool that returns price info (mocked for demo)
def get_price(product: str, city: str) -> str:
    """Mock tool returning a formatted table of prices."""
    # In real scenario, call external API or database
    return f"| Продукт | Цена (руб.) | Магазин  |
| {product} | 89          | Магнит   |"

# 3. Create ToolNode and graph
tool_node = ToolNode({"get_price": get_price})

class AgentState(dict):
    pass

graph = StateGraph(AgentState)

# Define nodes
@graph.node
def user_input(state: AgentState) -> Dict[str, Any]:
    return {"messages": state.get("messages", [])}

@graph.node
def llm_node(state: AgentState) -> Dict[str, Any]:
    # Append LLM response to messages
    new_msg = llm.invoke({"messages": state["messages"]})
    state["messages"].append(new_msg)
    return {"messages": state["messages"]}

@graph.node
def tool_node_wrapper(state: AgentState) -> Dict[str, Any]:
    # Execute any pending tool calls in the last message
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        for call in last.tool_calls:
            name = call["name"]
            args = eval(call["args"])  # simple eval for demo
            result = get_price(**args)
            tool_msg = llm.invoke({"messages": state["messages"] + [{"role": "tool", "name": name, "content": result}]})
            state["messages"].append(tool_msg)
    return {"messages": state["messages"]}

# Build graph
graph.add_node("user_input", user_input)
graph.add_node("llm", llm_node)
graph.add_node("tool", tool_node_wrapper)

# Define transitions
graph.set_entry_point("user_input")
graph.add_edge("user_input", "llm")
graph.add_conditional_edges(
    "llm",
    lambda x: "tool" if any(hasattr(m, 'tool_calls') and m.tool_calls for m in x["messages"]) else END,
)
graph.add_edge("tool", "llm")

# Compile graph with memory
memory = MemorySaver()
app = graph.compile(checkpointer=memory)

# 4. Stream execution
if __name__ == "__main__":
    user_query = input("Введите запрос: ")
    # Initialize state
    init_state = {"messages": [{"role": "human", "content": user_query}]}
    stream = app.stream(init_state, stream_mode=["messages", "updates"])

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
        call = message.tool_calls[0]
        return f"{call['name']}({call['args']})"

    for chunk_type, chunk_data in stream:
        if chunk_type == "messages":
            format_chunk_message(chunk_data)
        elif chunk_type == "updates":
            if chunk_data.get("model"):
                last_msg = chunk_data["model"]["messages"][-1]
                print(format_message(last_msg))

    print("\n--- завершено ---")
