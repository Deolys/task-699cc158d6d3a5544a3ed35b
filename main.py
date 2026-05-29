import os
from langgraph import Graph, State, ToolNode
from langgraph.tools import tool
from openai import OpenAI

# 1. Set up the OpenAI client (use environment variable for API key)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 2. Define a simple tool that returns a price lookup – in real life this would call an external service.
@tool
def get_price(product: str, city: str) -> str:
    """Return a mock price for a product in a given city."""
    prices = {
        ("молоко", "Казань"): "89",
        ("хлеб", "Казань"): "35",
        ("сахар", "Казань"): "45",
    }
    key = (product, city)
    price = prices.get(key, "неизвестно")
    return f"{price} руб."

# 3. Create a simple state that holds the conversation history.
class AgentState(State):
    messages: list[dict]

# 4. Build the graph – a single node that calls the LLM and may invoke tools.
async def llm_node(state: AgentState) -> AgentState:
    # Prepare messages for the model
    chat_messages = [
        {"role": msg.get("role", "assistant"), "content": msg["content"]}
        for msg in state.messages
    ]
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=chat_messages,
        stream=True,
    )
    # Yield tokens as they arrive
    async for chunk in response:
        if chunk.choices[0].delta.content is not None:
            yield {"role": "assistant", "content": chunk.choices[0].delta.content}
        elif chunk.choices[0].delta.tool_calls:
            # When a tool call starts, forward the whole tool call object
            yield {"role": "assistant", "tool_calls": chunk.choices[0].delta.tool_calls}
    return state

# 5. Assemble the graph with a single LLM node and the get_price tool.
graph = Graph(
    nodes={
        "llm": llm_node,
        "get_price": ToolNode(get_price),
    },
    edges=[("llm", "get_price"), ("get_price", "llm")],
)

# 6. Helper to format messages for printing.
def format_message(msg: dict) -> str:
    if msg.get("content"):
        return msg["content"]
    # tool call representation
    tc = msg.get("tool_calls", [{}])[0]
    name = tc.get("name", "unknown_tool")
    args = tc.get("args", {})
    return f"{name}({args})"

# 7. Main entry point – stream the conversation.
if __name__ == "__main__":
    # Initial user prompt
    user_prompt = "Покажи цены на молоко и хлеб в Казани."
    state = AgentState(messages=[{"role": "user", "content": user_prompt}])

    stream = graph.stream(state, stream_mode=["messages", "updates"])

    step = 1
    for chunk_type, chunk_data in stream:
        if chunk_type == "messages":
            message, meta = chunk_data
            # Detect step change
            current_step = meta.get("langgraph_step") or 0
            if current_step != step:
                step = current_step
                print("\n --- --- --- \n")
            if message.get("content"):
                print(message["content"], end="", flush=True)
        elif chunk_type == "updates":
            # When the model finishes a turn, we may want to show tool calls.
            if chunk_data.get("model"):
                last_msg = chunk_data["model"]["messages"][-1]
                print(format_message(last_msg))

    print("\n--- Конец диалога ---")
