from langchain import hub
from langchain_openai import ChatOpenAI
from langchain.tools import tool
import os

# Simple price lookup tool – in a real project this would query an API or database.
@tool("get_price", "Return the price of a product in a given city")
def get_price(product: str, city: str) -> str:
    # Dummy data for demonstration purposes
    prices = {
        ("молоко", "Казань"): 89,
        ("хлеб", "Казань"): 45,
        ("яблоки", "Москва"): 120,
    }
    key = (product, city)
    price = prices.get(key, None)
    if price is None:
        return f"Цена для {product} в городе {city} не найдена."
    return f"{price} руб." 

# LLM configuration – use your own OpenAI key via environment variable OPENAI_API_KEY
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Build the agent with function calling enabled
agent = llm.bind_tools([get_price])

# Example prompt – replace with actual user query if needed
prompt = {
    "messages": [
        {"role": "human", "content": "Покажи цены на молоко и хлеб в Казани."}
    ]
}

# Stream the response
stream = agent.stream(prompt, stream_mode=["messages", "updates"])

step = 1

def format_chunk_message(chunk):
    message, meta = chunk
    global step
    if meta.get("langgraph_step") != step:
        step = meta["langgraph_step"]
        print("\n --- --- --- \n")
    if message.content:
        print(message.content, end="", flush=True)

def format_message(message):
    if message.content:
        return message.content
    # If the tool call is finished, display it as a function call string
    if message.tool_calls and len(message.tool_calls) > 0:
        tc = message.tool_calls[0]
        return f"{tc['name']}({tc['args']})"
    return ""

for chunk in stream:
    chunk_type, chunk_data = chunk
    if chunk_type == "messages":
        format_chunk_message(chunk_data)
    elif chunk_type == "updates":
        # When the model finishes a step and may have called a tool
        if chunk_data.get("model"):
            last_msg = chunk_data["model"]["messages"][-1]
            print(format_message(last_msg))

print("\n--- Finished ---")
