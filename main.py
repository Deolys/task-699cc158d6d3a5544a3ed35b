import os
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, Tool, AgentType

# Define a simple echo tool that will be called by the agent

def echo_function(input_text: str) -> str:
    """Return the input text prefixed with 'Echo:'"""
    return f"Echo: {input_text}"

echo_tool = Tool(
    name="echo",
    description="Echoes back the provided input.",
    func=echo_function,
)

# Initialize LLM with streaming enabled
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, streaming=True)

# Create an agent executor that can use tools via OpenAI function calling
agent_executor = initialize_agent(
    [echo_tool],
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=False,
)

# Helper to format messages from the stream
step_counter = 1

def format_chunk_message(chunk):
    global step_counter
    message, meta = chunk
    if meta.get("langgraph_step") != step_counter:
        step_counter = meta["langgraph_step"]
        print("\n --- --- --- \n")
    if message.content:
        print(message.content, end="", flush=True)

# Helper to format a finished message (used for tool calls)
def format_message(message):
    if message.content:
        return message.content
    # If the message contains a tool call, display it nicely
    if hasattr(message, "tool_calls") and message.tool_calls:
        tc = message.tool_calls[0]
        return f"{tc.name}({tc.args})"
    return ""

if __name__ == "__main__":
    # Example user prompt that will trigger the echo tool
    user_prompt = "Hello, world!"
    stream = agent_executor.stream(
        {"input": user_prompt},
        stream_mode=["messages", "updates"],
    )

    for chunk in stream:
        chunk_type, chunk_data = chunk
        if chunk_type == "messages":
            format_chunk_message(chunk_data)
        elif chunk_type == "updates":
            # When the model finishes a step and may have called a tool
            if chunk_data.get("model"):
                last_msg = chunk_data["model"]["messages"][-1]
                print(format_message(last_msg))
    print()  # Ensure final newline
