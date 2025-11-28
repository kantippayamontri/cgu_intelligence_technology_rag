# 1. define tools and model
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from dotenv import load_dotenv

load_dotenv()

model = init_chat_model("gemini-2.5-flash",
                        model_provider="google_genai", temperature=0)

# Define tools
@tool
def multiply(a: int, b: int):
    """Multiply two integers and return the product."""
    return a * b


@tool
def add(a: int, b: int):
    """Add two integers and return the sum."""
    return a + b


@tool
def divide(a: int, b: int):
    """Divide a by b and return the quotient."""
    return a / b

# Augment the LLM with tools
tools = [add, multiply, divide]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools)

# 2. Define state
"""
the graph'state is used to store the message and llm_calls 
"""

from langchain.messages import AnyMessage
from typing_extensions import TypedDict, Annotated
import operator

class MessageState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int

# 3. Define model node
from langchain.messages import SystemMessage

def llm_call(state: dict):
    """LLM decide whether to call a tool or not"""
    return {
        "messages": [
            model_with_tools.invoke(
                [
                    SystemMessage(
                        content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
                    )
                ]
                + state["messages"]
            )
        ],
        "llm_calls": state.get('llm_calls', 0) + 1
    }

# 4. define tool node
# tool node is used to call the tools and return the results
from langchain.messages import ToolMessage

def tool_node(state: dict):
    """Performs the tool call"""
    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]) )
    return {"messages": result}

# 5. Define end logic
# the condition edge function is used to route to the tool node or end based upon whether the LLM made a tool call.

from typing import Literal
from langgraph.graph import StateGraph, START,END

def should_continue(state: MessageState) -> Literal["tool_node", END]:
    """decide if we should continue the loop or stop based upon wheter the LLM made a tool call"""
    messages = state["messages"]
    last_message = messages[-1]

    # if the LLM makes a tool call, then perform an action
    if last_message.tool_calls:
        return "tool_node"
    
    # Otherwise, we stop (reply to the user)
    return END

# 6. Build and compile the agent
# the agent is built using the `StateGraph` class and compiled using the `compile` method

# build workflow
agent_builder = StateGraph(MessageState)

# add nodes
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)

# add edges to connect nodes
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    ["tool_node", END]
)
agent_builder.add_edge("tool_node", "llm_call")

# Compile the agent
agent = agent_builder.compile()

# Show the agent
png_bytes = agent.get_graph(xray=True).draw_mermaid_png()

# Save to file
with open("agent_graph.png", "wb") as f:
    f.write(png_bytes)
# from IPython.display import Image, display
# display(Image(agent.get_graph(xray=True).draw_mermaid_png()))

# Invoke
from langchain.messages import HumanMessage
messages = [HumanMessage(content="Add 3 and 4")]
messages = agent.invoke({"messages": messages})
for m in messages["messages"]:
    m.pretty_print()
