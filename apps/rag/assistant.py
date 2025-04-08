from typing import Callable, Sequence, TypedDict, Annotated
from langchain_ollama import ChatOllama
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition


# Generate the AgentState and Agent graph
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def create_assistant_with_tools(tools: Sequence[Callable]):
    """
    Create an Agent with tools using 'qwen2.5-coder:7b-instruct'
    Agent can iterate between tools and message to return more information

    :param tools: Sequence[Callable]: Array of function tools
    """
    chat = ChatOllama(model="qwen2.5-coder:7b-instruct", verbose=True, temperature=1)
    chat_with_tools = chat.bind_tools(tools)

    def assistant(state: AgentState):
        return {
            "messages": [chat_with_tools.invoke(state["messages"])],
        }
    
    ## The graph
    builder = StateGraph(AgentState)

    # Define nodes: these do the work
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))

    # Define edges: these determine how the control flow moves
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges(
        "assistant",
        # If the latest message requires a tool, route to tools
        # Otherwise, provide a direct response
        tools_condition,
    )
    builder.add_edge("tools", "assistant")
    return builder.compile()