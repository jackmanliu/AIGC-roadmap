import json
from typing import Annotated
from typing_extensions import TypedDict, Literal
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.graph.message import add_messages
from langchain_core.messages import ToolMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from IPython.display import Image
import config    # Import the config module to load the configuration

class State(TypedDict):
    messages: Annotated[list, add_messages]

class BasicToolNode:

    def __init__(self, tools: list):
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No messages in inputs")

        output = []

        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(tool_call["args"])
            output.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"]
                )
            )

        return {"messages": output}


if __name__ == "__main__":
    # Create a chat model with OpenAI's GPT-4o model
    chat_openai = ChatOpenAI(model="gpt-4o-mini-2024-07-18")

    # chatbot function
    def chatbot(state: State):
        return {"messages": [chat_openai.invoke(state["messages"])]}

    # search tool
    tavily_search = TavilySearchResults(max_results=2)
    tools = [tavily_search]

    # bind the tools to the chat model
    chat_openai_with_tools = chat_openai.bind_tools(tools)

    # chatbot function with tools
    def chatbot_with_tools(state: State):
        return {"messages": [chat_openai_with_tools.invoke(state["messages"])]}

    # Create a basic tool node
    tool_node = BasicToolNode(tools=tools)

    # route function
    def route_tools(state) -> Literal["tools", "__end__"]:
        if isinstance(state, list):
            ai_message = state[-1]
        elif messages := state.get("messages"):
            ai_message = messages[-1]
        else:
            raise ValueError(f"No messages in {state}")

        if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
            return "tools"
        return "__end__"

    graph = StateGraph(State)
    graph.add_node("chatbot", chatbot_with_tools)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "chatbot")
    graph.add_conditional_edges("chatbot", route_tools, {"tools": "tools", "__end__": "__end__"})
    graph.add_edge("tools", "chatbot")
    compiled_graph = graph.compile()

    # Draw the graph
    Image(compiled_graph.get_graph().draw_mermaid_png(output_file_path="chatbot_with_tools.png"))

    while True:
        user_input = input("User:")

        if user_input.lower() in ["e", "q"]:
            print("Goodbye!")
            break

        for event in compiled_graph.stream({"messages": [("user",user_input)]}):
            for value in event.values():
                if isinstance(value["messages"][-1], BaseMessage):
                    print("AI:", value["messages"][-1].content)
