from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from IPython.display import Image
import config    # Import the config module to load the configuration

class State(TypedDict):
    messages: Annotated[list, add_messages]


if __name__ == "__main__":
    # Create a chat model with OpenAI's GPT-4o model
    chat_openai = ChatOpenAI(model="gpt-4o-mini-2024-07-18")

    # chatbot function
    def chatbot(state: State):
        return {"messages": [chat_openai.invoke(state["messages"])]}

    graph = StateGraph(State)
    graph.add_node("chatbot", chatbot)
    graph.add_edge(START, "chatbot")
    graph.add_edge("chatbot", END)
    compiled_graph = graph.compile()

    # Save the compiled graph as an image
    Image(compiled_graph.get_graph().draw_mermaid_png(output_file_path="chatbot_without_tools.png"))

    while True:
        user_input = input("User: ")

        if user_input.lower() in ["e", "q"]:
            print("Goodbye!")
            break

        for event in compiled_graph.stream({"messages": [("user",user_input)]}):
            for value in event.values():
                print("AI:", value["messages"][-1].content)
