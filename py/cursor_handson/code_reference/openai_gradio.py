# -*-  coding: utf-8  -*-
"""
@Time   : 29/10/2024 5:31 pm
@Author : jackmanliu@126.com
@File   : openai_gradio.py
"""

from openai import OpenAI
import gradio as gr


# Initialize OpenAI client with API key and base URL
client = OpenAI(
    api_key="xxxxxxxx",
    base_url="https://api.openai.com/v1"
)


def predict(message, history):
    """
    Process chat messages and generate AI responses
    Args:
        message: Current user input message
        history: List of previous conversation turns
    Returns:
        Generator yielding progressive response chunks
    """
    try:
        # Convert chat history to OpenAI's expected format
        history_openai_format = []
        for human, ai in history:
            history_openai_format.append({"role": "user", "content": human})
            history_openai_format.append({"role": "assistant", "content": ai})
        # Add current message to history
        history_openai_format.append({"role": "user", "content": message})

        try:
            # Make API call to OpenAI with streaming enabled
            response = client.chat.completions.create(
                model='gpt-4o-2024-08-06',
                messages=history_openai_format,
                temperature=1.0,  # Higher temperature means more creative/random responses
                stream=True       # Enable streaming for progressive response
            )

            # Process streaming response chunks
            partial_message = ""
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    partial_message += chunk.choices[0].delta.content
                    yield partial_message
        except Exception as api_error:
            yield f"API Error: {str(api_error)}"
    except Exception as format_error:
        yield f"Format Error: {str(format_error)}"


# Create and configure the Gradio chat interface
iface = gr.ChatInterface(
    predict,                     # Main prediction function
    chatbot=gr.Chatbot(height=500),  # Chat display component
    title="Openai Chat - Multi-turn Conversation",
    description="Chat with the gpt-4o AI model. Your conversation history will be maintained.",
    theme="soft",
    examples=["Hello, how are you?", "What's the weather like today?"],  # Example prompts
)

# Launch the web interface
iface.launch()
