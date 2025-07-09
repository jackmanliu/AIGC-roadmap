# -*-  coding: utf-8  -*-
"""
@Time   : 1/7/2025 8:33 am
@Author : jackmanliu@126.com
@File   : chat_completion.py
"""
from openai import OpenAI
import openai
import os
import json


if __name__ == "__main__":
    # Ensure the environment variables are set
    openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini-2024-07-18")

    # roles: system, user, assistant
    # system: provides context or instructions for the conversation
    # user: the person interacting with the model
    # assistant: the model's responses

    # 1. chat completion
    client = OpenAI(api_key=openai.api_key, base_url=openai.api_base)

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": "What is the capital of France?"},
        ]
    )
    print(json.dumps(completion.model_dump(), indent=2))
    print(completion.choices[0].message.content)

    # 2. chat completion - streaming
    stream_completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": "what is thee capital of France? and what is the history about this city?"}
        ],
        stream=True
    )
    # print(json.dumps(stream_completion.model_dump), indent=2)
    for chunk in stream_completion:
        if chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content, end='', flush=True)

    # 3. chat completion - response in JSON format
    #    the content of the messages must include the word "json" in some form. Or, use system prompt to indicate the JSON format.
    response = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages = [
            ## 1. use system prompt to indicate the JSON format
            # {"role": "system", "content": "you are a helpful assistant that always reponse in JSON format."},
            # {"role": "user", "content": "What is the capital of France?"}
            # 2. mention JSON format in the user message
            {"role": "user", "content": "What is the capital of France? please return the response in JSON format."}
        ]
    )
    print(response)

    # 4. can include multiple user messages in the messages block, but only the last message will be answered
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "user", "content": "What is the capital of Italy?"}
        ]
    )
    print(response.choices[0].message.content)

    # 5. few shot learning
    Q1 = "I have 3 apples. I bought 3 bags of apple and each bag has 5 apples. How many apples do I have in total?"
    A1 = "You have 18 apples in total."
    Q2 = "I have 2 balloons. I released 1 balloon and bought 2 balloons. How many balloons do I have in total?"
    response = client.chat.completions.create(
        model=model,
        messages = [
            # 1. put the few-shot examples in the system prompt
            # {"role": "system", "content": "Q: " + Q1 + " A: " + A1},
            # {"role": "user", "content": "Q: " + Q2},
            # 2. or put the few-shot examples in the user messages
            {"role": "user", "content": Q1},
            {"role": "assistant", "content": A1},
            {"role": "user", "content": Q2}
        ]
    )
    print(response.choices[0].message.content)

    # 6. zero-shot COT
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Please think following question step by step."},
            {"role": "user", "content": Q1}
        ]
    )
    print(response.choices[0].message.content)

     # 7. knowledge base in context
    knowledge_base = "Jerry is a project manager at OpenAI. He has a pet cat named Whiskers. Whiskers is very playful and loves to chase laser pointers."
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": knowledge_base},
            {"role": "user", "content": "does Jerry have a pet? If so, what is the name of his pet?"}
        ]
    )
    print(response.choices[0].message.content)
