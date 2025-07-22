# -*-  coding: utf-8  -*-
"""
@Time   : 13/7/2025 5:58 pm
@Author : jackmanliu@126.com
@File   : glm3_chat_completion.py
"""

import os
from transformers import AutoTokenizer, AutoModel
from openai import OpenAI
import requests
import json


def get_weather(loc):
    return json.dumps({"coord": {"lon": 116.3972, "lat": 39.9075}, "weather": [{"id": 801, "main": "Clouds", "description": "\u6674\uff0c\u5c11\u4e91", "icon": "02d"}], "base": "stations", "main": {"temp": 15.94, "feels_like": 14.09, "temp_min": 15.94, "temp_max": 15.94, "pressure": 1011, "humidity": 19, "sea_level": 1011, "grnd_level": 1005}, "visibility": 10000, "wind": {"speed": 4.98, "deg": 299, "gust": 8.94}, "clouds": {"all": 12}, "dt": 1700634535, "sys": {"type": 1, "id": 9609, "country": "CN", "sunrise": 1700607993, "sunset": 1700643269}, "timezone": 28800, "id": 1816670, "name": "Beijing", "cod": 200})


""" Demo 1
     1. git clone https://github.com/THUDM/ChatGLM3.git
     2. cd ChatGLM3
     3. pip install -r requirements.txt
"""
def zhipu_api_function_call():
    # model path
    pretrained_model = os.getenv("PRETRAIN_MODEL","THUDM/chatglm3-6b")
    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model, trust_remote_code=True)
    # model - FP16
    model_fp16 = AutoModel.from_pretrained(pretrained_model, trust_remote_code=True, device='cuda')
    # # model - INT8
    # model_int8 = AutoModel.from_pretrained(pretrained_model, trust_remote_code=True).quantize(8).cuda()
    # # model - INT4
    # model_int4 = AutoModel.from_pretrained(pretrained_model, trust_remote_code=True).quantize(4).cuda()

    # set to evaluation mode
    model_fp16 = model_fp16.eval()

   # function list
    function_list = [get_weather]

    weather_api = [
        {
            'name': 'get_weather',
            'description': 'get the current weather report according to the city name',
            'parameters': {
                'type': 'object',
                'properties': {
                    'loc': {
                        'description': "city name. For example，Beijing or Sydney",
                        'type': 'string',
                        'required': True
                    }
                }
            }
        }
    ]

    system_msg = {
        "role": "system",
        "content": "Answer below questions and you have access to the following tools: ",
        "tools": weather_api
    }

    history_msg = [system_msg]

    question = "What is the weather like in Beijing?"

    first_response, first_chat_history = model_fp16.chat(tokenizer, question, history=history_msg)

    print(f"first_response: {first_response}")
    print(f"first_chat_history: {first_chat_history}")
    print("==============")

    availale_funcntions = {func.__name__: func for func in function_list}
    function_name = first_response["name"]
    function_to_call = availale_funcntions[function_name]
    function_parameters = first_response["parameters"]
    function_response = function_to_call(**function_parameters)
    print(function_response)

    history_msg = []
    history_msg.append(
        {
            "role": "observation",
            "name": function_name,
            "content": function_response
        }
    )
    second_response, second_chat_history = model_fp16.chat(tokenizer, question, history_msg)
    print(f"second_response: {second_response}")
    print(f"second_chat_history{second_chat_history}")

""" Demo 2
    Enables distributed training and efficient model loading across multiple GPUs:
    when your model is too large to fit on a single GPU's memory, it allows you to split the model across multiple GPUs automatically.
    This helps in scenarios where each individual GPU has insufficient memory to hold the complete model.

    installs the Hugging Face Accelerate library:
      pip install accelerate
"""

"""Demo 3
    Usig OpenAI api:
      cd openai_api_demo
      python api_server.py
    API Endpoints:
    - "/v1/models": Lists the available models, specifically ChatGLM3-6B.
    - "/v1/chat/completions": Processes chat completion requests with options for streaming and regular responses.
"""
def openai_api_function_call():
    base_url = "http://127.0.0.1:8000/v1/"
    client = OpenAI(api_key="EMPTY", base_url=base_url)

    messages = [
        {"role":"user", "content":"What's the weather like in Shanghai, Sydney, and Wellington?"}
    ]

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        }
    ]

    response = client.chat.completions.create(
        model="chatglm3-6b",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )
    if response:
        content = response.choices[0].message.content
        print(content)
    else:
        print(response.status_code)


def openai_api_url():
    base_url = "http://127.0.0.1:8000/v1/chat/completions"
    model = "chatglm3-6b"
    messages = [
        {
            "role": "system",
            "content": "You are ChatGLM3, a large language model trained by Zhipu.AI. Follow the user's instructions carefully. Respond using markdown.",
        },
        {
            "role": "user",
            "content": "tell a story which is about 100 words"
        }
    ]

    data = {
        "model": model,
        "messages": messages,
        "max_tokens": 100,
        "temperature": 0.8,
        "top_p": 0.8
    }

    response = requests.post(base_url, json=data)
    decode_line = response.json()
    content = decode_line.get("choices",[{}])[0].get("message",{}).get("content", "")
    print(content)


if __name__ == "__main__":

    print("======= Demo 1 =======")
    zhipu_api_function_call()

    print("======= Demo 3 =======")
    openai_api_function_call()

    openai_api_url()



