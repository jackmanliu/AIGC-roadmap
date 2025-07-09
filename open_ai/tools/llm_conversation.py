# -*-  coding: utf-8  -*-
"""
@Time   : 9/7/2025 11:03 am
@Author : jackmanliu@126.com
@File   : llm_conversation.py
"""
import json
import os
from openai import OpenAI
from open_ai.tools.function_calliing_generaion import function_calling_gen


def proceed_conversation(messages, tool_list=None):
    """
    Proceed the conversation with the given messages and tool list.
    :param messages: List of messages(dict) in the conversation.
    :param tool_list: List of tools available for function calling.
    :return: The response from the LLM.
    """
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini-2024-07-18")
    client = OpenAI(api_key=openai_api_key, base_url=openai_base_url)
    if tool_list is None:
        response = client.chat.completions.create(
            model=openai_model,
            messages=messages
        )
        # result_message = response.choices[0].message
        result_content = response.choices[0].message.content
    else:
        functions_json_shema_list = function_calling_gen(tool_list)
        functions_list = {func.__name__: func for func in tool_list}
        response = client.chat.completions.create(
            model=openai_model,
            messages=messages,
            tools=functions_json_shema_list,
            tool_choice="auto"
        )
        tool_calls = response.choices[0].message.tool_calls
        result_message = response.choices[0].message
        if tool_calls:
            messages.append(result_message)
            for tool_call in tool_calls:
                func_name = tool_call.function.name
                func_params = json.loads(tool_call.function.arguments)
                func_to_call = functions_list[func_name]
                func_response = func_to_call(**func_params)
                messages.append(
                    {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": func_name,
                        "content": func_response
                    }
                )
        final_response = client.chat.completions.create(
            model=openai_model,
            messages=messages
        )
        result_content = final_response.choices[0].message.content
    return result_content

