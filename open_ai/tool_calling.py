# -*-  coding: utf-8  -*-
"""
@Time: 2/7/2025 8:32 am
@Author : jackmanliu@126.com
@File   : tool_calling.py
"""
import io
import json
import os
from openai import OpenAI
import pandas as pd

def multipleVector(data):
    '''
    For tool calling, to multiple vector
    :param data: input data to perform multiplying.It's a string and mandatory
    :return: a JSON format DataFrame object
    '''
    str = io.StringIO(data) #This allows the string to be treated as if it were a file, enabling it to be read by pandas
    input = pd.read_csv(str, sep='\s+', index_col=0)
    result = input * 10
    return json.dumps(result.to_string())

if __name__ == "__main__":
    # Ensure the environment variables are set
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
    openai_model = os.getenv("OPENAI_MODEL","gpt-4o-mini-2024-07-18")

    func_multiple_vector = {
        "type": "function",
        "function": {
            "name": "multipleVector",
            "description": "multiple a vector.",
            "parameters": {
                "type": "object",
                "properties": {
                    "data": {
                        "type": "string",
                        "description": "input data"
                    }
                },
                "required": [
                    "data"
                ],
            },
        }
    }

    tools = [func_multiple_vector]

    available_functions = {
        "multipleVector": multipleVector
    }

    df_str = pd.DataFrame({'a':[6,2], 'b':[3,7]}).to_string()

    messages=[
        {"role":"system", "content":"Here is the data set in string format: %s" % df_str},
        {"role":"user", "content":"please use vector multiply function on this data set."}
    ]

    client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)

    response = client.chat.completions.create(
        model=openai_model,
        messages=messages,
        tools = tools,
        tool_choice="auto"
    )
    print("1. first response message: %s" % response.choices[0].message)
    print("2. first response tool_calls: %s" % response.choices[0].message.tool_calls)

    tool_calls = response.choices[0].message.tool_calls
    if tool_calls:
        messages.append(response.choices[0].message)
        print("3. append first response message to message list: %s" % messages)
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_arguments = json.loads(tool_call.function.arguments)
            function_to_call = available_functions[function_name]
            function_response = function_to_call(**function_arguments) #  ** operator to unpack the dictionary into keyword(data) arguments
            print(" --function name: %s" % function_name)
            print(" --function argumenta: %s" % function_arguments)
            print(" --function response: %s" % function_response)
            messages.append(
                {
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": function_name,
                "content": function_response
                 }
            )
            print("4. Append tool call result to message list: %s" % messages)

        final_response = client.chat.completions.create(
            model=openai_model,
            messages=messages,
        )
        print("5. Final response after tool calling: %s" % final_response.choices[0].message.content)
