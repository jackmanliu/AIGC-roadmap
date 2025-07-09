# -*-  coding: utf-8  -*-
"""
@Time   : 9/7/2025 3:05 pm
@Author : jackmanliu@126.com
@File   : test_func.py
"""
import json
import os
import pandas as pd
from openai import OpenAI
from open_ai.tool_calling import multipleVector
from open_ai.tools.function_calliing_generaion import function_calling_gen
from open_ai.tools.llm_conversation import proceed_conversation

if __name__ == "__main__":

    func_list = [multipleVector]
    df_str = pd.DataFrame({'a': [6, 2], 'b': [3, 7]}).to_string()
    messages = [
        {"role": "system", "content": "Here is the data set in string format: %s" % df_str},
        {"role": "user", "content": "please use vector multiply function on this data set."}
    ]

    # test function_calling_generation.py
    print("======== function_calling_gen =========")
    max_try = 4
    num_attamps = 0
    while num_attamps < max_try:
        try:
            func_schema_list = function_calling_gen(func_list)
            break
        except Exception as e:
            num_attamps += 1
            print("error occurs: ", e)
            if num_attamps == max_try:
                print("reached max attampts... terminated")
                raise
            else:
                print("re-running...")

    for func_schema in func_schema_list:
        print(json.dumps(func_schema, indent=2))

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    response = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini-2024-07-18"),
        messages=messages,
        tools=func_schema_list,
        tool_choice="auto"
    )
    print(response.choices[0].message)

    # test llm_conversation.py
    print("======== proceed_conversation =========")
    func_list = [multipleVector]
    resp = proceed_conversation(messages, func_list)
    print(resp)