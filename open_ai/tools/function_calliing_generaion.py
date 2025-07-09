# -*-  coding: utf-8  -*-
"""
@Time: 3/7/2025 8:10 am
@Author : jackmanliu@126.com
@File   : function_calliing_generaion.py
"""
import os
from openai import OpenAI
import inspect
import json

def function_calling_gen(functions_list):
    """
    to generate function calling JSON schema
    :param functions: function list that is used for function calling,
                      make sure each function has a description for its usage and parameters
    :return: a list of JSON schema
    """
    # Ensure the environment variables are set
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini-2024-07-18")

    client = OpenAI(api_key=openai_api_key, base_url=openai_base_url)
    # to store function JSON schema
    functions_json_schema_list = []

    for func in functions_list:
        func_description = inspect.getdoc(func)
        func_name = func.__name__
        json_schema_prompt = ("""Please help me generate the JSON schema for function calling according to function's description with below structure:
                                     1. The JSON schema has three key value pair
                                     2. The first key is 'name', value is function's name: %s
                                     3. The second key is 'description', value is the description of the function
                                     4. The third key is 'parameters', value is a json schema that has parameters information
                                     5. Please include only JSON schema in the result not anything else""" % func_name)
        ai_response = client.chat.completions.create(
                            model=openai_model,
                            messages=[
                                {"role":"system", "content":"Here is a function's description: %s" % func_description},
                                {"role":"user", "content": json_schema_prompt}
                            ]
                       )
        # remove "```" and "json" from the Markdown format
        func_schema = ai_response.choices[0].message.content.replace("```","").replace("json","")
        json_schema = {"type": "function", "function": json.loads(func_schema)}
        functions_json_schema_list.append(json_schema)
    return functions_json_schema_list