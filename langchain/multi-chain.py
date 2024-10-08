import os
from operator import itemgetter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableParallel

# Set environment variables for LangSmith and OpenAI API keys
def setup_env():
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = "xxxxxxx"
    os.environ["LANGCHAIN_PROJECT"] = "xxxxxxx"
    os.environ["OPENAI_API_KEY"] = "xxxxxxx"

if __name__ == "__main__":
    # Set up environment variables
    setup_env()
    # Initialize a ChatOpenAI model
    llm = ChatOpenAI(model="gpt-4o-mini-2024-07-18")
    # Initialize a StrOutputParser
    str_output_parser = StrOutputParser()

    initializer = (
        ChatPromptTemplate.from_template("You are an senior programmer and please help to analyse the following task:\n {question}")
        | llm
        | str_output_parser
        | {"analysis": RunnablePassthrough()}
    )

    p1_program = (
        ChatPromptTemplate.from_template("please solve the task using Java according to the analysis:\n{analysis} \n\nPlease only provide the code.")
        | llm
        | str_output_parser
    )

    p2_program = (
        ChatPromptTemplate.from_template("please solve the task using Python according to the analysis:\n{analysis} \n\nPlease only provide the code.")
        | llm
        | str_output_parser
    )

    final_chain = (
        ChatPromptTemplate.from_template("You are an senior programmer and please help to analyse the following task:\n {question}")
    )

    chain = (
        initializer
        | RunnableParallel({"output_1": p1_program, "output_2": p2_program})
    )

    results = chain.invoke({"question": "what is fast sort?"})

    for _,value in results.items():
        print(value, end="\n\n=============================\n\n")
