
import os
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langsmith.client import Client
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from bs4 import SoupStrainer
from langchain_community.document_loaders import WebBaseLoader

# Set environment variables for LangSmith and OpenAI API keys
def setup_env():
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = "xxxxxxx"
    os.environ["LANGCHAIN_PROJECT"] = "xxxxx"
    os.environ["OPENAI_API_KEY"] = "xxxxxx"

def web_crawler(url: str) -> list:
    # Create a SoupStrainer to filter specific HTML elements
    strainer = SoupStrainer(class_=("mb-5 lg:mb-7.5 lg:mx-25","rich-text-body styles-module--richtext-wrapper--01207 lg:mx-25"))
    # Initialize a WebBaseLoader with the given URL and SoupStrainer
    loader = WebBaseLoader(
        web_path=url,
        bs_kwargs={"parse_only": strainer}
    )
    # Load and return the documents from the web page
    docs = loader.load()
    return docs

def list_retrieved_docs(retriever, query):
    print("====== Retrieved docs ======")
    retrieved_docs = retriever.invoke(query)
    print("there are ", len(retrieved_docs), " docs")
    for doc in retrieved_docs:
        print(doc.page_content, end="\n")

def format_docs(docs) -> str:
    # Join the content of all documents with double newlines
    return "\n\n".join(doc.page_content for doc in docs)

if __name__ == "__main__":
    # Set up environment variables
    setup_env()
    # Crawl the web page and retrieve documents
    docs = web_crawler("https://www.qlik.com/us/augmented-analytics/big-data-ai")
    # Initialize a text splitter with specific chunk size and overlap
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
    # Split the documents into smaller chunks
    all_splits = text_splitter.split_documents(docs)
    # Create a Chroma vector store from the document splits
    vector_store = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())
    # Create a retriever from the vector store with similarity search
    retriever = vector_store.as_retriever(search_type="similarity", top_k=3)
    # Initialize a ChatOpenAI model
    llm = ChatOpenAI(model="gpt-4o-mini-2024-07-18")
    # Initialize a LangSmith client
    langsmith_client = Client()
    # Pull a prompt from LangSmith
    qa_prompt = langsmith_client.pull_prompt("rlm/rag-prompt")
    # Initialize an output parser
    output_parser = StrOutputParser()

    # Test document retrieval
    list_retrieved_docs(retriever, "What is AI?")

    # Create a RAG chain with context, question, prompt, LLM, and output parser
    rag_chain = (
            {"context": retriever | format_docs,
             "question": RunnablePassthrough()
             }
            | qa_prompt
            | llm
            | output_parser
    )

    print("====== Answer ======")
    for chunk in rag_chain.stream("What is AI?"):
        print(chunk, end="", flush=True)



