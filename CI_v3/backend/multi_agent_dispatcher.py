from typing import Union
import warnings
import os
import json
from langchain import hub

from backend.query_context_resolver import (
    fetch_chat_history_from_mongo,
    get_relevant_docs,
    is_query_relevant_to_docs,
    load_vector_store,
    match_chat_history,
)

from backend.chat import save_chat_to_mongo
from agents import python_tool, react_tool


from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain


load_dotenv()

WORKSPACE_DIR = os.getenv("WORKSPACE_DIR")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("❌ Missing API keys! Set GOOGLE_API_KEY in .env.")

warnings.filterwarnings(
    "ignore", category=UserWarning, message=".*LangSmithMissingAPIKeyWarning.*"
)

# Configure Gemini AI
genai.configure(api_key=GOOGLE_API_KEY)
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")


def handle_doc_query(
    query: str, workspace_name: str, chat_history: list = None
) -> Union[dict, str]:
    if chat_history is None:
        chat_history = []

    # Load chat history from MongoDB
    if chat_history is None:
        chat_history = fetch_chat_history_from_mongo(workspace_name)

    # Load workspace metadata
    workspace_file = os.path.join(WORKSPACE_DIR, f"{workspace_name}.json")
    if not os.path.exists(workspace_file):
        return {"answer": "Workspace does not exist."}

    with open(workspace_file, "r") as f:
        workspace_data = json.load(f)

    index_path = workspace_data.get("index_path")
    if not index_path:
        return {"answer": "Workspace index path is missing."}

    # Set up vector store
    vector_store = vector_store = load_vector_store(index_path)

    retriever = vector_store.as_retriever()

    # Step 1: Check if query is relevant to the docs
    if not is_query_relevant_to_docs(query, retriever):
        return {
            "answer": "Your question doesn't seem related to the project documents."
        }

    # Step 2: Load the LLM + Prompt chain
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")

    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    chat_retriever_chain = create_history_aware_retriever(
        llm, retriever, rephrase_prompt
    )
    retrieval_chain = create_retrieval_chain(chat_retriever_chain, combine_docs_chain)

    # Get response
    response = retrieval_chain.invoke({"input": query, "chat_history": chat_history})

    if "answer" in response:
        save_chat_to_mongo(workspace_name, query, response["answer"])
        return response
    else:
        return {"answer": "No valid response received."}


def handle_idea_generation(prompt: str, workspace_name: str):
    relevant_docs = get_relevant_docs(prompt, workspace_name)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

    doc_context = "\n\n".join([doc.page_content for doc in relevant_docs])
    idea_prompt = f"Here are the project documents:\n{doc_context}\n\nPrompt: {prompt}\n\nGive me upgraded or professional ideas to improve this project."

    return {"answer": llm.invoke(idea_prompt)}


def handle_chat_reference(prompt: str, workspace_name: str):
    history = fetch_chat_history_from_mongo(workspace_name)
    match = match_chat_history(prompt, history)

    if match:
        return {"answer": f"Yes, you mentioned this before:\n\n{match}"}
    else:
        return {
            "answer": "I couldn't find any previous reference to that. Can you clarify?"
        }


def handle_code_generation(prompt: str, workspace_name: str):
    lower_prompt = prompt.lower()

    if "react" in lower_prompt or "component" in lower_prompt or "jsx" in lower_prompt:
        # Clean prompt for better code generation
        cleaned_prompt = prompt.replace("react", "", 1).strip()
        return {"answer": react_tool.func(cleaned_prompt)}

    elif "python" in lower_prompt or "py" in lower_prompt:
        cleaned_prompt = prompt.replace("python", "", 1).strip()
        return {"answer": python_tool.func(cleaned_prompt)}

    else:
        return {
            "answer": (
                "Please specify which code you want to generate: React or Python. "
                "For example: 'Generate a React component that...' or 'Write Python code to...'"
            )
        }


def handle_hybrid_prompt(prompt: str, workspace_name: str):
    history = fetch_chat_history_from_mongo(workspace_name)
    matched_history = match_chat_history(prompt, history)

    relevant_docs = get_relevant_docs(prompt, workspace_name)
    doc_context = "\n\n".join([doc.page_content for doc in relevant_docs])

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

    final_prompt = f"""
Prompt: {prompt}

Related Past Conversation:
{matched_history if matched_history else "None"}

Relevant Docs:
{doc_context if doc_context else "None"}

Generate appropriate Python/React code for this based on the context.
"""

    if "python" in prompt.lower():
        return {"answer": python_tool.func(final_prompt)}
    elif "react" in prompt.lower():
        return {"answer": react_tool.func(final_prompt)}
    else:
        return {"answer": llm.invoke(final_prompt)}
