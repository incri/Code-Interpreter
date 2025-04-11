from typing import Union
import warnings
import os
import json
from langchain import hub
import time
from backend.query_context_resolver import (
    fetch_chat_history_from_mongo,
    get_relevant_docs,
    is_query_relevant_to_docs,
    load_vector_store,
    match_chat_history,
)

from backend.chat import save_chat_to_mongo
from agents.python_tool import python_tool
from agents.react_tool import react_tool

from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, AgentType


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
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001")
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


def handle_idea_generation(prompt: str, workspace_name: str) -> dict:
    # Step 1: Retrieve relevant documents
    relevant_docs = get_relevant_docs(prompt, workspace_name)

    # Step 2: Initialize the LLM
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001")

    # Step 3: Construct the full context prompt
    doc_context = "\n\n".join([doc.page_content for doc in relevant_docs])
    idea_prompt = (
        f"Here are the project documents:\n{doc_context}\n\n"
        f"Prompt: {prompt}\n\n"
        "Give me upgraded or professional ideas to improve this project."
    )

    # Step 4: Get LLM response
    response = llm.invoke(idea_prompt)

    # Step 5: Save to MongoDB
    if response:
        save_chat_to_mongo(workspace_name, prompt, response.content)

    return {"answer": response}


def handle_chat_reference(prompt: str, workspace_name: str):
    history = fetch_chat_history_from_mongo(workspace_name)
    match = match_chat_history(prompt, history)

    if match:
        response = f"Yes, you mentioned something similar before:\n\n{match}"
    else:
        response = "I couldn't find any previous reference to that. Can you clarify?"

    # Save the chat to MongoDB
    save_chat_to_mongo(workspace_name, prompt, response)

    return {"answer": response}


# Optional: Predefined FAISS code shortcut
def predefined_faiss_code():
    return """
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

def store_chunks_in_faiss(chunks):
    model = SentenceTransformer('all-mpnet-base-v2')
    embeddings = model.encode(chunks)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype('float32'))
    return index, embeddings
"""


# Basic rate limiter for Gemini calls
last_gemini_call_time = 0


def rate_limited_gemini_call(llm, prompt: str, min_delay=1.5):
    global last_gemini_call_time
    elapsed = time.time() - last_gemini_call_time
    memory = ConversationBufferMemory(memory_key="chat_history")
    agent = initialize_agent(
        tools=[python_tool, react_tool],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        memory=memory,
    )
    if elapsed < min_delay:
        time.sleep(min_delay - elapsed)
    last_gemini_call_time = time.time()
    return agent.run(prompt)


def handle_code_generation(prompt: str, workspace_name: str):
    print(f"[Request] Prompt: {prompt}")
    # Gemini LLM and memory setup
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001")

    # Avoid Gemini for predefined code
    if "store chunks in faiss" in prompt.lower() or "faiss and pdf" in prompt.lower():
        print("[Info] Returning predefined FAISS code block.")
        return {"answer": predefined_faiss_code()}

    # Use LangChain tools and agents only when needed
    try:

        response = rate_limited_gemini_call(llm, prompt)
        if response:
            save_chat_to_mongo(workspace_name, prompt, response)
        return {"answer": response}

    except Exception as e:
        return {"answer": f"An error occurred during code generation: {str(e)}"}


def handle_hybrid_prompt(prompt: str, workspace_name: str):
    history = fetch_chat_history_from_mongo(workspace_name)
    matched_history = match_chat_history(prompt, history)

    relevant_docs = get_relevant_docs(prompt, workspace_name)
    doc_context = "\n\n".join([doc.page_content for doc in relevant_docs])

    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001")

    final_prompt = f"""
    Prompt: {prompt}

    Related Past Conversation:
    {matched_history if matched_history else "None"}

    Relevant Docs:
    {doc_context if doc_context else "None"}

    Generate appropriate Python/React code for this based on the context.
    """

    # Choose the right call method based on the type of request
    if any(keyword in prompt.lower() for keyword in ["python", "react"]):
        answer = rate_limited_gemini_call(llm, final_prompt)
    else:
        answer = llm.invoke(final_prompt)

    save_chat_to_mongo(workspace_name, prompt, answer)
    return {"answer": answer}
