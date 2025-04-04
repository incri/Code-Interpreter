import os
import json
import warnings
from typing import Union

from helper.db import chat_histories
from backend.chat import save_chat_to_mongo

from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain import hub
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS

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


# Step 1: Check if the query is relevant to the documents
def is_query_relevant_to_docs(query: str, retriever, threshold: float = 0.75) -> bool:
    results = retriever.get_relevant_documents(query)
    # If any document has a similarity score above threshold, we consider it relevant
    return len(results) > 0


# Step 2: Load chat history for the workspace
def fetch_chat_history_from_mongo(workspace_name: str):
    """Fetches the chat history from MongoDB for the specific workspace."""
    workspace_chat = chat_histories.find_one({"workspace_name": workspace_name})
    if workspace_chat:
        return workspace_chat["chat_history"]
    return []  # Return an empty list if no history exists


# Main function
def retrieve_answer(
    query: str, workspace_name: str, chat_history: list = None
) -> Union[dict, str]:
    if chat_history is None:
        chat_history = []

    # Load chat history from MongoDB
    chat_history = fetch_chat_history_from_mongo(workspace_name)

    # Load workspace metadata
    workspace_file = os.path.join(WORKSPACE_DIR, f"{workspace_name}.json")
    if not os.path.exists(workspace_file):
        return "Workspace does not exist."

    with open(workspace_file, "r") as f:
        workspace_data = json.load(f)

    index_path = workspace_data.get("index_path")
    if not index_path:
        return {"answer": "Workspace index path is missing."}

    # Set up vector store
    vector_store = FAISS.load_local(
        index_path, embeddings, allow_dangerous_deserialization=True
    )
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


def handle_chat(prompt: str, selected_workspace: str) -> str:
    """Handles the chat logic, processes user prompt and retrieves the answer."""
    # Make sure workspace is selected
    if not selected_workspace:
        return "Please select a workspace first."

    # Retrieve previous chat history from MongoDB
    chat_history = fetch_chat_history_from_mongo(selected_workspace)

    # Call retrieve_answer function to get the bot's response
    response = retrieve_answer(
        query=prompt,
        workspace_name=selected_workspace,
        chat_history=chat_history,
    )

    # Format the response with source links
    sources = {doc.metadata["source"] for doc in response["context"]}
    formatted_response = f"{response['answer']} \n\n 📌 **Sources:**\n" + "\n".join(
        f"{i+1}. {src}" for i, src in enumerate(sorted(list(sources)))
    )

    return formatted_response
