import os
import json
from functools import lru_cache

from helper.db import chat_histories

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

WORKSPACE_DIR = os.getenv("WORKSPACE_DIR")


embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")


@lru_cache(maxsize=10)
def load_vector_store(index_path: str):
    return FAISS.load_local(
        index_path, embeddings, allow_dangerous_deserialization=True
    )


# Step 1: Check if the query is relevant to the documents
def is_query_relevant_to_docs(query: str, retriever, threshold: float = 0.75) -> bool:
    results = retriever.get_relevant_documents(query)
    # If any document has a similarity score above threshold, we consider it relevant
    return len(results) > 0


# Step 2: Load chat history for the workspace
def fetch_chat_history_from_mongo(workspace_name: str):
    """Fetches the chat history from MongoDB for the specific workspace."""
    workspace_chat = chat_histories.find_one({"workspace_name": workspace_name})
    print(workspace_chat)
    if workspace_chat:
        print(workspace_chat["chat_"])

        return workspace_chat["chat_history"]
    return []  # Return an empty list if no history exists


def match_chat_history(prompt: str, chat_history: list):
    # Optional: use embedding similarity, or just keyword match
    for chat in reversed(chat_history[-10:]):  # recent 10 messages
        if any(kw in prompt.lower() for kw in chat["prompt"].lower().split()):
            return chat["prompt"] + "\n" + chat["response"]
    return None


def get_relevant_docs(query: str, workspace_name: str):
    workspace_file = os.path.join(WORKSPACE_DIR, f"{workspace_name}.json")
    with open(workspace_file, "r") as f:
        data = json.load(f)

    vector_store = FAISS.load_local(
        data["index_path"], embeddings, allow_dangerous_deserialization=True
    )
    retriever = vector_store.as_retriever()
    return retriever.get_relevant_documents(query)
