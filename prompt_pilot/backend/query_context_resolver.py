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
    if workspace_chat:
        return workspace_chat["chat_history"]
    return []  # Return an empty list if no history exists


def match_chat_history(prompt: str, chat_history: list):
    prompt_lower = prompt.lower()

    # Look through user messages in the last 10 entries
    for i in range(len(chat_history) - 1, -1, -1):
        chat = chat_history[i]
        if chat["role"] == "user":
            user_message = chat["content"]
            if any(word in prompt_lower for word in user_message.lower().split()):
                # Try to find the corresponding assistant reply
                if (
                    i + 1 < len(chat_history)
                    and chat_history[i + 1]["role"] == "assistant"
                ):
                    assistant_message = chat_history[i + 1]["content"]
                else:
                    assistant_message = "(No recorded response)"

                return (
                    f"User said: {user_message}\nAssistant replied: {assistant_message}"
                )

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
