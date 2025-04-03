import time
from helper.db import chat_histories


def save_chat_to_mongo(workspace_name: str, user_prompt: str, bot_response: str):
    """Saves chat history in MongoDB for the specific workspace."""

    # Format messages to align with LangChain's expectations (role and content)
    chat_message_user = {
        "role": "user",  # 'role' is user
        "content": user_prompt,  # 'content' is the user's message
        "timestamp": time.time(),
    }

    chat_message_bot = {
        "role": "assistant",  # 'role' is assistant
        "content": bot_response,  # 'content' is the bot's response
        "timestamp": time.time(),
    }

    # Check if the workspace already has a chat history document
    existing_chat = chat_histories.find_one({"workspace_name": workspace_name})

    if existing_chat:
        # If exists, update the chat history (append new messages)
        chat_histories.update_one(
            {"workspace_name": workspace_name},
            {
                "$push": {
                    "chat_history": chat_message_user,  # Save user message with role and content
                },
                "$set": {"last_updated": time.time()},
            },
        )

        # Also append the assistant's response to the history
        chat_histories.update_one(
            {"workspace_name": workspace_name},
            {
                "$push": {
                    "chat_history": chat_message_bot,  # Save bot message with role and content
                },
            },
        )

    else:
        # If the workspace doesn't exist, create a new document
        chat_histories.insert_one(
            {
                "workspace_name": workspace_name,
                "chat_history": [
                    chat_message_user,
                    chat_message_bot,
                ],  # Store both user and bot messages as an array of objects
                "last_updated": time.time(),
            }
        )
