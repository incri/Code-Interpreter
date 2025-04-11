import os

import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI

from backend.multi_agent_dispatcher import (
    handle_chat_reference,
    handle_code_generation,
    handle_doc_query,
    handle_hybrid_prompt,
    handle_idea_generation,
)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("❌ Missing API keys! Set GOOGLE_API_KEY in .env.")

genai.configure(api_key=GOOGLE_API_KEY)


def classify_prompt(prompt: str) -> str:
    system_prompt = """
You are a prompt classifier. Classify the given prompt into one of the following types:
1. 'doc_query' - if the prompt is asking about the uploaded documents.
2. 'idea_gen' - if the prompt asks for improvements, suggestions, or new ideas.
3. 'chat_followup' - if it refers to previous conversations.
4. 'code_gen' - if it asks to generate or fix code.
5. 'hybrid' - if it combines docs, chat history, and code generation.

Return ONLY the label.
"""
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001")
    result = llm.invoke(system_prompt + f"\nPrompt: {prompt}")
    return result


def handle_prompt(prompt: str, workspace_name: str) -> dict:
    prompt_type = classify_prompt(prompt)
    print(f"🧠 Prompt type identified: {prompt_type.content}")

    handler_map = {
        "doc_query": handle_doc_query,
        "idea_gen": handle_idea_generation,
        "chat_followup": handle_chat_reference,
        "code_gen": handle_code_generation,
        "hybrid": handle_hybrid_prompt,
    }

    handler = handler_map.get(prompt_type.content)
    if handler:
        print(f"✅ Dispatching to handler: {handler.__name__}")

        return handler(prompt, workspace_name)

    else:
        return {"answer": "Sorry, I couldn't classify your request."}


def handle_chat(prompt: str, selected_workspace: str) -> str:
    """Handles the chat logic, processes user prompt and retrieves the answer."""
    # Make sure workspace is selected
    if not selected_workspace:
        return "Please select a workspace first."

    # Call handle_prompt to get the bot's response
    response = handle_prompt(prompt, selected_workspace)

    # Check if response has the necessary fields
    if "answer" not in response:
        return "Sorry, there was an issue processing your request."

    # Return the answer without sources
    return response["answer"]
