import streamlit as st
import os
from backend.query_context_resolver import fetch_chat_history_from_mongo
from backend.prompt_controller import handle_chat
from backend.vector_store import create_workspace, list_workspaces, ingest_pdfs


def main():
    st.title("PromptPilot")

    # Sidebar for workspace management
    st.sidebar.header("Workspace Management")
    workspace_name = st.sidebar.text_input("Enter workspace name:")

    # Creating a workspace
    if st.sidebar.button("Create Workspace"):
        if workspace_name:
            create_workspace(workspace_name)
            st.sidebar.success(f"Workspace '{workspace_name}' created successfully.")
        else:
            st.sidebar.error("Please enter a workspace name.")

    # List and select existing workspaces
    st.sidebar.subheader("Existing Workspaces")
    workspaces = list_workspaces()  # Your function to list workspaces
    selected_workspace = st.sidebar.selectbox(
        "Select a workspace:", ["-- Select --"] + workspaces
    )

    # Clear the input box when workspace is switched
    if "last_workspace" not in st.session_state:
        st.session_state["last_workspace"] = None

    if selected_workspace != st.session_state["last_workspace"]:
        st.session_state["user_prompt_history"] = []
        st.session_state["chat_answer_history"] = []
        st.session_state["last_workspace"] = selected_workspace
        st.session_state["chat_fetched_from_mongo"] = False  # Reset flag
        st.rerun()  # Refresh the page on workspace switch

    # Proceed only if a workspace is selected
    if selected_workspace and selected_workspace != "-- Select --":
        # Handle PDF uploads and indexing for the selected workspace
        """Your AI-Powered Workspace Assistant for Smarter Prompts, Code, and Conversations."""
    if selected_workspace and selected_workspace != "-- Select --":
        st.subheader(f"Upload PDFs to Workspace: {selected_workspace}")
        uploaded_files = st.file_uploader(
            "Upload PDFs", type=["pdf"], accept_multiple_files=True
        )

        if st.button("Process PDFs"):
            if uploaded_files:
                save_paths = []
                os.makedirs("./uploads", exist_ok=True)
                for uploaded_file in uploaded_files:
                    file_path = os.path.join("./uploads", uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    save_paths.append(file_path)

                ingest_pdfs(selected_workspace, save_paths)
                st.success("PDFs processed and stored in vector database.")
            else:
                st.error("Please upload at least one PDF.")

        # Bot section - display below the workspace management section
        st.subheader(f"Helping Bot for Workspace: {selected_workspace}")
        st.markdown("**Ask questions related to the PDFs in this workspace.**")

        # Initialize session state for chat history if not already initialized
        if "user_prompt_history" not in st.session_state:
            st.session_state["user_prompt_history"] = []
        if "chat_answer_history" not in st.session_state:
            st.session_state["chat_answer_history"] = []

        with st.form(
            key="chat_input_form",
            clear_on_submit=True,
        ):
            prompt = st.text_input(
                "💬 **Ask me anything:**", placeholder="Type your question here..."
            )
            submit_button = st.form_submit_button("Send")

        st.markdown(
            """
        <style>
            [data-testid="stForm"] {
            position: fixed;
            bottom: 0px;
            left: 60%;
            transform: translateX(-50%);
            width: 800px;
            max-width: 100%;
            background-color: #ffffff;
            padding: 15px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            z-index:1000;
            }
            [data-testid="stTextInput"] {
            width: 765px;
            max-width: 100%;
            }
        </style>
        """,
            unsafe_allow_html=True,
        )

        # Handle submission and call the chat function
        if submit_button:
            if prompt.strip() == "":
                st.error("Please enter a valid question!")
            else:
                with st.spinner("Thinking..."):
                    try:
                        # Call the helper_bot to handle chat
                        response = handle_chat(prompt, selected_workspace)

                        # Append the new chat to session_state
                        st.session_state["user_prompt_history"].append(prompt)
                        st.session_state["chat_answer_history"].append(response)
                        st.session_state.prompt = ""  # Clear the form after submission
                    except Exception as e:
                        st.error(f"An error occurred: {e}")

        # If the workspace history is not fetched yet, fetch from MongoDB
        if not st.session_state.get("chat_fetched_from_mongo", False):
            chat_history = fetch_chat_history_from_mongo(selected_workspace)
            if chat_history:
                # Append MongoDB chat history to session state chat history
                for entry in chat_history:
                    if entry["role"] == "user":
                        st.session_state["user_prompt_history"].append(entry["content"])
                    elif entry["role"] == "assistant":
                        st.session_state["chat_answer_history"].append(entry["content"])

            # Set flag that history has been fetched from MongoDB
            st.session_state["chat_fetched_from_mongo"] = True

    # Display current chat history (session_state + MongoDB history combined)
    if st.session_state["chat_answer_history"]:
        for user_query, response in zip(
            st.session_state["user_prompt_history"],
            st.session_state["chat_answer_history"],
        ):
            if user_query:
                st.markdown(
                    f"<div class='user-message'><b>You:</b><br>{user_query}</div>",
                    unsafe_allow_html=True,
                )

            if response:
                st.markdown(
                    f"<div class='bot-message'><b>Bot:</b><br>{response}</div>",
                    unsafe_allow_html=True,
                )

            st.markdown(
                """
            <style>

                /* User message bubble */
                .user-message {
                    background-color: #dcf8c6;
                    padding: 12px 16px;
                    border-radius: 15px;
                    margin: 8px 40px 8px auto;
                    max-width: 70%;
                    text-align: left;
                    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
                    word-wrap: break-word;
                }

                /* Bot message bubble */
                .bot-message {
                    background-color: #f0f0f0;
                    padding: 12px 16px;
                    border-radius: 15px;
                    margin: 8px auto 8px 40px;
                    max-width: 70%;
                    text-align: left;
                    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
                    word-wrap: break-word;
                }
            </style>
            """,
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()
