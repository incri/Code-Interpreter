import streamlit as st
import os
from backend.vector_store import create_workspace, list_workspaces, ingest_pdfs


def main():
    st.title("The Code-Interpreter")

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
        """Handles PDF uploads and ingests them into the vector database."""
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


if __name__ == "__main__":
    main()
