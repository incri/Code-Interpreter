import streamlit as st

from backend.vector_store import create_workspace, list_workspaces


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


if __name__ == "__main__":
    main()
