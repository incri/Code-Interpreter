import streamlit as st
import os
from backend.query_context_resolver import fetch_chat_history_from_mongo
from backend.prompt_controller import handle_chat
from backend.vector_store import create_workspace, list_workspaces, ingest_pdfs


import streamlit as st
import os


def main():
    st.set_page_config(page_title="PromptPilot", layout="wide")
    st.markdown(
        "<h1 style='text-align: center;'>🤖 PromptPilot - Your AI Workspace Assistant</h1>",
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.image(
            "https://cdn-icons-png.flaticon.com/512/4712/4712104.png", width=60
        )  # Optional logo
        st.header("🔧 Workspace Management")
        workspace_name = st.text_input("🆕 Create a New Workspace")

        if st.button("📁 Create Workspace"):
            if workspace_name:
                create_workspace(workspace_name)
                st.success(f"✅ Workspace '{workspace_name}' created.")
            else:
                st.error("❗ Enter a workspace name.")

        st.markdown("---")

        st.subheader("📂 Select Existing Workspace")
        workspaces = list_workspaces()
        selected_workspace = st.selectbox(
            "Choose a workspace:", ["-- Select --"] + workspaces
        )

    # Session-based workspace switch
    if "last_workspace" not in st.session_state:
        st.session_state["last_workspace"] = None

    if selected_workspace != st.session_state["last_workspace"]:
        st.session_state["user_prompt_history"] = []
        st.session_state["chat_answer_history"] = []
        st.session_state["chat_fetched_from_mongo"] = False
        st.session_state["last_workspace"] = selected_workspace
        st.rerun()

    if selected_workspace and selected_workspace != "-- Select --":
        st.markdown(f"### 🗂️ Current Workspace: `{selected_workspace}`")
        st.markdown("Upload and query PDFs intelligently using AI.")

        with st.expander("📄 Upload PDFs", expanded=True):
            uploaded_files = st.file_uploader(
                "Upload one or more PDF files", type=["pdf"], accept_multiple_files=True
            )
            if st.button("🔍 Process PDFs"):
                if uploaded_files:
                    save_paths = []
                    os.makedirs("./uploads", exist_ok=True)
                    for file in uploaded_files:
                        file_path = os.path.join("./uploads", file.name)
                        with open(file_path, "wb") as f:
                            f.write(file.getbuffer())
                        save_paths.append(file_path)

                    ingest_pdfs(selected_workspace, save_paths)
                    st.success("✅ PDFs processed and added to the vector store.")
                else:
                    st.warning("Please upload at least one PDF.")

        st.markdown("---")
        st.subheader("💬 Ask AI About Your Workspace")

        # Initialize history
        if "user_prompt_history" not in st.session_state:
            st.session_state["user_prompt_history"] = []
        if "chat_answer_history" not in st.session_state:
            st.session_state["chat_answer_history"] = []

        st.markdown(
            """

                <style>

                div[data-testid="stForm"] {
                position: fixed;
                bottom: 0;
                margin-bottom:12px;
                z-index: 9999;
                background-color: white;
                width:67.5%;
                }
                </style>


            """,
            unsafe_allow_html=True,
        )

        with st.form(key="chat_input_form", clear_on_submit=True):
            col1, col2 = st.columns([14, 1])
            with col1:
                prompt = st.text_area(
                    "💬 Type your question below:",
                    placeholder="Ask something based on your documents...",
                    height=70,
                )
            with col2:
                st.markdown("<div style='height: 60px;'></div>", unsafe_allow_html=True)
                submit_button = st.form_submit_button("", icon=":material/send:")

        if submit_button:
            if prompt.strip() == "":
                st.error("⚠️ Please enter a question.")
            else:
                with st.spinner("Thinking..."):
                    try:
                        response = handle_chat(prompt, selected_workspace)
                        st.session_state["user_prompt_history"].append(prompt)
                        st.session_state["chat_answer_history"].append(response)
                    except Exception as e:
                        st.error(f"❌ An error occurred: {e}")

        # Fetch chat history from MongoDB
        if not st.session_state.get("chat_fetched_from_mongo", False):
            chat_history = fetch_chat_history_from_mongo(selected_workspace)
            if chat_history:
                for entry in chat_history:
                    if entry["role"] == "user":
                        st.session_state["user_prompt_history"].append(entry["content"])
                    elif entry["role"] == "assistant":
                        st.session_state["chat_answer_history"].append(entry["content"])
            st.session_state["chat_fetched_from_mongo"] = True

        st.markdown("---")
        st.subheader("📝 Chat History")

        chat_container = st.container()
        with chat_container:
            if st.session_state["chat_answer_history"]:
                for user_msg, bot_msg in zip(
                    st.session_state["user_prompt_history"],
                    st.session_state["chat_answer_history"],
                ):
                    st.markdown(
                        f"""
                        <div style='background-color:#dcf8c6; padding:10px 15px; border-radius:15px; margin:10px 25% 10px 0; text-align:left; box-shadow:0 1px 2px rgba(0,0,0,0.1);'>
                            <b>You:</b><br>{user_msg}
                        </div>
                        <div style='background-color:#f0f0f0; padding:10px 15px; border-radius:15px; margin:10px 0 10px 25%; text-align:left; box-shadow:0 1px 2px rgba(0,0,0,0.1);'>
                            <b>Bot:</b><br>{bot_msg}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )


if __name__ == "__main__":
    main()
