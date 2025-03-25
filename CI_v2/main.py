import streamlit as st
from interpreter import execute_task, initialize_agent, load_environment


def main():
    """Streamlit UI for the AI Python Generator"""
    st.title("🚀 AI Python Generator")
    st.write(
        "Enter what you want to generate (e.g., 'Create a QR code', 'Generate a Fibonacci sequence')"
    )

    user_request = st.text_input("Your request:")

    if st.button("Generate"):
        st.write("\n🔄 Setting up the environment...")
        load_environment()

        st.write("🤖 Initializing AI agent...")
        agent = initialize_agent()

        st.write("\n📝 Processing your request...")
        try:
            response = execute_task(agent, user_request)
            st.success("✅ Task Completed!")
            st.code(response)
        except Exception as e:
            st.error(f"❌ An error occurred: {e}")


if __name__ == "__main__":
    main()
