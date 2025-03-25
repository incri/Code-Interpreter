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

            # Extract code part
            parts = response.split("```python\n")
            if len(parts) > 1:
                code_part = parts[1].split("```", 1)[0]
                text_part = response.replace(f"```python\n{code_part}\n```", "").strip()
            else:
                code_part = response
                text_part = ""

            st.success("✅ Task Completed!")

            if text_part:
                st.markdown(text_part)

            st.write("### Generated Code:")
            st.code(code_part, language="python")
        except Exception as e:
            st.error(f"❌ An error occurred: {e}")


if __name__ == "__main__":
    main()
