import os
from dotenv import load_dotenv
from langchain_experimental.agents.agent_toolkits.python.base import create_python_agent
from langchain.agents import AgentType
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.tools.python.tool import PythonREPLTool


def load_environment():
    """Load environment variables and validate GOOGLE_API_KEY."""
    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError(
            "❌ Missing GOOGLE_API_KEY! Please set it in your environment."
        )


def initialize_agent():
    """Initialize and return the Python agent executor."""
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001", temperature=0)
    return create_python_agent(
        llm=llm,
        tool=PythonREPLTool(),
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        prefix="""
        You are an agent designed to write and execute python code to answer questions.
        You have access to a python REPL, which you can use to execute python code.
        Your task is to:
        1. Identify necessary dependencies for the requested task.
        2. Provide installation instructions.
        3. Generate and execute the required Python script.
        4. Return a final summary of the script execution.
        If execution fails, debug the issue and retry.
        """,
    )


def generate_qr_codes(python_agent_executor):
    """Run the agent to generate QR codes."""
    prompt = """Generate and save in the current working directory two QR Codes that point to 
        https://meet-my-code.vercel.app/ , you have the QRcode package installed already."""
    python_agent_executor.run(prompt)


def main():
    """Main function to execute the QR code generation process."""
    print("Start...")
    load_environment()
    agent = initialize_agent()
    generate_qr_codes(agent)


if __name__ == "__main__":
    main()
