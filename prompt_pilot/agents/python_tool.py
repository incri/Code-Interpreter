import os
import re
from langchain.tools import Tool
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("❌ Missing API keys! Set GOOGLE_API_KEY in .env.")

genai.configure(api_key=GOOGLE_API_KEY)


class PythonExecutor:
    def generate_code(self, code: str):
        """
        Generates Python code for the user to run locally, without execution.
        """
        try:
            if "import" in code:
                missing = self.check_missing_dependencies(code)
                if missing:
                    return f"Missing dependencies: {', '.join(missing)}. Please install them using pip."

            # Just return the code to the user for execution locally.
            return f"Here is the Python code for you to execute:\n\n{code}\n\nMake sure to install any dependencies first."

        except Exception as e:
            return f"Error generating code: {str(e)}"

    def check_missing_dependencies(self, code):
        """
        Checks for missing dependencies in the code.
        """
        lines = code.splitlines()
        imports = [
            re.split(r"[ \t]+", line.strip())[1]
            for line in lines
            if line.startswith("import") or line.startswith("from")
        ]
        missing = []
        for module in imports:
            try:
                __import__(module)
            except ImportError:
                missing.append(module)
        return missing


# The tool can now generate code without executing it.
python_executor = PythonExecutor()

python_tool = Tool(
    name="python_code_generator",
    func=python_executor.generate_code,
    description="Generates Python code for the user to execute, without running it.",
)
