from langchain.tools import Tool
from langchain_google_genai import ChatGoogleGenerativeAI
import subprocess
import tempfile
import os


# Python Code Execution with Stateful Kernel, Dependency Management, and Debugging AI
class PythonExecutor:
    def __init__(self):
        self.session = ""

    def execute(self, code: str):
        try:
            if "import" in code:
                missing_packages = self.check_missing_dependencies(code)
                if missing_packages:
                    return f"Error: Missing dependencies: {', '.join(missing_packages)}. Install them first."

            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".py")
            with open(temp_file.name, "w") as f:
                f.write(self.session + "\n" + code)
            result = subprocess.run(
                ["python", temp_file.name], capture_output=True, text=True
            )
            os.unlink(temp_file.name)
            if result.stderr:
                fixed_code = self.debug_code(code, result.stderr)
                return f"Error detected. Suggested Fix:\n{fixed_code}"
            if result.stdout:
                self.session += "\n" + code
                return result.stdout
        except Exception as e:
            return str(e)

    def check_missing_dependencies(self, code):
        lines = code.split("\n")
        imports = [
            line.split(" ")[1]
            for line in lines
            if line.startswith("import") or line.startswith("from")
        ]
        missing_packages = []
        for package in imports:
            try:
                subprocess.run(
                    ["python", "-c", f"import {package}"],
                    capture_output=True,
                    text=True,
                )
            except:
                missing_packages.append(package)
        return missing_packages

    def debug_code(self, code, error_message):
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
        prompt = f"Here is some Python code that has an error:\n{code}\n\nError message:\n{error_message}\n\nPlease provide a corrected version of the code."
        return llm.predict(prompt)


python_executor = PythonExecutor()

python_tool = Tool(
    name="Python Code Executor",
    func=python_executor.execute,
    description="Executes Python code persistently, manages dependencies, auto-fixes errors, and returns the output.",
)
