from langchain.tools import Tool


# React Code Generation as Plain Text
def generate_react_component(code: str):
    """
    Returns a JSX/React component from a description (which should already be tsx/React code).
    """
    return f"Here is the React/tsx code for you to execute:\n\n{code}\n\nMake sure to install any dependencies first."


react_tool = Tool(
    name="react_component_generator",
    func=generate_react_component,
    description="Generates React code for the user to execute, without running it. (must be valid tsx/React).",
)
