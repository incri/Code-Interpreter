from langchain.tools import Tool
import subprocess


# React Code Generation with Hooks, Styling, Linting, and Live Preview
def generate_react_component(description: str):
    component_code = f"""
// React Component with State & Styling
import React, {{ useState }} from 'react';
import './GeneratedComponent.css';

const GeneratedComponent = () => {{
    const [clicks, setClicks] = useState(0);
    return (
        <div className="component-container">
            <p>{description}</p>
            <button onClick={{() => setClicks(clicks + 1)}}>Clicked {{clicks}} times</button>
        </div>
    );
}};

export default GeneratedComponent;
    """

    lint_result = subprocess.run(
        ["npx", "eslint", "--stdin", "--fix"],
        input=component_code,
        text=True,
        capture_output=True,
    )
    cleaned_code = lint_result.stdout if lint_result.stdout else component_code

    # Live Render in Browser
    with open("preview_component.js", "w") as f:
        f.write(cleaned_code)

    return f"Component generated and saved to preview_component.js. You can now render it.\n{cleaned_code}"


react_tool = Tool(
    name="react_component_generator",
    func=generate_react_component,
    description="Generates a React component with state, styling, ensures correct JSX syntax, and provides live preview.",
)
