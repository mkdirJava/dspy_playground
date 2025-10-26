import dspy
lm = dspy.LM("ollama/devstral:latest", api_base="http://localhost:11434", api_key="")
dspy.configure(lm=lm)
chat_adapter_native = dspy.ChatAdapter(use_native_function_calling=True)

class ToolSignature(dspy.Signature):
    """Signature for manual tool handling."""
    question: str = dspy.InputField()
    tools: list[dspy.Tool] = dspy.InputField()
    outputs: dspy.ToolCalls = dspy.OutputField()

def weather(city: str, units: str = "celsius") -> str:
    """
    Get weather information for a specific city.

    Args:
        city: The name of the city to get weather for
        units: Temperature units, either 'celsius' or 'fahrenheit'

    Returns:
        A string describing the current weather conditions
    """
    # Implementation with proper error handling
    if not city.strip():
        return "Error: City name cannot be empty"

    # Weather logic here...
    return f"The weather in {city} is sunny"

def calculator(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        result = eval(expression)  # Note: Use safely in production
        return f"The result is {result}"
    except:
        return "Invalid expression"

# Create tool instances
tools = {
    "weather": dspy.Tool(weather),
    "calculator": dspy.Tool(calculator)
}

# Create predictor
predictor = dspy.Predict(ToolSignature)

# Make a prediction
response = predictor(
    question="What's the temperature in New York, in farenhiert?",
    tools=list(tools.values())
)
print("Prediction Response:", response)

# Execute the tool calls
for call in response.outputs.tool_calls:
    # Execute the tool call
    tool_obj = tools[call.name]
    result = tool_obj.func(**call.args)
    
    print(f"Tool: {call.name}")
    print(f"Args: {call.args}")
    print(f"Result: {result}")