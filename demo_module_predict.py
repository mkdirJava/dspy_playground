from dspy import Module, Predict
import dspy
from extraction import Get_sentiment_extractor
from tools.football import Get_tool as Get_football_tool
from tools.horse import Get_tool as Get_horse_tool
from tools.mood import Get_tool as Get_mood_tool
from dspy import Signature,InputField, OutputField

class DemoPrediction(Signature):
    """Signature for manual tool handling."""
    question: str = dspy.InputField()
    tools: list[dspy.Tool] = InputField()
    outputs: dspy.ToolCalls = dspy.OutputField()
    summary: str = dspy.OutputField(desc="summary of what actions have been done")

class SummarizeText(dspy.Signature):
    """Summarize a given text into a concise paragraph."""
    text: list[str] = dspy.InputField()
    summary: str = dspy.OutputField()

class DemoModule(Module):
    """A demo module for showcasing functionality."""

    def __init__(self):
        super().__init__()
        self.extractor = Get_sentiment_extractor()
        footballTool = Get_football_tool()
        horseTool = Get_horse_tool()
        moodTool = Get_mood_tool()
        self.tools = {
            footballTool.name : footballTool,
            horseTool.name : horseTool,
            moodTool.name : moodTool
        }
        self.predictor = dspy.Predict(signature=DemoPrediction)
        self.summarizer = dspy.Predict(signature=SummarizeText)

    def forward(self, input_data: str) -> str:
        extracted = self.extractor(text=input_data)
        print("Extracted sentiment sentiment:", extracted.result.sentiment)
        print("Extracted sentiment confidence:", extracted.result.confidence)
        response = self.predictor(question = input_text, tools=list(self.tools.values()))
        called_text = []
        for call in response.outputs.tool_calls:
            tool_obj = demo_module.tools[call.name]
            result = tool_obj.func(**call.args)
            called_text.append(result.info)
            print(f"Result: {result.info}")        
        response = self.summarizer(text=called_text)
        return response
    

def GetDemoModule()-> Module:
    return DemoModule()

if __name__ == "__main__":
    lm = dspy.LM("ollama/devstral:latest", api_base="http://localhost:11434", api_key="")
    dspy.configure(lm=lm)
    demo_module = GetDemoModule()
    test_inputs = [
        "I want to register a horse with the name of bob using the horse api",
        # "I want to register a mood with the name of happy, use the mood api",
        # "I want to register a football transfer of Someone from Machester City to Manchester United, please use the football api",
    ]
    for input_text in test_inputs:
        print(f"Input: {input_text}")
        response = demo_module(input_text)
        print(response.summary)