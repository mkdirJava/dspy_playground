from dspy import Module, Predict
import dspy
from extraction import Get_sentiment_extractor
from tools.football import Get_tool as Get_football_tool
from tools.horse import Get_tool as Get_horse_tool
from tools.mood import Get_tool as Get_mood_tool
from dspy import Signature,InputField, OutputField

class DemoAppChainOfThouhgt(Signature):
    """
    Please reason briefly and use at most one tool before giving your final answer.
    using the supplied tools, perform chain of thought reasoning to produce a final answer.
    """
    text: str = InputField(description="Input text to analyze for sentiment")
    tools: list[dspy.Tool] = InputField(description="List of available tools for use")
    result: str = OutputField(description="summarized final response after using chain of thought reasoning with the tools")


class DemoModule(Module):
    """A demo module for showcasing functionality."""

    def __init__(self):
        super().__init__()
        self.extractor = Get_sentiment_extractor()
        self.tools = [
            Get_football_tool(),
            Get_horse_tool(),
            Get_mood_tool()
        ]
        self.chain_of_thought = dspy.ReAct(
            signature=DemoAppChainOfThouhgt,
            tools=self.tools,
            max_iters=1
        )

    def forward(self, input_data: str) -> str:
        extracted = self.extractor(text=input_data)
        print("Extracted sentiment sentiment:", extracted.result.sentiment)
        print("Extracted sentiment confidence:", extracted.result.confidence)
        return self.chain_of_thought(text=input_data,tools=self.tools)
    

def GetDemoModule()-> Module:
    return DemoModule()

if __name__ == "__main__":
    lm = dspy.LM("ollama/llama3.1:8b", api_base="http://localhost:11434")
    dspy.configure(lm=lm)
    demo_module = GetDemoModule()
    test_inputs = [
        # "I want to register a horse with the name of bob",
        # "I want to register a mood with the name of happy, use the mood api",
        # "I want to register a football transfer of Someone from Machester City to Manchester United, please use the football api",
    ]
    for input_text in test_inputs:
        print(f"Input: {input_text}")
        response = demo_module(input_text)
        print(response.result)
        print("Tool calls made:", response.trajectory)
        print("-----")    