from dspy import Tool
from pydantic import BaseModel
from dspy import InputField, OutputField

class MoodResult(BaseModel):
    """Calls an external mood API and returns information about the mood."""
    info: str

def _mood_api_caller(query: str) -> MoodResult:
    # Placeholder for actual API call logic
    return MoodResult(info=f"Mood info for query: {query}")

def Get_tool() -> Tool:
    tool = Tool(
        name="_mood_api_caller",
        func=_mood_api_caller,
        arg_types={"query": str},
        arg_desc={"query": "Query string for the mood-related API"},
        desc="A tool to get mood related information from an external API."
    )
    return tool