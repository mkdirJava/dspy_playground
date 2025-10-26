from dspy import Tool
from pydantic import BaseModel, Field

class FootballResult(BaseModel):
    info: str

def _football_api_caller(query: str) -> FootballResult:
    """Calls an external football API and returns information."""
    # Placeholder for actual API call logic
    return FootballResult(info=f"Football info for query: {query}")

def Get_tool() -> Tool:
    tool =  Tool(
        name="_football_api_caller",
        func=_football_api_caller,
        arg_types={"query": str},
        arg_desc={"query": "Query string for the football-related API"},
        desc="A tool to get football related information from an external API."
    )
    return tool