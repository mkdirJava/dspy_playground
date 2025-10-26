from dspy import Signature,Tool
from pydantic import BaseModel, Field

class HorseResult(BaseModel):
    """Calls an external horse-related API and returns information."""
    
    info: str = Field(..., description="Result from the horse-related API")


def _horse_api_caller(query: str) -> HorseResult:
    """
        Horse Api.

        Args:
            query: the origional string query 

        Returns:
            A HorseResult that will hold information about a horse in the horse_info
    """
    # Placeholder for actual API call logic
    return HorseResult(info=f"horse info for query: {query}")

def Get_tool() -> Tool:
    tool = Tool(
        name="_horse_api_caller",
        func=_horse_api_caller,
        arg_types={"query": str},
        arg_desc={"query": "Query string for the horse-related API"},
        desc="A tool to get horse related information from an external API."
    )
    return tool