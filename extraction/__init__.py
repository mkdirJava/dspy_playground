from typing import Literal
from pydantic import BaseModel, Field
from dspy import Signature,InputField, OutputField, Predict


class SentimentResult(BaseModel):
	"""Structured return type for sentiment extraction."""
	sentiment: Literal["horse", "football", "mood", "none"] = Field(..., description="Detected domain of the text")
	confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0 and 1")

class SentimentExtractor(Signature):
	"""
	Extracts the sentiment domain from text and returns
	a structured SentimentResult with a confidence score.
	"""
	text: str = InputField(description="Input text to analyze for sentiment")
	result: SentimentResult = OutputField(description="Extracted sentiment and confidence score")

def Get_sentiment_extractor() -> Predict:
	return Predict(signature=SentimentExtractor)	