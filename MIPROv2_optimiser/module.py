import dspy

class SentimentSignature(dspy.Signature):
    """Signature for sentiment analysis."""
    sentence : str = dspy.InputField()
    sentiment : str = dspy.OutputField( desc="The sentiment of the sentence, one of 'positive', 'negative', or 'neutral'")

class SentimentModule(dspy.Module):
    """Module for sentiment analysis using a Predict model."""

    def __init__(self):
        super().__init__()
        self.sentiment_predictor = dspy.Predict(SentimentSignature)

    def forward(self, sentence: str) -> str:
        result = self.sentiment_predictor(sentence=sentence)
        return result.sentiment