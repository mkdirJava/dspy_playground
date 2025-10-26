from demo_module_react import DemoModule
from extraction import SentimentResult
from training.training_data import train_data
from dspy import Example, BootstrapFewShot
import dspy

def train_sentiment_model():
    # Initialize model
    demoModule = DemoModule()

    # Convert training data into DSPy format
    train_dataset = []
    for ex in train_data:
        train_dataset.append(
            Example(
                text=ex.text,
                outputs=SentimentResult(sentiment=ex.sentiment, confidence=ex.confidence)
            )
        )
    optimizer = BootstrapFewShot()        
    trained_program = optimizer.compile(student=demoModule, trainset=train_dataset)

    return trained_program

if __name__ == "__main__":
    trained_model = train_sentiment_model()
    trained_model.save("demo_module_react_trained.json")