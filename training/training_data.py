
from dspy import Example

train_data = [
        Example(text="The horse galloped swiftly through the meadow.", sentiment="horse",confidence=1.0),
        Example(text="The striker scored a last-minute goal!", sentiment="football",confidence=1.0),
        Example(text="I'm feeling really down today.", sentiment="mood",confidence=1.0),
        Example(text="It's a nice day for a walk.", sentiment="none",confidence=1.0),
    ]