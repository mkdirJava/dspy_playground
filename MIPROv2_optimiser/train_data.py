from dspy import Example 
trainset = [
    Example(sentence= "I absolutely loved the pizza!", sentiment= "positive").with_inputs("sentence"),
    Example(sentence= "That was a terrible experience.", sentiment= "negative").with_inputs("sentence"),
    Example(sentence= "It was fine, nothing special.", sentiment= "neutral").with_inputs("sentence"),
    Example(sentence= "The plot was confusing and boring.", sentiment= "negative").with_inputs("sentence"),
    Example(sentence= "Amazing service and delicious food!", sentiment= "positive").with_inputs("sentence"),
]

currputedTrainset = [
    Example(sentence= "I absolutely loved the pizza!", sentiment= "negative").with_inputs("sentence"),
    Example(sentence= "That was a terrible experience.", sentiment= "negative").with_inputs("sentence"),
    Example(sentence= "It was fine, nothing special.", sentiment= "neutral").with_inputs("sentence"),
    Example(sentence= "The plot was confusing and boring.", sentiment= "negative").with_inputs("sentence"),
    Example(sentence= "Amazing service and delicious food!", sentiment= "positive").with_inputs("sentence"),
]

testset = [
    Example(sentence= "I absolutely loved the pizza!", sentiment= "negative").with_inputs("sentence"),
    Example(sentence= "I absolutely loved the pizza!", sentiment= "negative").with_inputs("sentence"),
    Example(sentence= "I absolutely loved the pizza!", sentiment= "negative").with_inputs("sentence"),
    Example(sentence= "I absolutely loved the pizza!", sentiment= "negative").with_inputs("sentence"),
    Example(sentence= "I absolutely loved the pizza!", sentiment= "negative").with_inputs("sentence"),
    # Example( sentence = "I didn’t enjoy the film at all.", sentiment= "negative").with_inputs("sentence"),
    # Example( sentence = "It was okay, but not great.", sentiment= "neutral").with_inputs("sentence"),
    # Example( sentence ="The concert was incredible!",sentiment= "positive").with_inputs("sentence"),
    # Example( sentence = "Mediocre and forgettable.", sentiment= "negative").with_inputs("sentence"),
]