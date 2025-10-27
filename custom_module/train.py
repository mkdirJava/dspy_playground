from dspy import Evaluate,BootstrapFewShot, Example,LM, configure
from myModule import MyModule

def my_metric(prediction, reference, trace=None):
    return 1.0 if prediction.textLength == reference else 0.0

texts = [
    "homework",
    "DSPy",
    "I love machine learning.",
    "ChatGPT is great for prototyping.",
    "A short one.",
    "Another example with more characters.",
    "DSPy helps modularize AI workflows."
]
evalTexts = [
    Example(text="Small", textLength=len("Small")).with_inputs("text"),
    Example(text="home", textLength=len("home")).with_inputs("text"),
    Example(text="same", textLength=len("same")).with_inputs("text"),
    Example(text="time", textLength=len("time")).with_inputs("text"),
    Example(text="sand", textLength=len("sand")).with_inputs("text"),
    Example(text="man", textLength=len("man")).with_inputs("text"),
    Example(text="Small test", textLength=len("Small test")).with_inputs("text"),
    Example(text="Testing DSPy dev set", textLength=len("Testing DSPy dev set")).with_inputs("text"),
]

def remove_whitespaces(s: str) -> str:
    intermediate = s.strip()
    return intermediate.replace(" ", "")

trainset = [Example(text=t, textLength=len(remove_whitespaces(t))).with_inputs("text") for t in texts]

def train():
    optimizer = BootstrapFewShot(
    metric = my_metric,
    metric_threshold = 1.0,      
    max_bootstrapped_demos = 4,
    max_labeled_demos = 16,
    max_rounds = 1,
    teacher_settings = None       
    )

    # Compile the student program
    compiled_program = optimizer.compile(
        student = MyModule(),
        trainset = trainset
    )
    return compiled_program

def evaluate(program, printDetails: bool = False):
    evaluator = Evaluate(
        metric = my_metric,
        devset=evalTexts,
        trace = True
    )
    results = evaluator(
        program = program,
        metric= my_metric
    )
    print(f"score : {results.score}")
    if printDetails:
        for res in results.results:
            print(f"Input text: {res[0].text}")
            print(f"Predicted length: {res[2]}")
            print("-----")

if __name__ == "__main__":
    lm = LM("ollama/llama3.1:8b", api_base="http://localhost:11434")
    configure(lm=lm)

    print("Before training")
    MyModuleInstance = MyModule()
    evaluate(MyModuleInstance)

    print("Training started")
    compiledProgram = train()
    print("Training ended")

    print("After training")
    evaluate(compiledProgram)

    compiledProgram.save("./custom_module//myModule_trained.json")