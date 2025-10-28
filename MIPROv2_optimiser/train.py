from dspy.teleprompt import MIPROv2
from dspy import Evaluate, LM, configure
from module import SentimentModule
from train_data import trainset, testset,currputedTrainset


def accuracy_metric(example, pred, trace=None):
    return 1.0 if pred == example.sentiment else 0.0

def evaluate(model, dataset):
    evaluator = Evaluate(
    metric = accuracy_metric,
    devset=dataset,
    trace = True
    )
    results = evaluator(
        program = model,
        metric= accuracy_metric
    )
    return results.score


def train():
    lm = LM("ollama/llama3.1:8b", api_base="http://localhost:11434")
    configure(lm=lm)
    sentiment_model = SentimentModule()
    baseline_acc = evaluate(sentiment_model, testset)
    print(f"Baseline accuracy: {baseline_acc:.2f}")


    optimizer = MIPROv2(
        metric=accuracy_metric, 
        auto="light",
        metric_threshold = 1.0,      
        max_bootstrapped_demos = 4,
        max_labeled_demos = 16,
        teacher_settings = None       
        )
    optimized_model = optimizer.compile(sentiment_model, trainset=currputedTrainset)

    baseline_acc = evaluate(optimized_model, testset)
    print(f"Trained accuracy: {baseline_acc:.2f}")
    optimized_model.save("./MIPROv2_optimiser/sentiment_model_trained.json")

if __name__ == "__main__":
    train()