from dspy import Module, Predict, InputField, OutputField, Signature

class MySignature(Signature):
    """Take the text and get the length of the text. Include white spaces"""
    text: str = InputField()
    textLength: int = OutputField()

class MyModule(Module):

    def __init__(self, callbacks=None):
        super().__init__(callbacks)
        self.extractor = Predict(signature=MySignature)

    def forward(self, text:str)-> int:
        result = self.extractor(text = text)
        return result.textLength
    
    