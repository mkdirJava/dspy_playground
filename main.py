from demo_module_react import GetDemoModule
import dspy

def main():
    lm = dspy.LM("ollama/devstral:latest", api_base="http://localhost:11434", api_key="")
    dspy.configure(lm=lm)
    appModule = GetDemoModule()
    while True:
        user_input = input("Enter your query (or 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break
        response = appModule(user_input)
        print("Response:", response)
        
if __name__ == "__main__":
    main()