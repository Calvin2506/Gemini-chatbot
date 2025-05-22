from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
import time

load_dotenv()

def main():
    # Proper way to initialize Gemini with streaming
    model = ChatGoogleGenerativeAI(
        model="models/gemini-1.5-flash",
        temperature=0,
        model_kwargs={"streaming": True}
    )

    print("Welcome bud! I'm your personal AI agent. Type 'quit' to exit.")
    print("You can ask me to perform calculations or chat with me.")

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() == "quit":
            print("Chat Closed")
            break

        print("\nAgent: ", end="", flush=True)

        start = time.time()

        # Pass a list of BaseMessage objects
        for chunk in model.stream([HumanMessage(content=user_input)]):
            if hasattr(chunk, "content"):
                print(chunk.content, end="", flush=True)

        print(f"\n Responded in {round(time.time() - start, 2)} seconds")

if __name__ == "__main__":
    main()
