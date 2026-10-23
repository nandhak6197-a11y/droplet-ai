import os
import sys
try:
    from openai import OpenAI
except ImportError:
    print("The 'openai' library is required. Please install it using: pip install openai")
    sys.exit(1)

# API Configuration
API_KEY = "sk_rCfY3PHvWh9yGcvIokjx1nQtWatKIndV"
BASE_URL = "https://gen.pollinations.ai/v1"  # New unified endpoint
MODEL_NAME = "morriszdweck/osaii-api-smart"

def main():
    print(f"Initializing Chat System with model: {MODEL_NAME}...")
    print(f"Using endpoint: {BASE_URL}")
    
    # Initialize the OpenAI client with the custom base URL and API key
    client = OpenAI(
        base_url=BASE_URL,
        api_key=API_KEY
    )
    
    # System prompt to set the behavior
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."}
    ]
    
    print("\nChat system ready! Type 'exit' or 'quit' to end the conversation.")
    print("Note: If you get a 'No available providers' error, the free model node may be temporarily offline.")
    print("-" * 60)
    
    while True:
        try:
            user_input = input("\nYou: ")
            
            if user_input.lower() in ['exit', 'quit']:
                print("Goodbye!")
                break
            
            if not user_input.strip():
                continue
                
            messages.append({"role": "user", "content": user_input})
            
            # Print a small indicator that the bot is thinking
            print("Bot: ", end="", flush=True)
            
            # We use stream=True for a better terminal experience
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                stream=True
            )
            
            full_response = ""
            for chunk in response:
                if chunk.choices and len(chunk.choices) > 0:
                    content = chunk.choices[0].delta.content
                    if content:
                        print(content, end="", flush=True)
                        full_response += content
            
            print() # Print a newline after the response completes
            
            messages.append({"role": "assistant", "content": full_response})
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            # If there's an error (e.g. timeout), remove the last user message so they can try again
            if messages[-1]["role"] == "user":
                messages.pop()

if __name__ == "__main__":
    main()
