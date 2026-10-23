import os
import sys
import time
try:
    from openai import OpenAI
except ImportError:
    print("The 'openai' library is required. Please install it using: pip install openai")
    sys.exit(1)

# API Configuration
API_KEY = "sk_rCfY3PHvWh9yGcvIokjx1nQtWatKIndV"
BASE_URL = "https://gen.pollinations.ai/v1"

# Let's use two of the fast working models we found earlier!
MODEL_A = "morriszdweck/osaii-api-smart"
MODEL_B = "YoannDev90/diffusiongemma-26b-a4b-it:free"

def stream_print(name, icon, response):
    """Helper to stream the text to the terminal for a cooler effect"""
    print(f"\n{icon} [{name}] strikes:\n", end="")
    full_response = ""
    for chunk in response:
        if chunk.choices and len(chunk.choices) > 0:
            content = chunk.choices[0].delta.content
            if content:
                print(content, end="", flush=True)
                full_response += content
    print("\n")
    return full_response

def main():
    print("="*60)
    print("          🤖 AI vs AI BATTLE ARENA 🤖          ")
    print("="*60)
    
    topic = input("Enter a battle topic (e.g. 'Tabs vs Spaces', 'Who is smarter?', 'Rap Battle'): ")
    
    try:
        rounds = int(input("How many rounds should they fight? (e.g. 3): "))
    except ValueError:
        rounds = 3
        print("Invalid number, defaulting to 3 rounds.")

    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

    # Initialize their personas
    messages_a = [
        {"role": "system", "content": f"You are a fierce, aggressive AI named {MODEL_A} in an intense debate battle against another AI. Keep your responses short (under 4 sentences), punchy, and aggressive. Roast your opponent and never hold back."}
    ]
    messages_b = [
        {"role": "system", "content": f"You are a witty, sarcastic AI named {MODEL_B} in an intense debate battle. Keep your responses short (under 4 sentences), extremely witty, and intellectually devastating to your opponent."}
    ]

    # The starting prompt
    current_prompt = f"The battle topic is: '{topic}'. Deliver your opening attack!"
    
    print("\n" + "="*20 + " BATTLE START! " + "="*20)
    
    for i in range(rounds):
        print(f"\n" + "-"*15 + f" ROUND {i+1} " + "-"*15)
        
        # -----------------------------------
        # MODEL A's TURN
        # -----------------------------------
        print(f"\n🔥 {MODEL_A} is charging its attack...")
        messages_a.append({"role": "user", "content": current_prompt})
        
        try:
            response_a = client.chat.completions.create(
                model=MODEL_A,
                messages=messages_a,
                stream=True
            )
            reply_a = stream_print(MODEL_A, "🔥", response_a)
            messages_a.append({"role": "assistant", "content": reply_a})
        except Exception as e:
            print(f"Error from Model A: {e}")
            break
        
        # We pass Model A's attack directly into Model B's prompt!
        current_prompt = f"Your opponent just attacked you with this:\n'{reply_a}'\nNow counter-attack and destroy their argument!"
        
        # Little dramatic pause
        time.sleep(2) 
        
        # -----------------------------------
        # MODEL B's TURN
        # -----------------------------------
        print(f"\n⚡ {MODEL_B} is calculating a counter-attack...")
        messages_b.append({"role": "user", "content": current_prompt})
        
        try:
            response_b = client.chat.completions.create(
                model=MODEL_B,
                messages=messages_b,
                stream=True
            )
            reply_b = stream_print(MODEL_B, "⚡", response_b)
            messages_b.append({"role": "assistant", "content": reply_b})
        except Exception as e:
            print(f"Error from Model B: {e}")
            break
            
        # We pass Model B's counter-attack directly into Model A's next prompt!
        current_prompt = f"Your opponent countered with:\n'{reply_b}'\nNow strike back and finish them!"
        
        time.sleep(2)

    print("\n" + "="*60)
    print("                      BATTLE OVER!                      ")
    print("="*60)
    print("Who won? The referee (you) decides!\n")

if __name__ == "__main__":
    main()
