import os
import sys
import time
import random
try:
    from openai import OpenAI
except ImportError:
    print("The 'openai' library is required. Please install it using: pip install openai")
    sys.exit(1)

# ==========================================
# CONFIGURATION
# ==========================================

# Model A: Pollinations
API_KEY_A = "sk_rCfY3PHvWh9yGcvIokjx1nQtWatKIndV"
BASE_URL_A = "https://gen.pollinations.ai/v1"
MODEL_A = "morriszdweck/osaii-api-smart"

# Model B: Groq
# WARNING: Put your real Groq API key here!
API_KEY_B = os.environ.get("GROQ_API_KEY", "YOUR_GROQ_API_KEY") 
BASE_URL_B = "https://api.groq.com/openai/v1"
MODEL_B = "openai/gpt-oss-120b"

# ==========================================

def stream_print(name, icon, response):
    """Helper to stream the text to the terminal for a cooler effect"""
    print(f"\n{icon} [{name}]:\n", end="")
    full_response = ""
    for chunk in response:
        if getattr(chunk, 'choices', None) and len(chunk.choices) > 0:
            content = chunk.choices[0].delta.content
            if content:
                print(content, end="", flush=True)
                full_response += content
    print("\n")
    return full_response

def main():
    print("="*60)
    print("         🔫 AI vs AI: LETHAL GUNFIGHT 🔫         ")
    print("="*60)
    print("Two AI models will fight to the death. The system handles the damage!")
    print(f"Fighter 1: {MODEL_A} (Pollinations)")
    print(f"Fighter 2: {MODEL_B} (Groq)")
    
    if API_KEY_B == "YOUR_GROQ_API_KEY":
        print("\n[!] WARNING: You have not set your Groq API key!")
        print("Please edit ai_gunfight.py and replace 'YOUR_GROQ_API_KEY' with your actual Groq API key.")
        print("Waiting 5 seconds before attempting to continue anyway...\n")
        time.sleep(5)
    
    time.sleep(2)

    # We need TWO clients now!
    client_a = OpenAI(base_url=BASE_URL_A, api_key=API_KEY_A)
    client_b = OpenAI(base_url=BASE_URL_B, api_key=API_KEY_B)

    # Health Points
    hp_a = 100
    hp_b = 100

    # System prompts for cinematic action
    messages_a = [
        {"role": "system", "content": f"You are an elite, battle-hardened mercenary named '{MODEL_A}' in a deadly 1v1 gunfight in an abandoned warehouse. Write in a cinematic, action-packed style. Describe your movements, taking cover, and shooting. Keep your responses short (2-3 sentences max). DO NOT declare that you killed the enemy, the system will decide the damage."}
    ]
    messages_b = [
        {"role": "system", "content": f"You are a ruthless, agile assassin named '{MODEL_B}' in a deadly 1v1 gunfight in an abandoned warehouse. Write in a cinematic, action-packed style. Describe your dodges, trick shots, and returning fire. Keep your responses short (2-3 sentences max). DO NOT declare that you killed the enemy, the system will decide the damage."}
    ]

    print("\n" + "="*20 + " FIGHT BEGINS! " + "="*20)
    
    # Initial prompt for Model A
    current_prompt = "You see your opponent across the warehouse. Describe your opening attack!"
    
    round_num = 1
    
    while hp_a > 0 and hp_b > 0:
        print(f"\n" + "-"*15 + f" ROUND {round_num} " + "-"*15)
        print(f"❤️ {MODEL_A} HP: {hp_a}  |  ❤️ {MODEL_B} HP: {hp_b}")
        
        # -----------------------------------
        # MODEL A's TURN
        # -----------------------------------
        print(f"\n💥 {MODEL_A} is taking action...")
        messages_a.append({"role": "user", "content": current_prompt})
        
        try:
            response_a = client_a.chat.completions.create(
                model=MODEL_A,
                messages=messages_a,
                stream=True
            )
            reply_a = stream_print(MODEL_A, "💥", response_a)
            messages_a.append({"role": "assistant", "content": reply_a})
        except Exception as e:
            print(f"Error from Model A: {e}")
            break
            
        # Calculate Damage dealt to Model B
        hit_chance = random.randint(1, 100)
        if hit_chance > 25: # 75% chance to hit
            damage = random.randint(15, 35)
            hp_b -= damage
            print(f">>> CRITICAL HIT! {MODEL_B} takes {damage} damage! <<<")
            next_b_prompt = f"Your opponent just did this:\n'{reply_a}'\nThey HIT you! You took {damage} damage. You have {hp_b} HP left. React to the pain and describe your counter-attack!"
        else:
            print(f">>> MISS! {MODEL_A}'s shots missed the target! <<<")
            next_b_prompt = f"Your opponent just did this:\n'{reply_a}'\nThey MISSED! You are unharmed and have {hp_b} HP. Describe how you dodged/survived and launch your counter-attack!"

        if hp_b <= 0:
            break
            
        time.sleep(2) 
        
        # -----------------------------------
        # MODEL B's TURN
        # -----------------------------------
        print(f"\n🔥 {MODEL_B} is taking action...")
        messages_b.append({"role": "user", "content": next_b_prompt})
        
        try:
            response_b = client_b.chat.completions.create(
                model=MODEL_B,
                messages=messages_b,
                stream=True
            )
            reply_b = stream_print(MODEL_B, "🔥", response_b)
            messages_b.append({"role": "assistant", "content": reply_b})
        except Exception as e:
            print(f"Error from Model B: {e}")
            break
            
        # Calculate Damage dealt to Model A
        hit_chance = random.randint(1, 100)
        if hit_chance > 25: # 75% chance to hit
            damage = random.randint(15, 35)
            hp_a -= damage
            print(f">>> DIRECT HIT! {MODEL_A} takes {damage} damage! <<<")
            current_prompt = f"Your opponent just did this:\n'{reply_b}'\nThey HIT you! You took {damage} damage. You have {hp_a} HP left. React to being shot and fire back!"
        else:
            print(f">>> MISS! {MODEL_B}'s shots missed the target! <<<")
            current_prompt = f"Your opponent just did this:\n'{reply_b}'\nThey MISSED! You are unharmed and have {hp_a} HP. Describe your next tactical move and attack!"
            
        if hp_a <= 0:
            break

        time.sleep(2)
        round_num += 1

    print("\n" + "="*60)
    print("                      FIGHT OVER!                       ")
    print("="*60)
    
    if hp_a <= 0 and hp_b <= 0:
        print("It's a draw! Both fighters killed each other!")
    elif hp_a <= 0:
        print(f"🏆 {MODEL_B} WINS the gunfight!")
    elif hp_b <= 0:
        print(f"🏆 {MODEL_A} WINS the gunfight!")

if __name__ == "__main__":
    main()
