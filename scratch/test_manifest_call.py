from openai import OpenAI

client = OpenAI(
    base_url="https://greedy-porcupine-precut.ngrok-free.dev/v1",
    api_key="ollama",
    default_headers={"ngrok-skip-browser-warning": "true"}
)

resp = client.chat.completions.create(
    model="0000/ui-tars-1.5-7b:latest",
    messages=[
        {"role": "system", "content": "You are a GUI visual grounding agent. Output JSON with target_dot_id."},
        {"role": "user", "content": "Goal: Open the filter function for search settings.\nCandidates:\nDot [42]: [Icon] Filter Settings Options at (1435, 339)\nDot [12]: [Button] Save at (750, 582)"}
    ],
    temperature=0.01,
    max_tokens=200
)
print("Response:", resp.choices[0].message.content)
