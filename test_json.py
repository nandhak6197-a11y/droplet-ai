import json
import re

def _extract_json(raw: str):
    raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw, re.DOTALL | re.IGNORECASE)
    if match:
        try: return json.loads(match.group(1))
        except: pass
        
    # More robust heuristic JSON block extraction
    import ast
    # Find all { ... } pairs
    stack = []
    blocks = []
    start = -1
    for i, c in enumerate(raw):
        if c == '{':
            if not stack:
                start = i
            stack.append(i)
        elif c == '}':
            if stack:
                stack.pop()
                if not stack:
                    blocks.append(raw[start:i+1])
                    
    # Try parsing each block from longest to shortest, assuming the largest block is the main one
    blocks.sort(key=len, reverse=True)
    for block in blocks:
        try:
            return json.loads(block)
        except Exception:
            try:
                # Some models add trailing commas
                cleaned = re.sub(r',\s*([}\]])', r'\1', block)
                return json.loads(cleaned)
            except Exception:
                pass
                
    return None

raw = '''Here is the JSON:
```json
{
  "deliberation": {
    "intent_analysis": "open",
    "some_nested_dict": {"inner": "value"}
  },
  "target_dot_id": 15
}
```
Good luck!
'''
print(_extract_json(raw))

raw2 = '''
{
  "deliberation": {
    "intent_analysis": "open"
  },
  "target_dot_id": 15
}
{
  "dummy": 12
}
'''
print(_extract_json(raw2))
