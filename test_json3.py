import json
import re

def _extract_json(raw: str):
    if not raw: return None
    raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw, re.DOTALL | re.IGNORECASE)
    if match:
        try: return json.loads(match.group(1))
        except: pass
    start_indices = [i for i, c in enumerate(raw) if c == '{']
    blocks = []
    for start in start_indices:
        stack = 0
        for i in range(start, len(raw)):
            if raw[i] == '{': stack += 1
            elif raw[i] == '}': stack -= 1
            if stack == 0:
                blocks.append(raw[start:i+1])
                break
    blocks.sort(key=len, reverse=True)
    for block in blocks:
        try: return json.loads(block)
        except:
            try: return json.loads(re.sub(r',\s*([}\]])', r'\1', block))
            except: pass
    return None

raw = '''Here is the JSON { you requested:
```json
{
  "x": 10,
  "y": 20
}
```
'''
print(_extract_json(raw))

raw2 = '''Some text { unmatched
{
  "x": 10,
  "y": 20
}
And some { other text }
'''
print(_extract_json(raw2))
