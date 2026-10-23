import re

path = r'c:\Users\NANDHA K\Documents\antigravity\amazing-bose\spatial_droplet\reasoning\reasoner.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

helper = '''
    @staticmethod
    def _extract_json(raw: str):
        import json
        import re
        raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
        match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw, re.DOTALL | re.IGNORECASE)
        if match:
            try:
                return json.loads(match.group(1))
            except:
                pass
        fb = raw.find('{')
        lb = raw.rfind('}')
        if fb != -1 and lb != -1 and lb > fb:
            json_str = raw[fb:lb+1]
            try:
                return json.loads(json_str)
            except Exception:
                try:
                    cleaned = re.sub(r',\s*([}\]])', r'\\1', json_str)
                    return json.loads(cleaned)
                except:
                    pass
        return None
'''

if '_extract_json' not in content:
    content = content.replace('    @staticmethod\n    def _parse_groq_retry_seconds', helper + '\n    @staticmethod\n    def _parse_groq_retry_seconds')

# Patch ngrok parser
target_ngrok = '''raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
                raw = raw.replace('```json', '').replace('```', '').strip()
                fb = raw.find('{')
                lb = raw.rfind('}')
                if fb != -1 and lb != -1 and lb > fb:
                    try:
                        decision = json.loads(raw[fb:lb+1])
                    except json.JSONDecodeError:'''
replacement_ngrok = '''decision = self._extract_json(raw)
                if decision is not None:
                    if False:
                        pass
                else:
                    try:
                        raise json.JSONDecodeError("", "", 0)
                    except json.JSONDecodeError:'''
content = content.replace(target_ngrok, replacement_ngrok)

# Patch groq parser
target_groq = '''raw = response.choices[0].message.content or ''
                    raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
                    fb = raw.find('{')
                    lb = raw.rfind('}')
                    decision = None
                    if fb != -1 and lb != -1 and lb > fb:
                        json_str = raw[fb:lb+1]
                        try:
                            decision = json.loads(json_str)
                        except Exception:
                            try:
                                cleaned = re.sub(r',\s*([}\]])', r'\\1', json_str)
                                decision = json.loads(cleaned)
                            except Exception:
                                pass'''
replacement_groq = '''raw = response.choices[0].message.content or ''
                    decision = self._extract_json(raw)'''
content = content.replace(target_groq, replacement_groq)

# Explicit prompt instructions
prompt1 = 'Respond with ONLY valid JSON.'
prompt2 = 'Respond with ONLY a single valid JSON object. No preamble or markdown formatting.'
content = content.replace(prompt1, prompt2)

prompt3 = 'You are a JSON-only API. Output ONLY a single valid JSON object starting with { and ending with }.'
prompt4 = 'You are a JSON-only API. Output ONLY a single valid JSON object starting with { and ending with }. Do NOT output any markdown, preambles, or conversational text.'
content = content.replace(prompt3, prompt4)

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Patch applied.")
