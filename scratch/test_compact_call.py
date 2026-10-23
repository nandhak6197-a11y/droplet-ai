import asyncio
import cv2
import json
import sys
import base64
from pathlib import Path
sys.path.insert(0, r"c:\Users\NANDHA K\Documents\antigravity\amazing-bose")
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

from spatial_droplet.vision.sprayer import FullBleedLatticeSprayer
from spatial_droplet.reasoning.groq_rotator import GroqRotator

async def test_compact_call():
    img = cv2.imread(str(Path('osworld_g_images') / 'GDZVK92HHZ.png'))
    sprayer = FullBleedLatticeSprayer(48)
    droplets = await sprayer.spray(img)
    
    # 1. 512px Image
    sh, sw = img.shape[:2]
    scale = 512.0 / max(sw, sh)
    marked = cv2.resize(img, (int(sw * scale), int(sh * scale)), interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode(".jpg", marked, [cv2.IMWRITE_JPEG_QUALITY, 70])
    img_b64 = f"data:image/jpeg;base64,{base64.b64encode(buf).decode('utf-8')}"
    
    # 2. Compact Manifest of Top 40 Candidates
    active = [d for d in droplets if d.category != 'surface_probe']
    manifest_lines = [f"• [{d.id}] ({d.center_x},{d.center_y}) {d.label[:35]}" for d in active[:40]]
    manifest_text = "\n".join(manifest_lines)
    
    # 3. Compact System Prompt
    sys_prompt = """You are a Spatial AI Visual Grounding Brain controlling a Windows PC.
Examine the Set-of-Marks annotated screenshot and candidate manifest.
Choose the single best target Dot ID to fulfill the user's intent.

Respond with ONLY valid JSON:
{
  "deliberation": {
    "intent_analysis": "<step-by-step breakdown>",
    "spatial_anchoring": "<screen zone>",
    "candidate_evaluation": "<comparison and distractor elimination>",
    "action_plan": "<target Dot ID>"
  },
  "reasoning": "<1 sentence>",
  "target_dot_id": <int>,
  "action": "click"
}"""

    user_text = f"""USER REQUEST: "Click on the question icon of the save dialog."
CANDIDATE MANIFEST:
{manifest_text}"""

    print(f"Compact Sys prompt: {len(sys_prompt)} chars")
    print(f"Compact User prompt: {len(user_text)} chars")
    print(f"Compact Image base64: {len(img_b64)} chars ({len(img_b64)/1024:.1f} KB)")
    
    rotator = GroqRotator()
    client = rotator.key_pool[0]['client']
    
    print("Calling Groq Qwen 3.8 27B with compact payload...")
    import time
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model="qwen/qwen3.8-27b",
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": img_b64}}
            ]}
        ],
        temperature=0.1,
        max_tokens=600,
        timeout=30.0
    )
    dt = time.perf_counter() - t0
    print(f"🎉 SUCCESS! Received response in {dt:.2f}s!")
    print("Response Content:")
    print(resp.choices[0].message.content)

asyncio.run(test_compact_call())
