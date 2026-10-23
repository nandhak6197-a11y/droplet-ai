import asyncio
import cv2
import json
import sys
from pathlib import Path
sys.path.insert(0, r"c:\Users\NANDHA K\Documents\antigravity\amazing-bose")

from spatial_droplet.vision.sprayer import FullBleedLatticeSprayer
from spatial_droplet.reasoning.reasoner import DiagonalDropletReasoner
from spatial_droplet.core.models import WindowContext

async def count_tokens():
    img = cv2.imread(str(Path('osworld_g_images') / 'GDZVK92HHZ.png'))
    sprayer = FullBleedLatticeSprayer(48)
    droplets = await sprayer.spray(img)
    reasoner = DiagonalDropletReasoner()
    win = WindowContext(hwnd=0, title="OSWorld-G", rect=(0, 0, 1920, 1080), width=1920, height=1080, is_valid=True)
    
    # Mock create to capture kwargs
    captured = {}
    class MockChoices:
        def __init__(self):
            self.message = type('Msg', (), {'content': '{"target_dot_id": 1, "action": "click", "reasoning": "mock"}'})()
    class MockResp:
        def __init__(self):
            self.choices = [MockChoices()]
    class MockCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return MockResp()
    class MockClient:
        chat = type('Chat', (), {'completions': MockCompletions()})()
    for entry in reasoner.key_pool:
        entry['client'] = MockClient()
        
    await reasoner.decide(
        user_intent='Click on the question icon of the save dialog.',
        droplets=droplets,
        window_ctx=win,
        screenshot=img,
        mode="combined",
        target_model="qwen/qwen3.8-27b"
    )
    
    sys_msg = captured['messages'][0]['content']
    user_msg = captured['messages'][1]['content']
    print(f"System message char length: {len(sys_msg)}")
    
    # user_msg has text and image_url
    user_text = next(item['text'] for item in user_msg if item['type'] == 'text')
    print(f"User message text char length: {len(user_text)}")
    img_item = next((item for item in user_msg if item['type'] == 'image_url'), None)
    if img_item:
        b64 = img_item['image_url']['url']
        print(f"Image base64 length: {len(b64)} chars ({len(b64)/1024:.1f} KB)")
        
    # How many tokens is user_text + sys_msg roughly? (approx 4 chars per token)
    est_text_tokens = (len(sys_msg) + len(user_text)) / 4.0
    print(f"Estimated text tokens: {est_text_tokens:.0f}")

asyncio.run(count_tokens())
