import cv2
import winocr

img = cv2.imread(r"C:/Users/NANDHA K/.gemini/antigravity/brain/77172af1-65f6-41cc-bf5d-0a2f27cf94ff/.user_uploaded/media_1791998496786_2d52b8ac.png")
h, w = img.shape[:2]
bot = img[int(h*0.88):h, :]
cv2.imwrite("scratch/bottom_bar.png", bot)
bot_2x = cv2.resize(bot, (w*2, bot.shape[0]*2), interpolation=cv2.INTER_CUBIC)
cv2.imwrite("scratch/bottom_bar_2x.png", bot_2x)
op = winocr.recognize_cv2(bot_2x, lang='en')
res = op.get() if hasattr(op, 'get') else op
print("Bottom bar 2x lines:")
for l in res.lines:
    print(f"  \"{l.text}\" at ({l.words[0].bounding_rect.x}, {l.words[0].bounding_rect.y})")
