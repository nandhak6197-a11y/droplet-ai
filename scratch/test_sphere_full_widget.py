import os
import sys
import tkinter as tk
from PIL import Image, ImageTk, ImageDraw

def test_widget():
    root = tk.Tk()
    root.withdraw()

    # Generate assets if missing
    src = r'C:\Users\NANDHA K\.gemini\antigravity\brain\77172af1-65f6-41cc-bf5d-0a2f27cf94ff\.user_uploaded\media_1791987065371_67a86119.png'
    raw = Image.open(src).convert('RGBA')
    cx, cy, r = 160, 131, 131
    crop = raw.crop((cx - r, cy - r, cx + r, cy + r))

    def make_avatar(size, halo_color=(0, 229, 255)):
        scale = 4
        cs = (size * scale, size * scale)
        mask = Image.new('L', cs, 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((4, 4, cs[0] - 5, cs[1] - 5), fill=255)
        hr_crop = crop.resize(cs, Image.Resampling.LANCZOS)
        avatar = Image.new('RGBA', cs, (0, 0, 0, 0))
        avatar.paste(hr_crop, (0, 0), mask=mask)
        glow_draw = ImageDraw.Draw(avatar)
        glow_draw.ellipse((4, 4, cs[0] - 5, cs[1] - 5), outline=(*halo_color, 220), width=scale * 2)
        return avatar.resize((size, size), Image.Resampling.LANCZOS)

    os.makedirs('spatial_droplet/assets', exist_ok=True)
    orb_80 = make_avatar(80)
    orb_80.save('spatial_droplet/assets/sphere_orb_80.png')
    orb_56 = make_avatar(56)
    orb_56.save('spatial_droplet/assets/sphere_orb_56.png')

    # Test window creation
    win = tk.Toplevel(root)
    win.overrideredirect(True)
    win.wm_attributes("-topmost", True)
    win.wm_attributes("-transparentcolor", "#010101")
    win.config(bg="#010101")
    win.geometry("580x76+350+40")

    canvas = tk.Canvas(win, width=580, height=76, bg="#010101", highlightthickness=0)
    canvas.pack(fill="both", expand=True)

    # Test pill generation
    w, h, pill_r = 580, 76, 38
    scale = 3
    bg_img = Image.new('RGBA', (w * scale, h * scale), (1, 1, 1, 0))
    draw = ImageDraw.Draw(bg_img)
    draw.rounded_rectangle((3, 3, w * scale - 4, h * scale - 4), radius=pill_r * scale, fill=(13, 17, 23, 255), outline=(0, 229, 255, 255), width=scale * 2)
    bg_pill = bg_img.resize((w, h), Image.Resampling.LANCZOS)
    tk_bg = ImageTk.PhotoImage(bg_pill)
    canvas.create_image(w // 2, h // 2, image=tk_bg)

    tk_sphere_56 = ImageTk.PhotoImage(orb_56)
    canvas.create_image(42, 38, image=tk_sphere_56)

    print("Full widget rendering test completed successfully.")
    win.destroy()
    root.destroy()

if __name__ == '__main__':
    test_widget()
