import tkinter as tk
from PIL import Image, ImageTk
import os
import sys

root = tk.Tk()
root.withdraw()

win = tk.Toplevel(root)
win.overrideredirect(True)
win.wm_attributes("-topmost", True)
win.wm_attributes("-transparentcolor", "#010101")
win.config(bg="#010101")
win.geometry("84x84+400+100")

# Load sphere image
img_path = "spatial_droplet/assets/sphere_orb_80.png"
if os.path.exists(img_path):
    pil_img = Image.open(img_path)
    tk_img = ImageTk.PhotoImage(pil_img)
else:
    tk_img = None

canvas = tk.Canvas(win, width=84, height=84, bg="#010101", highlightthickness=0)
canvas.pack(fill="both", expand=True)

if tk_img:
    canvas.create_image(42, 42, image=tk_img)

print("Sphere widget test initialized successfully.")
win.destroy()
root.destroy()
