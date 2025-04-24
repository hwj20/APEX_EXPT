import os
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np

# Configuration
file_name = "turing_cat_llm_Hard_LLM_gpt-4o_20250420_201439"
video_path = f"{file_name}.mp4"  # Replace with actual path
output_dir = "frames"
output_image = f"./diff/timeline_comparison{file_name}.png"
frames_to_extract = 10
fps = 1  # extract 1 frame per second
images_per_row = 3
rows = 3

# === STEP 1: Extract frames from video ===
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
video_fps = cap.get(cv2.CAP_PROP_FPS)
interval = int(video_fps * (1 / fps))
extracted_paths = []

for i in range(frames_to_extract):
    cap.set(cv2.CAP_PROP_POS_FRAMES, i * interval)
    ret, frame = cap.read()
    if not ret:
        break
    frame_path = os.path.join(output_dir, f"frame_{i:02d}.png")
    cv2.imwrite(frame_path, frame)
    extracted_paths.append(frame_path)

cap.release()

# === STEP 2: Create a grid layout of the extracted frames ===
images = [Image.open(path) for path in extracted_paths]
width, height = images[0].size
grid_img = Image.new("RGB", (width * images_per_row, height * rows + 30), "white")
draw = ImageDraw.Draw(grid_img)

# Load font
try:
    font = ImageFont.truetype("arial.ttf", 16)
except:
    font = ImageFont.load_default()

for idx, img in enumerate(images):
    x = (idx % images_per_row) * width
    y = (idx // images_per_row) * height
    grid_img.paste(img, (x, y))
    draw.text((x + 10, y + height + 5), f"t={idx}s", fill=(0, 0, 0), font=font)

grid_img.save(output_image)
