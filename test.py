import os
import subprocess
from PIL import Image
from realesrgan_ncnn_py import Realesrgan

# Input and output paths
input_video = "generated_videos/video_1.mp4"
output_video = "output_videos/output1.mp4"

# Create temporary folders
os.makedirs("temp_frames", exist_ok=True)
os.makedirs("enhanced_frames", exist_ok=True)

# Step 1: Extract frames from the video
subprocess.run([
    "ffmpeg", "-i", input_video, "-q:v", "2", "temp_frames/frame_%04d.png"
], check=True)

# Step 2: Initialize Real-ESRGAN
realesrgan = Realesrgan(gpuid=-1, model=2, tilesize=128)  # Optimized for CPU

# Step 3: Enhance each frame
frame_files = sorted(os.listdir("temp_frames"))
print(f"Enhancing {len(frame_files)} frames...")

for fname in frame_files:
    in_path = os.path.join("temp_frames", fname)
    out_path = os.path.join("enhanced_frames", fname)
    with Image.open(in_path) as img:
        enhanced = realesrgan.process_pil(img)
        enhanced.save(out_path, quality=70)

# Step 4: Get original video FPS
fps_str = subprocess.check_output([
    "ffprobe", "-v", "0", "-select_streams", "v:0", "-show_entries",
    "stream=r_frame_rate", "-of", "default=noprint_wrappers=1:nokey=1", input_video
]).decode().strip()

# Convert to float using bash + bc
fps = float(subprocess.check_output([
    'bash', '-c', f'echo "{fps_str}" | bc -l'
]).decode().strip())

# Step 5: Recreate video from enhanced frames
subprocess.run([
    "ffmpeg", "-framerate", str(fps),
    "-i", "enhanced_frames/frame_%04d.png",
    "-c:v", "libx264", "-pix_fmt", "yuv420p",
    output_video
], check=True)

print("✅ Silent video enhanced successfully:", output_video)
