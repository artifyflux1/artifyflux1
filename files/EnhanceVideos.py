import os
import shutil
import subprocess
from PIL import Image
from tqdm import tqdm
from realesrgan_ncnn_py import Realesrgan

def EnhanceVideo(input_video_path, output_video_path):
    """
    Enhances a silent video using Real-ESRGAN (CPU only),
    saves output to given path, and cleans up temporary files.
    Shows a progress bar during enhancement.
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

    # Temporary directories
    temp_dir = "temp_frames"
    enhanced_dir = "enhanced_frames"

    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(enhanced_dir, exist_ok=True)

    try:
        # Step 1: Extract frames from video
        subprocess.run([
            "ffmpeg", "-i", input_video_path, "-q:v", "2", f"{temp_dir}/frame_%04d.png"
        ], check=True)

        # Step 2: Initialize Real-ESRGAN
        realesrgan = Realesrgan(gpuid=-1, model=2, tilesize=128)

        # Step 3: Enhance each frame with progress bar
        frame_files = sorted(os.listdir(temp_dir))
        print(f"Enhancing {len(frame_files)} frames...")

        for fname in tqdm(frame_files, desc="Enhancing Frames", unit="frame"):
            in_path = os.path.join(temp_dir, fname)
            out_path = os.path.join(enhanced_dir, fname)
            with Image.open(in_path) as img:
                enhanced = realesrgan.process_pil(img)
                enhanced.save(out_path, quality=70)

        # Step 4: Get video FPS
        fps_str = subprocess.check_output([
            "ffprobe", "-v", "0", "-select_streams", "v:0", "-show_entries",
            "stream=r_frame_rate", "-of", "default=noprint_wrappers=1:nokey=1",
            input_video_path
        ]).decode().strip()

        fps = float(subprocess.check_output([
            'bash', '-c', f'echo "{fps_str}" | bc -l'
        ]).decode().strip())

        # Step 5: Rebuild enhanced video
        subprocess.run([
            "ffmpeg", "-framerate", str(fps),
            "-i", f"{enhanced_dir}/frame_%04d.png",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            output_video_path
        ], check=True)

        print("✅ Silent video enhanced successfully:", output_video_path)

    finally:
        # Clean up temporary directories
        shutil.rmtree(temp_dir, ignore_errors=True)
        shutil.rmtree(enhanced_dir, ignore_errors=True)
