import cv2
import numpy as np
import os

def upscale_frame(frame, scale=2):
    return cv2.resize(frame, (frame.shape[1]*scale, frame.shape[0]*scale), interpolation=cv2.INTER_CUBIC)

def sharpen_frame(frame):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    return cv2.filter2D(frame, -1, kernel)

def enhance_contrast(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def denoise_frame(frame):
    return cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)

def enhance_frame(frame):
    frame = upscale_frame(frame)
    frame = enhance_contrast(frame)
    frame = sharpen_frame(frame)
    frame = denoise_frame(frame)
    return frame

def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video {input_path}.")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) * 2
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) * 2
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Ensure the directory exists for output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"🚀 Enhancing: {os.path.basename(input_path)} ({total_frames} frames)...")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        enhanced = enhance_frame(frame)
        out.write(enhanced)

        frame_count += 1
        if frame_count % 10 == 0:
            print(f"  ➤ {frame_count}/{total_frames} frames done")

    cap.release()
    out.release()
    print(f"✅ Saved enhanced video to: {output_path}")
    return True

def EnhanceVideos(input_video_folder_string, output_video_folder_string):
    try:
        input_folder = os.path.abspath(input_video_folder_string)
        output_folder = os.path.abspath(output_video_folder_string)
        print(f"🔍 Input folder: {input_folder}")
        print(f"📁 Output folder: {output_folder}")

        if not os.path.exists(input_folder):
            print("❌ Error: Input folder does not exist.")
            return False

        os.makedirs(output_folder, exist_ok=True)

        video_files = [f for f in os.listdir(input_folder) if f.endswith(('.mp4', '.avi', '.mov'))]
        if not video_files:
            print("❌ No supported video files found in the input folder.")
            return False

        for video_file in video_files:
            input_path = os.path.join(input_folder, video_file)
            output_path = os.path.join(output_folder, video_file)

            if not process_video(input_path, output_path):
                return False

        return True
    except Exception as e:
        print(f"❌ Exception: {str(e)}")
        return False
