import cv2
import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor

# --- Interpolation Functions (same as before) ---
def horizontal_interpolation(img: np.ndarray) -> np.ndarray:
    img_float = img.astype(np.float32)
    height, width, ch = img_float.shape
    new_width = 2 * width - 1
    new_img = np.zeros((height, new_width, ch), dtype=np.float32)

    new_img[:, ::2] = img_float
    new_img[:, 1::2] = (img_float[:, :-1] + img_float[:, 1:]) / 2  # Proper averaging

    return np.clip(new_img, 0, 255).astype(np.uint8)

def vertical_interpolation(img: np.ndarray) -> np.ndarray:
    img_float = img.astype(np.float32)
    height, width, ch = img_float.shape
    new_height = 2 * height - 1
    new_img = np.zeros((new_height, width, ch), dtype=np.float32)

    new_img[::2] = img_float
    new_img[1::2] = (img_float[:-1] + img_float[1:]) / 2

    return np.clip(new_img, 0, 255).astype(np.uint8)

# --- Edge Enhancement Function ---
def edge_enhance_in_memory(img: np.ndarray) -> np.ndarray:
    kernel = np.array([[-1, -1, -1, -1, -1],
                       [-1,  2,  2,  2, -1],
                       [-1,  2,  8,  2, -1],
                       [-1,  2,  2,  2, -1],
                       [-1, -1, -1, -1, -1]], dtype=np.float32) / 8.0
    return cv2.filter2D(img, -1, kernel)

# --- Denoising Function ---
def denoise_frame(frame: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoisingColored(frame, None, 7, 7, 7, 21)

# --- Video Processing Function ---
def process_single_video(args):
    input_path, output_path = args
    try:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {input_path}")

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Compute new dimensions
        new_width = 2 * width - 1
        new_height = 2 * height - 1

        # Set up VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Apply enhancement pipeline
            interpolated = horizontal_interpolation(frame)
            interpolated = vertical_interpolation(interpolated)
            denoised = denoise_frame(interpolated)
            enhanced = edge_enhance_in_memory(denoised)

            # Write enhanced frame
            out.write(enhanced)
            frame_count += 1

        cap.release()
        out.release()

        print(f"Processed {frame_count}/{total_frames} frames from '{input_path}'")
        return True, None

    except Exception as e:
        return False, str(e)

# --- Main Function for Video Batch Processing ---
def EnhanceVideos(input_folder: str, output_folder: str) -> bool:
    try:
        if not os.path.exists(input_folder):
            print(f"Input folder '{input_folder}' not found.")
            return False

        os.makedirs(output_folder, exist_ok=True)

        # Find all video files
        valid_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv')
        video_files = [
            f for f in os.listdir(input_folder)
            if f.lower().endswith(valid_extensions)
        ]

        if not video_files:
            print("No valid video files found in input folder.")
            return False

        args_list = []
        for filename in video_files:
            input_path = os.path.join(input_folder, filename)
            output_name = os.path.splitext(filename)[0] + ".mp4"
            output_path = os.path.join(output_folder, output_name)
            args_list.append((input_path, output_path))

        # Process videos in parallel
        with ProcessPoolExecutor() as executor:
            results = list(executor.map(process_single_video, args_list))

        all_success = True
        for success, error in results:
            if not success:
                print(f"Error: {error}")
                all_success = False

        return all_success

    except Exception as e:
        print(f"Critical error during processing: {str(e)}")
        return False