import cv2
import numpy as np
import os
from tqdm import tqdm

def is_mp4_file(filename):
    """
    Checks if the file is a .mp4 video.
    """
    return os.path.splitext(filename)[1].lower() == '.mp4'

def process_single_video(input_path, output_path):
    """
    Processes a single video to double its FPS using optical flow interpolation.
    Returns True on success, False on failure.
    """
    try:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            return False

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        input_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if width <= 0 or height <= 0 or input_fps <= 0:
            cap.release()
            return False

        output_fps = 2 * input_fps

        # Define video writer (MP4 format with 'mp4v' codec)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))

        if not writer.isOpened():
            cap.release()
            return False

        # Read and write the first frame
        success, frame_prev = cap.read()
        if not success:
            cap.release()
            writer.release()
            return False
        writer.write(frame_prev)

        # Precompute meshgrid
        x, y = np.meshgrid(np.arange(width), np.arange(height))

        # Sharpening kernel for post-processing
        sharpen_kernel = np.array([[0, -0.2, 0],
                                   [-0.2, 1.8, -0.2],
                                   [0, -0.2, 0]])

        with tqdm(total=total_frames, desc=f"Processing {os.path.basename(input_path)}", leave=False) as pbar:
            pbar.update(1)

            while True:
                success, frame_curr = cap.read()
                if not success:
                    break

                # Convert to grayscale
                gray_prev = cv2.cvtColor(frame_prev, cv2.COLOR_BGR2GRAY)
                gray_curr = cv2.cvtColor(frame_curr, cv2.COLOR_BGR2GRAY)

                # Compute optical flow
                flow = cv2.calcOpticalFlowFarneback(
                    gray_prev, gray_curr, None,
                    pyr_scale=0.7, levels=5, winsize=20,
                    iterations=10, poly_n=7, poly_sigma=1.5,
                    flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN
                )

                # Remap coordinates for warping
                map_x_forward = (x - 0.5 * flow[:, :, 0]).astype(np.float32)
                map_y_forward = (y - 0.5 * flow[:, :, 1]).astype(np.float32)
                map_x_backward = (x + 0.5 * flow[:, :, 0]).astype(np.float32)
                map_y_backward = (y + 0.5 * flow[:, :, 1]).astype(np.float32)

                # Apply high-quality warping
                warped_prev = cv2.remap(frame_prev, map_x_forward, map_y_forward, cv2.INTER_CUBIC)
                warped_curr = cv2.remap(frame_curr, map_x_backward, map_y_backward, cv2.INTER_CUBIC)

                # Blend frames
                interpolated_frame = ((warped_prev.astype(float) + warped_curr.astype(float)) / 2).astype(np.uint8)

                # Post-processing: blur to reduce artifacts, sharpen to restore detail
                interpolated_frame = cv2.GaussianBlur(interpolated_frame, (3, 3), 0)
                interpolated_frame = cv2.filter2D(interpolated_frame, -1, sharpen_kernel)

                # Write interpolated and current frame
                writer.write(interpolated_frame)
                writer.write(frame_curr)

                frame_prev = frame_curr
                pbar.update(1)

        cap.release()
        writer.release()
        return True

    except Exception as e:
        print(f"Error processing {input_path}: {str(e)}")
        return False

def IncreaseFPS(input_videos_folder_path, output_videos_folder_path):
    """
    Processes all .mp4 videos in the input folder, doubles their FPS,
    and saves the output videos in the output folder with the same name.

    Returns:
        bool: True if all videos were processed successfully, False otherwise.
    """
    try:
        # Validate input folder
        if not os.path.isdir(input_videos_folder_path):
            print("Error: Input folder does not exist.")
            return False

        # Ensure output folder exists
        os.makedirs(output_videos_folder_path, exist_ok=True)

        # Get all .mp4 video files
        video_files = [f for f in os.listdir(input_videos_folder_path) if is_mp4_file(f)]
        if not video_files:
            print("No .mp4 video files found in input folder.")
            return False

        # Process each video
        for video_file in video_files:
            input_path = os.path.join(input_videos_folder_path, video_file)
            output_path = os.path.join(output_videos_folder_path, video_file)

            print(f"Processing: {video_file}")
            if not process_single_video(input_path, output_path):
                print(f"Failed to process {video_file}. Aborting.")
                return False

        return True

    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return False