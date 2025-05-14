import cv2
import numpy as np
import os
from tqdm import tqdm

def is_mp4_file(filename):
    return os.path.splitext(filename)[1].lower() == '.mp4'

def process_single_video(input_path, output_path):
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
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))

        if not writer.isOpened():
            cap.release()
            return False

        # Read first frame
        success, frame_prev = cap.read()
        if not success:
            cap.release()
            writer.release()
            return False
        writer.write(frame_prev)

        x, y = np.meshgrid(np.arange(width), np.arange(height))
        sharpen_kernel = np.array([[0, -0.1, 0],
                                   [-0.1, 1.4, -0.1],
                                   [0, -0.1, 0]])

        with tqdm(total=total_frames, desc=f"Processing {os.path.basename(input_path)}", leave=False) as pbar:
            pbar.update(1)

            while True:
                success, frame_curr = cap.read()
                if not success:
                    break

                # Convert to grayscale with histogram equalization
                gray_prev = cv2.equalizeHist(cv2.cvtColor(frame_prev, cv2.COLOR_BGR2GRAY))
                gray_curr = cv2.equalizeHist(cv2.cvtColor(frame_curr, cv2.COLOR_BGR2GRAY))

                # Improved optical flow parameters
                flow = cv2.calcOpticalFlowFarneback(
                    gray_prev, gray_curr, None,
                    pyr_scale=0.5,
                    levels=3,
                    winsize=15,
                    iterations=3,
                    poly_n=5,
                    poly_sigma=1.2,
                    flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN
                )

                # Create forward and backward mappings with boundary handling
                map_x_forward = (x - 0.5 * flow[:, :, 0]).astype(np.float32)
                map_y_forward = (y - 0.5 * flow[:, :, 1]).astype(np.float32)
                map_x_backward = (x + 0.5 * flow[:, :, 0]).astype(np.float32)
                map_y_backward = (y + 0.5 * flow[:, :, 1]).astype(np.float32)

                # Warp with border replication
                warped_prev = cv2.remap(frame_prev, map_x_forward, map_y_forward,
                                       interpolation=cv2.INTER_CUBIC,
                                       borderMode=cv2.BORDER_REPLICATE)
                warped_curr = cv2.remap(frame_curr, map_x_backward, map_y_backward,
                                       interpolation=cv2.INTER_CUBIC,
                                       borderMode=cv2.BORDER_REPLICATE)

                # Improved blending with edge-aware combination
                mask = np.ones_like(warped_prev, dtype=np.float32)
                mask[warped_prev == 0] = 0  # Exclude pure black pixels (if any)
                mask[warped_curr == 0] = 0

                interpolated_frame = cv2.addWeighted(warped_prev, 0.5, warped_curr, 0.5, 0)
                
                # Post-processing pipeline
                interpolated_frame = cv2.medianBlur(interpolated_frame, 3)
                interpolated_frame = cv2.filter2D(interpolated_frame, -1, sharpen_kernel)
                interpolated_frame = cv2.fastNlMeansDenoisingColored(interpolated_frame, None, 3, 3, 7, 21)

                # Write frames
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
    try:
        if not os.path.isdir(input_videos_folder_path):
            print("Error: Input folder does not exist.")
            return False

        os.makedirs(output_videos_folder_path, exist_ok=True)

        video_files = [f for f in os.listdir(input_videos_folder_path) if is_mp4_file(f)]
        if not video_files:
            print("No .mp4 video files found in input folder.")
            return False

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