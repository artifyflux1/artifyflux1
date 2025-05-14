import os
import cv2

def CropClips(input_folder_string, output_folder_string):
    """
    Crops video files from the input folder to 405x720 resolution (centered crop),
    and saves them to the output folder with the same name and extension.
    
    Parameters:
        input_folder_string (str): Path to the folder containing input videos.
        output_folder_string (str): Path to the folder where cropped videos will be saved.
    
    Returns:
        bool: True if the operation was successful, False if any critical error occurred.
    """

    # Validate input folder
    if not os.path.isdir(input_folder_string):
        print("Error: Input folder does not exist.")
        return False

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder_string):
        try:
            os.makedirs(output_folder_string)
        except Exception as e:
            print(f"Error: Could not create output folder. {e}")
            return False

    # Process each file in the input folder
    for filename in os.listdir(input_folder_string):
        input_path = os.path.join(input_folder_string, filename)

        # Skip if not a file
        if not os.path.isfile(input_path):
            continue

        # Open video capture
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {filename}.")
            continue

        # Get video properties
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Validate resolution
        if original_width != 1280 or original_height != 720:
            print(f"Skipping {filename}: Expected 1280x720 resolution, got {original_width}x{original_height}")
            cap.release()
            continue

        # Calculate crop region (centered)
        target_width = 405
        x_start = (original_width - target_width) // 2

        # Prepare output path
        output_path = os.path.join(output_folder_string, filename)

        # Determine codec based on file extension
        _, ext = os.path.splitext(filename)
        ext = ext.lower()
        if ext == '.mp4':
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        elif ext == '.avi':
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
        else:
            print(f"Warning: Unknown file extension for {filename}, defaulting to mp4v codec.")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        # Create VideoWriter
        writer = cv2.VideoWriter(output_path, fourcc, fps, (target_width, original_height))
        if not writer.isOpened():
            print(f"Error: Could not create VideoWriter for {filename}.")
            cap.release()
            continue

        # Process each frame
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Crop frame from the center
            cropped_frame = frame[:, x_start:x_start + target_width]

            # Write the cropped frame
            writer.write(cropped_frame)

        # Release resources
        cap.release()
        writer.release()

    return True