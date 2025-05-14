import os
import ffmpeg
import logging

logging.basicConfig(level=logging.ERROR)

def CompileVideo(input_folder_string: str, 
                 output_video_name_with_extension_string: str, 
                 output_folder_string: str) -> bool:
    try:
        # Validate input folder
        if not os.path.isdir(input_folder_string):
            logging.error(f"Input folder not found: {input_folder_string}")
            return False

        # Create output folder if it doesn't exist
        os.makedirs(output_folder_string, exist_ok=True)

        # Get and sort input clips numerically
        clips = [f for f in os.listdir(input_folder_string) if f.endswith('.mp4')]
        if not clips:
            logging.error("No MP4 files found in input folder")
            return False

        # Sort clips numerically by filename (without extension)
        def numerical_sort(filename):
            return int(os.path.splitext(filename)[0])
        
        clips.sort(key=numerical_sort)

        # Process each clip and prepare concatenation
        processed_streams = []
        for clip in clips:
            input_path = os.path.join(input_folder_string, clip)
            
            stream = ffmpeg.input(input_path)
            
            # Convert FPS from 64 to 60
            stream = stream.filter('fps', fps=60)
            
            # Pad to 405x720 (original 404x720)
            stream = stream.filter('pad', 
                                 width=405, 
                                 height=720, 
                                 x='(405-iw)/2',  # Center horizontally
                                 y=0)
            
            processed_streams.append(stream)

        # Concatenate all processed streams
        concatenated = ffmpeg.concat(*processed_streams, v=1, a=0)

        # Prepare output path
        output_path = os.path.join(output_folder_string, 
                                 output_video_name_with_extension_string)

        # Run FFmpeg command
        (
            ffmpeg
            .output(concatenated, output_path, 
                   vcodec='libx264', 
                   r=60, 
                   aspect='9:16')
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )

        return True

    except Exception as e:
        logging.error(f"Error during video compilation: {str(e)}")
        return False