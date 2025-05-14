from moviepy.editor import VideoFileClip
import os

def ChangeDuration(clips_durations_text_file_path_string, input_videos_folder_path_string, output_videos_folder_path_string):
    try:
        # Ensure the output directory exists
        if not os.path.exists(output_videos_folder_path_string):
            os.makedirs(output_videos_folder_path_string)
            print(f"Created output directory: {output_videos_folder_path_string}")

        # Read the text file containing video filenames and durations
        with open(clips_durations_text_file_path_string, 'r') as f:
            lines = f.readlines()

        # Process each line in the text file
        for line in lines:
            try:
                # Parse filename and desired duration
                filename, duration_str = line.strip().split()
                desired_duration = float(duration_str)

                # Construct input and output file paths
                input_video_path = os.path.join(input_videos_folder_path_string, filename)
                output_video_path = os.path.join(output_videos_folder_path_string, filename)

                # Load the video clip
                clip = VideoFileClip(input_video_path)
                original_duration = clip.duration  # Should be 5 seconds

                # Define time transformation to adjust speed
                time_transform = lambda t: t * (original_duration / desired_duration)

                # Create new clip with adjusted speed and duration
                new_clip = clip.fl_time(time_transform).set_duration(desired_duration)

                # Write the modified video to the output folder at 32 FPS
                new_clip.write_videofile(output_video_path, fps=32)

                # Close clips to free resources
                clip.close()
                new_clip.close()

            except Exception as e:
                print(f"Error processing {filename}: {e}")
                return False

        return True

    except Exception as e:
        print(f"Error in ChangeDuration: {e}")
        return False