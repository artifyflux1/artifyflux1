import os
import shutil
from huggingface_hub import InferenceClient

def GenerateVideos(video_generator_prompts_file_string, output_folder_string, api_key_string_1, api_key_string_2, api_key_string_3):
    """
    Generates videos from prompts in a text file using Hugging Face's Inference API.
    
    Parameters:
        video_generator_prompts_file_string (str): Path to the prompt text file.
        output_folder_string (str): Path to the output folder for generated videos.
        api_key_string_1 (str): Primary API key.
        api_key_string_2 (str): Fallback API key.
        api_key_string_3 (str): Final fallback API key.

    Returns:
        bool: True if all videos are successfully generated, False otherwise.
    """
    
    # List of API keys in order of priority
    api_keys = [api_key_string_1, api_key_string_2, api_key_string_3]

    # Step 1: Ensure output folder exists and is empty
    try:
        if os.path.exists(output_folder_string):
            shutil.rmtree(output_folder_string)
        os.makedirs(output_folder_string)
    except Exception as e:
        print(f"Error initializing output folder: {e}")
        return False

    # Step 2: Read the prompt file
    try:
        with open(video_generator_prompts_file_string, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading prompt file: {e}")
        return False

    # Step 3: Parse prompts and line numbers
    prompts = []
    for idx, line in enumerate(lines, start=1):
        line = line.strip()
        if not line or '.' not in line:
            continue

        parts = line.split('.', 1)
        if len(parts) < 2:
            continue

        number_str, prompt_text = parts[0].strip(), parts[1].strip()
        if not number_str.isdigit():
            continue

        prompt_number = int(number_str)
        if prompt_text:
            prompts.append((prompt_number, prompt_text))

    if not prompts:
        print("No valid prompts found in the file.")
        return False

    # Step 4: Generate videos for each prompt
    for prompt_number, prompt_text in prompts:
        video_saved = False
        for api_key in api_keys:
            try:
                client = InferenceClient(provider="fal-ai", api_key=api_key)
                video_bytes = client.text_to_video(
                    prompt=prompt_text,
                    model="Wan-AI/Wan2.1-T2V-14B"
                )
                output_path = os.path.join(output_folder_string, f"{prompt_number}.mp4")
                with open(output_path, 'wb') as f:
                    f.write(video_bytes)
                video_saved = True
                break
            except Exception as e:
                print(f"Failed with API key: {e}")
                continue

        if not video_saved:
            print(f"Failed to generate video for prompt: {prompt_text}")
            return False

    return True