import re
import os
from files.GeneratePrompts import GeneratePrompts
from files.ExtractPrompts import ExtractPrompts
from files.GenerateImages import ImageGen
from files.GenerateVideos import VideoGen
from files.EnhanceVideos import EnhanceVideos

HF_TOKEN_1 = ""

# ===============================

def read_file_as_string(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        return file.read()

def run_chain(functions):
    for func in functions:
        result = func()

        # If function returns a bool, respect it
        if isinstance(result, bool):
            if not result:
                print(f"{func.__name__} failed. Halting chain.")
                return
        else:
            # Void function or other return types are treated as success
            continue

# ==== Wrapper Functions ====

def GeneratePromptsWrapper():
    prompt_script = read_file_as_string("script.txt")
    return GeneratePrompts(HF_TOKEN_1, prompt_script, "temp.txt")

def ExtractPromptsWrapper():
    return ExtractPrompts("temp.txt", "prompts.txt")

def GenerateImagesWrapper():
    return ImageGen("prompts.txt", "generated_images")

def VideoGenWrapper():
    return VideoGen("prompts.txt", "generated_images", "generated_videos")

def EnhanceVideosWrapper():
    return EnhanceVideos("generated_videos", "videos")

# ==== Execute the chain ====

run_chain([
    #GeneratePromptsWrapper,
    #ExtractPromptsWrapper,
    #GenerateImagesWrapper,
    #VideoGenWrapper,
    EnhanceVideosWrapper,
])