from files.VideoPrompts.VideoPrompts import VideoPrompts
from files.GenerateVideos.GenerateVideos import GenerateVideos
from files.CropClips.CropClips import CropClips
from files.GenerateAudio.GenerateAudio import GenerateAudio
from files.IncreaseFPS.IncreaseFPS import IncreaseFPS
from files.ClipDurations.ClipDurations import ClipDurations
from files.ChangeDuration.ChangeDuration import ChangeDuration

import re
import os
import sys

HF_TOKEN_1 = ""
HF_TOKEN_2 = ""
HF_TOKEN_3 = ""
EL_API_KEY = ""

# ===============================

def count_sentences(filename):
    with open(filename, 'r') as file:
        text = file.read().replace('\n', ' ')
    sentence_endings = re.findall(r'[.!?]+(?=\s+|$)', text)
    return len(sentence_endings)

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

def VideoPromptsWrapper():
    with open("script.txt", "r") as file:
        script = file.read()
    sentence_num = count_sentences("script.txt")
    return VideoPrompts(script, HF_TOKEN_1, sentence_num)

def GenerateVideosWrapper():
    return GenerateVideos("video_generator_prompts.txt", "output_1", HF_TOKEN_1, HF_TOKEN_2, HF_TOKEN_3)

def CropClipsWrapper():
    return CropClips("output_1", "output_2")

def GenerateAudioWrapper():
    with open("script.txt", "r") as file:
        script = file.read()
    return GenerateAudio(script, EL_API_KEY, "audio.mp3")

def ClipDurationsWrapper():
    return ClipDurations("audio.mp3", "clip_durations.txt")

def IncreaseFPSWrapper1():
    return IncreaseFPS("output_2", "output_3")

def ChangeDurationWrapper():
    return ChangeDuration("clip_durations.txt", "output_3", "output_4")

def IncreaseFPSWrapper2():
    return IncreaseFPS("output_4", "output_5")

# ==== Execute the chain ====

run_chain([
    #VideoPromptsWrapper,
    #GenerateVideosWrapper,
    #CropClipsWrapper,
    #GenerateAudioWrapper,
    #ClipDurationsWrapper,
    #IncreaseFPSWrapper1,
    #ChangeDurationWrapper,
    #IncreaseFPSWrapper2,
])