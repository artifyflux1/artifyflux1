import whisper

def ClipDurations(input_audio_file_string: str, output_text_file_string: str) -> bool:
    try:
        # Load the Whisper model
        model = whisper.load_model("base")
        
        # Transcribe the audio file
        result = model.transcribe(input_audio_file_string)

        # Combine segments to improve sentence matching
        durations = []
        current_start = None
        current_end = None
        for segment in result["segments"]:
            if current_start is None:
                current_start = segment["start"]
            current_end = segment["end"]
            # Check if this segment ends a sentence
            if segment["text"].strip().endswith(('.', '?', '!', '."')):
                duration = current_end - current_start
                durations.append(duration)
                current_start = None  # Reset for next sentence

        # Add 1 second to the last duration if applicable
        if durations:
            durations[-1] += 1

        # Write durations to the output file
        with open(output_text_file_string, "w") as file:
            for idx, duration in enumerate(durations, 1):
                line = f"{idx}.mp4 {duration:.2f}"
                if idx != len(durations):
                    line += "\n"
                file.write(line)

        return True  # Success
    
    except Exception as e:
        print(f"Error: {e}")
        return False  # Failure