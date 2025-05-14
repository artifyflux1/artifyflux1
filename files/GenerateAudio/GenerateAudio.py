from elevenlabs import ElevenLabs
import requests
import os

def GenerateAudio(text_prompt_string, eleven_labs_api_key_string, output_filename_with_extension_string):
    """
    Generates audio from text using ElevenLabs API and saves it to a file.
    
    Args:
        text_prompt_string: The text to convert to speech
        eleven_labs_api_key_string: Your ElevenLabs API key
        output_filename_with_extension_string: Output filename with extension (e.g., "audio.mp3")
    
    Returns:
        bool: True if successful, False if any error occurs
    """
    try:
        # Initialize the ElevenLabs client
        client = ElevenLabs(
            api_key=eleven_labs_api_key_string,
        )
        
        # Generate the audio (returns a generator)
        audio_stream = client.text_to_speech.convert(
            voice_id="pNInz6obpgDQGcFmaJgB",
            output_format="mp3_44100_128",
            text=text_prompt_string,
            model_id="eleven_multilingual_v2",
        )
        
        # Write the audio data to file by consuming the generator
        with open(output_filename_with_extension_string, "wb") as f:
            for chunk in audio_stream:
                if chunk:
                    f.write(chunk)
        
        # Verify the file was created
        if os.path.exists(output_filename_with_extension_string) and os.path.getsize(output_filename_with_extension_string) > 0:
            return True
        return False
        
    except requests.exceptions.RequestException as e:
        print(f"HTTP/API Error occurred: {e}")
        return False
    except Exception as e:
        print(f"Error generating audio: {e}")
        return False