from huggingface_hub import InferenceClient

def VideoPrompts(user_prompt_string: str,
                huggingface_api_key_string: str,
                sentence_count: int) -> bool:
    # Dynamically build the system prompt with the sentence count
    system_prompt = f"""YOU WILL RECEIVE EXACTLY {sentence_count} SENTENCES AND MUST OUTPUT EXACTLY ONE VIDEO GENERATOR PROMPT PER SENTENCE, NUMBERED 1. 2. 3. ETC. EACH VIDEO GENERATION PROMPT MUST APPEAR ON ITS OWN LINE WITH NO EMPTY LINES BETWEEN PROMPTS AND BE ENCLOSED IN A SINGLE CODE BLOCK NOT WRAPPED IN DOUBLE QUOTES. PROMPTS SHOULD AND MUST CONTAIN ONLY PLAIN WORDS AND COMMAS. SPECIAL CHARACTERS LIKE HYPHENS, QUOTES AND DOUBLE QUOTES SHOULD STICTLY BE AVOIDED. EACH PROMPT MUST BE 15 TO 30 WORDS IN LENGTH, ACCURATELY DESCRIBE THE SCENE ELEMENTS OF ITS CORRESPONDING SENTENCE, AND INCLUDE NO EXTRA INTERPRETATION OR FANCIFUL LANGUAGE. MAKE SURE THE PROMPT IS EXTREMELY SIMPLE AND EASY WITHOUT ANY HARD WORDS. MAKE SURE THE PROMPT IS WELL OPTIMIZED FOR AI VIDEO GENERATION. FOLLOW THESE RULES WITHOUT EXCEPTION."""

    try:
        client = InferenceClient(
            provider="novita",
            api_key=huggingface_api_key_string,
        )

        completion = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-V3-0324",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt_string}
            ],
            max_tokens=512,
        )

        ai_response = completion.choices[0].message.content

        # Extract code block content
        parts = ai_response.split('```')
        if len(parts) < 3:
            print("No code block found in response")
            return False

        code_block = parts[1].strip()

        # Remove potential language identifier (first line if single word)
        code_lines = code_block.split('\n')
        if code_lines:
            first_line = code_lines[0].strip()
            if first_line and ' ' not in first_line:
                code_lines = code_lines[1:]

        # Process lines to remove empty lines and trailing spaces
        processed_lines = []
        for line in code_lines:
            stripped_line = line.rstrip()
            if stripped_line:
                processed_lines.append(stripped_line)

        if not processed_lines:
            print("Empty code block after processing")
            return False

        final_code = '\n'.join(processed_lines).strip()

        # Write processed code to file
        with open("video_generator_prompts.txt", "w") as file:
            file.write(final_code)

        return True

    except Exception as e:
        print(f"An error occurred: {e}")
        return False