import os

def ExtractPrompts(input_file_string, output_file_string):
    try:
        with open(input_file_string, 'r', encoding='utf-8') as file:
            content = file.read()

        start_tag = "<visuals>"
        end_tag = "</visuals>"

        start_index = content.find(start_tag)
        end_index = content.find(end_tag)

        if start_index == -1 or end_index == -1:
            return False

        # Include the closing tag in the selection
        end_index += len(end_tag)
        extracted = content[start_index:end_index]

        # Remove empty lines and strip extra surrounding whitespace
        cleaned = "\n".join(line for line in extracted.splitlines() if line.strip())

        with open(output_file_string, 'w', encoding='utf-8') as outfile:
            outfile.write(cleaned.strip() + '\n')

        return True

    except Exception:
        return False
