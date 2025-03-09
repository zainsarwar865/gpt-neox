import json

def txt_to_jsonl(input_file, output_file, words_per_chunk=10000):
    """
    Efficiently converts a large text file into JSONL format by splitting it into chunks of words.
    
    Args:
        input_file (str): Path to the large .txt file (e.g., 70GB file).
        output_file (str): Path to the output .jsonl file.
        words_per_chunk (int): Number of words per JSONL entry.
    """
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        buffer = []  # Stores words before writing a JSON chunk
        word_count = 0

        for line in infile:
            words = line.strip().split()  # Split line into words
            buffer.extend(words)  # Add words to buffer
            word_count += len(words)

            # If buffer reaches the threshold, write to JSONL
            if word_count >= words_per_chunk:
                json.dump({"text": " ".join(buffer)}, outfile)
                outfile.write("\n")  # Newline for JSONL format
                
                # Reset buffer
                buffer = []
                word_count = 0

        # Write remaining words if any
        if buffer:
            json.dump({"text": " ".join(buffer)}, outfile)
            outfile.write("\n")

    print(f"✅ Conversion completed! JSONL saved at: {output_file}")

# Example usage
txt_to_jsonl("/scratch/bdgs/zsarwar/datasets/raw/fineweb_edu_raw_12B.txt", "/scratch/bdgs/zsarwar/datasets/raw/fineweb_edu_12B.jsonl", words_per_chunk=2048)