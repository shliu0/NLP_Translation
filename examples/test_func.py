import os
from tqdm import tqdm

# Function to translate text
def translate(text, src_lang, tgt_lang, model, tokenizer):
    tokenizer.src_lang = src_lang
    encoded_text = tokenizer(text, return_tensors="pt").to(model.device)
    generated_tokens = model.generate(**encoded_text, forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang])
    return tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

def postprocess(input_file_path, output_file_path):
    try:
        # Read the input file and remove double quotes from each line
        with open(input_file_path, 'r', encoding='utf-8') as infile:
            lines = [line.replace('"', '') for line in infile]

        # Write the processed lines to a new TXT file
        with open(output_file_path, 'w', encoding='utf-8') as outfile:
            outfile.writelines(lines)

        print(f"File processed successfully. Output saved to {output_file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

def test(model, tokenizer, output_dir, eval_batch_size=16):
    # Define source and target languages
    source_language = "zh_CN" # Replace with the source language code
    mtarl2tarl = {"en_XX":"en", "es_XX":"es", "fr_XX":"fr", "ru_RU":"ru"}
    test_files = ["datasets/test/test-en.zh.txt",
                "datasets/test/test-es.zh.txt",
                "datasets/test/test-fr.zh.txt",
                "datasets/test/test-ru.zh.txt",
                ]
    tar_langs = ["en_XX", "es_XX", "fr_XX", "ru_RU"]
    # Translate each line using GPU
    translated_lines = []
    for target_language, file_path in zip(tar_langs, test_files):
        cnt = 0
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        for line in tqdm(lines):
            translated_lines.append(translate(line.strip(), source_language, target_language, model, tokenizer))
        translated_lines = [f"{i+1+(cnt*500)}\t{line}\t{mtarl2tarl[target_language]}" for i,line in enumerate(translated_lines)]
        cnt += 1

    # Output the translated lines
    output_file_path = os.path.join(output_dir, f"test.txt")
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    with open(output_file_path, 'w', encoding='utf-8') as file:
        file.write("id\ttext\ttype\n")  # Writing the header
        file.writelines("\n".join(translated_lines))

    print(f"Translation completed. Output saved to {output_file_path}")

    submit_file_path = os.path.join(output_dir, f"submit.txt")
    postprocess(output_file_path, submit_file_path)
