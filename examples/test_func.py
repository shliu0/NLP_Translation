import os
from tqdm import tqdm
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    DataCollatorForSeq2Seq,
    default_data_collator,
)
from torch.utils.data import DataLoader

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
    total_lines_count = 0 
    all_translated_lines = [] 
    for target_language, file_path in zip(tar_langs, test_files):
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        for line in tqdm(lines):
            translated_line = translate(line.strip(), source_language, target_language, model, tokenizer)
            all_translated_lines.append(f"{total_lines_count+1}\t{translated_line}\t{mtarl2tarl[target_language]}")
            total_lines_count += 1

    # Output the translated lines
    output_file_path = os.path.join(output_dir, f"test.txt")
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    with open(output_file_path, 'w', encoding='utf-8') as file:
        file.write("id\ttext\ttype\n")  # Writing the header
        file.writelines("\n".join(all_translated_lines))

    print(f"Translation completed. Output saved to {output_file_path}")

    submit_file_path = os.path.join(output_dir, f"submit.txt")
    postprocess(output_file_path, submit_file_path)


def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]

        return preds, labels


def test_mgpu(model, tokenizer, output_dir, accelerator, args, gen_kwargs=None):
    # Define source and target languages
    source_lang = "zh_CN" # Replace with the source language code
    mtarl2tarl = {"en_XX":"en", "es_XX":"es", "fr_XX":"fr", "ru_RU":"ru"}
    test_files = ["datasets/test/test-en.zh.txt",
                "datasets/test/test-es.zh.txt",
                "datasets/test/test-fr.zh.txt",
                "datasets/test/test-ru.zh.txt",
                ]
    tar_langs = ["en_XX", "es_XX", "fr_XX", "ru_RU"]

    total_lines_count = 0 
    all_translated_lines = [] 
    for test_file, target_lang in zip(test_files, tar_langs):
        translated_lines = test_per_lang(model, tokenizer, accelerator, test_file, source_lang, target_lang, args, gen_kwargs)
        for line in tqdm(translated_lines):
            all_translated_lines.append(f"{total_lines_count+1}\t{line}\t{target_lang}")
            total_lines_count += 1

        # Output the translated lines
    output_file_path = os.path.join(output_dir, f"test.txt")
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    with open(output_file_path, 'w', encoding='utf-8') as file:
        file.write("id\ttext\ttype\n")  # Writing the header
        file.writelines("\n".join(all_translated_lines))

    print(f"Translation completed. Output saved to {output_file_path}")

    submit_file_path = os.path.join(output_dir, f"submit.txt")
    postprocess(output_file_path, submit_file_path)


def test_per_lang(model, tokenizer, accelerator, test_file, source_lang, tgt_lang, args, gen_kwargs=None):
    model.eval()

    target_lang = tgt_lang.split("_")[0]
    # prepare dataset
    extension = test_file.split(".")[-1]
    data_files = {}
    data_files["test"] = test_file
    raw_datasets = load_dataset(extension, data_files=data_files)

    prefix = args.source_prefix if args.source_prefix is not None else ""
    padding = "max_length" if args.pad_to_max_length else False
    # Temporarily set max_target_length for training.
    max_target_length = args.max_target_length
    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function(examples):
        inputs = [ex[source_lang] for ex in examples["translation"]]
        targets = [ex[target_lang] for ex in examples["translation"]]
        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=args.max_source_length, padding=padding, truncation=True)

        # Tokenize targets with the `text_target` keyword argument
        labels = tokenizer(text_target=targets, max_length=max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    column_names = raw_datasets["test"].column_names
    with accelerator.main_process_first():
        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
    test_dataset = processed_datasets["test"]

    # DataLoaders creation:
    label_pad_token_id = -100 if args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if accelerator.use_fp16 else None,
        )
    test_dataloader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
    test_dataloader = accelerator.prepare(test_dataloader)

    samples_seen = 0
    all_translated_lines = [] 
    for step, batch in enumerate(test_dataloader):
        print(f"batch keys:{batch.keys()}")
        with torch.no_grad():
            print(f"input_ids & labels:{batch['input_ids'][0]}, {batch['labels'][0]}")
            generated_tokens = accelerator.unwrap_model(model).generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang]
                **gen_kwargs,
            )

            generated_tokens = accelerator.pad_across_processes(
                generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
            )

            generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()

            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

            # postprocess
            decoded_preds = [pred.strip() for pred in decoded_preds]
            #decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

            # If we are in a multiprocess environment, the last batch has duplicates
            if accelerator.num_processes > 1:
                if step == len(test_dataloader) - 1:
                    decoded_preds = decoded_preds[: len(test_dataloader.dataset) - samples_seen]
                else:
                    samples_seen += len(decoded_preds)

            all_translated_lines.extend(decoded_preds)
    print(f"len(all_translated_lines):{len(all_translated_lines)}")
    assert len(all_translated_lines) == 500

    return all_translated_lines