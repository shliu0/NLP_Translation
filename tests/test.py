# from transformers import MBartForConditionalGeneration, MBart50TokenizerFast, MBart50Tokenizer
# from datasets import load_dataset

# article_zh = "这是中文"

# tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
# model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
# tokenizer1 = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")


# # translate zh to en
# # tokenizer.src_lang = "zh_CN"
# tokenizer.src_lang = "zh_CN"
# tokenizer1.src_lang = 'en_XX'

# encoded_ar = tokenizer(article_zh, return_tensors="pt")
# generated_tokens = model.generate(encoded_ar['input_ids'], num_beams=4, max_length=200,
#                                   forced_bos_token_id=tokenizer.lang_code_to_id["ru_RU"])  # target_lang: "en_XX", "es_XX", "fr_XX", "ru_RU"

# generated_tokens1 = model.generate(**encoded_ar,
#                                   forced_bos_token_id=tokenizer.lang_code_to_id["es_XX"]) 

# breakpoint()
# print(f'output_ru:{tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)}')
# print(f'output_es:{tokenizer.batch_decode(generated_tokens1, skip_special_tokens=True)}')

import evaluate

# 加载BLEU评价指标
# bleu = evaluate.load("bleu")
bleu = evaluate.load("sacrebleu")

# 设置参考翻译和机器翻译
references = [["这是一个参考翻译的例子。"]]
candidates = ["这是机器翻译的一个例子。"]

# 计算BLEU分数
results = bleu.compute(predictions=candidates, references=references)

# 输出结果
print("BLEU分数:", results)
