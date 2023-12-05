# python ../examples/run_translation_no_trainer.py \
#     --model_name_or_path facebook/mbart-large-50-many-to-many-mmt  \
#     --train_file ../datasets/train/train.zh2en.json \
#     --validation_file ../datasets/dev/dev.zh2en.json \
#     --source_lang zh_CN \
#     --target_lang en_XX \
#     --output_dir ../outputs/tst-translation \
#     --num_train_epochs 10 \
#     --per_device_train_batch_size 4 \
#     --per_device_eval_batch_size 4 \
#     --predict_with_generate True \

accelerate launch ../examples/run_translation_no_trainer.py \
    --model_name_or_path facebook/mbart-large-50-many-to-many-mmt  \
    --train_file ../datasets/train/train.zh2en.json \
    --validation_file ../datasets/dev/dev.zh2en.json \
    --source_lang zh_CN \
    --target_lang en_XX \
    --output_dir ../outputs/tst-translation \
    --num_train_epochs 10 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --predict_with_generate True \
