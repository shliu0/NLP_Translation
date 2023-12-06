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
export MODEL_DIR=$(pwd)/weights
ln -s /home/users/sihan.liu/weights ./
# CUDA_VISIBLE_DEVICES="1," accelerate launch --config_file="./configs/default_config.yaml"
# CUDA_VISIBLE_DEVICES="1,2,3" accelerate launch --config_file="./configs/configs/acc_mgpu_config_cluster.yaml"
export TRAIN_BS = 16 # 4 in 3090, 16 in a800
export EVAL_BS = 16 # 4 in 3090, 16 in a800
CUDA_VISIBLE_DEVICES="1,2,3" accelerate launch --config_file="configs/acc_mgpu_config_dev.yaml" examples/run_translation_no_trainer.py \
    --model_name_or_path weights/mbart-large-50-many-to-many-mmt  \
    --train_file datasets/train/train.zh2en.json \
    --validation_file datasets/dev/dev.zh2en.json \
    --source_lang zh_CN \
    --target_lang en_XX \
    --output_dir outputs/tst-translation/1205/zh2en \
    --num_train_epochs 10 \
    --per_device_train_batch_size ${TRAIN_BS} \
    --per_device_eval_batch_size ${EVAL_BS} \
    --predict_with_generate True \
    --report_to tensorboard --with_tracking \
    --checkpointing_steps epoch

CUDA_VISIBLE_DEVICES="1,2,3" accelerate launch --config_file="configs/acc_mgpu_config_dev.yaml" examples/run_translation_no_trainer.py \
    --model_name_or_path weights/mbart-large-50-many-to-many-mmt  \
    --train_file datasets/train/train.zh2en.json \
    --validation_file datasets/dev/dev.zh2en.json \
    --source_lang zh_CN \
    --target_lang en_XX \
    --output_dir outputs/tst-translation/1205/zh2en \
    --num_train_epochs 10 \
    --per_device_train_batch_size ${TRAIN_BS} \
    --per_device_eval_batch_size ${EVAL_BS} \
    --predict_with_generate True \
    --report_to tensorboard --with_tracking \
    --checkpointing_steps epoch

CUDA_VISIBLE_DEVICES="1,2,3" accelerate launch --config_file="configs/acc_mgpu_config_dev.yaml" examples/run_translation_no_trainer.py \
    --model_name_or_path weights/mbart-large-50-many-to-many-mmt  \
    --train_file datasets/train/train.zh2en.json \
    --validation_file datasets/dev/dev.zh2en.json \
    --source_lang zh_CN \
    --target_lang en_XX \
    --output_dir outputs/tst-translation/1205/zh2en \
    --num_train_epochs 10 \
    --per_device_train_batch_size ${TRAIN_BS} \
    --per_device_eval_batch_size ${EVAL_BS} \
    --predict_with_generate True \
    --report_to tensorboard --with_tracking \
    --checkpointing_steps epoch

CUDA_VISIBLE_DEVICES="1,2,3" accelerate launch --config_file="configs/acc_mgpu_config_dev.yaml" examples/run_translation_no_trainer.py \
    --model_name_or_path weights/mbart-large-50-many-to-many-mmt  \
    --train_file datasets/train/train.zh2en.json \
    --validation_file datasets/dev/dev.zh2en.json \
    --source_lang zh_CN \
    --target_lang en_XX \
    --output_dir outputs/tst-translation/1205/zh2en \
    --num_train_epochs 10 \
    --per_device_train_batch_size ${TRAIN_BS} \
    --per_device_eval_batch_size ${EVAL_BS} \
    --predict_with_generate True \
    --report_to tensorboard --with_tracking \
    --checkpointing_steps epoch

rm -r ${MODEL_DIR}

