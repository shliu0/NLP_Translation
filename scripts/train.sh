export WANDB_PROJECT=test
export CUDA_VISIBLE_DEVICES=0,1,2
python -m torch.distributed.launch --nproc_per_node=3 --master_port=29527 --nnodes=1 ../examples/run_translation.py \
    --report_to wandb \
    --model_name_or_path facebook/mbart-large-50-many-to-many-mmt  \
    --do_train \
    --do_eval \
    --train_file ../datasets/train/train.zh2en.json \
    --validation_file ../datasets/dev/dev.zh2en.json \
    --source_lang zh_CN \
    --target_lang en_XX \
    --output_dir /root/autodl-tmp/nlp_saved_models/tst_translation \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --evaluation_strategy steps \
    --eval_steps 500 \
    --predict_with_generate True \
    --seed 777 \
    --save_total_limit 1 \
    --use_auth_token False \
