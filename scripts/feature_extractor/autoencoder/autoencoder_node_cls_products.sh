cd src/feature_extractor/autoencoder/vec2text

python run.py \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 16 \
--max_seq_length 256 \
--model_name_or_path t5-base \
--dataset_name Egbertjing/products \
--embedder_model_name gtr_base \
--num_repeat_tokens 16 \
--embedder_no_grad True \
--num_train_epochs 100 \
--max_eval_samples 500 \
--eval_steps 2000 \
--warmup_steps 10000 \
--bf16=1 \
--use_wandb=0 \
--use_frozen_embeddings_as_input False \
--experiment inversion \
--lr_scheduler_type constant_with_warmup \
--exp_group_name arxiv-gtr \
--learning_rate 0.0001 \
--output_dir ./saves/autoencoder \
--save_steps 5000 \
--overwrite_output_dir \
--use_infonce_loss \
--auto_encoder_name sentence-transformers/sentence-t5-base \
--infonce_loss_weight  1.0