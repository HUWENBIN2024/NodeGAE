

cd src/feature_extractor/lm_finetune
python feature_extractor_lm_finetune_products.py \
--model_path save_lm_finetune/t5base_mode_cls.pt \
--save_path ../../../emb/lm_finetune_products.pt