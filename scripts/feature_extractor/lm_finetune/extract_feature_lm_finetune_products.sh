ln -s /home/hjingaa/github/NodeGAE/src/feature_extractor/autoencoder/vec2text/simteg /home/hjingaa/github/NodeGAE/src/feature_extractor/lm_finetune
ln -s /home/hjingaa/github/NodeGAE/src/feature_extractor/autoencoder/vec2text/dataset /home/hjingaa/github/NodeGAE/src/feature_extr
actor/lm_finetune
python src/feature_extractor/lm_finetune/feature_extractor_lm_finetune_products.py \
--model_path save_lm_finetune/t5base_mode_cls.pt \
--save_path emb/lm_finetune.pt