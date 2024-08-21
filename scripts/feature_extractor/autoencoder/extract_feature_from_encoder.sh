cd src/feature_extractor/autoencoder/vec2text
iteration=200000

python feature_extractor.py \
--model_path saves/autoencoder/checkpoint-$iteration \
--save_path ../../../../emb/nodegae_feature_emb.pt