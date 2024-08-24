python -m src.classifier.node_classification.products.mlp_node_cls \
--emb_path  emb/nodegae_feature_emb_products.pt \
--lr 0.01 \
--hidden_channels 256 

python -m src.classifier.node_classification.products.mlp_node_cls \
--emb_path emb/lm_finetune_products.pt \
--lr 0.01 \
--hidden_channels 256 

python -m src.classifier.node_classification.products.mlp_node_cls \
--emb_path emb/sent_emb_products.pt \
--lr 0.01 \
--hidden_channels 256 


python -m src.classifier.node_classification.products.mlp_node_cls \
--emb_path ogb \
--lr 0.01 \
--hidden_channels 256 
