python -m src.classifier.node_classification.products.clusterGCN_cls \
--emb_path  emb/nodegae_feature_emb.pt \
--lr 0.01 \
--hidden_size_gnn 256

python -m src.classifier.node_classification.products.clusterGCN_cls \
--emb_path emb/lm_finetune.pt \
--lr 0.01 \
--hidden_size_gnn 256 \
--batch_size 1024


python -m src.classifier.node_classification.products.clusterGCN_cls \
--emb_path emb/sent_emb.pt \
--lr 0.01 \
--hidden_size_gnn 256 \
--batch_size 1024

python -m src.classifier.node_classification.products.clusterGCN_cls \
--emb_path ogb \
--lr 0.01 \
--hidden_size_gnn 256 \
--batch_size 1024