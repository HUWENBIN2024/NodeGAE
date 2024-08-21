bash scripts/revgat_node_gae.sh
bash scripts/revgat_simteg.sh
bash scripts/revgat_gpt_pred.sh

logits0=.cache_revgat/node_gae/cached_embs
logits1=.cache_revgat/simteg/cached_embs
logits2=.cache_revgat/gpt_pred/cached_embs

python -m src.classifier.node_classification.revgat.compute_ensemble \
    --list_logits "${logits0} ${logits1} ${logits2}" \
    --weights 1 1 1 \
    --start_seed 1
