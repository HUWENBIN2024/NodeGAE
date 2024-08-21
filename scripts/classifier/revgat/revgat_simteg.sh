output_dir=.cache_revgat/simteg
ckpt_dir=${output_dir}/ckpt
mkdir -p ${output_dir}
mkdir -p ${ckpt_dir}
wget -nc -O simteg_official_x_embs.pt https://huggingface.co/datasets/vermouthdky/SimTeG/resolve/main/ogbn-arxiv/e5-large/main/cached_embs/x_embs.pt
bert_x_dir=emb/simteg_official_x_embs.pt

python -m src.classifier.node_classification.revgat.main \
    --use-norm \
    --no-attn-dst \
    --mode teacher \
    --gpu 0 \
    --dropout 0.58 \
    --edge-drop 0.46 \
    --group 1 \
    --input-drop 0.37 \
    --label_smoothing_factor 0.02 \
    --n-heads 2 \
    --n-hidden 256 \
    --n-label-iters 2 \
    --n-layers 2 \
    --use-labels \
    --use_bert_x \
    --bert_x_dir $bert_x_dir \
    --ckpt_dir $ckpt_dir \
    --output_dir $output_dir \
    --save_pred \
    --n-runs 10 \
    --kd ${output_dir}/kd
    # --n-epochs 200
    # --seed 3407 \


# self kd
python -m src.classifier.node_classification.revgat.main \
    --use-norm \
    --no-attn-dst \
    --mode student \
    --gpu 0 \
    --dropout 0.58 \
    --edge-drop 0.46 \
    --group 1 \
    --input-drop 0.37 \
    --label_smoothing_factor 0.02 \
    --n-heads 2 \
    --n-hidden 256 \
    --n-label-iters 2 \
    --n-layers 2 \
    --use-labels \
    --use_bert_x \
    --bert_x_dir $bert_x_dir \
    --ckpt_dir $ckpt_dir \
    --output_dir $output_dir \
    --save_pred \
    --n-runs 10 \
    --kd ${output_dir}/kd
    # --n-epochs 200
    # --seed 3407 \