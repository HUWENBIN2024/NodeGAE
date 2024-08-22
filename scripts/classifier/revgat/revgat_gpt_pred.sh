output_dir=.cache_revgat/gpt_pred
ckpt_dir=${output_dir}/ckpt
mkdir -p ${output_dir}
mkdir -p ${ckpt_dir}

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
    --use_gpt_preds \
    --ckpt_dir $ckpt_dir \
    --output_dir $output_dir \
    --save_pred \
    --n-runs 10 \
    --kd ${output_dir}/kd \
    # --seed 3401

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
    --use_gpt_preds \
    --ckpt_dir $ckpt_dir \
    --output_dir $output_dir \
    --save_pred \
    --n-runs 10 \
    --kd ${output_dir}/kd \
    # --seed 3401