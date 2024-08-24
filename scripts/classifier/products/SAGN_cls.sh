cd src/classifier/node_classification/products/ogbn-products

python pre_processing.py --num_hops 3 --dataset ogbn-products --giant_path "emb/sent_emb_products.pt"

python main.py \
--method SAGN \
--stages 400 \
--train-num-epochs 0 \
--input-drop 0.2 \
--att-drop 0.4 \
--pre-process \
--residual \
--dataset ogbn-products \
--num-runs 10 \
--eval 10 \
--batch_size 50000 \
--patience 300 \
--tem 0.5 \
--lam 0.5 \
--ema \
--mean_teacher \
--ema_decay 0.0 \
--lr 0.001 \
--adap \
--gap 20 \
--warm_up 100 \
--top 0.85 \
--down 0.8 \
--kl \
--kl_lam 0.2 \
--hidden 256 \
--zero-inits \
--dropout 0.5 \
--num-heads 1  \
--label-drop 0.5 \
--mlp-layer 1 \
--num_hops 3 \
--label_num_hops 9 \
--giant
