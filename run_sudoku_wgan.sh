echo ">>> Building Sudoku 30% mask dataset..."
./.venv/bin/python dataset/build_custom_sudoku_dataset.py \
    --output-dir data/sudoku-mask30 \
    --subsample-size 1000 \
    --num-aug 1000 \
    --mask-ratio 0.3

echo ">>> Training TRM with GAN loss..."
./.venv/bin/python pretrain_adv.py \
    --config-name cfg_sudoku_wgan \
    data_paths="[data/sudoku-mask30]" \
    +run_name=sudoku_mask30_wgan \
    +checkpoint_path=checkpoints/sudoku_mask30_wgan \
    global_batch_size=128 \
    epochs=1000\
    eval_interval=100 \
    wgan.d_iters=30 \
    wgan.d_hidden_size=128 \
    wgan.adv_weight=0.1 \
    ema=True
