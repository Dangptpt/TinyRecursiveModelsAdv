echo "Make Sudoku dataset with different mask ratios"

python dataset/build_custom_sudoku_dataset.py --output-dir data/sudoku-mask30 --subsample-size 1000 --num-aug 1000 --mask-ratio 0.3
python dataset/build_custom_sudoku_dataset.py --output-dir data/sudoku-mask40 --subsample-size 100 --num-aug 0 --mask-ratio 0.4 --skip-train
python dataset/build_custom_sudoku_dataset.py --output-dir data/sudoku-mask50 --subsample-size 100 --num-aug 0 --mask-ratio 0.5 --skip-train
python dataset/build_custom_sudoku_dataset.py --output-dir data/sudoku-mask60 --subsample-size 100 --num-aug 0 --mask-ratio 0.6 --skip-train

echo "Train model on Sudoku 30% mask"
python pretrain.py \
arch=trm \
data_paths="[data/sudoku-mask30]" \
evaluators="[]" \
global_batch_size=256 \
epochs=50000 eval_interval=5000 \
lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
arch.mlp_t=True arch.pos_encodings=none \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=6 \
+run_name=sudoku_mask30 ema=True +checkpoint_path=checkpoints/sudoku_mask30