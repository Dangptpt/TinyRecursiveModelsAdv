import os
import torch
import yaml
from torch.utils.data import DataLoader
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig
from utils.functions import load_model_class
import argparse
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Benchmark Sudoku model on higher mask ratios")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the checkpoint file (e.g., checkpoints/sudoku_mask30/step_195312)")
    parser.add_argument("--batch-size", type=int, default=256, help="Inference batch size")
    parser.add_argument("--max-samples", type=int, default=10000, help="Maximum samples to evaluate")
    args = parser.parse_args()

    checkpoint_dir = os.path.dirname(args.checkpoint)
    config_path = os.path.join(checkpoint_dir, "all_config.yaml")
    
    if not os.path.exists(config_path):
        print(f"Can't find {checkpoint_dir}")
        return

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    
    arch_config = config_dict['arch']
    
    print(f"Loading model: {arch_config['name']}")
    model_cls = load_model_class(arch_config['name'])
    loss_head_cls = load_model_class(arch_config['loss']['name'])
    
    model_cfg = {
        **{k: v for k, v in arch_config.items() if k not in ['name', 'loss']},
        "batch_size": args.batch_size,
        "vocab_size": 11,
        "seq_len": 81,
        "num_puzzle_identifiers": 1,
        "causal": False
    }
    
    model = model_cls(model_cfg).cuda()
    
    loss_kwargs = {k: v for k, v in arch_config['loss'].items() if k != 'name'}
    model = loss_head_cls(model, **loss_kwargs).cuda()
    
    print(f"Loading checkpoint: {args.checkpoint}")
    state_dict = torch.load(args.checkpoint, map_location="cuda")
    
    new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    
    model.load_state_dict(new_state_dict)
    model.eval()

    test_datasets = [
        "data/sudoku-mask30", 
        "data/sudoku-mask40",
        "data/sudoku-mask50",
        "data/sudoku-mask60"
    ]

    print("\n" + "="*50)
    print(f"{'Dataset':<25} | {'Exact Accuracy (%)':<20}")
    print("-" * 50)

    for data_path in test_datasets:
        if not os.path.exists(data_path):
            print(f"Skip {data_path} (not found)")
            continue
            
        ds_config = PuzzleDatasetConfig(
            seed=42,
            dataset_paths=[data_path],
            global_batch_size=args.batch_size,
            test_set_mode=True, #
            epochs_per_iter=1,
            rank=0,
            num_replicas=1
        )
        
        dataset = PuzzleDataset(ds_config, split="test")
        dataloader = DataLoader(dataset, batch_size=None)

        total_steps = (min(dataset.metadata.total_puzzles, args.max_samples) + args.batch_size - 1) // args.batch_size

        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for set_name, batch, _ in tqdm(dataloader, total=total_steps, desc=f"Evaluating {data_path.split('/')[-1]}", leave=False):
                if total_samples >= args.max_samples:
                    break
                
                batch = {k: v.cuda() for k, v in batch.items()}
                
                with torch.device("cuda"):
                    carry = model.initial_carry(batch)
                
                while True:
                    carry, loss, metrics, preds, all_finish = model(carry=carry, batch=batch, return_keys=[])
                    if all_finish:
                        break
                
                total_correct += metrics['exact_accuracy'].item()
                total_samples += metrics['count'].item()

        accuracy = (total_correct / total_samples) * 100 if total_samples > 0 else 0
        print(f"{data_path:<25} | {accuracy:>18.2f}%")

    print("="*50)

if __name__ == "__main__":
    main()
