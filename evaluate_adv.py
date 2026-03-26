import os
import torch
import yaml
from torch.utils.data import DataLoader
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig
from utils.functions import load_model_class
import argparse
from tqdm import tqdm
import math

def main():
    parser = argparse.ArgumentParser(description="Benchmark TRM WGAN model on multiple mask ratios")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the G checkpoint file (e.g., checkpoints/sudoku_mask30_wgan/step_1000)")
    parser.add_argument("--batch-size", type=int, default=128, help="Inference batch size")
    parser.add_argument("--max-samples", type=int, default=10000, help="Maximum samples to evaluate per ratio")
    args = parser.parse_args()

    checkpoint_dir = os.path.dirname(args.checkpoint)
    config_path = os.path.join(checkpoint_dir, "all_config.yaml")
    
    if not os.path.exists(config_path):
        config_path = os.path.join(os.path.dirname(checkpoint_dir), "all_config.yaml")
        if not os.path.exists(config_path):
            print(f"Error: Can't find all_config.yaml in {checkpoint_dir} or its parent.")
            return

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    
    arch_config = config_dict['arch']
    
    print(f"Loading TRM model: {arch_config['name']}")
    model_cls = load_model_class(arch_config['name'])
    
    loss_name = arch_config['loss']['name']
    if "ACTLossHead" in loss_name and "WGAN" not in loss_name:
        loss_name = loss_name.replace("ACTLossHead", "ACTWGANLossHead")
        print(f"Overriding loss head to: {loss_name}")
    
    loss_head_cls = load_model_class(loss_name)
    
    model_cfg = {
        **{k: v for k, v in arch_config.items() if k not in ['name', 'loss']},
        "batch_size": args.batch_size,
        "vocab_size": 11,
        "seq_len": 81,
        "num_puzzle_identifiers": 1,
        "causal": False
    }
    
    with torch.device("cuda"):
        model = model_cls(model_cfg)
        loss_kwargs = {k: v for k, v in arch_config['loss'].items() if k != 'name'}
        model = loss_head_cls(model, **loss_kwargs)
        model = model.cuda()
    
    print(f"Loading model state from: {args.checkpoint}")
    state_dict = torch.load(args.checkpoint, map_location="cuda")
    
    # Handle both compiled and uncompiled state dicts
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace("_orig_mod.", "")
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict)
    model.eval()

    test_datasets = {
        "30% Mask": "data/sudoku-mask30", 
        "40% Mask": "data/sudoku-mask40",
        "50% Mask": "data/sudoku-mask50",
        "60% Mask": "data/sudoku-mask60"
    }

    print("\n" + "="*60)
    print(f"{'Dataset Ratio':<20} | {'Exact Accuracy (%)':<20} | {'Samples':<10}")
    print("-" * 60)

    for name, data_path in test_datasets.items():
        if not os.path.exists(data_path):
            print(f"{name:<20} | Skip (not found)")
            continue
            
        ds_config = PuzzleDatasetConfig(
            seed=42,
            dataset_paths=[data_path],
            global_batch_size=args.batch_size,
            test_set_mode=True,
            epochs_per_iter=1,
            rank=0,
            num_replicas=1
        )
        
        try:
            dataset = PuzzleDataset(ds_config, split="test")
            dataloader = DataLoader(dataset, batch_size=None)
        except Exception as e:
            print(f"{name:<20} | Error loading: {e}")
            continue

        max_batches = math.ceil(args.max_samples / args.batch_size)
        total_correct = 0
        total_samples = 0
        n_batches = 0
        
        with torch.no_grad():
            for _, batch, _ in tqdm(dataloader, total=max_batches, desc=f"Eval {name}", leave=False):
                if n_batches >= max_batches:
                    break
                
                batch = {k: v.cuda() for k, v in batch.items()}
                
                with torch.device("cuda"):
                    carry = model.initial_carry(batch)
                
                while True:
                    carry, _, metrics, _, all_finish = model(carry=carry, batch=batch, return_keys=[])
                    if all_finish:
                        break
                
                total_correct += metrics.get('exact_accuracy', torch.tensor(0)).item()
                total_samples += metrics.get('count', torch.tensor(1)).item()
                n_batches += 1

        accuracy = (total_correct / total_samples) * 100 if total_samples > 0 else 0
        print(f"{name:<20} | {accuracy:>18.2f}% | {total_samples:>10}")

    print("="*60)

if __name__ == "__main__":
    main()
