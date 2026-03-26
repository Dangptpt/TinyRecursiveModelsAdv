
from typing import Optional, Any, Sequence, List, Dict
from dataclasses import dataclass, field
import os
import math
import yaml
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from torch.utils.tensorboard import SummaryWriter
import coolname
import hydra
import pydantic
from omegaconf import DictConfig
from adam_atan2 import AdamATan2

from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig, PuzzleDatasetMetadata
from utils.functions import load_model_class, get_model_source_path
from models.sparse_embedding import CastedSparseEmbeddingSignSGD_Distributed
from models.ema import EMAHelper
from models.sudoku_discriminator import SudokuDiscriminator
from models.losses import make_real_soft_distribution


class LossConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str


class ArchConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str
    loss: LossConfig


class WGANConfig(pydantic.BaseModel):
    """Hyperparameters riêng cho WGAN discriminator và adversarial loss."""
    d_hidden_size: int = 128       # Discriminator width
    d_iters: int = 30              # Số vòng lặp đệ quy (iters)
    d_lr: float = 5e-5             # Learning rate D (RMSprop)
    d_iters_per_step: int = 1    # Số lần cập nhật D trên mỗi batch
    weight_clip: float = 0.01    # WGAN weight clipping bound
    adv_weight: float = 0.1      # adv: cân bằng adversarial vs reconstruction
    noise_std: float = 1.0       # Độ noise thêm vào real distribution cho D


class PretrainAdvConfig(pydantic.BaseModel):
    arch: ArchConfig
    data_paths: List[str]
    data_paths_test: List[str] = []

    global_batch_size: int
    epochs: int

    lr: float
    lr_min_ratio: float
    lr_warmup_steps: int

    weight_decay: float
    beta1: float
    beta2: float

    puzzle_emb_lr: float
    puzzle_emb_weight_decay: float

    wgan: WGANConfig = pydantic.Field(default_factory=WGANConfig)

    project_name: Optional[str] = None
    run_name: Optional[str] = None
    load_checkpoint: Optional[str] = None
    checkpoint_path: Optional[str] = None

    seed: int = 0
    checkpoint_every_eval: bool = False
    eval_interval: Optional[int] = None
    min_eval_interval: Optional[int] = 0
    eval_samples: int = 10000

    ema: bool = False
    ema_rate: float = 0.999
    freeze_weights: bool = False
    grad_clip: float = 1.0  # Thêm tham số Grad Clip


@dataclass
class TrainAdvState:
    model: nn.Module                        # G: ACTWGANLossHead(TRM)
    d_net: SudokuDiscriminator              # D: WGAN critic
    optimizers: Sequence[torch.optim.Optimizer]  # G optimizers
    optimizer_d: torch.optim.Optimizer           # D optimizer (RMSprop)
    optimizer_lrs: Sequence[float]

    carry: Any = None                       # ACT carry state
    step: int = 0
    total_steps: int = 0



def cosine_lr(current_step: int, *, base_lr: float, warmup_steps: int, total_steps: int, min_ratio: float = 0.0) -> float:
    if current_step < warmup_steps:
        return base_lr * current_step / max(1, warmup_steps)
    progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * (min_ratio + max(0.0, (1 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))))


def compute_lr(base_lr: float, config: PretrainAdvConfig, step: int, total_steps: int) -> float:
    return cosine_lr(step, base_lr=base_lr, warmup_steps=round(config.lr_warmup_steps),
                     total_steps=total_steps, min_ratio=config.lr_min_ratio)



def create_dataloader(config: PretrainAdvConfig, split: str, rank: int, world_size: int, **kwargs):
    data_paths = (
        config.data_paths_test
        if len(config.data_paths_test) > 0 and split == "test"
        else config.data_paths
    )
    dataset = PuzzleDataset(PuzzleDatasetConfig(
        seed=config.seed, dataset_paths=data_paths,
        rank=rank, num_replicas=world_size, **kwargs
    ), split=split)
    loader = DataLoader(dataset, batch_size=None, num_workers=1,
                        prefetch_factor=8, pin_memory=True, persistent_workers=True)
    return loader, dataset.metadata



def load_checkpoint(model: nn.Module, config: PretrainAdvConfig):
    if config.load_checkpoint is None:
        return
    print(f"Loading checkpoint: {config.load_checkpoint}")
    state_dict = torch.load(config.load_checkpoint, map_location="cuda")
    puzzle_emb_name = "_orig_mod.model.inner.puzzle_emb.weights"
    if puzzle_emb_name in state_dict:
        expected_shape = model.model.puzzle_emb.weights.shape  # type: ignore
        if state_dict[puzzle_emb_name].shape != expected_shape:
            print(f"Resizing puzzle emb: {state_dict[puzzle_emb_name].shape} → {expected_shape}")
            state_dict[puzzle_emb_name] = (
                torch.mean(state_dict[puzzle_emb_name], dim=0, keepdim=True).expand(expected_shape).contiguous()
            )
    model.load_state_dict(state_dict, assign=True)


def build_model_and_optimizer(
    config: PretrainAdvConfig,
    train_metadata: PuzzleDatasetMetadata,
    rank: int, world_size: int,
):
    model_cfg = dict(
        **config.arch.__pydantic_extra__,  # type: ignore
        batch_size=config.global_batch_size // world_size,
        vocab_size=train_metadata.vocab_size,
        seq_len=train_metadata.seq_len,
        num_puzzle_identifiers=train_metadata.num_puzzle_identifiers,
        causal=False,
    )

    model_cls = load_model_class(config.arch.name)
    loss_name = config.arch.loss.name
    if "ACTLossHead" in loss_name and "WGAN" not in loss_name:
        loss_name = loss_name.replace("ACTLossHead", "ACTWGANLossHead")
        print(f"[WGAN] Overriding loss head to: {loss_name}")
    loss_head_cls = load_model_class(loss_name)

    with torch.device("cuda"):
        model: nn.Module = model_cls(model_cfg)
        model = loss_head_cls(model, **config.arch.loss.__pydantic_extra__)  # type: ignore
        
        model = model.cuda()

        # if "DISABLE_COMPILE" not in os.environ:
        #     model = torch.compile(model)  # type: ignore

        if rank == 0:
            load_checkpoint(model, config)

        if world_size > 1:
            with torch.no_grad():
                for param in list(model.parameters()) + list(model.buffers()):
                    dist.broadcast(param, src=0)

    # Discriminator
    d_net = SudokuDiscriminator(
        seq_len=train_metadata.seq_len,
        vocab_size=train_metadata.vocab_size,
        hidden_size=config.wgan.d_hidden_size,
        iters=config.wgan.d_iters,
    ).cuda()

    # Optimizers G
    puzzle_emb_ndim = config.arch.__pydantic_extra__.get("puzzle_emb_ndim", 0)  # type: ignore
    if puzzle_emb_ndim == 0 or config.freeze_weights:
        optimizers = [AdamATan2(model.parameters(), lr=0,
                                weight_decay=config.weight_decay,
                                betas=(config.beta1, config.beta2))]
        optimizer_lrs = [config.lr]
    else:
        optimizers = [
            CastedSparseEmbeddingSignSGD_Distributed(
                model.model.puzzle_emb.buffers(),  # type: ignore
                lr=0, weight_decay=config.puzzle_emb_weight_decay, world_size=world_size
            ),
            AdamATan2(model.parameters(), lr=0,
                      weight_decay=config.weight_decay, betas=(config.beta1, config.beta2))
        ]
        optimizer_lrs = [config.puzzle_emb_lr, config.lr]

    # Optimizer D: RMSprop (standard WGAN)
    optimizer_d = torch.optim.RMSprop(d_net.parameters(), lr=config.wgan.d_lr)

    return model, d_net, optimizers, optimizer_lrs, optimizer_d



def train_one_batch(
    config: PretrainAdvConfig,
    state: TrainAdvState,
    batch: Dict[str, torch.Tensor],
    global_batch_size: int,
    rank: int,
    world_size: int,
) -> Optional[Dict[str, float]]:
    state.step += 1
    if state.step > state.total_steps:
        return None

    wgan = config.wgan
    batch = {k: v.cuda() for k, v in batch.items()}
    inputs: torch.Tensor   = batch["inputs"]    # [B, seq_len]  
    targets: torch.Tensor  = batch["labels"]    # [B, seq_len]  
    vocab_size: int        = state.d_net.vocab_size


    if state.carry is None:
        with torch.device("cuda"):
            state.carry = state.model.initial_carry(batch)  # type: ignore

    state.model.eval()
    with torch.no_grad(), torch.device("cuda"):
        carry_d = state.model.initial_carry(batch)  # type: ignore
        all_done = False
        g_logits_for_d = None
        _step_d = 0
        while not all_done:
            carry_d, _, _, preds_d, all_done = state.model(
                carry=carry_d, batch=batch, return_keys=["logits"]
            )
            if preds_d and "logits" in preds_d:
                g_logits_for_d = preds_d["logits"]   # [B, seq_len, vocab_size]
            _step_d += 1
            if _step_d > 32:  # safety cap
                break
        if g_logits_for_d is None:
            g_logits_for_d = torch.zeros(inputs.shape[0], inputs.shape[1], vocab_size, device="cuda")

    g_soft_detach = torch.softmax(g_logits_for_d.float(), dim=-1)  # [B, seq_len, vocab_size], no grad

    state.model.eval()     
    state.d_net.train()

    x_real = make_real_soft_distribution(targets, vocab_size, wgan.noise_std)  # [B, seq_len, vocab_size]

    total_loss_d = torch.tensor(0.0, device="cuda")
    for _d_i in range(wgan.d_iters_per_step):
        # WGAN weight clipping
        for p in state.d_net.parameters():
            p.data.clamp_(-wgan.weight_clip, wgan.weight_clip)

        state.optimizer_d.zero_grad()

        score_real = state.d_net(inputs, x_real)        # [B]
        score_fake = state.d_net(inputs, g_soft_detach)  # [B]

        loss_d = -score_real.mean() + score_fake.mean()
        loss_d.backward()
        
        # Gradient Clipping cho D
        torch.nn.utils.clip_grad_norm_(state.d_net.parameters(), config.grad_clip)
        
        state.optimizer_d.step()
        total_loss_d = total_loss_d + loss_d.detach()

    loss_d_avg = total_loss_d / wgan.d_iters_per_step


    state.model.train()
    state.d_net.eval()  

    for optim in state.optimizers:
        optim.zero_grad()

    with torch.device("cuda"):
        carry_g = state.model.initial_carry(batch)  # type: ignore

    g_logits_with_grad = None
    all_done_g = False
    rec_loss_total = torch.tensor(0.0, device="cuda", requires_grad=True)
    _step_g = 0

    while not all_done_g:
        carry_g, rec_loss_step, g_metrics, preds_g, all_done_g = state.model(
            carry=carry_g, batch=batch, return_keys=["logits_with_grad"]
        )
        rec_loss_total = rec_loss_step
        if preds_g and "logits_with_grad" in preds_g:
            g_logits_with_grad = preds_g["logits_with_grad"]  # [B, seq_len, vocab_size], HAS grad
        _step_g += 1
        if _step_g > 32:
            break

    # Adversarial loss
    loss_g_adv = torch.tensor(0.0, device="cuda")
    if g_logits_with_grad is not None:
        g_soft_grad = torch.softmax(g_logits_with_grad.float(), dim=-1)   # [B, seq_len, vocab_size]
        with torch.no_grad():
            pass
        state.d_net.requires_grad_(False)
        score_g = state.d_net(inputs, g_soft_grad) 
        loss_g_adv = -score_g.mean()                
        state.d_net.requires_grad_(True)

    loss_g_total = rec_loss_total + wgan.adv_weight * loss_g_adv
    (loss_g_total / global_batch_size).backward()

    # Gradient Clipping 
    torch.nn.utils.clip_grad_norm_(state.model.parameters(), config.grad_clip)

    if world_size > 1:
        for p in state.model.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad)

    lr_this_step = None
    for optim, base_lr in zip(state.optimizers, state.optimizer_lrs):
        lr_this_step = compute_lr(base_lr, config, state.step, state.total_steps)
        for pg in optim.param_groups:
            pg['lr'] = lr_this_step
        optim.step()


    metrics: Dict[str, float] = {
        "train/loss_d":       loss_d_avg.item(),
        "train/loss_g_adv":   loss_g_adv.item() if torch.is_tensor(loss_g_adv) else 0.0,
        "train/loss_rec":     rec_loss_total.item() if torch.is_tensor(rec_loss_total) else 0.0,
        "train/loss_g_total": loss_g_total.item() if torch.is_tensor(loss_g_total) else 0.0,
        "train/lr":           lr_this_step or 0.0,
        "train/score_real":   score_real.mean().item(),
        "train/score_fake":   score_fake.mean().item(),
    }
    if "exact_accuracy" in g_metrics and "count" in g_metrics:
        cnt = max(g_metrics["count"].item(), 1)
        metrics["train/exact_accuracy"] = g_metrics["exact_accuracy"].item() / cnt * 100

    return metrics



@torch.inference_mode()
def evaluate(
    state: TrainAdvState,
    eval_loader,
    max_batches: int = 200,
) -> Dict[str, float]:
    state.model.eval()
    total_correct = 0
    total_count = 0
    n_batches = 0

    for set_name, batch, _ in eval_loader:
        if n_batches >= max_batches:
            break
        batch = {k: v.cuda() for k, v in batch.items()}
        with torch.device("cuda"):
            carry = state.model.initial_carry(batch)  # type: ignore

        all_done = False
        while not all_done:
            carry, _, metrics, _, all_done = state.model(carry=carry, batch=batch, return_keys=[])

        total_correct += metrics.get("exact_accuracy", torch.tensor(0)).item()
        total_count   += max(metrics.get("count", torch.tensor(1)).item(), 1)
        n_batches += 1

    acc = total_correct / max(total_count, 1) * 100
    return {"eval/exact_accuracy": acc, "eval/total_samples": total_count}



def save_checkpoint(config: PretrainAdvConfig, state: TrainAdvState, suffix: str = ""):
    if config.checkpoint_path is None:
        return
    os.makedirs(config.checkpoint_path, exist_ok=True)
    g_path = os.path.join(config.checkpoint_path, f"step_{state.step}{suffix}")
    d_path = os.path.join(config.checkpoint_path, f"step_{state.step}{suffix}_dnet")
    torch.save(state.model.state_dict(), g_path)
    torch.save(state.d_net.state_dict(), d_path)
    print(f"[Checkpoint] Saved G → {g_path}")
    print(f"[Checkpoint] Saved D → {d_path}")


def load_synced_config(hydra_config: DictConfig, rank: int, world_size: int) -> PretrainAdvConfig:
    objects = [None]
    if rank == 0:
        cfg = PretrainAdvConfig(**hydra_config)  # type: ignore
        if cfg.project_name is None:
            cfg.project_name = f"{os.path.basename(cfg.data_paths[0]).capitalize()}-WGAN"
        if cfg.run_name is None:
            cfg.run_name = f"wgan-{coolname.generate_slug(2)}"
        if cfg.checkpoint_path is None:
            cfg.checkpoint_path = os.path.join("checkpoints", cfg.project_name, cfg.run_name)
        objects = [cfg]
    if world_size > 1:
        dist.broadcast_object_list(objects, src=0)
    return objects[0]  # type: ignore



@hydra.main(config_path="config", config_name="cfg_sudoku_wgan", version_base=None)
def launch(hydra_config: DictConfig):
    RANK = 0
    WORLD_SIZE = 1
    CPU_PROCESS_GROUP = None

    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        RANK = dist.get_rank()
        WORLD_SIZE = dist.get_world_size()
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        CPU_PROCESS_GROUP = dist.new_group(backend="gloo")

    config = load_synced_config(hydra_config, rank=RANK, world_size=WORLD_SIZE)
    torch.random.manual_seed(config.seed + RANK)

    # Data
    train_epochs_per_iter = config.eval_interval if config.eval_interval else config.epochs
    train_epochs_per_iter = min(train_epochs_per_iter, config.epochs)
    total_iters = config.epochs // train_epochs_per_iter

    train_loader, train_meta = create_dataloader(
        config, "train", RANK, WORLD_SIZE,
        test_set_mode=False, epochs_per_iter=train_epochs_per_iter,
        global_batch_size=config.global_batch_size
    )
    eval_loader = eval_meta = None
    try:
        eval_loader, eval_meta = create_dataloader(
            config, "test", RANK, WORLD_SIZE,
            test_set_mode=True, epochs_per_iter=1,
            global_batch_size=config.global_batch_size
        )
    except Exception as e:
        print(f"No eval data: {e}")

    # Build model & state
    total_steps = int(
        config.epochs * train_meta.total_groups * train_meta.mean_puzzle_examples / config.global_batch_size
    )
    model, d_net, optimizers, optimizer_lrs, optimizer_d = build_model_and_optimizer(
        config, train_meta, RANK, WORLD_SIZE
    )

    if RANK == 0:
        g_params = sum(p.numel() for p in model.parameters()) / 1e6
        d_params = sum(p.numel() for p in d_net.parameters()) / 1e6
        print(f"G: {g_params:.2f}M params | D: {d_params:.2f}M params")
        print(f"Total training steps: {total_steps}")

    state = TrainAdvState(
        model=model, d_net=d_net,
        optimizers=optimizers, optimizer_d=optimizer_d, optimizer_lrs=optimizer_lrs,
        step=0, total_steps=total_steps
    )

    # EMA
    ema_helper = None
    if config.ema:
        ema_helper = EMAHelper(mu=config.ema_rate)
        ema_helper.register(state.model)

    # Logger
    writer = None
    progress_bar = None
    best_acc = -1.0
    if RANK == 0:
        wandb.init(
            project=config.project_name, name=config.run_name,
            config=config.model_dump(),
            settings=wandb.Settings(_disable_stats=True)
        )
        writer = SummaryWriter(log_dir=os.path.join("runs", config.run_name or "wgan"))
        progress_bar = tqdm(total=total_steps, desc="Training WGAN")
        if config.checkpoint_path:
            os.makedirs(config.checkpoint_path, exist_ok=True)
            with open(os.path.join(config.checkpoint_path, "all_config.yaml"), "w") as f:
                yaml.dump(config.model_dump(), f)

    for _iter in range(total_iters):
        if RANK == 0:
            print(f"\n══ Iter {_iter+1}/{total_iters} | epoch {_iter * train_epochs_per_iter} ══")

        state.model.train()
        state.d_net.train()
        state.carry = None  # reset carry mỗi iter

        for set_name, batch, global_bs in train_loader:
            metrics = train_one_batch(config, state, batch, global_bs, RANK, WORLD_SIZE)

            if RANK == 0 and metrics:
                wandb.log(metrics, step=state.step)
                if writer:
                    for k, v in metrics.items():
                        writer.add_scalar(k, v, state.step)
                if progress_bar:
                    progress_bar.update(1)
                    progress_bar.set_postfix({
                        "D":   f"{metrics.get('train/loss_d', 0):+.3f}",
                        "Adv": f"{metrics.get('train/loss_g_adv', 0):+.3f}",
                        "Rec": f"{metrics.get('train/loss_rec', 0):.3f}",
                        "Acc": f"{metrics.get('train/exact_accuracy', 0):.1f}%",
                    })

            if ema_helper:
                ema_helper.update(state.model)

        should_eval = _iter >= (config.min_eval_interval or 0) and eval_loader is not None
        if should_eval:
            if RANK == 0:
                print("  Evaluating...")
            eval_state = copy.deepcopy(state)
            if config.ema and ema_helper:
                eval_state.model = ema_helper.ema_copy(eval_state.model)
            
            max_eval_batches = math.ceil(config.eval_samples / config.global_batch_size)
            eval_metrics = evaluate(eval_state, eval_loader, max_batches=max_eval_batches)

            if RANK == 0:
                wandb.log(eval_metrics, step=state.step)
                if writer:
                    for k, v in eval_metrics.items():
                        writer.add_scalar(k, v, state.step)
                acc = eval_metrics.get("eval/exact_accuracy", 0.0)
                print(f"  Eval exact accuracy: {acc:.2f}%")

                # Save best
                is_best = acc > best_acc
                if is_best:
                    best_acc = acc
                    save_checkpoint(config, eval_state, suffix="_best")
                    print(f"  ★ New best: {best_acc:.2f}%")

                if config.checkpoint_every_eval or _iter == total_iters - 1:
                    save_checkpoint(config, eval_state)

    if dist.is_initialized():
        dist.destroy_process_group()
    if RANK == 0:
        wandb.finish()
        if writer:
            writer.close()
        print(f"\nTraining done. Best eval accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    launch()
