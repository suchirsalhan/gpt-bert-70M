# --- Imports --- #
import os
import json
import shutil
import time
from pathlib import Path
import torch
import wandb
import os
import os.path
import argparse
from tqdm import tqdm
from itertools import count
from socket import gethostname
from tokenizers import Tokenizer
from statistics import mean
import json
import math
import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from lamb import Lamb
from model_extra import Bert
from utils import cosine_schedule_with_warmup_cooldown, is_main_process, get_rank, seed_everything, get_world_size
from dataset import MaskedDataset, CausalDataset, ValidationDataset
from model_logging import ModelLogger


from transformers import Trainer, TrainingArguments
from transformers.trainer_callback import (
    TrainerCallback,
    TrainerState,
    TrainerControl,
)
from transformers.trainer_utils import (
    HubStrategy,
    PREFIX_CHECKPOINT_DIR,
)
from transformers.utils import (
    CONFIG_NAME,
    WEIGHTS_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
)

from huggingface_hub import upload_folder



"""
BABYLM CHECKPOINTING LOGIC
"""

# --- Constants and Helpers --- #
TRAIN_EPOCHS = 10
GLOBAL_BATCH_SIZE = 64  # NOTE: 64 * 16k = 1M tokens per batch (if seq_len=16k)

def get_deepspeed_config(accumulation_steps, num_devices):
    return {
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {"device": "cpu"},
            "offload_param": {"device": "cpu"},
        },
        "train_batch_size": GLOBAL_BATCH_SIZE // num_devices,
        "gradient_accumulation_steps": accumulation_steps,
        "bf16": {"enabled": True},
    }

# --- Custom Checkpoint Callback --- #

class CustomCheckpointingCallback(TrainerCallback):
    """
    Implements BabyLM checkpointing:
    - Every 1M words until 10M
    - Every 10M words until 100M
    - Every 100M words until 1B
    (Uses token counts internally, assuming a fixed token-to-word ratio.)
    """
    def __init__(self, total_steps, seq_len):
        super().__init__()

        self.seq_len = seq_len
        total_tokens = total_steps * GLOBAL_BATCH_SIZE * seq_len
        self.token_to_word_ratio = total_tokens / 1_000_000_000

        self.checkpoint_tokens = (
            [int(self.token_to_word_ratio * i * 1_000_000) for i in range(1, 11)] +      # 1M–10M
            [int(self.token_to_word_ratio * i * 10_000_000) for i in range(2, 11)] +     # 20M–100M
            [int(self.token_to_word_ratio * i * 100_000_000) for i in range(2, 11)]      # 200M–1B
        )
        self.next_checkpoint_idx = 0

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        tokens_seen = state.global_step * GLOBAL_BATCH_SIZE * self.seq_len
        if (self.next_checkpoint_idx < len(self.checkpoint_tokens) and
            tokens_seen >= self.checkpoint_tokens[self.next_checkpoint_idx]):
            print(f"\nDEBUG: Triggering checkpoint at step {state.global_step}")
            control.should_save = True

            words_seen = int(self.checkpoint_tokens[self.next_checkpoint_idx] / self.token_to_word_ratio)
            print(
                f"DEBUG: Checkpoint at {words_seen:,} words "
                f"({self.checkpoint_tokens[self.next_checkpoint_idx]:,} tokens) "
                f"| Step {state.global_step:,} "
                f"| Progress: {self.next_checkpoint_idx + 1}/{len(self.checkpoint_tokens)}"
            )
            self.next_checkpoint_idx += 1
        return control

# --- Custom Trainer with Upload Logic --- #

class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        self.seq_len = kwargs.pop("seq_len")
        self.total_steps = kwargs.pop("total_steps")

        total_tokens = self.total_steps * GLOBAL_BATCH_SIZE * self.seq_len
        self.token_to_word_ratio = total_tokens / 1_000_000_000

        self.checkpoint_words = (
            [int(i * 1_000_000) for i in range(1, 11)] +      # 1M–10M
            [int(i * 10_000_000) for i in range(2, 11)] +     # 20M–100M
            [int(i * 100_000_000) for i in range(2, 11)]      # 200M–1B
        )
        self.next_checkpoint_idx = 0

        super().__init__(*args, **kwargs)

    def _push_from_checkpoint(self, checkpoint_folder):
        if not self.is_world_process_zero() or self.args.hub_strategy == HubStrategy.END:
            return

        if not self.args.hub_always_push and self.push_in_progress is not None and not self.push_in_progress.is_done():
            return

        output_dir = self.args.output_dir
        modeling_files = [CONFIG_NAME, WEIGHTS_NAME, SAFE_WEIGHTS_NAME]

        for index_file in [WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_INDEX_NAME]:
            index_path = os.path.join(checkpoint_folder, index_file)
            if os.path.isfile(index_path):
                modeling_files.append(index_file)
                with open(index_path) as f:
                    index = json.load(f)
                modeling_files.extend(set(index["weight_map"].values()))

        for modeling_file in modeling_files:
            src = os.path.join(checkpoint_folder, modeling_file)
            dst = os.path.join(output_dir, modeling_file)
            if os.path.isfile(src):
                shutil.copy(src, dst)

        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)

        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

        commit_message = (
            f"Checkpoint step: {self.state.global_step:,} | "
            f"Target words: {self.checkpoint_words[self.next_checkpoint_idx]:,} | "
            f"Actual tokens: {self.state.global_step * GLOBAL_BATCH_SIZE * self.seq_len:,} | "
            f"Actual words: {self.state.global_step * GLOBAL_BATCH_SIZE * self.seq_len / self.token_to_word_ratio:,} | "
            f"Progress: {self.next_checkpoint_idx + 1}/{len(self.checkpoint_words)}"
        )
        self.next_checkpoint_idx += 1

        # Upload main output dir
        for attempt in range(5):
            try:
                upload_folder(
                    repo_id=self.hub_model_id,
                    folder_path=output_dir,
                    commit_message=commit_message,
                    token=self.args.hub_token,
                    run_as_future=False,
                    ignore_patterns=["_*", f"{PREFIX_CHECKPOINT_DIR}-*"],
                )
                break
            except Exception as e:
                print(f"Upload attempt {attempt+1} failed: {e}")
                if attempt < 4:
                    time.sleep(15)
                else:
                    print("Max upload retries reached. Skipping this checkpoint push.")

        # Optionally upload the raw checkpoint folder
        if self.args.hub_strategy in [HubStrategy.CHECKPOINT, HubStrategy.ALL_CHECKPOINTS]:
            path_in_repo = (
                "last-checkpoint" if self.args.hub_strategy == HubStrategy.CHECKPOINT
                else Path(checkpoint_folder).name
            )
            upload_folder(
                repo_id=self.hub_model_id,
                folder_path=checkpoint_folder,
                path_in_repo=path_in_repo,
                commit_message=commit_message + ", checkpoint",
                token=self.args.hub_token,
                run_as_future=False,
            )



"""
Computes whether each GPU should do masked or causal training using a hybrid ratio:
if rank * denom < numer * world_size:
    use_masked
else:
    use_causal

Loads a JSON config file (e.g., transformer dimensions, layers) and appends it to the args.
"""

"""
prepare_model_and_optimizer(args)
Loads config and creates the Bert model. Moves it to the correct GPU (args.device).
Sets up weight decay filtering for optimizer.
Uses either AdamW or Lamb as optimizer.
Sets up cosine learning rate scheduler with warmup and cooldown.
Wraps model in DistributedDataParallel with memory-efficient settings.
Creates an EMA (exponential moving average) copy of the model.
Loads from checkpoint if specified.
"""


"""
get_batch(...)
Gets a batch from the dataset, Moves it to the device
Returns tensors in the format required for BERT-style training: input_ids, target_ids, attention_mask, and mask_p (the masking probability for that batch)
"""


# ====================
# Trainer
# ====================
def setup_training(args, tokenizer):
    assert torch.cuda.is_available()
    args.n_gpu = torch.cuda.device_count()

    args.world_size = int(os.environ["WORLD_SIZE"])
    args.rank = int(os.environ["SLURM_PROCID"])
    args.gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
    assert args.gpus_per_node == torch.cuda.device_count()
    print(f"Hello from rank {args.rank} of {args.world_size} on {gethostname()} where there are {args.gpus_per_node} allocated GPUs per node.", flush=True)

    assert args.world_size % args.hybrid_denominator == 0

    # if args.rank / args.world_size < args.hybrid_numerator / args.hybrid_denominator:
    if args.rank * args.hybrid_denominator < args.hybrid_numerator * args.world_size:
        args.dataset_type = "masked"
    else:
        args.dataset_type = "causal"

    print(f"Dataset type: {args.dataset_type}", flush=True)

    seed_everything(args.seed + args.rank)

    torch.distributed.init_process_group(backend="nccl", rank=args.rank, world_size=args.world_size)
    if args.rank == 0:
        print(f"Group initialized? {torch.distributed.is_initialized()}", flush=True)

    args.local_rank = args.rank - args.gpus_per_node * (args.rank // args.gpus_per_node)
    torch.cuda.set_device(args.local_rank)
    args.device = torch.device("cuda", args.local_rank)
    print(f"RCCL started on device {args.device}", flush=True)
    print(f"host: {gethostname()}, rank: {args.rank}, local_rank: {args.local_rank}")

    if is_main_process():
        print(f"Training for {args.max_steps:,} steps with {get_world_size()} GPUs")
        print(f"In total, the model will be trained on 'steps'({args.max_steps:,}) x 'GPUs'({get_world_size()}) x 'batch_size'({args.local_batch_size:,}) x 'seq_len'({args.seq_length:,}) = {args.max_steps * get_world_size() * args.local_batch_size * args.seq_length:,} subword instances")

    args.vocab_size = tokenizer.get_vocab_size()

    if is_main_process():
        wandb.init(
            name=args.name,
            project="BabyLM-v2",
            entity="nor-ret"
        )

# ====================
# Load Config
# ====================
def load_model_config(args):
    with open(args.config_file) as f:
        config = json.load(f)
    for k, v in config.items():
        setattr(args, k, v)
    return args


# ====================
# Model & Optimizer
# ====================
def prepare_model_and_optimizer(args):
    args = load_config(args)
    model = Bert(args)

    if is_main_process():
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        wandb.config.update(args)
        wandb.config.update({"n_params": n_params})
        print(model)
        print(f"NUMBER OF PARAMETERS: {n_params}\n", flush=True)

    model.to(args.device)

    no_decay = ['bias', 'layer_norm']
    decay_params = [(n, p) for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)]
    no_decay_params = [(n, p) for n, p in model.named_parameters() if any(nd in n for nd in no_decay)]
    optimizer_grouped_parameters = [
        {'params': [p for _, p in decay_params], 'weight_decay': args.weight_decay},
        {'params': [p for _, p in no_decay_params], 'weight_decay': 0.0}
    ]

    if is_main_process():
        print("Parameters without weight decay:")
        for n, _ in no_decay_params:
            print(n)
        print()
        print("Parameters with weight decay:")
        for n, _ in decay_params:
            print(n)
        print(flush=True)

    if args.optimizer == "adam" or args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            betas=(args.optimizer_beta1, args.optimizer_beta2),
            eps=args.optimizer_eps,
        )
    elif args.optimizer == "lamb":
        optimizer = Lamb(
            optimizer_grouped_parameters,
            args.learning_rate,
            betas=(args.optimizer_beta1, args.optimizer_beta2),
            eps=args.optimizer_eps,
        )

    scheduler = cosine_schedule_with_warmup_cooldown(
        optimizer,
        int(args.max_steps * args.warmup_proportion),
        int(args.max_steps * args.cooldown_proportion),
        args.max_steps,
        0.1
    )

    model = DistributedDataParallel(
        model,
        device_ids=[args.local_rank],
        bucket_cap_mb=torch.cuda.get_device_properties(args.device).total_memory,
        broadcast_buffers=False,
        gradient_as_bucket_view=True,
        static_graph=True
    )

    ema_model: nn.Module = copy.deepcopy(model.module)
    for param in ema_model.parameters():
        param.requires_grad = False

    global_step, epoch = 0, 0
    if args.checkpoint_filename is not None:
        state_dict = torch.load(args.checkpoint_filename, map_location="cpu")
        model.load_state_dict(state_dict["model"])
        ema_model.load_state_dict(state_dict["ema_model"])
        optimizer.load_state_dict(state_dict["optimizer"])
        scheduler.load_state_dict(state_dict["scheduler"])
        global_step = state_dict["global_step"]
        epoch = state_dict["epoch"]

    return model, ema_model, optimizer, scheduler, global_step, epoch


def get_batch(dataloader, device, global_step):
    dataloader._dataset.set_global_step(global_step)
    batch = next(dataloader)
    input_ids, target_ids, attention_mask, mask_p = [t.pin_memory().to(device, non_blocking=True) for t in batch]
    input_ids, target_ids = input_ids.t(), target_ids.t()
    mask_p = mask_p.mean()

    return input_ids, attention_mask, target_ids, mask_p



# ====================
#  TRAINING ARGS
# ====================

def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--tokenizer_path", type=str)

    # Model
    parser.add_argument("--config_file", type=str)
    parser.add_argument("--model_type", type=str, default="bert")
    parser.add_argument("--n_special_tokens", type=int, default=16)

    # Optimizer
    parser.add_argument("--lr", "--learning_rate", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--optimizer", type=str, choices=["adamw", "lamb"], default="adamw")
    parser.add_argument("--optimizer_eps", type=float, default=1e-8)
    parser.add_argument("--optimizer_beta1", type=float, default=0.9)
    parser.add_argument("--optimizer_beta2", type=float, default=0.98)
    parser.add_argument("--max_gradient", type=float, default=2.0)

    # Training
    parser.add_argument("--seq_len", "--seq_length", type=int, default=128)
    parser.add_argument("--batch_size", "--local_batch_size", type=int, default=128)
    parser.add_argument("--global_batch_size", type=int, default=32768)
    parser.add_argument("--batch_reduction", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=31250 // 2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--mixed_precision", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)

    # EMA
    parser.add_argument("--ema_decay", type=float, default=0.999)

    # Warmup / Cooldown
    parser.add_argument("--warmup_proportion", type=float, default=0.016)
    parser.add_argument("--cooldown_proportion", type=float, default=0.016)

    # MLM settings
    parser.add_argument("--mask_p_start", type=float, default=0.3)
    parser.add_argument("--mask_p_end", type=float, default=0.15)
    parser.add_argument("--mask_random_p", type=float, default=0.1)
    parser.add_argument("--mask_keep_p", type=float, default=0.1)
    parser.add_argument("--token_weighted_loss", action="store_true", default=False)
    parser.add_argument("--z_loss_weight", type=float, default=1e-4)

    # Logging & Checkpointing
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--hub_token", type=str)
    parser.add_argument("--hub_model_id", type=str)
    parser.add_argument("--hub_strategy", type=str, default="checkpoint")

    return parser.parse_args()

# ====================
# Main Training Script
# ====================
def train(model, ema_model, tokenizer, optimizer, scheduler, args):
    # Initialize the global training step counter
    global_step = 0

    # Placeholders for train and validation dataloaders (they will be reloaded each epoch)
    train_dataloader, valid_dataloader = None, None

    # Loop over all training epochs
    for epoch in range(args.epochs):
        # Load or update the training and validation datasets for the current epoch
        train_dataloader, valid_dataloader = load_datasets(
            args, tokenizer, epoch, global_step, train_dataloader, valid_dataloader
        )

        # Set model to training mode and zero out gradients
        model.train()
        optimizer.zero_grad(set_to_none=True)

        # Determine number of steps this epoch (respecting total max steps and accumulation)
        num_steps = min(len(train_dataloader), (args.max_steps - global_step) * args.accumulate_steps)
        train_dataloader = iter(train_dataloader)  # Make dataloader iterable

        # Initialize running metrics for logging
        total_loss = total_accuracy = total_z_loss = total_mask_p = total_grad_norm = 0.0

        # Pre-fetch the first batch
        input_ids_, attention_mask_, target_ids_, mask_p_ = get_batch(train_dataloader, args.device, global_step)

        # Training loop for the current epoch
        for local_step in tqdm(
            range(num_steps),
            desc="Train iteration",
            initial=global_step,
            total=args.max_steps,
            disable=not is_main_process()
        ):
            # Get the current batch (from prefetch)
            input_ids, attention_mask, target_ids, mask_p = input_ids_, attention_mask_, target_ids_, mask_p_

            # Forward pass with automatic mixed precision (bfloat16 if enabled)
            with torch.cuda.amp.autocast(args.mixed_precision, dtype=torch.bfloat16):
                with ModelLogger(enable=global_step % 100 == 0, module=model):  # Optional model stats logging
                    loss, accuracy, z_loss, num_tokens = model(input_ids, attention_mask, target_ids)

            # Pre-fetch the next batch (except for the last step)
            if local_step < num_steps - 1:
                input_ids_, attention_mask_, target_ids_, mask_p_ = get_batch(train_dataloader, args.device, global_step)

            # Compute loss weight if using token-based scaling (distributed-safe)
            if args.token_weighted_loss:
                total_tokens = torch.tensor(num_tokens, device=args.device, dtype=torch.long)
                torch.distributed.all_reduce(total_tokens, torch.distributed.ReduceOp.SUM)
                weight = args.world_size * num_tokens / total_tokens / args.accumulate_steps
            else:
                weight = 1.0 / args.accumulate_steps  # Uniform weighting for accumulated steps

            # Backward pass: scale loss (including z_loss) and accumulate gradients
            ((loss + args.z_loss_weight * z_loss) * weight).backward()

            # Accumulate metrics (scaled)
            total_loss += loss.detach() * weight
            total_accuracy += accuracy * weight
            total_z_loss += z_loss * weight
            total_mask_p += mask_p * weight

            # Only perform optimization after accumulate_steps steps
            if (local_step + 1) % args.accumulate_steps != 0:
                continue

            # Clip gradient norms to stabilize training and avoid exploding gradients
            total_grad_norm += nn.utils.clip_grad_norm_(model.parameters(), args.max_gradient) * weight

            # Optimizer step and learning rate scheduler step
            optimizer.step()
            scheduler.step()

            # Update Exponential Moving Average (EMA) model parameters
            with torch.no_grad():
                for param_q, param_k in zip(model.module.parameters(), ema_model.parameters()):
                    param_k.data.mul_(args.ema_decay).add_((1.0 - args.ema_decay) * param_q.detach().data)

                # Split loss and metrics into MLM and CLM depending on hybrid ratio and dataset type
                if args.dataset_type == "masked":
                    total_mlm_loss = total_loss / (args.hybrid_numerator / args.hybrid_denominator)
                    total_clm_loss = torch.zeros_like(total_mlm_loss)
                    total_mask_p = total_mask_p / (args.hybrid_numerator / args.hybrid_denominator)
                else:
                    total_clm_loss = total_loss / (1 - args.hybrid_numerator / args.hybrid_denominator)
                    total_mlm_loss = torch.zeros_like(total_clm_loss)
                    total_mask_p = torch.zeros_like(total_mask_p)

                # Aggregate metrics across processes in distributed training
                metrics = torch.stack([
                    total_loss,
                    total_accuracy,
                    total_z_loss,
                    total_mask_p,
                    total_mlm_loss,
                    total_clm_loss
                ])
                torch.distributed.all_reduce(metrics, torch.distributed.ReduceOp.AVG)

                # Unpack final metrics after reduction
                total_loss, total_accuracy, total_z_loss, total_mask_p, total_mlm_loss, total_clm_loss = metrics.tolist()

            # Log training statistics to Weights & Biases (only from main process)
            if is_main_process():
                wandb.log(
                    {
                        "epoch": epoch,
                        "train/loss": total_loss,
                        "train/z_loss": total_z_loss,
                        "train/perplexity": math.exp(total_loss),
                        "train/accuracy": total_accuracy * 100.0,
                        "train/mlm_loss": total_mlm_loss,
                        "train/clm_loss": total_clm_loss,
                        "stats/learning_rate": optimizer.param_groups[0]['lr'],
                        "stats/grad_norm": total_grad_norm,
                        "stats/seq_length": train_dataloader.dataset.seq_length,
                        "stats/global_batch_size": args.current_global_batch_size,
                        "stats/local_batch_size": args.current_local_batch_size,
                        "stats/accumulate_steps": args.accumulate_steps,
                        "stats/mask_p": total_mask_p,
                        "global_step": global_step
                    },
                    commit=True
                )

            # Reset gradients and accumulated metrics
            optimizer.zero_grad(set_to_none=True)
            total_loss = total_accuracy = total_z_loss = total_mask_p = total_grad_norm = 0.0

            # Save checkpoint periodically
            if global_step % args.save_every == 0:
                save(model, ema_model, optimizer, scheduler, global_step, epoch, args)

            # Run validation periodically
            if (global_step + 1) % args.validate_every == 0:
                validation_epoch(model, valid_dataloader, epoch, args)
                model.train()  # Switch back to training mode

            # Update global step counter
            global_step += 1

            # Exit training loop if max steps reached
            if global_step >= args.max_steps:
                return

def main():
    args = parse_arguments()
    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    setup_training(args, tokenizer)

    model, ema_model, optimizer, scheduler = prepare_model_and_optimizer(args, tokenizer)

    train(model, ema_model, tokenizer, optimizer, scheduler, args)

if __name__ == "__main__":
    main()
