import argparse
import gzip
import json
import math
import os
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.metrics import log_loss
from torch.utils.data import IterableDataset
from transformers import TrainingArguments, default_data_collator, TrainerCallback
from transformers.trainer_utils import get_last_checkpoint
import wandb

from intentrcmd.metrics import safe_auc
from intentrcmd.modules.avazu_batch_processor import (
    AVAZU_CATEGORICAL_FEATURES,
    AvazuBatchIterableDataset,
)
from intentrcmd.utils.hf_utils import Trainer
from src.models.avazu_unified_model import AvazuUniGCRConfig, AvazuUniGCRModel


class StepUpdateCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        model = kwargs["model"]
        model.max_steps = state.max_steps

    def on_step_begin(self, args, state, control, **kwargs):
        model = kwargs["model"]
        model.global_step = state.global_step


def load_avazu_vocab(avazu_vocab_path: str) -> Tuple[Dict[str, Dict[str, int]], Dict]:
    with open(avazu_vocab_path) as f:
        payload = json.load(f)
    if isinstance(payload, dict) and "vocab" in payload:
        return payload["vocab"], payload.get("meta", {})
    return payload, {}


def load_model_group_config(model_config_path: str) -> Dict:
    if not model_config_path:
        return {}
    with open(model_config_path) as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("model config json must be a dict")
    return payload


class BatchToSampleIterableDataset(IterableDataset):
    """Convert batched iterable dataset output into per-sample items for HF Trainer."""

    def __init__(self, batched_dataset: AvazuBatchIterableDataset):
        super().__init__()
        self.batched_dataset = batched_dataset

    def __iter__(self):
        for batch in self.batched_dataset:
            bsz = int(batch["label"].shape[0])
            for i in range(bsz):
                sample = {k: v[i] for k, v in batch.items()}
                # Trainer labels consumed by model.forward.
                sample["labels_click"] = sample["label"].float()
                sample["labels_c14"] = sample["C14"].long()
                yield sample


def preprocess_logits_for_metrics(logits, labels):
    # Trainer may pass tuple(logits, ...) if model returns multiple outputs.
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits


def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    # labels come in the same order as label_names.
    if isinstance(labels, (list, tuple)) and len(labels) >= 1:
        labels_click = labels[0]
    else:
        labels_click = labels

    logits = predictions
    probs = 1.0 / (1.0 + np.exp(-logits))
    y_true = np.asarray(labels_click).reshape(-1)
    y_score = np.asarray(probs).reshape(-1)

    auc = safe_auc(y_true=y_true, y_score=y_score)
    y_score_clip = np.clip(y_score, 1e-7, 1.0 - 1e-7)
    ll = float(log_loss(y_true=y_true, y_pred=y_score_clip, labels=[0, 1]))
    return {
        "eval_auc": float(auc) if auc == auc else float("nan"),
        "eval_log_loss": ll,
    }


def count_data_rows(file_path: str) -> int:
    """Count number of data rows in csv/csv.gz (excluding header)."""
    opener = gzip.open if file_path.endswith(".gz") else open
    with opener(file_path, "rt", encoding="utf-8", errors="ignore") as f:
        line_count = sum(1 for _ in f)

    # Avazu readers in this script use has_header=True.
    return max(line_count - 1, 0)


def main():
    parser = argparse.ArgumentParser(description="Train Avazu CTR model with HuggingFace Trainer")
    parser.add_argument("--train_data_path", type=str, required=True, help="Path to Avazu train csv/csv.gz")
    parser.add_argument("--valid_data_path", type=str, required=True, help="Path to Avazu valid csv/csv.gz")
    parser.add_argument("--avazu_vocab_path", type=str, required=True, help="Path to avazu vocab json")
    parser.add_argument("--model_config_path", type=str, default="", help="Optional model config json with feature_groups")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for model/checkpoints")

    parser.add_argument("--batch_size", type=int, default=10000, help="Rows per data chunk")
    parser.add_argument("--num_epochs", type=float, default=3.0, help="Number of passes over the training data")
    parser.add_argument("--max_steps", type=int, default=-1, help="Optional override for total train steps; -1 means infer from num_epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--model_resume_training", action="store_true", help="Whether to resume training from last checkpoint in output_dir")

    parser.add_argument("--embedding_dim", type=int, default=40, help="Embedding dim per categorical feature")
    parser.add_argument("--d_model", type=int, default=40, help="Token/model dimension for HSTU and heads")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout in model")
    parser.add_argument("--lambda_gen", type=float, default=0.1, help="Weight of C14 generative loss")
    parser.add_argument("--lambda_ctr", type=float, default=1.0, help="Weight of CTR loss")
    parser.add_argument("--ctr_hidden_units", type=int, nargs="+", default=[40, 20], help="Hidden units for CTR tower")
    parser.add_argument("--ctr_shallow_shortcut", action="store_true", help="Enable shallow shortcut in CTR tower")
    parser.add_argument("--gen_loss_decay", action="store_true", help="Enable auxiliary gen loss decay")
    parser.add_argument("--hstu_num_heads", type=int, default=2, help="Number of heads in HSTU")
    parser.add_argument("--hstu_num_blocks", type=int, default=4, help="Number of blocks in HSTU")
    parser.add_argument("--use_post_hstu_moe", action="store_true", help="Use MoE layer after HSTU")
    parser.add_argument("--moe_num_experts", type=int, default=4, help="Number of experts in MoE layer")
    parser.add_argument("--moe_top_k", type=int, default=1, help="Number of experts to activate per token in MoE layer")
    parser.add_argument("--moe_load_balance_weight", type=float, default=0.01, help="Weight for MoE load balancing loss")
    parser.add_argument("--moe_ffn_dim", type=int, default=0, help="FFN dimension for MoE layer")
    parser.add_argument("--eval_steps", type=int, default=500, help="Evaluate every N steps")
    parser.add_argument("--save_steps", type=int, default=500, help="Save every N steps")
    parser.add_argument("--max_train_samples", type=int, default=0, help="Cap train rows, 0 for all")
    parser.add_argument("--max_valid_samples", type=int, default=0, help="Cap valid rows, 0 for all")
    parser.add_argument("--test_data_path", type=str, required=True, help="Path to Avazu test csv/csv.gz")
    parser.add_argument("--max_test_samples", type=int, default=0, help="Cap test rows, 0 for all")
    parser.add_argument("--num_workers", type=int, default=2, help="Dataloader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    parser.add_argument("--wandb_project", type=str, default="", help="Wandb project")
    parser.add_argument("--wandb_run_name", type=str, default="", help="Wandb run name")
    parser.add_argument("--exp", type=str, default="local", help="Experiment tag for file names")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    vocab_dict, vocab_meta = load_avazu_vocab(args.avazu_vocab_path)
    model_group_config = load_model_group_config(args.model_config_path)
    for name in AVAZU_CATEGORICAL_FEATURES:
        if name not in vocab_dict:
            raise ValueError(f"Missing feature in vocab dict: {name}")

    if isinstance(vocab_meta, dict) and "embedding_dim" in vocab_meta:
        print(f"Vocab meta embedding_dim={vocab_meta['embedding_dim']}, using embedding_dim={args.embedding_dim}")

    train_batched = AvazuBatchIterableDataset(
        file_path=args.train_data_path,
        chunk_size=args.batch_size,
        mode="train",
        max_samples=args.max_train_samples,
        random_seed=args.seed,
        sparse_vocab_dict=vocab_dict,
        build_vocab_on_the_fly=False,
        has_header=True,
    )
    valid_batched = AvazuBatchIterableDataset(
        file_path=args.valid_data_path,
        chunk_size=args.batch_size,
        mode="eval",
        max_samples=args.max_valid_samples,
        random_seed=args.seed,
        sparse_vocab_dict=vocab_dict,
        build_vocab_on_the_fly=False,
        has_header=True,
    )
    test_batched = AvazuBatchIterableDataset(
        file_path=args.test_data_path,
        chunk_size=args.batch_size,
        mode="eval",
        max_samples=args.max_test_samples,
        random_seed=args.seed,
        sparse_vocab_dict=vocab_dict,
        build_vocab_on_the_fly=False,
        has_header=True,
    )

    train_dataset = BatchToSampleIterableDataset(train_batched)
    valid_dataset = BatchToSampleIterableDataset(valid_batched)
    test_dataset = BatchToSampleIterableDataset(test_batched)

    user_feature_names = [k for k in AVAZU_CATEGORICAL_FEATURES if k != "C14"]
    feature_vocab_sizes = {
        name: ((max(vocab_dict[name].values()) + 1) if len(vocab_dict[name]) > 0 else 2)
        for name in AVAZU_CATEGORICAL_FEATURES
    }

    model_config = AvazuUniGCRConfig(
        feature_vocab_sizes=feature_vocab_sizes,
        user_feature_names=user_feature_names,
        feature_groups=model_group_config.get("feature_groups", {}),
        c14_feature_name="C14",
        emb_size=args.embedding_dim,
        d_model=args.d_model,
        dropout=args.dropout,
        ctr_hidden_units=args.ctr_hidden_units,
        hstu_num_heads=args.hstu_num_heads,
        hstu_num_blocks=args.hstu_num_blocks,
        lambda_ctr=args.lambda_ctr,
        lambda_gen=args.lambda_gen,
        ctr_shallow_shortcut=args.ctr_shallow_shortcut,
        gen_loss_decay=args.gen_loss_decay,
        use_post_hstu_moe=args.use_post_hstu_moe,
        moe_num_experts=args.moe_num_experts,
        moe_top_k=args.moe_top_k,
        moe_load_balance_weight=args.moe_load_balance_weight,
        moe_ffn_dim=args.moe_ffn_dim,
    )
    model = AvazuUniGCRModel(model_config)

    run_name = args.wandb_run_name or f"avazu_ctr_{datetime.now().strftime('%m%d_%H%M%S')}"
    report_to = "wandb" if (args.wandb_project and os.getenv("WANDB_API_KEY")) else "none"

    wandb.login()
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config={
            "train_data_path": args.train_data_path,
            "valid_data_path": args.valid_data_path,
            "embedding_dim": args.embedding_dim,
            "d_model": args.d_model,
            "lambda_gen": args.lambda_gen,
            "feature_groups": model_group_config.get("feature_groups", {}),
            "batch_size": args.batch_size,
            "num_epochs": args.num_epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
        },
    )

    # IterableDataset has no reliable __len__, so infer max_steps from epochs by default.
    if args.max_steps > 0:
        max_steps = args.max_steps
        print(f"Using explicit max_steps={max_steps}")
    else:
        if args.max_train_samples > 0:
            train_rows = args.max_train_samples
            print(f"Inferring steps from capped train rows: max_train_samples={train_rows}")
        else:
            train_rows = count_data_rows(args.train_data_path)
            if train_rows <= 0:
                raise ValueError(f"Could not infer train rows from file: {args.train_data_path}")
            print(f"Inferring steps from full train file rows={train_rows}")

        max_steps = math.ceil((train_rows * args.num_epochs) / max(args.batch_size, 1))
        max_steps = max(max_steps, 1)
        print(
            f"Epoch-driven schedule: num_epochs={args.num_epochs}, "
            f"batch_size={args.batch_size}, inferred_max_steps={max_steps}"
        )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        max_steps=max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=args.weight_decay,
        max_grad_norm=1.0,
        eval_strategy="steps",
        logging_strategy="steps",
        save_strategy="steps",
        eval_steps=max(args.eval_steps, 1),
        logging_steps=max(args.eval_steps, 1),
        save_steps=max(args.save_steps, 1),
        load_best_model_at_end=True,
        save_total_limit=2,
        metric_for_best_model="eval_auc",
        greater_is_better=True,
        report_to=report_to,
        remove_unused_columns=False,
        label_names=["labels_click", "labels_c14"],
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=True,
        ignore_data_skip=True,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=default_data_collator,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=[StepUpdateCallback()],
    )

    # Resume from last checkpoint, but will not recover data iterator offset.
    last_checkpoint = None
    if args.model_resume_training:
        if os.path.isdir(training_args.output_dir):
            last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None:
            print(f"[WARNING] failed to resume checkpoint from: {training_args.output_dir}")

    if last_checkpoint is not None:
        print(f"[INFO] Resuming training from checkpoint: {last_checkpoint}")
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        trainer.train()

    # Save best/final checkpoints with stable names.
    best_state_dict = trainer.model.state_dict()
    torch.save(best_state_dict, os.path.join(args.output_dir, f"best_avazu_ctr_model_{args.exp}.bin"))

    metrics = trainer.evaluate()
    test_metrics = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
    metrics.update(test_metrics)

    with open(os.path.join(args.output_dir, f"metrics_{args.exp}.json"), "w") as f:
        json.dump(metrics, f)
    print(f"Final eval metrics: {metrics}")


if __name__ == "__main__":
    main()
