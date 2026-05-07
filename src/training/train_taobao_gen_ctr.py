import argparse
import gzip
import json
import math
import os
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import torch
import wandb
from sklearn.metrics import log_loss
from torch.utils.data import IterableDataset
from transformers import TrainingArguments, default_data_collator, TrainerCallback

from src.models.taobao_unified_gen_model import TaobaoAuGRGenConfig, TaobaoAuGRGenModel
from intentrcmd.metrics import safe_auc
from intentrcmd.modules.taobao_batch_processor import (
    TAOBAO_BASE_CATEGORICAL_FEATURES,
    TaobaoBatchIterableDataset,
)
from intentrcmd.utils.hf_utils import Trainer


class StepUpdateCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        model = kwargs["model"]
        model.max_steps = state.max_steps

    def on_step_begin(self, args, state, control, **kwargs):
        model = kwargs["model"]
        model.global_step = state.global_step


def parse_feature_list(raw: str) -> List[str]:
    if not raw:
        return []
    return [name.strip() for name in raw.split(",") if name.strip()]


def load_taobao_vocab(taobao_vocab_path: str) -> Tuple[Dict[str, Dict[str, int]], Dict]:
    with open(taobao_vocab_path) as f:
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


def align_missing_history_vocabs(vocab_dict: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, int]]:
    """Backfill missing history vocab entries from paired base features.

    This keeps training robust when an older vocab json is missing one side
    of the aligned feature-history pairs.
    """
    pairs = [
        ("brand", "brand_his"),
        ("btag", "btag_his"),
        ("cate_id", "cate_his"),
    ]

    out = dict(vocab_dict)
    for base_key, history_key in pairs:
        has_base = base_key in out
        has_history = history_key in out
        if has_base and not has_history:
            out[history_key] = dict(out[base_key])
            print(f"[INFO] Backfilled missing vocab key '{history_key}' from '{base_key}'")
        elif has_history and not has_base:
            out[base_key] = dict(out[history_key])
            print(f"[INFO] Backfilled missing vocab key '{base_key}' from '{history_key}'")
    return out


class BatchToSampleIterableDataset(IterableDataset):
    """Convert batched iterable dataset output into per-sample items for HF Trainer."""

    def __init__(self, batched_dataset: TaobaoBatchIterableDataset):
        super().__init__()
        self.batched_dataset = batched_dataset

    def __iter__(self):
        for batch in self.batched_dataset:
            bsz = int(batch["label"].shape[0])
            for i in range(bsz):
                sample = {k: v[i] for k, v in batch.items()}
                sample["labels_click"] = sample["label"].float()
                yield sample


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits


def compute_metrics(eval_pred):
    predictions, labels = eval_pred

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
    opener = gzip.open if file_path.endswith(".gz") else open
    with opener(file_path, "rt", encoding="utf-8", errors="ignore") as f:
        line_count = sum(1 for _ in f)
    return max(line_count - 1, 0)


def main():
    parser = argparse.ArgumentParser(description="Train TaoBao CTR model with HuggingFace Trainer")
    parser.add_argument("--train_data_path", type=str, required=True, help="Path to TaoBao train csv/csv.gz")
    parser.add_argument("--valid_data_path", type=str, required=True, help="Path to TaoBao valid csv/csv.gz")
    parser.add_argument("--test_data_path", type=str, required=True, help="Path to TaoBao test csv/csv.gz")
    parser.add_argument("--taobao_vocab_path", type=str, required=True, help="Path to TaoBao vocab json")
    parser.add_argument("--model_config_path", type=str, default="", help="Optional model config json with feature_groups")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for model/checkpoints")

    parser.add_argument("--label_col", type=str, default="clk", help="Label column name")
    parser.add_argument("--target_feature_name", type=str, default="cate_id", help="Fallback item feature when item group is not provided")
    parser.add_argument(
        "--item_features",
        type=str,
        default="",
        help="Optional comma-separated item features for item token construction in CTR head",
    )
    parser.add_argument(
        "--categorical_features",
        type=str,
        default="",
        help="Optional comma-separated categorical features (overrides vocab metadata)",
    )
    parser.add_argument(
        "--sequence_features",
        type=str,
        default="",
        help="Optional comma-separated sequence features (overrides vocab metadata)",
    )
    parser.add_argument(
        "--sequence_separator",
        type=str,
        default="^",
        help="Sequence token separator in input files",
    )
    parser.add_argument("--sequence_max_len", type=int, default=50, help="Maximum sequence length per sequence feature")
    parser.add_argument("--sequence_pooling_type", type=str, default="self_attention", help="Sequence pooling type")

    parser.add_argument("--batch_size", type=int, default=10000, help="Rows per data chunk")
    parser.add_argument("--num_epochs", type=float, default=3.0, help="Number of passes over the training data")
    parser.add_argument("--max_steps", type=int, default=-1, help="Optional override for total train steps; -1 means infer from num_epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")

    parser.add_argument("--embedding_dim", type=int, default=40, help="Embedding dim per categorical feature")
    parser.add_argument("--d_model", type=int, default=40, help="Token/model dimension for HSTU and heads")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout in model")
    parser.add_argument("--lambda_gen", type=float, default=0.1, help="Weight of user-head CTR loss")
    parser.add_argument("--lambda_ctr", type=float, default=1.0, help="Weight of item-head CTR loss")
    parser.add_argument("--gen_loss_decay", action="store_true", help="Apply exponential decay to generation loss weight over training steps")
    parser.add_argument("--ctr_hidden_units", type=int, nargs="+", default=[40, 20], help="Hidden units for CTR tower")
    parser.add_argument("--ctr_shallow_shortcut", action="store_true", help="Enable shallow shortcut in CTR tower")
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

    vocab_dict, vocab_meta = load_taobao_vocab(args.taobao_vocab_path)
    vocab_dict = align_missing_history_vocabs(vocab_dict)
    model_group_config = load_model_group_config(args.model_config_path)

    meta_categorical = vocab_meta.get("categorical_features", []) if isinstance(vocab_meta, dict) else []
    meta_sequence = vocab_meta.get("sequence_features", []) if isinstance(vocab_meta, dict) else []
    categorical_features = parse_feature_list(args.categorical_features) or meta_categorical or TAOBAO_BASE_CATEGORICAL_FEATURES
    sequence_features = parse_feature_list(args.sequence_features) or meta_sequence or []

    if args.target_feature_name not in categorical_features:
        categorical_features = list(categorical_features) + [args.target_feature_name]
        print(f"[INFO] Appended target feature to categorical_features: {args.target_feature_name}")

    if isinstance(vocab_meta, dict) and "sequence_separator" in vocab_meta and args.sequence_separator == "^":
        sequence_separator = vocab_meta["sequence_separator"]
    else:
        sequence_separator = args.sequence_separator
    if isinstance(vocab_meta, dict) and "sequence_max_len" in vocab_meta and args.sequence_max_len == 50:
        sequence_max_len = int(vocab_meta["sequence_max_len"])
    else:
        sequence_max_len = args.sequence_max_len

    all_sparse_features = categorical_features + [name for name in sequence_features if name not in categorical_features]
    feature_groups = model_group_config.get("feature_groups", {})
    default_item_features = []
    if isinstance(feature_groups, dict):
        maybe_item_group = feature_groups.get("item_group", [])
        if isinstance(maybe_item_group, list):
            default_item_features = [name for name in maybe_item_group if isinstance(name, str) and name]
    item_feature_names = parse_feature_list(args.item_features) or default_item_features or [args.target_feature_name]

    for feature_name in item_feature_names:
        if feature_name not in categorical_features:
            raise ValueError(f"item feature must be a categorical feature, got: {feature_name}")
    for name in all_sparse_features:
        if name not in vocab_dict:
            raise ValueError(f"Missing feature in vocab dict: {name}")

    train_batched = TaobaoBatchIterableDataset(
        file_path=args.train_data_path,
        chunk_size=args.batch_size,
        mode="train",
        max_samples=args.max_train_samples,
        random_seed=args.seed,
        sparse_vocab_dict=vocab_dict,
        build_vocab_on_the_fly=False,
        has_header=True,
        categorical_features=categorical_features,
        sequence_features=sequence_features,
        sequence_separator=sequence_separator,
        sequence_max_len=sequence_max_len,
        label_col=args.label_col,
    )
    valid_batched = TaobaoBatchIterableDataset(
        file_path=args.valid_data_path,
        chunk_size=args.batch_size,
        mode="eval",
        max_samples=args.max_valid_samples,
        random_seed=args.seed,
        sparse_vocab_dict=vocab_dict,
        build_vocab_on_the_fly=False,
        has_header=True,
        categorical_features=categorical_features,
        sequence_features=sequence_features,
        sequence_separator=sequence_separator,
        sequence_max_len=sequence_max_len,
        label_col=args.label_col,
    )
    test_batched = TaobaoBatchIterableDataset(
        file_path=args.test_data_path,
        chunk_size=args.batch_size,
        mode="eval",
        max_samples=args.max_test_samples,
        random_seed=args.seed,
        sparse_vocab_dict=vocab_dict,
        build_vocab_on_the_fly=False,
        has_header=True,
        categorical_features=categorical_features,
        sequence_features=sequence_features,
        sequence_separator=sequence_separator,
        sequence_max_len=sequence_max_len,
        label_col=args.label_col,
    )

    train_dataset = BatchToSampleIterableDataset(train_batched)
    valid_dataset = BatchToSampleIterableDataset(valid_batched)
    test_dataset = BatchToSampleIterableDataset(test_batched)

    user_feature_names = [name for name in categorical_features if name not in set(item_feature_names)]
    feature_vocab_sizes = {
        name: ((max(vocab_dict[name].values()) + 1) if len(vocab_dict[name]) > 0 else 2)
        for name in all_sparse_features
    }

    model_config = TaobaoAuGRGenConfig(
        feature_vocab_sizes=feature_vocab_sizes,
        user_feature_names=user_feature_names,
        item_feature_names=item_feature_names,
        sequence_feature_names=sequence_features,
        feature_groups=feature_groups,
        target_feature_name=args.target_feature_name,
        emb_size=args.embedding_dim,
        d_model=args.d_model,
        dropout=args.dropout,
        sequence_pooling_type=args.sequence_pooling_type,
        ctr_hidden_units=args.ctr_hidden_units,
        hstu_num_heads=args.hstu_num_heads,
        hstu_num_blocks=args.hstu_num_blocks,
        lambda_ctr=args.lambda_ctr,
        lambda_gen=args.lambda_gen,
        ctr_shallow_shortcut=args.ctr_shallow_shortcut,
        use_post_hstu_moe=args.use_post_hstu_moe,
        moe_num_experts=args.moe_num_experts,
        moe_top_k=args.moe_top_k,
        moe_load_balance_weight=args.moe_load_balance_weight,
        moe_ffn_dim=args.moe_ffn_dim,
    )
    model = TaobaoAuGRGenModel(model_config)

    run_name = args.wandb_run_name or f"taobao_ctr_{datetime.now().strftime('%m%d_%H%M%S')}"
    report_to = "wandb" if (args.wandb_project and os.getenv("WANDB_API_KEY")) else "none"

    if report_to == "wandb":
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
                "lambda_ctr": args.lambda_ctr,
                "gen_loss_decay": args.gen_loss_decay,
                "post_hstu_moe": args.use_post_hstu_moe,
                "feature_groups": feature_groups,
                "item_features": item_feature_names,
                "sequence_features": sequence_features,
                "batch_size": args.batch_size,
                "num_epochs": args.num_epochs,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
            },
        )

    if args.max_steps > 0:
        max_steps = args.max_steps
    else:
        if args.max_train_samples > 0:
            train_rows = args.max_train_samples
        else:
            train_rows = count_data_rows(args.train_data_path)
            if train_rows <= 0:
                raise ValueError(f"Could not infer train rows from file: {args.train_data_path}")

        max_steps = math.ceil((train_rows * args.num_epochs) / max(args.batch_size, 1))
        max_steps = max(max_steps, 1)

    training_args = TrainingArguments(
        run_name=run_name,
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
        label_names=["labels_click"],
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

    trainer.train()

    best_state_dict = trainer.model.state_dict()
    torch.save(best_state_dict, os.path.join(args.output_dir, f"best_taobao_ctr_model_{args.exp}.bin"))

    metrics = trainer.evaluate()
    test_metrics = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
    metrics.update(test_metrics)

    with open(os.path.join(args.output_dir, f"metrics_{args.exp}.json"), "w") as f:
        json.dump(metrics, f)
    print(f"Final eval metrics: {metrics}")


if __name__ == "__main__":
    main()
