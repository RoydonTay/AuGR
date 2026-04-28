import random
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import IterableDataset, get_worker_info


AVAZU_ID_COL = "id"
AVAZU_LABEL_COL = "click"
AVAZU_RAW_HOUR_COL = "hour"

# Raw columns in Avazu csv/gz examples.
AVAZU_RAW_COLUMNS = [
    AVAZU_ID_COL,
    AVAZU_LABEL_COL,
    AVAZU_RAW_HOUR_COL,
    "C1",
    "banner_pos",
    "site_id",
    "site_domain",
    "site_category",
    "app_id",
    "app_domain",
    "app_category",
    "device_id",
    "device_ip",
    "device_model",
    "device_type",
    "device_conn_type",
    "C14",
    "C15",
    "C16",
    "C17",
    "C18",
    "C19",
    "C20",
    "C21",
]

# Processed categorical features used for embeddings.
AVAZU_CATEGORICAL_FEATURES = [
    "hour",
    "weekday",
    "is_weekend",
    "C1",
    "banner_pos",
    "site_id",
    "site_domain",
    "site_category",
    "app_id",
    "app_domain",
    "app_category",
    "device_id",
    "device_ip",
    "device_model",
    "device_type",
    "device_conn_type",
    "C14",
    "C15",
    "C16",
    "C17",
    "C18",
    "C19",
    "C20",
    "C21",
]

OOV_TOKEN = "<OOV>"


class EncodedFeat:
    """Encoded feature parameters and configurations."""

    def __init__(self, shape: List, dim: int, type: str = "sparse"):
        self.shape = shape
        self.dim = dim
        self.type = type


def build_sparse_embedding_and_dropout_layers(target_model_config, fid_config_dict, dropout: float = 0.2):
    """Build embedding/dropout layers from sparse feature config."""
    embedding_dict = {}
    dropout_dict = {}
    encoded_feat_dict = {}

    for k, v in target_model_config.items():
        if fid_config_dict[k]["type"] == "sparse":
            embedding_dict[k] = nn.Embedding(fid_config_dict[k]["num_embeddings"], v["d_model"])
            dropout_dict[k] = nn.Dropout(dropout)

        encoded_feat_dict[k] = EncodedFeat(
            shape=fid_config_dict[k]["shape"],
            dim=v["d_model"],
            type=fid_config_dict[k]["type"],
        )

    return nn.ModuleDict(embedding_dict), nn.ModuleDict(dropout_dict), encoded_feat_dict


def decode_avazu_hour(raw_hour: str):
    """Parse Avazu YYMMDDHH integer-like value into hour, weekday, is_weekend."""
    s = str(raw_hour).strip()
    if not s:
        return "0", "0", "0"

    # Avazu hour is YYMMDDHH, e.g. 14102100.
    try:
        dt = datetime.strptime(s[-8:], "%y%m%d%H")
    except ValueError:
        return "0", "0", "0"

    hour = str(dt.hour)
    weekday = str(dt.weekday())
    is_weekend = "1" if dt.weekday() >= 5 else "0"
    return hour, weekday, is_weekend


class AvazuBatchIterableDataset(IterableDataset):
    """Stream Avazu data in chunks and yield tensor batches for CTR training."""

    def __init__(
        self,
        file_path: str,
        chunk_size: int = 8192,
        mode: str = "train",
        max_samples: int = 0,
        random_seed: int = 42,
        sparse_vocab_dict: Optional[Dict[str, Dict[str, int]]] = None,
        max_vocab_size_per_feature: int = 0,
        build_vocab_on_the_fly: bool = False,
        has_header: bool = True,
        min_category_count: int = 1,
    ):
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.mode = mode
        self.max_samples = max_samples
        self.random_seed = random_seed
        self.max_vocab_size_per_feature = max_vocab_size_per_feature
        self.build_vocab_on_the_fly = build_vocab_on_the_fly
        self.has_header = has_header
        self.min_category_count = min_category_count

        if sparse_vocab_dict is None:
            sparse_vocab_dict = {k: {} for k in AVAZU_CATEGORICAL_FEATURES}
        self.sparse_vocab_dict = sparse_vocab_dict
        self._next_sparse_index = {
            k: (max(v.values()) + 1 if len(v) > 0 else 1)
            for k, v in self.sparse_vocab_dict.items()
        }

    def __iter__(self):
        worker = get_worker_info()
        worker_id = worker.id if worker else 0
        worker_count = worker.num_workers if worker else 1

        rng = random.Random(self.random_seed + worker_id)
        yielded = 0

        reader = pd.read_csv(
            self.file_path,
            chunksize=self.chunk_size,
            dtype=object,
            keep_default_na=False,
            header=0 if self.has_header else None,
            names=None if self.has_header else AVAZU_RAW_COLUMNS,
        )

        for chunk_idx, chunk in enumerate(reader):
            if chunk_idx % worker_count != worker_id:
                continue

            batch = self.process_chunk(
                chunk,
                update_vocab=(self.mode == "train" and self.build_vocab_on_the_fly),
            )
            batch_size = int(batch["label"].shape[0])
            if batch_size == 0:
                continue

            if self.mode == "train":
                perm = torch.as_tensor(rng.sample(range(batch_size), k=batch_size), dtype=torch.long)
                batch = {k: v[perm] for k, v in batch.items()}

            if self.max_samples:
                remaining = self.max_samples - yielded
                if remaining <= 0:
                    return
                if batch_size > remaining:
                    batch = {k: v[:remaining] for k, v in batch.items()}
                    batch_size = remaining

            yield batch
            yielded += batch_size
            if self.max_samples and yielded >= self.max_samples:
                return

    def process_chunk(self, chunk: pd.DataFrame, update_vocab: bool = False) -> Dict[str, torch.Tensor]:
        """Convert one Avazu chunk to model-ready tensors."""
        labels = pd.to_numeric(chunk[AVAZU_LABEL_COL], errors="coerce").fillna(0).astype(np.float32).to_numpy()
        sparse_tokens = self._build_processed_categorical_frame(chunk)
        sparse_np = self._encode_sparse_features(sparse_tokens, update_vocab=update_vocab)

        out = {
            "label": torch.from_numpy(labels),
            "sparse_features": torch.from_numpy(sparse_np),
        }
        for i, name in enumerate(AVAZU_CATEGORICAL_FEATURES):
            out[name] = out["sparse_features"][:, i].reshape(-1, 1)
        return out

    def build_sparse_feature_config(self, embedding_dim: int = 40) -> Dict[str, Dict]:
        """Generate sparse feature config from current vocab sizes."""
        feature_config = {}
        for name in AVAZU_CATEGORICAL_FEATURES:
            feature_config[name] = {
                "type": "sparse",
                "shape": [1],
                "num_embeddings": self._next_sparse_index[name],
                "d_model": embedding_dim,
            }
        return feature_config

    def _build_processed_categorical_frame(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """Drop id and expand raw hour into derived categorical features."""
        cat_df = pd.DataFrame(index=chunk.index)

        decoded = chunk[AVAZU_RAW_HOUR_COL].astype(str).apply(decode_avazu_hour)
        cat_df["hour"] = decoded.apply(lambda x: x[0])
        cat_df["weekday"] = decoded.apply(lambda x: x[1])
        cat_df["is_weekend"] = decoded.apply(lambda x: x[2])

        for name in AVAZU_CATEGORICAL_FEATURES:
            if name in ("hour", "weekday", "is_weekend"):
                continue
            cat_df[name] = chunk[name].astype(str)

        cat_df = cat_df.fillna(OOV_TOKEN).replace("", OOV_TOKEN)
        return cat_df

    def _encode_sparse_features(self, cat_df: pd.DataFrame, update_vocab: bool) -> np.ndarray:
        encoded_columns = []
        for name in AVAZU_CATEGORICAL_FEATURES:
            encoded = self._encode_sparse_column(
                cat_df[name].astype(str).to_numpy(),
                feature_name=name,
                update_vocab=update_vocab,
            )
            encoded_columns.append(encoded)
        return np.stack(encoded_columns, axis=1).astype(np.int64)

    def _encode_sparse_column(self, values: np.ndarray, feature_name: str, update_vocab: bool) -> np.ndarray:
        vocab = self.sparse_vocab_dict[feature_name]
        next_idx = self._next_sparse_index[feature_name]
        max_vocab = self.max_vocab_size_per_feature

        encoded = np.zeros(len(values), dtype=np.int64)
        for i, token in enumerate(values):
            idx = vocab.get(token)
            if idx is None and update_vocab:
                can_extend = (max_vocab <= 0) or (len(vocab) < max_vocab)
                if can_extend:
                    idx = next_idx
                    vocab[token] = idx
                    next_idx += 1
            encoded[i] = 0 if idx is None else idx

        self._next_sparse_index[feature_name] = next_idx
        return encoded
