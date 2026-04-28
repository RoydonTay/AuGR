import random
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import IterableDataset, get_worker_info


TAOBAO_LABEL_COL = "clk"
ITEM_CLASS_COL = "item_class"

TAOBAO_BASE_CATEGORICAL_FEATURES = [
    "final_gender_code",
    "age_level",
    "pvalue_level",
    "shopping_level",
    "occupation",
    "new_user_class_level",
    "adgroup_id",
    "pid",
    "price",
    "brand",
    "campaign_id",
    "cate_id",
    ITEM_CLASS_COL,
    "customer",
]

OOV_TOKEN = "<OOV>"


def _build_item_class_series(chunk: pd.DataFrame) -> pd.Series:
    if "cate_id" not in chunk.columns:
        raise ValueError("Cannot derive item_class: missing cate_id column")
    
    if "brand" not in chunk.columns:
        raise ValueError("Cannot derive item_class: missing brand column")

    cate_values = chunk["cate_id"].astype(str).fillna(OOV_TOKEN).replace("", OOV_TOKEN)
    brand_values = chunk["brand"].astype(str).fillna(OOV_TOKEN).replace("", OOV_TOKEN)

    item_class = cate_values.str.cat(brand_values, sep="|")
    invalid_mask = (cate_values == OOV_TOKEN) | (brand_values == OOV_TOKEN)
    item_class = item_class.mask(invalid_mask, OOV_TOKEN)
    return item_class


def _normalize_sequence_tokens(value: str, separator: str) -> List[str]:
    s = str(value).strip()
    if not s:
        return []
    if s.lower() == "nan" or s == OOV_TOKEN:
        return []

    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1].strip()
        if not s:
            return []

    if separator and separator in s:
        raw_tokens = s.split(separator)
    elif "," in s:
        raw_tokens = s.split(",")
    elif " " in s:
        raw_tokens = s.split(" ")
    else:
        raw_tokens = [s]

    return [tok.strip() for tok in raw_tokens if tok.strip() and tok.strip() != OOV_TOKEN]


class TaobaoBatchIterableDataset(IterableDataset):
    """Stream TaoBao csv/csv.gz data in chunks and yield tensor batches for CTR training."""

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
        categorical_features: Optional[List[str]] = None,
        sequence_features: Optional[List[str]] = None,
        sequence_separator: str = "^",
        sequence_max_len: int = 50,
        label_col: str = TAOBAO_LABEL_COL,
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
        self.categorical_features = categorical_features or TAOBAO_BASE_CATEGORICAL_FEATURES
        self.sequence_features = sequence_features or []
        self.sequence_separator = sequence_separator
        self.sequence_max_len = sequence_max_len
        self.label_col = label_col

        self.all_sparse_features = self.categorical_features + [
            name for name in self.sequence_features if name not in self.categorical_features
        ]

        if sparse_vocab_dict is None:
            sparse_vocab_dict = {name: {} for name in self.all_sparse_features}
        for name in self.all_sparse_features:
            sparse_vocab_dict.setdefault(name, {})
        self.sparse_vocab_dict = sparse_vocab_dict
        self._next_sparse_index = {
            name: (max(v.values()) + 1 if len(v) > 0 else 1)
            for name, v in self.sparse_vocab_dict.items()
            if name in self.all_sparse_features
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
            compression="infer",
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
        """Convert one TaoBao chunk to model-ready tensors."""
        if self.label_col not in chunk.columns:
            raise ValueError(f"Missing label column in input chunk: {self.label_col}")

        labels = pd.to_numeric(chunk[self.label_col], errors="coerce").fillna(0).astype(np.float32).to_numpy()
        cat_df = self._build_categorical_frame(chunk)
        sparse_np = self._encode_categorical_features(cat_df, update_vocab=update_vocab)

        out = {
            "label": torch.from_numpy(labels),
            "sparse_features": torch.from_numpy(sparse_np),
        }
        for i, name in enumerate(self.categorical_features):
            out[name] = out["sparse_features"][:, i].reshape(-1, 1)

        for name in self.sequence_features:
            if name not in chunk.columns:
                raise ValueError(f"Missing sequence feature column in input chunk: {name}")
            seq_np = self._encode_sequence_column(
                chunk[name].astype(str).to_numpy(),
                feature_name=name,
                update_vocab=update_vocab,
            )
            out[name] = torch.from_numpy(seq_np)

        return out

    def build_sparse_feature_config(self, embedding_dim: int = 40) -> Dict[str, Dict]:
        """Generate sparse feature config from current vocab sizes."""
        feature_config = {}
        for name in self.all_sparse_features:
            shape = [self.sequence_max_len] if name in self.sequence_features else [1]
            feature_config[name] = {
                "type": "sparse",
                "shape": shape,
                "num_embeddings": self._next_sparse_index[name],
                "d_model": embedding_dim,
            }
        return feature_config

    def _build_categorical_frame(self, chunk: pd.DataFrame) -> pd.DataFrame:
        cat_df = pd.DataFrame(index=chunk.index)
        for name in self.categorical_features:
            if name == ITEM_CLASS_COL and name not in chunk.columns:
                cat_df[name] = _build_item_class_series(chunk)
                continue

            if name not in chunk.columns:
                raise ValueError(f"Missing categorical feature column in input chunk: {name}")
            cat_df[name] = chunk[name].astype(str)

        cat_df = cat_df.fillna(OOV_TOKEN).replace("", OOV_TOKEN)
        return cat_df

    def _encode_categorical_features(self, cat_df: pd.DataFrame, update_vocab: bool) -> np.ndarray:
        encoded_columns = []
        for name in self.categorical_features:
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

    def _encode_sequence_column(self, values: np.ndarray, feature_name: str, update_vocab: bool) -> np.ndarray:
        vocab = self.sparse_vocab_dict[feature_name]
        next_idx = self._next_sparse_index[feature_name]
        max_vocab = self.max_vocab_size_per_feature
        max_len = max(self.sequence_max_len, 1)

        encoded = np.zeros((len(values), max_len), dtype=np.int64)
        for i, value in enumerate(values):
            tokens = _normalize_sequence_tokens(value=value, separator=self.sequence_separator)
            if len(tokens) > max_len:
                tokens = tokens[-max_len:]

            for j, token in enumerate(tokens):
                idx = vocab.get(token)
                if idx is None and update_vocab:
                    can_extend = (max_vocab <= 0) or (len(vocab) < max_vocab)
                    if can_extend:
                        idx = next_idx
                        vocab[token] = idx
                        next_idx += 1
                encoded[i, j] = 0 if idx is None else idx

        self._next_sparse_index[feature_name] = next_idx
        return encoded
