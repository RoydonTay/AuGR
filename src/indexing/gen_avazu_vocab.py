import argparse
import gzip
import json
import os
from collections import Counter
from datetime import datetime
from typing import Dict

import pandas as pd


AVAZU_ID_COL = "id"
AVAZU_LABEL_COL = "click"
AVAZU_RAW_HOUR_COL = "hour"

AVAZU_COLUMNS = [
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


def decode_avazu_hour(raw_hour: str):
    """Parse YYMMDDHH string and return derived categorical fields."""
    s = str(raw_hour).strip()
    if not s:
        return "0", "0", "0"
    try:
        dt = datetime.strptime(s[-8:], "%y%m%d%H")
    except ValueError:
        return "0", "0", "0"

    hour = str(dt.hour)
    weekday = str(dt.weekday())
    is_weekend = "1" if dt.weekday() >= 5 else "0"
    return hour, weekday, is_weekend


def read_avazu_chunks(input_path: str, chunk_size: int):
    """Read Avazu csv or csv.gz as chunk iterator with object dtype."""
    compression = "gzip" if input_path.endswith(".gz") else "infer"
    return pd.read_csv(
        input_path,
        chunksize=chunk_size,
        dtype=object,
        keep_default_na=False,
        compression=compression,
    )


def build_processed_categorical_frame(chunk: pd.DataFrame) -> pd.DataFrame:
    """Drop id/click semantics and derive categorical features for embedding vocab."""
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


def build_avazu_vocab_dict(
    input_path: str,
    chunk_size: int,
    min_category_count: int,
    max_vocab_size_per_feature: int,
) -> Dict[str, Dict[str, int]]:
    """Build deterministic vocab dicts for Avazu categorical features.

    - Index 0 is reserved for OOV/unknown categories.
    - Categories with count <= min_category_count are filtered into OOV.
    """
    counters = {k: Counter() for k in AVAZU_CATEGORICAL_FEATURES}

    for chunk in read_avazu_chunks(input_path=input_path, chunk_size=chunk_size):
        cat_df = build_processed_categorical_frame(chunk)
        for name in AVAZU_CATEGORICAL_FEATURES:
            counters[name].update(cat_df[name].astype(str).tolist())

    vocab_dict: Dict[str, Dict[str, int]] = {}
    for name in AVAZU_CATEGORICAL_FEATURES:
        kept_tokens = [
            token
            for token, cnt in counters[name].items()
            if cnt > min_category_count and token != OOV_TOKEN
        ]
        kept_tokens = sorted(kept_tokens)
        if max_vocab_size_per_feature > 0:
            kept_tokens = kept_tokens[:max_vocab_size_per_feature]

        # Reserve 0 for OOV and start known categories from 1.
        feature_vocab = {OOV_TOKEN: 0}
        feature_vocab.update({token: idx + 1 for idx, token in enumerate(kept_tokens)})
        vocab_dict[name] = feature_vocab

    return vocab_dict


def main():
    parser = argparse.ArgumentParser(description="Build Avazu categorical vocab dictionaries")
    parser.add_argument("--input_path", type=str, required=True, help="Path to Avazu train csv/csv.gz")
    parser.add_argument("--output_path", type=str, required=True, help="Output json path")
    parser.add_argument("--chunk_size", type=int, default=200000, help="Rows per chunk")
    parser.add_argument(
        "--min_category_count",
        type=int,
        default=1,
        help="Categories with count <= this threshold are mapped to OOV",
    )
    parser.add_argument(
        "--max_vocab_size_per_feature",
        type=int,
        default=0,
        help="Maximum kept categories per feature, 0 means unlimited",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=40,
        help="Reference embedding dimension to store in output metadata",
    )
    args = parser.parse_args()

    vocab_dict = build_avazu_vocab_dict(
        input_path=args.input_path,
        chunk_size=args.chunk_size,
        min_category_count=args.min_category_count,
        max_vocab_size_per_feature=args.max_vocab_size_per_feature,
    )

    output_payload = {
        "meta": {
            "embedding_dim": args.embedding_dim,
            "min_category_count": args.min_category_count,
            "oov_token": OOV_TOKEN,
            "categorical_features": AVAZU_CATEGORICAL_FEATURES,
        },
        "vocab": vocab_dict,
    }

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(output_payload, f)

    print("Feature vocab sizes:")
    for name in AVAZU_CATEGORICAL_FEATURES:
        print(f"- {name}: {len(vocab_dict[name])}")
    print(f"Saved vocab json to: {args.output_path}")


if __name__ == "__main__":
    main()
