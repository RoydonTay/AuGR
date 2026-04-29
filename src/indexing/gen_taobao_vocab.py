import argparse
import json
import math
import os
from collections import Counter
from typing import Dict, List

import pandas as pd

TAOBAO_LABEL_COL = "clk"
ITEM_CLASS_COL = "item_class" # alternative gen target to cate_id

TAOBAO_BASE_CATEGORICAL_FEATURES = [
    "final_gender_code", "age_level", "pvalue_level", "shopping_level", "occupation", "new_user_class_level", "adgroup_id", "pid", "price", "brand", "campaign_id", "cate_id", "customer", "cms_segid", "cms_group_id"
]

OOV_TOKEN = "<OOV>"

def parse_feature_list(raw: str) -> List[str]:
    if not raw:
        return []
    return [name.strip() for name in raw.split(",") if name.strip()]


def ensure_item_class_feature(categorical_features: List[str], sequence_features: List[str]) -> List[str]:
    """Auto-include item_class when it can be derived from available features.

    This keeps backward compatibility with old feature lists that omit item_class
    while still enabling the new generative target vocab.
    """
    features = list(categorical_features)
    if ITEM_CLASS_COL in features:
        return features

    has_cate = "cate_id" in features or "cate_id" in sequence_features
    has_brand = (
        "brand" in features
        or "brand_id" in features
        or "brand" in sequence_features
        or "brand_id" in sequence_features
    )
    if has_cate and has_brand:
        features.append(ITEM_CLASS_COL)
    return features


def normalize_sequence_tokens(value: str, separator: str) -> List[str]:
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



def build_item_class_series(chunk: pd.DataFrame) -> pd.Series:
    if "cate_id" not in chunk.columns:
        raise ValueError("Cannot derive item_class: missing cate_id column")
    if "brand" not in chunk.columns:
        raise ValueError("Cannot derive item_class: missing brand column")

    cate_values = chunk["cate_id"].astype(str).fillna(OOV_TOKEN).replace("", OOV_TOKEN)
    brand_values = chunk["brand"].astype(str).fillna(OOV_TOKEN).replace("", OOV_TOKEN)

    item_class = cate_values.str.cat(brand_values, sep="|")
    invalid_mask = (cate_values == OOV_TOKEN) | (brand_values == OOV_TOKEN)
    return item_class.mask(invalid_mask, OOV_TOKEN)


def align_feature_pair_vocab_indices(
    vocab_dict: Dict[str, Dict[str, int]],
    counters: Dict[str, Counter],
    min_category_count: int,
    max_vocab_size_per_feature: int,
    feature_key: str,
    history_key: str,
):
    """Force shared ids for overlapping tokens between a feature and its history.

    Each feature still has its own vocab dict, but the same token receives the
    same index in both dicts when present in both features after filtering.
    """
    if feature_key not in counters or history_key not in counters:
        return

    def _kept(counter: Counter) -> List[str]:
        tokens = [
            token
            for token, count in counter.items()
            if count > min_category_count and token != OOV_TOKEN
        ]
        tokens = sorted(tokens)
        if max_vocab_size_per_feature > 0:
            tokens = tokens[:max_vocab_size_per_feature]
        return tokens

    feature_tokens = _kept(counters[feature_key])
    history_tokens = _kept(counters[history_key])

    merged_tokens = sorted(set(feature_tokens) | set(history_tokens))
    if max_vocab_size_per_feature > 0:
        merged_tokens = merged_tokens[:max_vocab_size_per_feature]

    shared_vocab = {OOV_TOKEN: 0}
    shared_vocab.update({token: idx + 1 for idx, token in enumerate(merged_tokens)})
    vocab_dict[feature_key] = dict(shared_vocab)
    vocab_dict[history_key] = dict(shared_vocab)


def compute_top_half_length_threshold(
    input_path: str,
    chunk_size: int,
    behavior_col: str,
    sequence_separator: str,
) -> int:
    """Return minimum sequence length needed to keep top 50% most-active rows."""
    length_hist = Counter()
    total_rows = 0

    reader = pd.read_csv(
        input_path,
        chunksize=chunk_size,
        dtype=object,
        keep_default_na=False,
        compression="infer",
    )

    for chunk in reader:
        if behavior_col not in chunk.columns:
            raise ValueError(f"Missing behavior feature column for filtering: {behavior_col}")

        lengths = chunk[behavior_col].astype(str).apply(
            lambda value: len(normalize_sequence_tokens(value=value, separator=sequence_separator))
        )
        counts = lengths.value_counts()
        for length, cnt in counts.items():
            length_hist[int(length)] += int(cnt)
        total_rows += int(len(lengths))

    if total_rows == 0:
        return 0

    keep_rows = max(1, int(math.ceil(total_rows * 0.5)))
    running = 0
    for length in sorted(length_hist.keys(), reverse=True):
        running += length_hist[length]
        if running >= keep_rows:
            return int(length)

    return 0


def build_taobao_vocab_dict(
    input_path: str,
    chunk_size: int,
    categorical_features: List[str],
    sequence_features: List[str],
    sequence_separator: str,
    min_category_count: int,
    max_vocab_size_per_feature: int,
    filter_top_half_by_btag_len: bool = False,
    behavior_col: str = "btag_his",
) -> Dict[str, Dict[str, int]]:
    counters = {name: Counter() for name in categorical_features}
    for name in sequence_features:
        counters.setdefault(name, Counter())

    min_behavior_len = None
    if filter_top_half_by_btag_len:
        min_behavior_len = compute_top_half_length_threshold(
            input_path=input_path,
            chunk_size=chunk_size,
            behavior_col=behavior_col,
            sequence_separator=sequence_separator,
        )
        print(
            f"Applying top-50% behavior filter using {behavior_col}: "
            f"keep rows with initial length >= {min_behavior_len}"
        )

    reader = pd.read_csv(
        input_path,
        chunksize=chunk_size,
        dtype=object,
        keep_default_na=False,
        compression="infer",
    )

    for chunk in reader:
        if min_behavior_len is not None:
            if behavior_col not in chunk.columns:
                raise ValueError(f"Missing behavior feature column for filtering: {behavior_col}")
            lengths = chunk[behavior_col].astype(str).apply(
                lambda value: len(normalize_sequence_tokens(value=value, separator=sequence_separator))
            )
            chunk = chunk[lengths >= min_behavior_len]
            if chunk.empty:
                continue

        for name in categorical_features:
            if name == ITEM_CLASS_COL and name not in chunk.columns:
                values = build_item_class_series(chunk)
            else:
                if name not in chunk.columns:
                    raise ValueError(f"Missing categorical feature column in train data: {name}")
                values = chunk[name].astype(str).fillna(OOV_TOKEN).replace("", OOV_TOKEN)
            counters[name].update(values.tolist())

        for name in sequence_features:
            if name not in chunk.columns:
                raise ValueError(f"Missing sequence feature column in train data: {name}")
            for value in chunk[name].astype(str).tolist():
                tokens = normalize_sequence_tokens(value=value, separator=sequence_separator)
                counters[name].update(tokens)

    vocab_dict: Dict[str, Dict[str, int]] = {}
    for name, counter in counters.items():
        kept_tokens = [
            token for token, count in counter.items() if count > min_category_count and token != OOV_TOKEN
        ]
        kept_tokens = sorted(kept_tokens)
        if max_vocab_size_per_feature > 0:
            kept_tokens = kept_tokens[:max_vocab_size_per_feature]

        feature_vocab = {OOV_TOKEN: 0}
        feature_vocab.update({token: idx + 1 for idx, token in enumerate(kept_tokens)})
        vocab_dict[name] = feature_vocab

    align_feature_pair_vocab_indices(
        vocab_dict=vocab_dict,
        counters=counters,
        min_category_count=min_category_count,
        max_vocab_size_per_feature=max_vocab_size_per_feature,
        feature_key="brand",
        history_key="brand_his",
    )
    align_feature_pair_vocab_indices(
        vocab_dict=vocab_dict,
        counters=counters,
        min_category_count=min_category_count,
        max_vocab_size_per_feature=max_vocab_size_per_feature,
        feature_key="btag",
        history_key="btag_his",
    )
    align_feature_pair_vocab_indices(
        vocab_dict=vocab_dict,
        counters=counters,
        min_category_count=min_category_count,
        max_vocab_size_per_feature=max_vocab_size_per_feature,
        feature_key="cate_id",
        history_key="cate_his",
    )

    return vocab_dict


def main():
    parser = argparse.ArgumentParser(description="Build TaoBao categorical/sequence vocab dictionaries")
    parser.add_argument("--input_path", type=str, required=True, help="Path to TaoBao train csv/csv.gz")
    parser.add_argument("--output_path", type=str, required=True, help="Output json path")
    parser.add_argument("--chunk_size", type=int, default=200000, help="Rows per chunk")
    parser.add_argument(
        "--categorical_features",
        type=str,
        default=",".join(TAOBAO_BASE_CATEGORICAL_FEATURES),
        help="Comma-separated categorical feature names",
    )
    parser.add_argument(
        "--sequence_features",
        type=str,
        default="btag_his,cate_his,brand_his",
        help="Comma-separated sequence feature names (optional)",
    )
    parser.add_argument(
        "--sequence_separator",
        type=str,
        default="^",
        help="Separator used in sequence feature columns",
    )
    parser.add_argument(
        "--sequence_max_len",
        type=int,
        default=50,
        help="Reference max sequence length to store in output metadata",
    )
    parser.add_argument(
        "--min_category_count",
        type=int,
        default=10,
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
    parser.add_argument(
        "--filter_top_half_by_btag_len",
        action="store_true",
        help="If set, keep only the top 50% rows by initial btag_his length before building vocab",
    )
    parser.add_argument(
        "--behavior_col",
        type=str,
        default="btag_his",
        help="Behavior sequence column used for top-50% filtering",
    )
    args = parser.parse_args()

    categorical_features = parse_feature_list(args.categorical_features)
    sequence_features = parse_feature_list(args.sequence_features)
    categorical_features = ensure_item_class_feature(categorical_features, sequence_features)
    all_features = categorical_features + [name for name in sequence_features if name not in categorical_features]
    if not all_features:
        raise ValueError("At least one categorical or sequence feature must be provided")

    vocab_dict = build_taobao_vocab_dict(
        input_path=args.input_path,
        chunk_size=args.chunk_size,
        categorical_features=categorical_features,
        sequence_features=sequence_features,
        sequence_separator=args.sequence_separator,
        min_category_count=args.min_category_count,
        max_vocab_size_per_feature=args.max_vocab_size_per_feature,
        filter_top_half_by_btag_len=args.filter_top_half_by_btag_len,
        behavior_col=args.behavior_col,
    )

    output_payload = {
        "meta": {
            "embedding_dim": args.embedding_dim,
            "min_category_count": args.min_category_count,
            "oov_token": OOV_TOKEN,
            "categorical_features": categorical_features,
            "sequence_features": sequence_features,
            "sequence_separator": args.sequence_separator,
            "sequence_max_len": args.sequence_max_len,
            "filter_top_half_by_btag_len": bool(args.filter_top_half_by_btag_len),
            "behavior_col": args.behavior_col,
        },
        "vocab": vocab_dict,
    }

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(output_payload, f)

    print("Feature vocab sizes:")
    for name in all_features:
        print(f"- {name}: {len(vocab_dict[name])}")
    print(f"Saved vocab json to: {args.output_path}")


if __name__ == "__main__":
    main()
