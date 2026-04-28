import torch
from typing import List, Dict
from torch.utils.data import IterableDataset, get_worker_info
import torch.nn.functional as F
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import os
import random

from intentrcmd.datasets.util import gen_intent_indexed_values, gen_intent_scores, gen_bias_indexed_values, gen_mtl_label_indexed_values
from intentrcmd.datasets.base_listwise_dataset import BIAS_FEATURE_DICT


class UniGCRDataset(IterableDataset):
    """IterableDataset for UniGCR V2: Full-vocab scoring model.

    Each sample is a user session with:
    - User features (solo + seq + seq_group) — session-level, not per-candidate
    - labels_intents: (V+1,) full-vocab click labels
    - impressions: (V+1,) full-vocab impression indicators
    - gen_target_intent: the clicked intent id for the generative head
    - scores: (V+1,) full-vocab online scores (for online metrics baseline)

    Intent features are NOT included per sample — they are stored as model
    buffers (intent_batch) and encoded inside model.forward() for all V+1
    intents at once, matching online model architecture.
    """

    def __init__(self, file_path: str, date_list: List[str], intent_vocab_dict: Dict,
                 mtl_vocab_dicts: Dict, user_feature_config: Dict,
                 intent_feature_config: Dict = None, max_samples=-1,
                 for_evaluation=False, random_seed=42, shuffle_buffer=2048,
                 data_version='00', text_embedding_dict: Dict = None,
                 intent_text_config: Dict = None):
        self.file_path = file_path
        self.date_list = date_list
        self.intent_vocab_dict = intent_vocab_dict
        self.mtl_vocab_dicts = mtl_vocab_dicts
        self.user_feature_config = user_feature_config
        self.max_samples = max_samples if max_samples > 0 else 0
        self.for_evaluation = for_evaluation
        self.random_seed = random_seed
        self.shuffle_buffer = shuffle_buffer
        self.data_version = data_version

        self.parquet_files = self._list_parquet_files(file_path, date_list)
        self.total_rows = self._count_total_rows()

        if self.max_samples and self.total_rows > self.max_samples:
            print(f"[UniGCRDataset] Limiting dataset size to {self.max_samples} "
                  f"(Total available: {self.total_rows})")

    def _list_parquet_files(self, file_path: str, date_list: List[str]) -> List[str]:
        all_dates = os.listdir(file_path)
        file_list = []
        for date in all_dates:
            if date not in date_list:
                continue
            date_path = os.path.join(file_path, date, self.data_version)
            if not os.path.isdir(date_path):
                print(f"[WARNING] Skipping missing date directory: {date_path}")
                continue
            for file in os.listdir(date_path):
                if file.endswith('_SUCCESS') or not file.endswith(".parquet"):
                    continue
                full_path = os.path.join(date_path, file)
                if os.path.getsize(full_path) == 0:
                    print(f"[WARNING] Skipping 0-byte parquet file: {full_path}")
                    continue
                file_list.append(full_path)
        return file_list

    def _count_total_rows(self) -> int:
        total = 0
        for f in self.parquet_files:
            pf = pq.ParquetFile(f)
            total += pf.metadata.num_rows
        return total

    def __len__(self):
        if self.max_samples:
            return min(self.max_samples, self.total_rows)
        return self.total_rows

    def __iter__(self):
        worker = get_worker_info()
        files = list(self.parquet_files)
        max_samples = self.max_samples
        if worker:
            files = files[worker.id::worker.num_workers]
            if max_samples:
                max_samples = max_samples // worker.num_workers

        rng = random.Random(self.random_seed)
        if not self.for_evaluation:
            rng.shuffle(files)

        yielded = 0
        buffer = []

        for filepath in files:
            pf = pq.ParquetFile(filepath)

            for batch in pf.iter_batches(batch_size=256):
                cols = {
                    name: batch.column(i)
                    for i, name in enumerate(batch.schema.names)
                }

                for i in range(batch.num_rows):
                    row = {k: cols[k][i].as_py() for k in cols}
                    sample = self._process_row(row)

                    if not self.for_evaluation and self.shuffle_buffer > 0:
                        if len(buffer) < self.shuffle_buffer:
                            buffer.append(sample)
                            continue
                        # Replacement-style shuffle
                        idx = rng.randint(0, len(buffer) - 1)
                        yield buffer[idx]
                        buffer[idx] = sample
                    else:
                        yield sample

                    yielded += 1
                    if max_samples and yielded >= max_samples:
                        # Flush remaining buffer
                        if not self.for_evaluation:
                            rng.shuffle(buffer)
                            for s in buffer:
                                yield s
                        return

            del pf

        # Flush remaining buffer
        if not self.for_evaluation and buffer:
            rng.shuffle(buffer)
            for sample in buffer:
                yield sample

    def _process_row(self, row):
        # --- Intent candidate list (used to build full-vocab labels) ---
        intent_id_list = row.get('intent_id_list', [])
        if not isinstance(intent_id_list, (list, np.ndarray)):
            intent_id_list = [intent_id_list] if intent_id_list is not None else []
        elif isinstance(intent_id_list, np.ndarray):
            intent_id_list = intent_id_list.tolist()

        mapped_intent_ids = [self.intent_vocab_dict.get(int(x), 0) for x in intent_id_list]
        seq_len = len(mapped_intent_ids)

        def get_list(key, length, default=0.0):
            val = row.get(key, [])
            if isinstance(val, (list, np.ndarray)):
                if isinstance(val, np.ndarray):
                    val = val.tolist()
                if len(val) < length:
                    val = list(val) + [default] * (length - len(val))
                return val[:length]
            return [default] * length

        is_click_list = get_list('is_click_list', seq_len, 0.0)
        is_impression_list = get_list('is_impression_list', seq_len, 0.0)

        # # Fix missing impressions when click is present
        # for i in range(len(is_click_list)):
        #     if is_click_list[i] > 0:
        #         is_impression_list[i] = 1.0

        # --- Full-vocab labels (V+1,) — same dimension as online model ---
        raw_intent_id_list = row.get('intent_id_list', [])
        if isinstance(raw_intent_id_list, np.ndarray):
            raw_intent_id_list = raw_intent_id_list.tolist()

        full_labels = gen_intent_indexed_values(
            raw_intent_id_list, is_click_list, self.intent_vocab_dict)
        full_impressions = gen_intent_indexed_values(
            raw_intent_id_list, is_impression_list, self.intent_vocab_dict)

        sample = {
            'labels_intents': torch.FloatTensor(full_labels),        # (V+1,)
            'impressions': torch.FloatTensor(full_impressions),      # (V+1,)
        }

        # Mask: at least one valid click within vocab
        mask_intents = (sum(full_labels[1:]) >= 1)
        sample['masks_intents'] = torch.BoolTensor([mask_intents])   # (1,)

        # --- RC label (case category) ---
        if self.mtl_vocab_dicts and 'level3_case_category_name' in self.mtl_vocab_dicts:
            label_rc, mask_rc = gen_mtl_label_indexed_values(
                row.get('level3_case_category_name', None),
                self.mtl_vocab_dicts['level3_case_category_name'])
            sample['labels_rc'] = torch.FloatTensor(label_rc)    # (num_label_rc,)
            sample['masks_rc'] = torch.BoolTensor([mask_rc])     # (1,)

        # --- Live label (to live agent) ---
        label_live = int(row.get('is_to_live_agent', 0) == 1)
        mask_live = (mask_intents or label_live == 1)
        sample['labels_live'] = torch.FloatTensor([label_live])  # (1,)
        sample['masks_live'] = torch.BoolTensor([mask_live])     # (1,)

        # Bias features for debias confidence predictor
        source_list = row.get('source_list', [])
        if not isinstance(source_list, (list, np.ndarray)):
            source_list = []
        elif isinstance(source_list, np.ndarray):
            source_list = source_list.tolist()
        location_list = row.get('location_list', [])
        if not isinstance(location_list, (list, np.ndarray)):
            location_list = []
        elif isinstance(location_list, np.ndarray):
            location_list = location_list.tolist()

        sources = gen_bias_indexed_values(
            raw_intent_id_list, source_list,
            BIAS_FEATURE_DICT['source'], self.intent_vocab_dict)
        locations = gen_bias_indexed_values(
            raw_intent_id_list, location_list,
            BIAS_FEATURE_DICT['location'], self.intent_vocab_dict)
        sample['sources'] = torch.IntTensor(sources)       # (V+1,)
        sample['locations'] = torch.IntTensor(locations)   # (V+1,)

        # Full-vocab online scores (for online metrics baseline)
        if self.for_evaluation:
            raw_scores = row.get('scores', [])
            if isinstance(raw_scores, (list, np.ndarray)) and len(raw_scores) > 0:
                full_scores = gen_intent_scores(
                    raw_intent_id_list, row.get('item_id_list', []),
                    raw_scores, self.intent_vocab_dict)
            else:
                full_scores_col = row.get('full_scores', [])
                full_item_ids = row.get('full_item_id_list', [])
                if isinstance(full_scores_col, (list, np.ndarray)) and len(full_scores_col) > 0:
                    full_scores = gen_intent_scores(
                        raw_intent_id_list, full_item_ids,
                        full_scores_col, self.intent_vocab_dict)
                else:
                    full_scores = [0.0] * (len(self.intent_vocab_dict) + 1)
            sample['scores'] = torch.FloatTensor(full_scores)

        # Session ID
        if 'session_id' in row:
            sid = row['session_id']
            sample['session_id'] = sid if isinstance(sid, str) else torch.tensor(sid, dtype=torch.long)

        # --- User features ---
        for fid_k, fid_config in self.user_feature_config.items():
            if fid_config.get('type') != 'sparse':
                continue
            fid_v = row.get(fid_k, 0)
            shape = fid_config['shape']
            if isinstance(fid_v, (list, np.ndarray)) and len(fid_v) > 0:
                if len(shape) == 1 and shape[0] == 1:
                    fid_v = fid_v[0]
            if len(shape) == 1:
                if fid_v is None or (isinstance(fid_v, float) and np.isnan(fid_v)):
                    fid_v = 0
                if shape[0] == 1:
                    if fid_config.get('dtype', '') == 'int32':
                        fid_v = torch.IntTensor([fid_v]).reshape(-1)
                    else:
                        fid_v = torch.LongTensor([fid_v]).reshape(-1)
                else:
                    if not isinstance(fid_v, (list, np.ndarray)):
                        fid_v = [0] * shape[0]
                    elif isinstance(fid_v, np.ndarray):
                        fid_v = fid_v.tolist()
                    fid_v = fid_v[:shape[0]]
                    if fid_config.get('dtype', '') == 'int32':
                        fid_v = torch.IntTensor(fid_v)
                    else:
                        fid_v = torch.LongTensor(fid_v)
                sample[fid_k] = F.pad(fid_v, pad=(0, shape[0] - len(fid_v)), mode='constant', value=0)

        # --- Generative target (clicked intent for gen head prediction) ---
        clicked_intent_id = 0
        for i in range(seq_len):
            if is_click_list[i] > 0 and mapped_intent_ids[i] > 0:
                clicked_intent_id = mapped_intent_ids[i]
                break

        sample['gen_target_intent'] = torch.LongTensor([clicked_intent_id])

        return sample

