import torch
import zlib


class UniGCRDataCollator:
    """Data collator for UniGCR V2 full-vocab model.

    All tensors are fixed-size (labels/impressions are (V+1,) dimension),
    so collation is simple stacking — no padding needed.

    Handles session ID hashing for GAUC group computation.
    """

    def __init__(self, user_feature_config, intent_feature_config=None):
        self.user_feature_config = user_feature_config

        self.user_keys = set()
        if self.user_feature_config:
            self.user_keys.update(
                k for k, v in self.user_feature_config.items()
                if v.get('type') in ('sparse', 'dense'))

        self._has_logged = False

    def collate_fn(self, batch):
        return self.__call__(batch)

    def __call__(self, batch):
        if not batch:
            return {}

        elem = batch[0]
        collated = {}
        batch_size = len(batch)

        if not self._has_logged:
            print(f"[UniGCRDataCollator] Sample keys: {list(elem.keys())}")
            print(f"[UniGCRDataCollator] Batch size: {batch_size}")

        # --- Fixed-size tensors: just stack ---
        # labels_intents (V+1,), impressions (V+1,), scores (V+1,), gen_target_intent (1,)
        # sources (V+1,), locations (V+1,), masks_intents (1,)
        stack_keys = {'labels_intents', 'impressions', 'scores', 'gen_target_intent',
                      'sources', 'locations', 'masks_intents',
                      'labels_rc', 'masks_rc', 'labels_live', 'masks_live'}
        for key in stack_keys:
            if key in elem:
                collated[key] = torch.stack([s[key] for s in batch])

        # --- User features: stack (B, shape) ---
        for key in self.user_keys:
            if key not in elem:
                continue
            collated[key] = torch.stack([s[key] for s in batch])

        # --- Session / Group ID for GAUC ---
        if 'session_id' in elem:
            session_ids = [s['session_id'] for s in batch]
            first_sid = session_ids[0]
            if isinstance(first_sid, str):
                hashed = [zlib.adler32(s.encode('utf-8')) for s in session_ids]
                collated['group_id'] = torch.tensor(hashed, dtype=torch.long)
            else:
                collated['group_id'] = torch.stack(session_ids)

        if not self._has_logged:
            self._has_logged = True
            for k, v in collated.items():
                if torch.is_tensor(v):
                    print(f"  {k}: {v.shape} {v.dtype}")

        return collated