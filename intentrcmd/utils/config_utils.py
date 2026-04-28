import os
import json


def load_intent_issue_mapping(intent_issue_mapping_path):
    """Load intent issue mapping."""

    with open(intent_issue_mapping_path) as f:
        intent_issue_mapping = json.load(f)
        intent_issue_mapping = {int(intent_id): [int(issue_id) for issue_id in issue_ids] \
            for intent_id, issue_ids in intent_issue_mapping.items()}
    return intent_issue_mapping


def load_feature_dict(feature_dict_path, grass_region):
    """Load regional feature dict."""

    with open(feature_dict_path) as f:
        feature_dict = json.load(f)
        for key in feature_dict:
            if not key.endswith(grass_region.lower()):
                continue
            feature_dict = feature_dict[key]
    return feature_dict


def load_both_model_config(model_config_path):
    """Load both user and intent model config"""

    with open(model_config_path) as f:
        model_config = json.load(f)
        user_model_config = model_config['user']
        intent_model_config = model_config['intent']

    print('len(user_model_config):', len(user_model_config))
    print('len(intent_model_config):', len(intent_model_config))
    return user_model_config, intent_model_config


def load_both_feature_config(feature_config_path, user_feature_dict, intent_feature_dict):
    """Load both user and intent feature config.

    input format: {
        'user': {
            'fid1': {'type': 'dense', ...}
            'fid2': {'type': 'sparse', ...}
            'fid3': {'type': 'emb', ...}
        },
        'intent': { ... }
    }
    """

    def proc_sparse_feature_config(fid, sparse_feature_config, feature_dict):
        """Set embedding size according to how the sparse feature is generated.

        - For mapping operator processed features, embedding size is length of feature dict.
        - For operators like bucket, timestamp, embedding size should be specified manually by `index_max + 1` .
        - For operators like ctime, embedding size should be specified manually by `clip_max - clip_min + 2`.
        """
        if 'index_max' in sparse_feature_config:
            sparse_feature_config['num_embeddings'] = sparse_feature_config['index_max'] + 1
        elif 'clip_min' in sparse_feature_config:
            clip_min, clip_max = sparse_feature_config['clip_min'], sparse_feature_config['clip_max']
            sparse_feature_config['num_embeddings'] = clip_max - clip_min + 2
        else:
            fid_dict = user_feature_dict[fid.lstrip('fid')]
            sparse_feature_config['num_embeddings'] = len(fid_dict.values()) + 1
        return sparse_feature_config

    with open(feature_config_path) as f:
        feature_config = json.load(f)

        user_feature_config = feature_config['user']
        for fid in user_feature_config:
            if user_feature_config[fid]['type'] == 'sparse':
                user_feature_config[fid] = proc_sparse_feature_config(fid, user_feature_config[fid], user_feature_dict)

        intent_feature_config = feature_config['intent']
        for fid in intent_feature_config:
            if intent_feature_config[fid]['type'] == 'sparse':
                intent_feature_config[fid] = proc_sparse_feature_config(fid, intent_feature_config[fid], intent_feature_dict)

    return user_feature_config, intent_feature_config
