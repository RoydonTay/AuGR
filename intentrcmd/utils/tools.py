import os
import pandas as pd
import torch
from tqdm import tqdm
from datetime import datetime, timedelta

from .hf_utils import compute_listwise_metrics
from .torch_utils import export_onnx


def gen_date_list(start_date, end_date):
    date_list = []
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")

    current_date = start_date
    while current_date <= end_date:
        date_list.append(current_date.strftime("%Y-%m-%d"))
        current_date += timedelta(days=1)
    return date_list


def calc_online_metrics(test_loader, metrics_dict, grass_region):
    """Calculate online metrics for test data."""
    clicks = []
    impressions = []
    scores = []

    for data in test_loader:
        clicks.append(data['labels_intents'].tolist())
        impressions.append(data['impressions'].tolist())
        scores.append(data['scores'].tolist())
    # Flatten the lists
    clicks = [item for sublist in clicks for item in sublist]
    impressions = [item for sublist in impressions for item in sublist]
    scores = [item for sublist in scores for item in sublist]

    online_metrics = compute_listwise_metrics(metrics_dict)([
        scores,
        [clicks, impressions]
    ])
    metrics_result = {"region": [grass_region], "model": ["online"]}
    metrics_result.update({k: [v] for k, v in online_metrics.items()})
    metrics_df = pd.DataFrame(metrics_result)
    return metrics_df


def export_task_model(task_model, intent_feature_config, test_loader, intent_batch, model_output_path):
    """Export task model to onnx format"""
    for t in test_loader:
        break

    # convert t to single sample
    data_sample = {}
    for k, v in t.items():
        v = v[0:1, :]  # reduce batch size to 1.
        data_sample[k] = v.reshape(-1)
    for k, v in intent_batch.items():
        v = v[0:1, :]  # reduce batch size to 1.
        data_sample[k] = v.reshape(-1)

    # pop out unused keys for model
    for k in list(data_sample.keys()):
        if not k.startswith('fid'):
            data_sample.pop(k)

    data_sample = {k: v.unsqueeze(0) for k, v in data_sample.items()}

    input_names = list(data_sample.keys())
    dynamic_axes = {k: {0: "batch_size"} for k in input_names}
    dynamic_axes.update({"output": {0: "batch_size"}})
    print(dynamic_axes)

    output_file = os.path.join(model_output_path, 'model.onnx')
    export_onnx(
        model=task_model,
        input_names=input_names,
        output_names=["output"],
        output_file=output_file,
        input_tensor_batch_sample=data_sample,
        dynamic_axes=dynamic_axes,
    )
    print('ONNX exported to:', output_file)