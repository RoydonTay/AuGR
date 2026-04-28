import os
from typing import Any, Dict, List, Optional, Text

import numpy as np
import onnxruntime
import torch
from torch import Tensor, nn


def get_shape(x: Tensor):
    shape = list(x.size())
    return shape


def get_device():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return device


def load_reusable_weights_from_old_model(new_model, old_model_bin_path, exclude_keys=[]):
    """Load re-usable weights from pre-trained old model.

    Args:
        new_model: pytorch model
        old_model_bin_path: string, path to pretrained model bin file.
        exclude_keys: []string, list of weight keys to be excluded from loading.
    Returns:
        reusable_state_dict
    """
    new_model_dict = new_model.state_dict()
    old_model_dict = torch.load(old_model_bin_path, map_location=torch.device('cpu'))

    # Fiter out unneccessary keys
    reusable_state_dict = {}
    for k in old_model_dict:
        # 0. Skip exclude key.
        if k in exclude_keys:
            print('skip exclude weight, k:', k)
            continue
        # 1. Skip different key.
        if k not in new_model_dict:
            print('skip non-exist weight, k:', k)
            continue
        # 2. Re-use already trained embeddings.
        # Attention that this only works when feature dicts are consistent so that vocab are incrementally updated.
        if 'embedding_dict' in k:
            new_vocab_size = new_model_dict[k].shape[0]
            old_vocab_size = old_model_dict[k].shape[0]
            assert new_vocab_size >= old_vocab_size
            reusable_state_dict[k] = torch.cat((old_model_dict[k], new_model_dict[k][old_vocab_size:, :]), dim=0)
            continue
        # 3. Skip weights with different shape (exclude embedding).
        if old_model_dict[k].shape != new_model_dict[k].shape:
            print('skip diff-shape weight, k:', k)
            continue
        # 4. Set same weights to same key with same shape.
        reusable_state_dict[k] = old_model_dict[k]

    print(f'new model dict size: {len(new_model_dict)}, old model dict size: {len(old_model_dict)}, reusable dict size: {len(reusable_state_dict)}')
    return reusable_state_dict


def export_onnx(
    model: nn.Module,
    input_names: List[Text],
    output_names: List[Text],
    output_file: Text,
    input_tensor_batch_sample: Dict[Text, Tensor],
    dynamic_axes: Optional[Dict[Text, Any]] = None,
    opset_version: int = 11,
) -> None:
    model.eval()
    device = get_device()

    input_dict = {
        k: v.to(device)[:1]
        for k, v in input_tensor_batch_sample.items()
        if k in input_names
    }
    # just to make sure the order is the same
    input_names = list(input_dict.keys())

    class OnionWrapper(torch.nn.Module):
        def __init__(self, model, feature_keys):
            super().__init__()
            self.model = model
            self.feature_keys = feature_keys

        def forward(self, *inputs):
            kwargs = {k: v for k, v in zip(self.feature_keys, inputs)}
            return self.model(**kwargs)
    
    wrapped_model = OnionWrapper(model, input_names)
    tuple_args = tuple(input_dict.values())

    torch.onnx.export(
        model=wrapped_model,
        args=tuple_args,
        f=output_file,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=False,
        training=torch.onnx.TrainingMode.EVAL,
    )

    # verify prediction results
    model.eval()
    r1 = model(**input_dict)
    if torch.cuda.is_available():
        r1 = [p.detach().cpu().numpy() for p in r1]
        input_feed = {k: v.detach().cpu().numpy() for k, v in input_dict.items()}
    else:
        r1 = [p.detach().numpy() for p in r1]
        input_feed = {k: v.detach().numpy() for k, v in input_dict.items()}

    sess_options = onnxruntime.SessionOptions()
    sess_options.intra_op_num_threads = 1
    sess_options.log_severity_level = 3
    ort_session = onnxruntime.InferenceSession(output_file, sess_options=sess_options)

    for input in ort_session.get_inputs():
        print(
            f"{input.name}|onnx: dtype={input.type}, shape={input.shape}, |input: dtype={input_feed[input.name].dtype}, shape={input_feed[input.name].shape}"
        )

    print("*" * 20)
    for output in ort_session.get_outputs():
        print(
            f"{output.name}|onnx: dtype={output.type}, shape={output.shape}, |output: dtype={r1[0].dtype}, shape={r1[0].shape}"
        )

    r2 = ort_session.run(["output"], input_feed)

    model.train()

    if np.allclose(r1, r2, atol=1e-5):
        print("ONNX prediction is quite close to trained model prediction")
    else:
        print(f"Model prediction: {r1}")
        print(f"ONNX prediction: {r2}")
        raise AssertionError(
            "ONNX prediction is different from trained model prediction"
        )


def load_onnx(
    model_file: Text, intra_op_num_threads: int = 1, log_severity_level: int = 3
) -> onnxruntime.InferenceSession:
    model = None
    if model_file and os.path.exists(model_file):
        sess_options = onnxruntime.SessionOptions()
        # due to k8s NUMA arch, set threads = 1 or 2
        sess_options.intra_op_num_threads = intra_op_num_threads
        # disable warnings: output shape verification
        sess_options.log_severity_level = log_severity_level
        model = onnxruntime.InferenceSession(model_file, sess_options=sess_options)
    return model
