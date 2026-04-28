import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from transformers import Trainer as baseTrainer

from ..metrics import safe_auc,safe_ndcg


def compute_auc_metrics(eval_pred):
    return {"auc": roc_auc_score(y_true=eval_pred[1], y_score=eval_pred[0])}


"""
metrics_dict = {
    "eval_auc": {
        "metrics": safe_auc,
        "k": 0,
        "ignore_ties": False
    },
    "eval_gauc": {
        "metrics": safe_auc,
        "k": 0,
        "ignore_ties": False
    },
    "eval_recall_3": {
        "metrics": safe_recall,
        "k": 0,
        "ignore_ties": False
    },
}
"""

def compute_listwise_metrics(metrics_dict, mtl=False):
    def compute_metrics(eval_pred):
        if mtl:
            predictions, la = eval_pred
            predictions = predictions[0]
            labels, is_impression = la[0], la[-1]
        else:
            predictions, la = eval_pred
            labels, is_impression = la

        # make sure numpy
        predictions = np.asarray(predictions)
        labels = np.asarray(labels)
        is_impression = np.asarray(is_impression)

        # ---------- metrics helper ----------
        def reduce_listwise(metric_fn, k=0, ignore_ties=False, use_impression=False):
            vals = []
            for i in range(labels.shape[0]):
                if use_impression:
                    mask = is_impression[i].astype(bool)
                    if not mask.any():
                        continue
                    y_true = labels[i][mask]
                    y_score = predictions[i][mask]
                else:
                    y_true = labels[i]
                    y_score = predictions[i]

                vals.append(
                    metric_fn(
                        y_true=y_true,
                        y_score=y_score,
                        k=k,
                        ignore_ties=ignore_ties,
                    )
                )
            return float(np.nanmean(vals)) if vals else float("nan")

        # ---------- compute ----------
        result = {}
        for name, cfg in metrics_dict.items():
            if name == "eval_auc":
                result[name] = safe_auc(
                    y_true=labels.ravel(),
                    y_score=predictions.ravel(),
                )
            else:
                result[name] = reduce_listwise(
                    cfg["metrics"],
                    k=cfg.get("k", 0),
                    ignore_ties=cfg.get("ignore_ties", False),
                    use_impression=(cfg["metrics"] == safe_ndcg),
                )

        del predictions, labels, is_impression
        return result

    return compute_metrics


def compute_pointwise_metrics(metrics_dict, group_=False):
    def compute_metrics(eval_pred):
        def metrics(
            eval_pred, metrics_func, k=0, ignore_ties=False, group=None, filter_key=None
        ):
            predictions, label_mix = eval_pred
            groups = None
            if group_ or (filter_key is not None):
                labels, groups = label_mix
            else:
                labels = label_mix
            if filter_key is not None and filter_key != -1:
                predictions = predictions[groups == filter_key]
                labels = labels[groups == filter_key]
            if group is None or not group or groups is None:
                return metrics_func(
                    y_true=labels.reshape(
                        len(labels),
                    ),
                    y_score=predictions.reshape(
                        len(labels),
                    ),
                    k=k,
                    ignore_ties=False,
                )
            eval = pd.DataFrame(
                {
                    "groups": list(
                        map(
                            str,
                            groups.reshape(
                                len(labels),
                            ),
                        )
                    ),
                    "score": predictions.reshape(
                        len(labels),
                    ),
                    "label": labels.reshape(
                        len(labels),
                    ),
                }
            )
            gind = np.nanmean(
                eval.groupby(by="groups").apply(
                    lambda x: metrics_func(
                        y_true=x["label"].values,
                        y_score=x["score"].values,
                        k=k,
                        ignore_ties=ignore_ties,
                    )
                )
            )
            return gind

        result = {}
        for k, v in metrics_dict.items():
            result[k] = metrics(
                eval_pred,
                v["metrics"],
                k=v.get("k", 0),
                ignore_ties=v.get("ignore_ties", False),
                group=v.get("group", False),
                filter_key=v.get("filter_key"),
            )
        return result

    return compute_metrics


class Trainer(baseTrainer):
    def dump_best_eval_metric(self, metrics, model, metrics_dict):
        best = self.state.best_metric
        eval_best_metrics = [
            d
            for d in self.state.log_history
            if f"eval_{metrics}" in d and d[f"eval_{metrics}"] == best
        ][0]
        result = {"model": [model]}
        result.update(
            {k: [v] for k, v in eval_best_metrics.items() if k in metrics_dict}
        )
        return result

    
def compute_pointwise_group_metrics(metrics_dict, only_intent_prediction_click=True, only_intent_prediction_imp=False):
    def compute_metrics(eval_pred):
        predictions, is_click, is_impression = eval_pred
        if only_intent_prediction_imp:
            predictions = [predictions[i] for i in range(len(is_impression)) if is_impression[i] == 1]
            is_click = [is_click[i] for i in range(len(is_impression)) if is_impression[i] == 1]
            is_impression = [is_impression[i] for i in range(len(is_impression)) if is_impression[i] == 1]

        if only_intent_prediction_click:
            is_click = [int(is_click[i] & is_impression[i]) for i in range(len(is_click))]

        def metrics(metrics_func, k=0, ignore_ties=False):
            return metrics_func(
                y_true=is_click,
                y_score=predictions,
                k=k,
                ignore_ties=ignore_ties,
            )
        result = {}
        for k, v in metrics_dict.items():
            result[k] = metrics(
                v["metrics"],
                k=v.get("k", 0),
                ignore_ties=v.get("ignore_ties", False),
            )
        return result

    return compute_metrics
