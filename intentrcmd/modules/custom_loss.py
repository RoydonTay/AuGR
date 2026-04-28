import torch
from torch import nn
import torch.nn.functional as F


def softmax_loss(logits, labels, scale=20, reduction='mean'):
    """Softmax cross entropy loss function"""
    EPS = 1e-8
    labels = labels / (labels.sum(axis=1, keepdim=True) + EPS)

    logits = logits * scale
    log_p = F.log_softmax(logits, dim=1)
    loss = -(labels * log_p).sum(axis=1)

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'none':
        return loss
    return loss.mean()

def softmax_loss_with_weights(logits, labels, weights, scale=20, reduction='mean'):
    """Softmax cross-entropy loss with per-label confidence weights (only for label==1)"""
    EPS = 1e-8
    # Step 1: 只保留 label==1 的 weights，其他置为 0
    effective_weights = (weights * 0.8 + 0.2) * labels  # shape: [B, N]

    # Step 2: 行归一化 effective_weights，作为 soft labels
    soft_labels = effective_weights / (effective_weights.sum(dim=1, keepdim=True) + EPS)

    # Step 3: softmax + log
    logits = logits * scale
    log_probs = F.log_softmax(logits, dim=1)

    # Step 4: 加权 loss
    loss = -(soft_labels * log_probs).sum(dim=1)  # shape: [B]

    # Step 5: reduction
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'none':
        return loss
    return loss.mean()

def listnet_loss(logits, labels, scale=20, reduction='mean'):
    """ListNet loss with regularization"""
    EPS = 1e-8

    P_label = F.softmax(labels * scale, dim=1)
    P_pred = F.softmax(logits * scale, dim=1)
    loss = - (P_label * torch.log(P_pred + EPS)).sum(dim=1)

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'none':
        return loss
    return loss.mean()

def listnet_loss_with_weights(logits, labels, weights, scale=20, reduction='mean'):
    """ListNet loss with per-click weights"""
    EPS = 1e-8

    # Step 1: 只保留 label == 1 的位置的 weights
    effective_weights = (weights * 0.5 + 0.5) * labels  # [B, N]

    # Step 2: soft label = 归一化 effective weights
    P_label_weighted = effective_weights / (effective_weights.sum(dim=1, keepdim=True) + EPS)

    # Step 3: softmax logits -> predicted distribution
    P_pred = F.softmax(logits * scale, dim=1)

    # Step 4: KL loss (P_label_weighted is the target distribution)
    loss = - (P_label_weighted * torch.log(P_pred + EPS)).sum(dim=1)

    # Step 5: reduction
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'none':
        return loss
    return loss.mean()


def impression_focused_listnet_loss(logits, labels, impressions, scale=20, reduction='mean'):
    """ListNet loss computed only on impressed items per sample.

    For each sample, select only the items with impressions > 0,
    then compute a standard ListNet (cross-entropy over softmax) on that subset.
    Samples with fewer than 2 impressed items or no positive label in the
    impression subset are skipped (loss = 0).

    Args:
        logits: (B, N) predicted scores
        labels: (B, N) binary click labels
        impressions: (B, N) impression indicators (>0 means impressed)
        scale: temperature for softmax
        reduction: 'mean' | 'sum' | 'none'
    Returns:
        loss scalar or (B,) tensor
    """
    EPS = 1e-8
    B, N = logits.shape
    imp_mask = (impressions > 0)  # (B, N)

    losses = torch.zeros(B, device=logits.device)
    for i in range(B):
        mask_i = imp_mask[i]  # (N,)
        if mask_i.sum() < 2:
            continue
        labels_i = labels[i][mask_i]  # (n_imp,)
        if labels_i.sum() == 0:
            continue
        logits_i = logits[i][mask_i]  # (n_imp,)

        # Target distribution: uniform over clicked items within impression set
        P_label = labels_i / (labels_i.sum() + EPS)
        # Predicted distribution
        P_pred = F.softmax(logits_i * scale, dim=0)
        # Cross-entropy
        losses[i] = -(P_label * torch.log(P_pred + EPS)).sum()

    if reduction == 'mean':
        # Mean over samples that have valid impression loss
        valid = (losses > 0).sum()
        return losses.sum() / (valid + EPS)
    elif reduction == 'sum':
        return losses.sum()
    elif reduction == 'none':
        return losses
    return losses.sum() / ((losses > 0).sum() + EPS)


def listnet_ctr_loss_with_weights(logits, labels, weights, scale=20, reduction='mean'):
    """CTR-friendly ListNet loss with weights"""
    EPS = 1e-8

    # Step 1: Compute effective weights only for label==1
    effective_weights = (weights * 0.5 + 0.5) * labels  # [B, N]

    # Step 2: Construct label distribution (target)
    P_label = effective_weights / (effective_weights.sum(dim=1, keepdim=True) + EPS)  # [B, N]

    # Step 3: Compute predicted CTR probabilities (after sigmoid)
    ctr_pred = torch.sigmoid(logits * scale)  # [B, N], values in (0, 1)

    # Step 4: Normalize predicted CTRs to form a distribution
    P_pred = ctr_pred / (ctr_pred.sum(dim=1, keepdim=True) + EPS)  # [B, N]

    # Step 5: KL-divergence loss (P_label is target)
    loss = - (P_label * torch.log(P_pred + EPS)).sum(dim=1)  # [B]

    # Step 6: Reduction
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'none':
        return loss
    return loss.mean()


def softmax_with_hardneg_loss(scores, labels, scale=20, k=10):
    """Softmax cross entropy loss on hard negative intents"""
    EPS = 1e-8
    NEG_INF = -1e8

    loss = torch.zeros(scores.size(0))
    for i in range(scores.size(0)):
        sample_scores = scores[i]
        sample_labels = labels[i]

        # Find top k highest scored negative cases, which are hard negatives
        neg_indices = torch.nonzero(sample_labels == 0).squeeze(axis=1)
        _, topk_neg_indices = torch.topk(sample_scores[neg_indices], k=k)
        topk_neg_indices = [neg_indices[i].item() for i in topk_neg_indices]

        # Keep all positives and only top hard negatives
        valid = (sample_labels != 0).float()
        valid[topk_neg_indices] = 1

        sample_logits = sample_scores * scale
        sample_logits = sample_logits + (1 - valid) * NEG_INF

        sample_labels = sample_labels / (sample_labels.sum() + EPS)
        log_p = F.log_softmax(sample_logits, dim=0)
        sample_loss = -(sample_labels * log_p).sum()
        loss[i] = sample_loss
    return loss.mean()


def softmax_with_impressions_loss(scores, labels, impressions, scale=20):
    """Softmax cross entropy loss function on only impression intents"""
    EPS = 1e-8
    NEG_INF = -1e8
    valid = ((impressions+labels) != 0).float()
    labels = labels * valid
    labels = labels / (labels.sum(axis=1, keepdim=True) + EPS)

    logits = scores * scale
    logits = logits + (1 - valid) * NEG_INF
    log_p = F.log_softmax(logits, dim=1)
    loss = -(labels * log_p).sum(axis=1).mean()
    return loss


def softmax_with_impressions_sampling_loss(scores, labels, impressions, scale=20, k=10):
    """Softmax cross entropy loss function on impression intents and randomly sampled intents"""
    EPS = 1e-8
    NEG_INF = -1e8
    valid = ((impressions+labels) != 0).float()

    # Random select k intents as valid samples
    for i in range(valid.size(0)):
        indices = torch.randperm(valid.size(1))[:k]
        valid[i, indices] = 1

    labels = labels * valid
    labels = labels / (labels.sum(axis=1, keepdim=True) + EPS)

    logits = scores * scale
    logits = logits + (1 - valid) * NEG_INF
    log_p = F.log_softmax(logits, dim=1)
    loss = -(labels * log_p).sum(axis=1).mean()
    return loss


def multi_softmax_loss(scores, labels, scale=20):
    """Multiple softmax cross entropy loss function"""
    NEG_INF = -1e8

    loss = torch.zeros(scores.size(0))
    for i in range(scores.size(0)):
        sample_loss = 0.0
        sample_logits = scores[i] * scale
        sample_labels = labels[i]

        pos_indices = torch.nonzero(sample_labels == 1).squeeze(axis=1)
        for pos_index in pos_indices:
            _logits = sample_logits.clone()
            _labels = sample_labels.clone()

            _logits[pos_indices] = NEG_INF
            _logits[pos_index] = sample_logits[pos_index]

            _labels[pos_indices] = 0
            _labels[pos_index] = 1

            _log_p = F.log_softmax(_logits, dim=0)
            sample_loss += -(_labels * _log_p).sum()
        loss[i] = sample_loss
    return loss.mean()


def bce_loss(scores, labels):
    """Binary cross entropy loss"""
    scores = (scores + 1.0) / 2
    loss = nn.BCEWithLogitsLoss()(scores, labels)
    return loss


def bce_with_sigmoid_loss(scores, labels, scale=5):
    """Binary cross entropy loss with sigmoid"""
    scores = torch.sigmoid(scores * scale)
    loss = nn.BCEWithLogitsLoss()(scores, labels)
    return loss


# Refer to the following paper for unsupervised contrastive representation losses:
# https://arxiv.org/pdf/2005.10242
def lalign(x, y, alpha=2):
    loss = (x - y).norm(dim=1).pow(alpha).mean()
    return loss

def lunif(x, t=2):
    sq_pdist = torch.pdist(x, p=2).pow(2)
    loss = sq_pdist.mul(-t).exp().mean().log()
    return loss

def embedding_unsup_loss(embeddings_u, embeddings_i, labels):
    # According to paper, total loss = lalign(x, y) + lam * (lunif(x) + lunif(y)) / 2,
    # But here we should first adapt embedding shapes.

    uniform_loss_u = lunif(embeddings_u)
    uniform_loss_i = lunif(embeddings_i)

    expanded_u = embeddings_u.repeat_interleave(labels.sum(dim=1).to(torch.int), dim=0)
    expanded_i = embeddings_i[torch.nonzero(labels, as_tuple=False)[:, 1]]
    align_loss = lalign(expanded_u, expanded_i)

    #print('embedding_unsup_loss:', uniform_loss_u, uniform_loss_i, align_loss)
    loss = align_loss + (uniform_loss_u + uniform_loss_i) / 2
    return loss


def calc_aux_loss_weight(global_step, max_steps):
    decay_start = int(0.1 * max_steps)
    decay_end = int(0.5 * max_steps)
    if global_step <= decay_start:
        return 1.0
    elif global_step <= decay_end:
        return 1.0 - (global_step - decay_start) / (decay_end - decay_start)
    return 0.0


# =========================================================================
# ApproxNDCG Loss  (Qin et al., 2010 — "A General Approximation Framework
# for Direct Optimization of Information Retrieval Measures")
# =========================================================================

def _approx_ndcg_loss(logits, labels, weights=None, scale=20, temperature=1.0, reduction='mean'):
    """Differentiable approximation to NDCG for listwise ranking.

    Uses a smooth approximation of the sorting operator: the rank of item j
    is approximated as  r_j ≈ 1 + Σ_{k≠j} sigmoid((s_k - s_j) / τ)
    where τ = temperature controls the smoothness.

    Args:
        logits: (B, N) predicted scores
        labels: (B, N) relevance labels (binary or graded)
        weights: (B, N) optional per-position confidence weights (for debiasing).
                 Applied multiplicatively to relevance gains.
        scale: float, pre-multiplier on logits before computing approx ranks
        temperature: float, smoothness of the sigmoid approximation (lower = sharper)
        reduction: 'mean' | 'sum' | 'none'

    Returns:
        loss: scalar or (B,) per-sample losses
    """
    EPS = 1e-10

    # Scale logits for sharper rank approximation
    scores = logits * scale  # (B, N)

    # Approximate ranks via pairwise sigmoid
    # diff[b, i, j] = scores[b, j] - scores[b, i]
    diff = scores.unsqueeze(1) - scores.unsqueeze(2)  # (B, N, N)
    approx_indicator = torch.sigmoid(diff / temperature)  # (B, N, N)

    # Rank of position i ≈ 1 + sum over j≠i of sigmoid((s_j - s_i)/τ)
    # approx_indicator[:, i, j] = P(item j ranked above item i)
    # We want rank of i, so sum over dim=2 (all j), exclude self
    eye = torch.eye(approx_indicator.size(1), device=logits.device).unsqueeze(0)
    approx_ranks = 1.0 + (approx_indicator * (1.0 - eye)).sum(dim=2)  # (B, N)

    # DCG gains: (2^rel - 1) / log2(1 + rank)
    gains = labels  # For binary labels, 2^1 - 1 = 1, 2^0 - 1 = 0
    if weights is not None:
        # Apply confidence weights: modulate the gain of each positive
        effective_weights = (weights * 0.5 + 0.5) * labels
        gains = effective_weights

    discount = torch.log2(1.0 + approx_ranks)  # (B, N)
    approx_dcg = (gains / discount).sum(dim=1)  # (B,)

    # Ideal DCG: sort labels descending, compute DCG at ideal positions
    sorted_labels, _ = labels.sort(dim=1, descending=True)
    if weights is not None:
        # For ideal DCG, use the best possible gains (sorted by effective weight)
        effective_gains_ideal = (weights * 0.5 + 0.5) * labels
        sorted_gains, _ = effective_gains_ideal.sort(dim=1, descending=True)
    else:
        sorted_gains = sorted_labels
    ideal_ranks = torch.arange(1, labels.size(1) + 1, device=logits.device, dtype=logits.dtype)
    ideal_discount = torch.log2(1.0 + ideal_ranks).unsqueeze(0)  # (1, N)
    ideal_dcg = (sorted_gains / ideal_discount).sum(dim=1)  # (B,)

    # ApproxNDCG = approx_dcg / ideal_dcg
    approx_ndcg = approx_dcg / (ideal_dcg + EPS)  # (B,)

    # Loss = 1 - ApproxNDCG (minimize)
    loss = 1.0 - approx_ndcg

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'none':
        return loss
    return loss.mean()
