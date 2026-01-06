from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def _gaussian_kernel(
    source: torch.Tensor,
    target: torch.Tensor,
    kernel_mul: float = 2.0,
    kernel_num: int = 5,
    fix_sigma: Optional[float] = None,
) -> torch.Tensor:
    n_samples = int(source.size(0) + target.size(0))
    total = torch.cat([source, target], dim=0)

    total0 = total.unsqueeze(0).expand(total.size(0), -1, -1)
    total1 = total.unsqueeze(1).expand(-1, total.size(0), -1)
    l2_distance = ((total0 - total1) ** 2).sum(2)

    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(l2_distance.data) / (n_samples ** 2 - n_samples + 1e-8)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]

    kernel_val = torch.zeros_like(l2_distance)
    for bandwidth_temp in bandwidth_list:
        kernel_val += torch.exp(-l2_distance / (bandwidth_temp + 1e-8))
    return kernel_val


def gdd_loss(
    source: torch.Tensor,
    target: torch.Tensor,
    kernel_mul: float = 2.0,
    kernel_num: int = 5,
    fix_sigma: Optional[float] = None,
) -> torch.Tensor:
    """Global Domain Discrepancy loss.

    Corresponds to the global domain alignment term in the paper, estimated via
    multi-scale Gaussian kernels over source and target feature distributions.
    """
    n = source.size(0)
    m = target.size(0)
    kernels = _gaussian_kernel(source, target, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    xx = kernels[:n, :n]
    yy = kernels[n:, n:]
    xy = kernels[:n, n:]
    yx = kernels[n:, :n]

    xx = torch.div(xx, n * n + 1e-8).sum(dim=1, keepdim=True)
    yy = torch.div(yy, m * m + 1e-8).sum(dim=1, keepdim=True)
    xy = torch.div(xy, -n * m + 1e-8).sum(dim=1, keepdim=True)
    yx = torch.div(yx, -m * n + 1e-8).sum(dim=1, keepdim=True)
    loss = (xx + xy).sum() + (yx + yy).sum()
    return loss


def _class_indicator(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    eye = torch.eye(num_classes, device=labels.device)
    return eye[labels.long()]


def lsd_loss(
    source: torch.Tensor,
    target: torch.Tensor,
    source_label: torch.Tensor,
    target_label: torch.Tensor,
    sample_weight: Optional[torch.Tensor] = None,
    num_classes: int = 2,
    kernel_mul: float = 2.0,
    kernel_num: int = 5,
    fix_sigma: Optional[float] = None,
) -> torch.Tensor:
    """Local Sub-domain Discrepancy (LSD) with optional confidence weighting."""
    n = source.size(0)
    m = target.size(0)
    kernels = _gaussian_kernel(source, target, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    xx = kernels[:n, :n]
    yy = kernels[n:, n:]
    xy = kernels[:n, n:]
    yx = kernels[n:, :n]

    src_onehot = _class_indicator(source_label, num_classes)
    tgt_onehot = _class_indicator(target_label, num_classes)
    if sample_weight is not None:
        sample_weight = sample_weight.unsqueeze(-1)
        tgt_onehot = tgt_onehot * sample_weight

    total_cls = torch.cat([src_onehot, tgt_onehot], dim=0)
    coef = total_cls @ total_cls.t()

    ss = (coef[:n, :n] * xx)
    tt = (coef[n:, n:] * yy)
    st = (coef[:n, n:] * xy)
    ts = (coef[n:, :n] * yx)

    ss_norm = ss.sum() / (ss.numel() + 1e-8)
    tt_norm = tt.sum() / (tt.numel() + 1e-8)
    st_norm = st.sum() / (st.numel() + 1e-8)
    ts_norm = ts.sum() / (ts.numel() + 1e-8)
    return ss_norm + tt_norm - st_norm - ts_norm


def reliability_weighted_ce(
    logits: torch.Tensor,
    pseudo_labels: torch.Tensor,
    reliability: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    ce = F.cross_entropy(logits, pseudo_labels, reduction="none")
    weight = reliability.detach()
    loss = ce * weight
    if reduction == "mean":
        denom = weight.sum().clamp(min=1e-6)
        return loss.sum() / denom
    if reduction == "sum":
        return loss.sum()
    return loss


def cosine_prototype_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    prototypes: torch.Tensor,
    temperature: float = 0.5,
) -> torch.Tensor:
    """ProtoAlign loss (Equation L_proto in MuseFuse).

    L_proto = -log( exp(sim(h_f, P_y)/τ) / Σ_c exp(sim(h_f, P_c)/τ) )

    Args:
        features: Fusion (or branch) embeddings h_f ∈ R^{B×D}.
        labels: Ground-truth emotion labels y.
        prototypes: Prototype bank P ∈ R^{C×D}.
        temperature: τ_proto (default: 0.5).
    """
    logits = torch.matmul(F.normalize(features, dim=1), F.normalize(prototypes, dim=1).t())
    logits = logits / temperature
    return F.cross_entropy(logits, labels)


def distill_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """TriDistill KL divergence between student and teacher logits.

    Used for symmetric KL distillation in TriDistill (Eqs. L_FM, L_MF, L_FC, L_CF),
    with temperature scaling to stabilise the teacher distribution.
    """
    t = temperature
    log_p = F.log_softmax(student_logits / t, dim=1)
    q = F.softmax(teacher_logits / t, dim=1)
    return F.kl_div(log_p, q, reduction="batchmean") * (t * t)

