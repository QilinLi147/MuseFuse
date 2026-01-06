import argparse
import json
import os
from datetime import datetime
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from lib import (
    get_default_config,
    MusicEmotionDataset,
    build_dataloaders,
    MuseFuseModel,
    gdd_loss,
    lsd_loss,
    reliability_weighted_ce,
    cosine_prototype_loss,
    distill_kl,
    set_seed,
    MetricsTracker,
    GradientScaler,
    compute_dynamic_tradeoff,
    get_dataset_stats,
)
from lib.vis_data_collector import VisualizationDataCollector


def simple_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(np.int64)
    scores = scores.astype(np.float64)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    all_scores = np.concatenate([pos, neg])
    ranks = all_scores.argsort().argsort() + 1
    r_pos = ranks[: len(pos)].sum()
    n_pos = len(pos)
    n_neg = len(neg)
    U = r_pos - n_pos * (n_pos + 1) / 2.0
    auc = U / (n_pos * n_neg)
    return float(auc)


def precision_recall_f1(preds: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    tp = np.sum((preds == 1) & (labels == 1))
    fp = np.sum((preds == 1) & (labels == 0))
    fn = np.sum((preds == 0) & (labels == 1))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def evaluate(model: nn.Module, loader, device: torch.device) -> Dict[str, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total = 0
    correct = 0
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for batch in loader:
            mel = batch["mel"].to(device)
            coch = batch["coch"].to(device)
            labels = batch["label"].to(device)
            outputs = model(mel, coch)
            logits = outputs["fusion_logits"]
            loss = criterion(logits, labels)
            total_loss += float(loss.item()) * mel.size(0)
            total += mel.size(0)
            probs = F.softmax(logits, dim=1)[:, 1]
            preds = logits.argmax(dim=1)
            correct += int((preds == labels).sum().item())
            all_labels.append(labels.detach().cpu().numpy())
            all_probs.append(probs.detach().cpu().numpy())
    model.train()
    y_true = np.concatenate(all_labels) if all_labels else np.zeros((0,))
    y_prob = np.concatenate(all_probs) if all_probs else np.zeros((0,))
    y_pred = (y_prob >= 0.5).astype(np.int64)
    acc = correct / max(1, total)
    auc = simple_auc(y_true, y_prob) if y_true.size > 0 else 0.5
    prf = precision_recall_f1(y_pred, y_true) if y_true.size > 0 else {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    return {
        "loss": total_loss / max(1, total),
        "acc": acc,
        "auc": auc,
        **prf,
    }


def train(args: argparse.Namespace) -> None:
    cfg = get_default_config(**vars(args))
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = MusicEmotionDataset(cfg.dataset, label_mode=cfg.label_mode, data_root=cfg.data_root)
    train_loader, val_loader = build_dataloaders(
        dataset,
        batch_size=cfg.batch_size,
        train_ratio=1.0 - cfg.val_ratio,
        seed=cfg.seed,
        num_workers=cfg.num_workers,
    )

    stats = get_dataset_stats(dataset)

    sample_item = dataset[0]
    input_dim = sample_item["mel"].size(-1)
    mel_dim = dataset.mel_nodes * dataset.feat_dim
    coch_dim = dataset.coch_nodes * dataset.feat_dim
    model = MuseFuseModel(
        mel_dim=mel_dim,
        coch_dim=coch_dim,
        hidden_mel=cfg.mel_hidden_dim,
        hidden_coch=cfg.coch_hidden_dim,
        fusion_hidden=cfg.fusion_hidden_dim,
        num_classes=cfg.num_classes,
    ).to(device)

    optimizer = (
        torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        if cfg.optimizer == "adam"
        else torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)
    )
    scheduler = None
    if cfg.cosine_scheduler:
        total_steps = max(1, len(train_loader) * cfg.epochs)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    scaler = GradientScaler(enabled=cfg.mixed_precision)
    criterion = nn.CrossEntropyLoss()
    metrics = MetricsTracker()
    os.makedirs(cfg.log_path, exist_ok=True)
    log_file = os.path.join(cfg.log_path, cfg.log_file)
    with open(log_file, "a") as f:
        f.write(f"Training configuration: {json.dumps(cfg.to_dict(), ensure_ascii=False)}\n")
        f.write(f"Dataset statistics: {json.dumps(stats, ensure_ascii=False)}\n")
        f.write(f"Device used: {'cuda' if device.type == 'cuda' else 'cpu'}\n")

    best_acc = 0.0
    os.makedirs(cfg.out_dir, exist_ok=True)
    global_step = 0
    
    vis_collector = None
    collect_vis_data = getattr(args, 'collect_vis_data', False)
    if collect_vis_data:
        vis_data_dir = os.path.join(cfg.log_path, 'vis_data')
        vis_collector = VisualizationDataCollector(vis_data_dir, collect_interval=5)
        print(f"Visualisation data collector initialised, output directory: {vis_data_dir}")

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        running_src = 0.0
        running_pseudo = 0.0
        running_domain = 0.0
        running_consistency = 0.0
        running_proto = 0.0
        running_distill = 0.0
        running_relia = 0.0
        correct = 0
        seen = 0

        for step, batch in enumerate(train_loader, start=1):
            mel = batch["mel"].to(device)
            coch = batch["coch"].to(device)
            labels = batch["label"].to(device)

            with scaler.autocast():
                outputs = model(mel, coch)
                mel_logits = outputs["mel_logits"]
                coch_logits = outputs["coch_logits"]
                fusion_logits = outputs["fusion_logits"]

                src_loss = criterion(mel_logits, labels) + 0.5 * criterion(fusion_logits, labels)

                with torch.no_grad():
                    coch_probs = F.softmax(coch_logits, dim=1)
                    confidence, pseudo_labels = coch_probs.max(dim=1)
                    entropy = -(coch_probs * (coch_probs + 1e-8).log()).sum(dim=1)
                schedule_ratio = 1.0 - (epoch - 1) / max(1, cfg.epochs)
                threshold = max(cfg.pseudo_threshold_min, cfg.pseudo_threshold * schedule_ratio)
                confident_mask = confidence >= threshold
                if cfg.use_relia_pseudo:
                    if cfg.relia_reliability == "entropy":
                        norm = torch.log(torch.tensor(float(cfg.num_classes), device=coch_logits.device))
                        reliability = 1.0 - entropy / norm
                    else:
                        reliability = confidence
                    reliability = reliability.detach().clamp(min=1e-6)
                    if cfg.relia_soft_mask:
                        pseudo_loss = reliability_weighted_ce(coch_logits, pseudo_labels, reliability)
                    elif confident_mask.sum() > 0:
                        pseudo_loss = reliability_weighted_ce(
                            coch_logits[confident_mask],
                            pseudo_labels[confident_mask],
                            reliability[confident_mask],
                        )
                    else:
                        pseudo_loss = coch_logits.new_tensor(0.0)
                elif confident_mask.sum() > 0:
                    pseudo_loss = criterion(coch_logits[confident_mask], pseudo_labels[confident_mask])
                else:
                    pseudo_loss = coch_logits.new_tensor(0.0)

                if confident_mask.sum() > 0:
                    lsd_val = lsd_loss(
                        outputs["mel_feat"],
                        outputs["coch_feat"][confident_mask],
                        labels,
                        pseudo_labels[confident_mask],
                        confidence[confident_mask],
                        num_classes=cfg.num_classes,
                    )
                else:
                    lsd_val = coch_logits.new_tensor(0.0)

                gdd_val = gdd_loss(outputs["mel_feat"], outputs["coch_feat"])
                consistency = F.mse_loss(outputs["mel_feat"], outputs["coch_feat"])

                alpha = compute_dynamic_tradeoff(gdd_val.detach(), lsd_val.detach())
                domain_loss = (1 - alpha) * gdd_val + alpha * lsd_val

                proto_loss = coch_logits.new_tensor(0.0)
                pseudo_proto_loss = coch_logits.new_tensor(0.0)
                if cfg.use_proto_align and cfg.proto_weight > 0:
                    proto_terms = []
                    if cfg.proto_on_fusion:
                        proto_terms.append(
                            cosine_prototype_loss(
                                outputs["fusion_feat"],
                                labels,
                                model.prototypes,
                                temperature=cfg.proto_temperature,
                            )
                        )
                    if cfg.proto_on_mel:
                        proto_terms.append(
                            cosine_prototype_loss(
                                outputs["mel_feat"],
                                labels,
                                model.prototypes,
                                temperature=cfg.proto_temperature,
                            )
                        )
                    if cfg.proto_on_coch:
                        proto_terms.append(
                            cosine_prototype_loss(
                                outputs["coch_feat"],
                                labels,
                                model.prototypes,
                                temperature=cfg.proto_temperature,
                            )
                        )
                    if proto_terms:
                        proto_loss = cfg.proto_weight * sum(proto_terms) / len(proto_terms)

                if (
                    cfg.use_proto_align
                    and cfg.proto_use_pseudo
                    and cfg.proto_pseudo_weight > 0
                    and confident_mask.sum() > 0
                ):
                    pseudo_proto_loss = cosine_prototype_loss(
                        outputs["coch_feat"][confident_mask],
                        pseudo_labels[confident_mask],
                        model.prototypes,
                        temperature=cfg.proto_temperature,
                    )
                    if cfg.proto_pseudo_conf_soft:
                        pseudo_proto_loss = pseudo_proto_loss * confidence[confident_mask].mean().detach()
                    pseudo_proto_loss = cfg.proto_pseudo_weight * pseudo_proto_loss

                distill_loss = coch_logits.new_tensor(0.0)
                if cfg.distill_weight > 0 and epoch > cfg.distill_warmup_epochs:
                    t = cfg.distill_temperature
                    distill_core = (
                        distill_kl(fusion_logits, mel_logits, t)
                        + distill_kl(mel_logits, fusion_logits, t)
                        + distill_kl(fusion_logits, coch_logits, t)
                        + distill_kl(coch_logits, fusion_logits, t)
                    ) * 0.25
                    distill_loss = cfg.distill_weight * distill_core

                total_loss = (
                    src_loss
                    + cfg.beta_tradeoff * domain_loss
                    + cfg.gamma_pseudo * pseudo_loss
                    + cfg.consistency_weight * consistency
                    + proto_loss
                    + pseudo_proto_loss
                    + distill_loss
                )

            optimizer.zero_grad()
            scaled_loss = scaler.scale(total_loss)
            scaled_loss.backward()
            if cfg.max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()

            if cfg.use_proto_align and cfg.proto_use_ema:
                with torch.no_grad():
                    momentum = cfg.proto_momentum
                    for cls in labels.unique():
                        cls_mask = labels == cls
                        if cls_mask.sum() == 0:
                            continue
                        mean_feat = outputs["fusion_feat"][cls_mask].mean(dim=0)
                        model.prototypes[cls] = model.prototypes[cls] * momentum + mean_feat * (1 - momentum)

            running_loss += float(total_loss.item())
            running_src += float(src_loss.item())
            running_pseudo += float(pseudo_loss.item())
            running_domain += float(domain_loss.item())
            running_consistency += float(consistency.item())
            running_proto += float((proto_loss + pseudo_proto_loss).item())
            running_distill += float(distill_loss.item())
            if cfg.use_relia_pseudo:
                running_relia += float(pseudo_loss.item())
            preds = fusion_logits.argmax(dim=1)
            correct += int((preds == labels).sum().item())
            seen += labels.size(0)
            global_step += 1

            if step % cfg.log_interval == 0:
                msg = (
                    f"epoch {epoch} step {step}/{len(train_loader)} "
                    f"loss {total_loss.item():.4f} src {src_loss.item():.4f} "
                    f"domain {domain_loss.item():.4f} pseudo {pseudo_loss.item():.4f} "
                    f"cons {consistency.item():.4f} proto {(proto_loss + pseudo_proto_loss).item():.4f} "
                    f"distill {distill_loss.item():.4f} thr {threshold:.2f} acc {correct / max(1, seen):.4f}"
                )
                print(msg)
                with open(log_file, "a") as f:
                    f.write(msg + "\n")

        train_acc = correct / max(1, seen)
        metrics.update(train_loss=running_loss / max(1, len(train_loader)), train_acc=train_acc)

        val_metrics = evaluate(model, val_loader, device)
        val_line = (
            f"Epoch {epoch} - training loss: {running_loss / max(1, len(train_loader)):.4f}, "
            f"validation loss: {val_metrics['loss']:.4f}\n"
            f"Epoch {epoch} - training accuracy: {train_acc:.4f}, validation accuracy: {val_metrics['acc']:.4f}\n"
            f"Epoch {epoch} - precision: {val_metrics['precision']:.4f}, recall: {val_metrics['recall']:.4f}, "
            f"F1-score: {val_metrics['f1']:.4f}, AUC: {val_metrics['auc']:.4f}"
        )
        print(val_line)
        with open(log_file, "a") as f:
            f.write(val_line + "\n")
        
        if vis_collector is not None:
            vis_collector.collect_prototype_evolution(epoch, model, val_loader, device)
            
            model.eval()
            all_confidence = []
            all_entropy = []
            all_pseudo_labels = []
            all_labels = []
            with torch.no_grad():
                for batch in train_loader:
                    mel = batch["mel"].to(device)
                    coch = batch["coch"].to(device)
                    labels = batch["label"].to(device)
                    outputs = model(mel, coch)
                    coch_logits = outputs["coch_logits"]
                    coch_probs = F.softmax(coch_logits, dim=1)
                    confidence, pseudo_labels = coch_probs.max(dim=1)
                    entropy = -(coch_probs * (coch_probs + 1e-8).log()).sum(dim=1)
                    
                    all_confidence.append(confidence)
                    all_entropy.append(entropy)
                    all_pseudo_labels.append(pseudo_labels)
                    all_labels.append(labels)
            
            if len(all_confidence) > 0:
                all_confidence = torch.cat(all_confidence)
                all_entropy = torch.cat(all_entropy)
                all_pseudo_labels = torch.cat(all_pseudo_labels)
                all_labels = torch.cat(all_labels)
                
                schedule_ratio = 1.0 - (epoch - 1) / max(1, cfg.epochs)
                threshold = max(cfg.pseudo_threshold_min, cfg.pseudo_threshold * schedule_ratio)
                confident_mask = all_confidence >= threshold
                
                vis_collector.collect_pseudo_label_stats(
                    epoch=epoch,
                    confidence=all_confidence,
                    entropy=all_entropy,
                    threshold=threshold,
                    confident_mask=confident_mask,
                    pseudo_labels=all_pseudo_labels,
                    true_labels=all_labels
                )
            model.train()

        if val_metrics["acc"] >= best_acc:
            best_acc = val_metrics["acc"]
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            ckpt_path = os.path.join(
                cfg.out_dir,
                f"musefuse_{cfg.dataset}_{cfg.label_mode}_{ts}.pt",
            )
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": cfg.to_dict(),
                    "val_metrics": val_metrics,
                },
                ckpt_path,
            )
            with open(log_file, "a") as f:
                f.write(f"Model checkpoint saved to {ckpt_path}\n")
        torch.cuda.empty_cache()

    summary = metrics.summary()
    final_line = (
        f"Training completed. Best validation accuracy: {best_acc:.4f}, "
        f"training summary: {json.dumps(summary, ensure_ascii=False)}"
    )
    print(final_line)
    with open(log_file, "a") as f:
        f.write(final_line + "\n")
    
    if vis_collector is not None:
        print("Collecting gating weight statistics on the full dataset...")
        from torch.utils.data import DataLoader, SequentialSampler
        test_loader = DataLoader(
            dataset, 
            batch_size=cfg.batch_size, 
            sampler=SequentialSampler(dataset),
            num_workers=cfg.num_workers,
            pin_memory=True
        )
        vis_collector.collect_gating_weights(model, test_loader, device, max_samples=1000)
        dataset_name = f"{cfg.dataset}_{cfg.label_mode}"
        vis_collector.save(dataset_name)


def build_argparser() -> argparse.ArgumentParser:
    """Build argument parser for MuseFuse Memo2496-only training."""
    parser = argparse.ArgumentParser("Train MuseFuse on Memo2496 Dataset")

    # Dataset & task (fixed dataset)
    parser.add_argument(
        "--label_mode",
        type=str,
        default="a",
        choices=["a", "v"],
        help="Emotion dimension: 'a' for Arousal, 'v' for Valence",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/data/qilin.li/dataset",
        help="Root directory containing Memo2496 dataset",
    )

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)

    # Module weights (all modules enabled by default in config)
    parser.add_argument(
        "--proto_weight",
        type=float,
        default=0.15,
        help="ProtoAlign loss weight",
    )
    parser.add_argument(
        "--distill_weight",
        type=float,
        default=0.2,
        help="TriDistill KL loss weight",
    )

    # Computational
    parser.add_argument(
        "--mixed_precision",
        action="store_true",
        default=True,
        help="Enable mixed-precision training (AMP)",
    )
    parser.add_argument("--num_workers", type=int, default=4)

    # Output
    parser.add_argument("--log_file", type=str, default="musefuse.log")
    parser.add_argument("--out_dir", type=str, default="checkpoints_musefuse")

    # Visualization data collection (optional)
    parser.add_argument(
        "--collect_vis_data",
        action="store_true",
        help="Collect visualization statistics (prototypes, gating, pseudo-labels)",
    )

    return parser


if __name__ == "__main__":
    parser = build_argparser()
    args = parser.parse_args()
    train(args)

