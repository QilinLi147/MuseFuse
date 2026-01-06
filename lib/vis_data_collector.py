"""Visualisation data collector for MuseFuse training.

This module records prototype evolution, gating behaviour, and pseudo-label dynamics
for subsequent advanced visual analyses.
"""
import os
import json
import torch
import numpy as np
from typing import Dict, List, Optional
from collections import defaultdict


class VisualizationDataCollector:
    """Collect visualisation-related statistics during training."""
    
    def __init__(self, save_dir: str, collect_interval: int = 5):
        """
        Args:
            save_dir: Output directory for visualisation data.
            collect_interval: Collect prototype-related statistics every N epochs.
        """
        self.save_dir = save_dir
        self.collect_interval = collect_interval
        os.makedirs(save_dir, exist_ok=True)
        
        self.prototype_evolution = {
            'epochs': [],
            'prototypes': [],  # List of [num_classes, hidden_dim] tensors
            'inter_class_cosine_sim': [],  # List of cosine similarities between class pairs
            'prototype_l2_norms': [],  # List of L2 norms per class
            'intra_class_distances': defaultdict(list),  # {epoch: {class: [distances]}}
        }
        
        self.gating_data = {
            'gating_weights': [],  # List of [batch_size, hidden_dim] tensors
            'predictions': [],  # List of predictions
            'labels': [],  # List of ground truth labels
            'confidences': [],  # List of prediction confidences
            'fusion_features': [],  # For Arousal/Valence analysis if needed
        }
        
        self.pseudo_label_data = {
            'epochs': [],
            'confidence_mean': [],
            'confidence_std': [],
            'entropy_mean': [],
            'entropy_std': [],
            'threshold': [],
            'coverage': [],
            'pseudo_quality': defaultdict(dict),  # {epoch: {high_conf: {...}, low_conf: {...}}}
        }
        
    def collect_prototype_evolution(self, epoch: int, model, val_loader, device: torch.device):
        """Collect prototype evolution statistics.

        Data are recorded for epoch 1 and then every ``collect_interval`` epochs thereafter.
        """
        if epoch % self.collect_interval != 0 and epoch != 1:
            return
            
        model.eval()
        prototypes = model.prototypes.detach()
        num_classes = prototypes.size(0)
        
        prototypes_norm = torch.nn.functional.normalize(prototypes, dim=1)
        cosine_sim = torch.matmul(prototypes_norm, prototypes_norm.t())
        
        l2_norms = torch.norm(prototypes, dim=1)
        
        intra_distances = defaultdict(list)
        with torch.no_grad():
            for batch in val_loader:
                mel = batch["mel"].to(device)
                coch = batch["coch"].to(device)
                labels = batch["label"].to(device)
                outputs = model(mel, coch)
                fusion_feat = outputs["fusion_feat"]
                
                fusion_norm = torch.nn.functional.normalize(fusion_feat, dim=1)
                for cls in range(num_classes):
                    cls_mask = labels == cls
                    if cls_mask.sum() == 0:
                        continue
                    cls_feat = fusion_norm[cls_mask]
                    cls_proto = prototypes_norm[cls:cls+1]
                    cosine_sim_samples = torch.matmul(cls_feat, cls_proto.t()).squeeze()
                    cosine_dist = 1.0 - cosine_sim_samples
                    intra_distances[cls].extend(cosine_dist.cpu().tolist())
        
        self.prototype_evolution['epochs'].append(epoch)
        self.prototype_evolution['prototypes'].append(prototypes.cpu().numpy().tolist())
        self.prototype_evolution['inter_class_cosine_sim'].append(cosine_sim.cpu().numpy().tolist())
        self.prototype_evolution['prototype_l2_norms'].append(l2_norms.cpu().numpy().tolist())
        self.prototype_evolution['intra_class_distances'][epoch] = {
            str(cls): dists for cls, dists in intra_distances.items()
        }
        
        model.train()
    
    def collect_gating_weights(self, model, test_loader, device: torch.device, max_samples: int = 1000):
        """Collect gating weights and prediction statistics on the test set."""
        model.eval()
        gating_weights_list = []
        predictions_list = []
        labels_list = []
        confidences_list = []
        
        sample_count = 0
        with torch.no_grad():
            for batch in test_loader:
                if sample_count >= max_samples:
                    break
                mel = batch["mel"].to(device)
                coch = batch["coch"].to(device)
                labels = batch["label"].to(device)
                
                mel_flat = mel.view(mel.size(0), -1)
                coch_flat = coch.view(coch.size(0), -1)
                
                mel_feat = model.mel_encoder(mel_flat)
                coch_feat = model.coch_encoder(coch_flat)
                
                gate = model.gating(torch.cat([mel_feat, coch_feat], dim=-1))
                
                fusion_feat = gate * mel_feat + (1 - gate) * coch_feat
                fusion_logits = model.fusion_head(fusion_feat)
                
                probs = torch.nn.functional.softmax(fusion_logits, dim=1)
                preds = fusion_logits.argmax(dim=1)
                conf = probs.max(dim=1)[0]
                
                batch_size = min(gate.size(0), max_samples - sample_count)
                gating_weights_list.append(gate[:batch_size].cpu().numpy())
                predictions_list.append(preds[:batch_size].cpu().numpy())
                labels_list.append(labels[:batch_size].cpu().numpy())
                confidences_list.append(conf[:batch_size].cpu().numpy())
                
                sample_count += batch_size
        
        self.gating_data['gating_weights'] = np.concatenate(gating_weights_list, axis=0).tolist()
        self.gating_data['predictions'] = np.concatenate(predictions_list, axis=0).tolist()
        self.gating_data['labels'] = np.concatenate(labels_list, axis=0).tolist()
        self.gating_data['confidences'] = np.concatenate(confidences_list, axis=0).tolist()
        
        model.train()
    
    def collect_pseudo_label_stats(
        self, 
        epoch: int, 
        confidence: torch.Tensor, 
        entropy: torch.Tensor,
        threshold: float,
        confident_mask: torch.Tensor,
        pseudo_labels: Optional[torch.Tensor] = None,
        true_labels: Optional[torch.Tensor] = None
    ):
        """Collect pseudo-label statistics once per epoch to avoid duplication."""
        # Skip if the epoch has already been recorded
        if epoch in self.pseudo_label_data['epochs']:
            return
        
        conf_np = confidence.detach().cpu().numpy()
        ent_np = entropy.detach().cpu().numpy()
        confident_mask_cpu = confident_mask.detach().cpu()
        
        self.pseudo_label_data['epochs'].append(epoch)
        self.pseudo_label_data['confidence_mean'].append(float(conf_np.mean()))
        self.pseudo_label_data['confidence_std'].append(float(conf_np.std()))
        self.pseudo_label_data['entropy_mean'].append(float(ent_np.mean()))
        self.pseudo_label_data['entropy_std'].append(float(ent_np.std()))
        self.pseudo_label_data['threshold'].append(threshold)
        self.pseudo_label_data['coverage'].append(float(confident_mask_cpu.sum().item() / max(1, confident_mask_cpu.numel())))
        
        if pseudo_labels is not None and true_labels is not None:
            pseudo_cpu = pseudo_labels.detach().cpu()
            true_cpu = true_labels.detach().cpu()
            confidence_cpu = confidence.detach().cpu()
            
            pseudo_np = pseudo_cpu.numpy()
            true_np = true_cpu.numpy()
            
            high_conf_mask = confidence_cpu >= 0.8
            low_conf_mask = (confidence_cpu >= 0.4) & (confidence_cpu < 0.6)
            
            quality = {}
            if high_conf_mask.sum() > 0:
                high_conf_mask_np = high_conf_mask.numpy()
                high_pseudo = pseudo_np[high_conf_mask_np]
                high_true = true_np[high_conf_mask_np]
                quality['high_conf'] = {
                    'accuracy': float((high_pseudo == high_true).mean()),
                    'count': int(high_conf_mask.sum().item())
                }
            
            if low_conf_mask.sum() > 0:
                low_conf_mask_np = low_conf_mask.numpy()
                low_pseudo = pseudo_np[low_conf_mask_np]
                low_true = true_np[low_conf_mask_np]
                quality['low_conf'] = {
                    'accuracy': float((low_pseudo == low_true).mean()),
                    'count': int(low_conf_mask.sum().item())
                }
            
            self.pseudo_label_data['pseudo_quality'][epoch] = quality
    
    def save(self, dataset_name: str):
        """Persist all collected visualisation statistics to a JSON file."""
        save_path = os.path.join(self.save_dir, f"vis_data_{dataset_name}.json")
        
        data = {
            'prototype_evolution': {
                'epochs': self.prototype_evolution['epochs'],
                'prototypes': self.prototype_evolution['prototypes'],
                'inter_class_cosine_sim': self.prototype_evolution['inter_class_cosine_sim'],
                'prototype_l2_norms': self.prototype_evolution['prototype_l2_norms'],
                'intra_class_distances': {
                    str(epoch): dists 
                    for epoch, dists in self.prototype_evolution['intra_class_distances'].items()
                }
            },
            'gating_data': self.gating_data,
            'pseudo_label_data': {
                'epochs': self.pseudo_label_data['epochs'],
                'confidence_mean': self.pseudo_label_data['confidence_mean'],
                'confidence_std': self.pseudo_label_data['confidence_std'],
                'entropy_mean': self.pseudo_label_data['entropy_mean'],
                'entropy_std': self.pseudo_label_data['entropy_std'],
                'threshold': self.pseudo_label_data['threshold'],
                'coverage': self.pseudo_label_data['coverage'],
                'pseudo_quality': {
                    str(epoch): quality 
                    for epoch, quality in self.pseudo_label_data['pseudo_quality'].items()
                }
            }
        }
        
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Visualisation data saved to: {save_path}")
        return save_path

