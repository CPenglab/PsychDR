import torch
import torch.nn as nn
import random
from sklearn.metrics import roc_auc_score

class BPRLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, pos_score, neg_score):
        return -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-8).mean()

def sample_neg_pairs_no_leak(pos_dst, all_positive_set, num_drugs, device, neg_ratio=1):
    """严格避免数据泄漏的负采样"""
    neg_src, neg_dst = [], []
    for d in pos_dst.unique().tolist():
        num_pos = (pos_dst == d).sum().item()
        num_neg = int(num_pos * neg_ratio)
        candidates = [drug for drug in range(num_drugs) if (drug, d) not in all_positive_set]
        if not candidates: continue
        num_neg = min(num_neg, len(candidates))
        sampled_drugs = random.sample(candidates, num_neg)
        neg_src.extend(sampled_drugs)
        neg_dst.extend([d] * num_neg)
    return torch.tensor(neg_src, dtype=torch.long, device=device), torch.tensor(neg_dst, dtype=torch.long, device=device)

def train_one_epoch(ddi_edge_train, edge_index_dicts_train, feature_reducer, model, predictor, optimizer,all_positive_set,
                    disease_sim_idx=None, disease_sim_wts=None, gate_c=None,
                    lambda_rank=0.5):
    model.train(); predictor.train(); feature_reducer.train()
    optimizer.zero_grad()
    x_dict_raw = {k: data[k].x.to(device) for k in data.node_types}
    x_dict = feature_reducer(x_dict_raw)
    edge_index_dicts_train_dev = move_edge_index_dicts_to_device(edge_index_dicts_train, device)
    h_dict = model(x_dict, edge_index_dicts_train_dev, disease_sim_idx, disease_sim_wts, gate_c)
    pos_edge = ddi_edge_train.to(device)
    if pos_edge.dim() == 1:
        pos_edge = pos_edge.unsqueeze(1)
    pos_src, pos_dst = pos_edge[0].view(-1), pos_edge[1].view(-1)
    neg_src, neg_dst = sample_neg_pairs_no_leak(pos_dst,all_positive_set,num_drugs=data['drug'].num_nodes,neg_ratio=1)
    pos_score = predictor(h_dict['drug'][pos_src], h_dict['disease'][pos_dst])
    if neg_src.numel() > 0:
        neg_score = predictor(h_dict['drug'][neg_src], h_dict['disease'][neg_dst])
        # ===== BCE loss =====
        labels = torch.cat([
            torch.ones_like(pos_score),
            torch.zeros_like(neg_score)
        ])
        scores = torch.cat([pos_score, neg_score])
        bce_loss = bce_loss_fn(scores, labels)
        # ===== Ranking loss (BPR) =====
        # 对齐正负样本（随机匹配）
        min_len = min(len(pos_score), len(neg_score))
        pos_sample = pos_score[:min_len]
        neg_sample = neg_score[:min_len]
        rank_loss = bpr_loss_fn(pos_sample, neg_sample)
        # ===== 总 loss =====
        loss = bce_loss + lambda_rank * rank_loss
    else:
        loss = bce_loss_fn(pos_score, torch.ones_like(pos_score))
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate_on_set(
        ddi_edge_eval,
        edge_index_dicts_eval,
        feature_reducer,
        model,
        predictor,
        fixed_neg_src,
        fixed_neg_dst,
        disease_sim_idx=None,
        disease_sim_wts=None,
        gate_c=None):
    model.eval()
    predictor.eval()
    feature_reducer.eval()
    x_dict_raw = {
        k: data[k].x.to(device)
        for k in data.node_types
    }
    x_dict = feature_reducer(x_dict_raw)
    edge_index_dicts_eval_dev = move_edge_index_dicts_to_device(
        edge_index_dicts_eval,
        device
    )
    h_dict = model(
        x_dict,
        edge_index_dicts_eval_dev,
        disease_sim_idx,
        disease_sim_wts,
        gate_c
    )
    pos_edge = ddi_edge_eval.to(device)
    pos_src = pos_edge[0].view(-1)
    pos_dst = pos_edge[1].view(-1)
    pos_scores = predictor(
        h_dict['drug'][pos_src],
        h_dict['disease'][pos_dst]
    )
    neg_scores = predictor(
        h_dict['drug'][fixed_neg_src],
        h_dict['disease'][fixed_neg_dst]
    )
    labels = torch.cat([
        torch.ones_like(pos_scores),
        torch.zeros_like(neg_scores)
    ]).cpu().numpy()
    preds = torch.cat([
        pos_scores,
        neg_scores
    ]).cpu().numpy()
    auc = roc_auc_score(labels, preds)
    return auc, None
